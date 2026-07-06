"""Sliding-layer ternarization: CAT-Q quantizer on the SliderQuant framework.

Mechanics follow the SliderQuant reference implementation
(docs/reference_impl/SliderQuant, W2A16 config) with CAT-Q's LM+ST quantizer
replacing the uniform LWC quantizer:

  - Window schedule: progressive expansion over the first `fill_window_size`
    layers ([0], [0,1], ...), fixed {window_size, stride} windows over the
    middle, progressive contraction over the last `fill_window_size` layers.
  - Dual streams: a full-precision stream and a quantized stream are advanced
    separately. Each window's target is the FP teacher output on the FP
    stream, while the student consumes the quantized stream — so every window
    actively corrects accumulated quantization error (reference
    `use_base_loss="last"`).
  - Intra-layer sliding: the whole schedule runs once per quant rate in
    `quant_rates` (partial ternarization of the first fraction of input
    channels), with `epochs` split evenly across the passes. Streams reset to
    the embeddings at the start of each pass.
  - Layers are never finalized mid-run; every CATQLinear stays live (warm
    parameters) and all layers are hard-ternarized and baked at the end.
  - No channel scaling for weight-only quantization (reference uses
    quant_mode=lora_only with scale_lr=0 for W2A16), and LR groups are scaled
    by the batch size (reference `lr_factor`).

CAT-Q specifics kept from the paper: LM factors + ST transition per group,
ST time state t advancing over each layer's whole participation lifetime
(all windows in all passes), reaching t=1 in its final window.
"""

import logging
import math
from dataclasses import dataclass, field

import torch
from torch import nn

from catq.config import CATQConfig
from catq.module import CATQLinear

logger = logging.getLogger(__name__)


@dataclass
class WindowLog:
    pass_index: int
    quant_rate: float
    window_start: int
    layers: list[int]
    hard_init_loss: float
    initial_loss: float
    final_loss: float


@dataclass
class RunResult:
    windows: list[WindowLog]
    layer_stats: dict[str, dict[str, float]] = field(default_factory=dict)


def window_loss_fn(out: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    out, target = out.float(), target.float()
    if kind == "token_rms":
        norm = target.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return torch.nn.functional.mse_loss(out / norm, target / norm)
    return torch.nn.functional.mse_loss(out, target)


def build_schedule(n_layers: int, config: CATQConfig) -> list[list[int]]:
    """Window layer lists, per the reference `fill_window_size` scheduler."""
    s, i, fill = config.window_size, config.stride, config.fill_window_size
    if n_layers < 2 * fill + s:
        windows = []
        start = 0
        while True:
            windows.append(list(range(start, min(start + s, n_layers))))
            if start + s >= n_layers:
                break
            start += i
        return windows

    start_windows = [list(range(k + 1)) for k in range(fill)]
    end_windows = [list(range(n_layers - fill + k, n_layers)) for k in range(fill)]
    start_len = fill - i
    end_len = fill - i
    mid_len = n_layers - start_len - end_len
    mid_round = math.ceil((mid_len - s) / i) + 1
    mid_windows = [
        list(range(r * i + start_len, min(r * i + s + start_len, n_layers)))
        for r in range(mid_round)
    ]
    return start_windows + mid_windows + end_windows


def _target_linears(layer: nn.Module, suffixes: tuple[str, ...]) -> dict[str, nn.Linear]:
    found: dict[str, nn.Linear] = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in suffixes:
            found[name] = module
    return found


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


class SlidingWindowQuantizer:
    """Runs CAT-Q over the decoder layers of a HF causal LM."""

    def __init__(self, model: nn.Module, config: CATQConfig) -> None:
        self.model = model
        self.config = config
        self.layers: nn.ModuleList = model.model.layers
        self.device = torch.device(config.device)
        self.dtype = model.dtype
        # layer_idx -> {module_name: CATQLinear}
        self.wrapped: dict[int, dict[str, CATQLinear]] = {}

    # ---------------------------------------------------------------- setup

    @torch.no_grad()
    def _build_inputs(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Embed all calibration samples and precompute rotary embeddings.

        Sequences are full-length without padding, so attention_mask=None makes
        SDPA use is_causal=True, and cos/sin (batch dim 1) broadcast over batch.
        """
        embed = self.model.model.embed_tokens
        hidden = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], self.model.config.hidden_size),
            dtype=self.dtype, device=self.device,
        )
        for i in range(0, input_ids.shape[0], self.config.batch_size):
            ids = input_ids[i : i + self.config.batch_size].to(self.device)
            hidden[i : i + ids.shape[0]] = embed(ids)

        position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(hidden[:1], position_ids)
        return hidden, position_embeddings

    def _wrap_layer(self, idx: int) -> None:
        if idx in self.wrapped:
            return
        cfg = self.config
        wrapped: dict[str, CATQLinear] = {}
        for name, linear in _target_linears(self.layers[idx], cfg.target_suffixes).items():
            module = CATQLinear(
                linear,
                group_size=cfg.group_size,
                delta0=cfg.delta0,
                s0=cfg.s0,
                gamma=cfg.gamma,
                lora_rank=cfg.lora_rank,
                cs_enabled=cfg.cs_enabled,
            ).to(self.device)
            _set_submodule(self.layers[idx], name, module)
            wrapped[name] = module
        self.wrapped[idx] = wrapped

    def _all_modules(self) -> list[CATQLinear]:
        return [m for mods in self.wrapped.values() for m in mods.values()]

    def _set_fp_mode(self, layer_indices: list[int], fp_mode: bool) -> None:
        for idx in layer_indices:
            if idx in self.wrapped:
                for module in self.wrapped[idx].values():
                    module.fp_mode = fp_mode

    def _window_params(self, layer_indices: list[int]) -> list[dict[str, object]]:
        cfg = self.config
        lr_factor = cfg.batch_size if cfg.scale_lr_by_batch else 1
        factors: list[nn.Parameter] = []
        lora: list[nn.Parameter] = []
        for idx in layer_indices:
            for m in self.wrapped[idx].values():
                factors += [m.rho_mu, m.rho_alpha, m.rho_delta]
                if m.cs_scale is not None:
                    factors.append(m.cs_scale)
                if m.lora_A is not None:
                    lora += [m.lora_A, m.lora_B]
        groups: list[dict[str, object]] = [
            {"params": factors, "lr": cfg.lr * lr_factor, "weight_decay": 0.0}
        ]
        if lora:
            groups.append({"params": lora, "lr": cfg.lora_lr * lr_factor, "weight_decay": 0.0})
        return groups

    # ------------------------------------------------------------- forward

    def _forward_window(
        self,
        layer_indices: list[int],
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        for idx in layer_indices:
            hidden = self.layers[idx](
                hidden, attention_mask=None, position_embeddings=position_embeddings
            )
        return hidden

    @torch.no_grad()
    def _forward_window_batched(
        self,
        layer_indices: list[int],
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        fp_mode: bool = False,
        inplace: bool = False,
    ) -> torch.Tensor:
        """Batched no-grad window forward; fp_mode selects the FP teacher.

        With inplace=True the result overwrites `hidden` batch by batch
        (used for stream advancement, avoiding a full-size extra buffer).
        """
        if fp_mode:
            self._set_fp_mode(layer_indices, True)
        out = hidden if inplace else torch.empty_like(hidden)
        for i in range(0, hidden.shape[0], self.config.batch_size):
            batch = hidden[i : i + self.config.batch_size]
            out[i : i + batch.shape[0]] = self._forward_window(
                layer_indices, batch, position_embeddings
            )
        if fp_mode:
            self._set_fp_mode(layer_indices, False)
        return out

    @torch.no_grad()
    def _window_loss(
        self,
        layer_indices: list[int],
        hidden: torch.Tensor,
        target: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        t: float | None = None,
    ) -> float:
        saved = [(m, m.t) for idx in layer_indices for m in self.wrapped[idx].values()]
        if t is not None:
            for m, _ in saved:
                m.t = t
        total = 0.0
        for i in range(0, hidden.shape[0], self.config.batch_size):
            batch = hidden[i : i + self.config.batch_size]
            out = self._forward_window(layer_indices, batch, position_embeddings)
            total += window_loss_fn(
                out, target[i : i + batch.shape[0]], self.config.loss
            ).item() * batch.shape[0]
        for m, t_saved in saved:
            m.t = t_saved
        return total / hidden.shape[0]

    # ----------------------------------------------------------------- run

    def run(self, input_ids: torch.Tensor) -> RunResult:
        cfg = self.config
        n_layers = len(self.layers)
        schedule = build_schedule(n_layers, cfg)
        num_passes = len(cfg.quant_rates)
        pass_epochs = max(cfg.epochs // num_passes, 1)
        num_samples = input_ids.shape[0]
        logs: list[WindowLog] = []

        # Lifetime ST schedule: t advances across all windows (in all passes)
        # containing the layer and reaches 1.0 in its final window.
        windows_per_layer = {
            idx: sum(idx in layers for layers in schedule) for idx in range(n_layers)
        }
        total_epochs = {
            idx: pass_epochs * num_passes * windows_per_layer[idx] for idx in range(n_layers)
        }
        done_epochs = {idx: 0 for idx in range(n_layers)}

        # Main passes anneal t over each layer's lifetime; the optional polish
        # pass re-runs the schedule with t pinned at 1.0 (hard STE).
        pass_specs: list[tuple[float, int, bool]] = [
            (rate, pass_epochs, True) for rate in cfg.quant_rates
        ]
        if cfg.polish_epochs > 0:
            pass_specs.append((1.0, cfg.polish_epochs, False))

        for pass_index, (quant_rate, cur_epochs, anneal) in enumerate(pass_specs):
            hidden_q, position_embeddings = self._build_inputs(input_ids)
            hidden_fp = hidden_q.clone()
            for m in self._all_modules():
                m.set_quant_rate(quant_rate)

            for w, layer_indices in enumerate(schedule):
                for idx in layer_indices:
                    self._wrap_layer(idx)
                    for m in self.wrapped[idx].values():
                        m.set_quant_rate(quant_rate)

                # Target: FP teacher on the FP stream (error-correcting objective).
                target = self._forward_window_batched(
                    layer_indices, hidden_fp, position_embeddings, fp_mode=True
                )
                hard_init_loss = self._window_loss(
                    layer_indices, hidden_q, target, position_embeddings, t=1.0
                )

                optimizer = torch.optim.AdamW(self._window_params(layer_indices))
                steps_per_epoch = (num_samples + cfg.batch_size - 1) // cfg.batch_size
                total_steps = cur_epochs * steps_per_epoch
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda step: 1.0 - step / total_steps
                )

                generator = torch.Generator().manual_seed(cfg.seed + pass_index * 1000 + w)
                initial_loss = final_loss = float("nan")
                for epoch in range(1, cur_epochs + 1):
                    for idx in layer_indices:
                        t = (done_epochs[idx] + epoch) / total_epochs[idx] if anneal else 1.0
                        for module in self.wrapped[idx].values():
                            module.t = t
                    perm = torch.randperm(num_samples, generator=generator)
                    epoch_loss = 0.0
                    for i in range(0, num_samples, cfg.batch_size):
                        idx_batch = perm[i : i + cfg.batch_size]
                        out = self._forward_window(
                            layer_indices, hidden_q[idx_batch], position_embeddings
                        )
                        loss = window_loss_fn(out, target[idx_batch], cfg.loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        epoch_loss += loss.item() * idx_batch.shape[0]
                    epoch_loss /= num_samples
                    if epoch == 1:
                        initial_loss = epoch_loss
                    final_loss = epoch_loss
                    if epoch % 10 == 0 or epoch == 1:
                        logger.info(
                            "pass %d/%d (rate %.2f) window %d/%d layers=[%d..%d] epoch %d/%d loss=%.6f",
                            pass_index + 1, len(pass_specs), quant_rate, w + 1, len(schedule),
                            layer_indices[0], layer_indices[-1], epoch, cur_epochs, epoch_loss,
                        )

                if anneal:
                    for idx in layer_indices:
                        done_epochs[idx] += pass_epochs

                # Advance both streams up to the next window's start.
                next_start = schedule[w + 1][0] if w + 1 < len(schedule) else layer_indices[0]
                advance = list(range(layer_indices[0], next_start))
                if advance:
                    self._forward_window_batched(
                        advance, hidden_fp, position_embeddings, fp_mode=True, inplace=True
                    )
                    self._forward_window_batched(
                        advance, hidden_q, position_embeddings, inplace=True
                    )
                del target
                torch.cuda.empty_cache()

                logs.append(
                    WindowLog(
                        pass_index=pass_index,
                        quant_rate=quant_rate,
                        window_start=layer_indices[0],
                        layers=layer_indices,
                        hard_init_loss=hard_init_loss,
                        initial_loss=initial_loss,
                        final_loss=final_loss,
                    )
                )
                logger.info(
                    "pass %d window %d/%d done (hard-init %.6f, epoch loss %.6f -> %.6f)",
                    pass_index + 1, w + 1, len(schedule), hard_init_loss, initial_loss, final_loss,
                )

            del hidden_q, hidden_fp
            torch.cuda.empty_cache()

        # Bake: hard-ternarize every wrapped linear with its final parameters.
        layer_stats: dict[str, dict[str, float]] = {}
        for idx in sorted(self.wrapped):
            assert done_epochs[idx] == total_epochs[idx]
            for name, module in self.wrapped[idx].items():
                linear, stats = module.finalize()
                _set_submodule(self.layers[idx], name, linear)
                layer_stats[f"layer{idx}.{name}"] = stats
        self.wrapped.clear()
        logger.info("baked %d ternary linears", len(layer_stats))
        return RunResult(windows=logs, layer_stats=layer_stats)
