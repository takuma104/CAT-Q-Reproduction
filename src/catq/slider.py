"""Sliding-layer ternarization optimization (CAT-Q Section 2.4 + SliderQuant).

Window schedule follows SliderQuant (docs/papers/slider-quant-paper.md, §3.2):
  - PESW over the Ls shallow layers: windows [0], [0,1], ..., [0..Ls-1] with
    layer 0 as anchor; layers 0..Ls-2 are finalized after the last expansion.
  - FSSW {s=window_size, i=1} over intermediate layers, one overlapped layer
    at each boundary; the window head is finalized each step.
  - PCSW over the Ld deep layers: windows [n-Ld..n-1], ..., [n-1] with the
    last layer as anchor; the first window layer is finalized each step.

Remaining interpretation notes (not fully specified across the two papers):
  - The target is F(W, X) with pristine FP weights on the same X coming from
    already-quantized upstream layers (Eq. 7 uses a single X for both terms).
  - The 60 calibration epochs run per window position. A layer's ST time
    state t advances over its whole participation lifetime (all windows that
    contain it), reaching t=1 exactly when it is finalized. Re-annealing a
    warm-started layer from t=0 each window destroys its hard-stage solution
    (verified empirically: held-out perplexity exploded), so t never resets.
  - SliderQuant's intra-layer sliding (gamma=0.5, N=2) is not implemented; its
    coupling with ST's annealing schedule is unclear from the papers.
"""

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from catq.config import CATQConfig
from catq.module import CATQLinear

logger = logging.getLogger(__name__)


@dataclass
class WindowLog:
    window_start: int
    layers: list[int]
    finalized: list[int]
    # Hard-ternarization loss before this window's optimization; for the very
    # first window this equals an absmean-style static baseline.
    hard_init_loss: float
    initial_loss: float
    final_loss: float
    layer_stats: dict[str, dict[str, float]] = field(default_factory=dict)


def build_schedule(n_layers: int, config: CATQConfig) -> list[tuple[list[int], list[int]]]:
    """Returns [(window_layer_indices, finalize_indices), ...]."""
    s = config.window_size
    if config.schedule == "fixed" or n_layers < config.shallow_layers + config.deep_layers + 2:
        return [
            (list(range(i, min(i + s, n_layers))), [i]) for i in range(n_layers)
        ]

    ls, ld = config.shallow_layers, config.deep_layers
    schedule: list[tuple[list[int], list[int]]] = []
    # PESW: expand from [0] to [0..ls-1]; finalize 0..ls-2 after the last step,
    # leaving layer ls-1 as the overlapped layer with the intermediate phase.
    for k in range(1, ls + 1):
        finalize = list(range(ls - 1)) if k == ls else []
        schedule.append((list(range(k)), finalize))
    # FSSW: [i, i+s-1] for i in [ls-1, n-ld-1]; the last window overlaps the
    # deep phase by one layer (n-ld).
    for i in range(ls - 1, n_layers - ld):
        schedule.append((list(range(i, min(i + s, n_layers))), [i]))
    # PCSW: contract from [n-ld..n-1] to [n-1], finalizing the head each step.
    for i in range(n_layers - ld, n_layers):
        schedule.append((list(range(i, n_layers)), [i]))
    return schedule


def window_loss_fn(out: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    out, target = out.float(), target.float()
    if kind == "token_rms":
        norm = target.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return torch.nn.functional.mse_loss(out / norm, target / norm)
    return torch.nn.functional.mse_loss(out, target)


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

    def _finalize_layer(self, idx: int) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for name, module in self.wrapped.pop(idx).items():
            linear, module_stats = module.finalize()
            _set_submodule(self.layers[idx], name, linear)
            stats[f"layer{idx}.{name}"] = module_stats
        return stats

    def _set_time(self, layer_indices: list[int], t: float) -> None:
        for idx in layer_indices:
            for module in self.wrapped[idx].values():
                module.t = t

    def _set_fp_mode(self, layer_indices: list[int], fp_mode: bool) -> None:
        for idx in layer_indices:
            for module in self.wrapped[idx].values():
                module.fp_mode = fp_mode

    def _window_params(self, layer_indices: list[int]) -> list[dict[str, object]]:
        cfg = self.config
        factors: list[nn.Parameter] = []
        lora: list[nn.Parameter] = []
        for idx in layer_indices:
            for m in self.wrapped[idx].values():
                factors += [m.rho_mu, m.rho_alpha, m.rho_delta]
                if m.cs_scale is not None:
                    factors.append(m.cs_scale)
                if m.lora_A is not None:
                    lora += [m.lora_A, m.lora_B]
        groups: list[dict[str, object]] = [{"params": factors, "lr": cfg.lr}]
        if lora:
            groups.append({"params": lora, "lr": cfg.lora_lr})
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
    def _compute_targets(
        self,
        layer_indices: list[int],
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """FP window outputs on the same X: pristine pretrained weights."""
        self._set_fp_mode(layer_indices, True)
        target = torch.empty_like(hidden)
        for i in range(0, hidden.shape[0], self.config.batch_size):
            batch = hidden[i : i + self.config.batch_size]
            target[i : i + batch.shape[0]] = self._forward_window(
                layer_indices, batch, position_embeddings
            )
        self._set_fp_mode(layer_indices, False)
        return target

    @torch.no_grad()
    def _window_loss(
        self,
        layer_indices: list[int],
        hidden: torch.Tensor,
        target: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        t: float,
    ) -> float:
        self._set_time(layer_indices, t)
        total = 0.0
        for i in range(0, hidden.shape[0], self.config.batch_size):
            batch = hidden[i : i + self.config.batch_size]
            out = self._forward_window(layer_indices, batch, position_embeddings)
            total += window_loss_fn(
                out, target[i : i + batch.shape[0]], self.config.loss
            ).item() * batch.shape[0]
        return total / hidden.shape[0]

    # ----------------------------------------------------------------- run

    def run(self, input_ids: torch.Tensor) -> list[WindowLog]:
        cfg = self.config
        n_layers = len(self.layers)
        schedule = build_schedule(n_layers, cfg)
        hidden, position_embeddings = self._build_inputs(input_ids)
        num_samples = hidden.shape[0]
        logs: list[WindowLog] = []

        # Per-layer lifetime epoch budget: t advances across all windows that
        # contain the layer and hits 1.0 in its last window (see module note).
        total_epochs = {
            idx: cfg.epochs * sum(idx in layers for layers, _ in schedule)
            for idx in range(n_layers)
        }
        done_epochs = {idx: 0 for idx in range(n_layers)}

        for w, (layer_indices, finalize_indices) in enumerate(schedule):
            for idx in layer_indices:
                self._wrap_layer(idx)

            target = self._compute_targets(layer_indices, hidden, position_embeddings)
            hard_init_loss = self._window_loss(
                layer_indices, hidden, target, position_embeddings, t=1.0
            )

            optimizer = torch.optim.AdamW(self._window_params(layer_indices))
            steps_per_epoch = (num_samples + cfg.batch_size - 1) // cfg.batch_size
            total_steps = cfg.epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 1.0 - step / total_steps
            )

            generator = torch.Generator().manual_seed(cfg.seed + w)
            initial_loss = final_loss = float("nan")
            for epoch in range(1, cfg.epochs + 1):
                for idx in layer_indices:
                    t = (done_epochs[idx] + epoch) / total_epochs[idx]
                    for module in self.wrapped[idx].values():
                        module.t = t
                perm = torch.randperm(num_samples, generator=generator)
                epoch_loss = 0.0
                for i in range(0, num_samples, cfg.batch_size):
                    idx = perm[i : i + cfg.batch_size]
                    out = self._forward_window(layer_indices, hidden[idx], position_embeddings)
                    loss = window_loss_fn(out, target[idx], cfg.loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item() * idx.shape[0]
                epoch_loss /= num_samples
                if epoch == 1:
                    initial_loss = epoch_loss
                final_loss = epoch_loss
                if epoch % 20 == 0 or epoch == 1:
                    logger.info(
                        "window %d/%d layers=%s epoch %d/%d loss=%.6f",
                        w + 1, len(schedule), layer_indices, epoch, cfg.epochs, epoch_loss,
                    )

            for idx in layer_indices:
                done_epochs[idx] += cfg.epochs

            layer_stats: dict[str, dict[str, float]] = {}
            for idx in finalize_indices:
                layer_stats.update(self._finalize_layer(idx))
                with torch.no_grad():
                    for i in range(0, num_samples, cfg.batch_size):
                        batch = hidden[i : i + cfg.batch_size]
                        hidden[i : i + batch.shape[0]] = self.layers[idx](
                            batch, attention_mask=None, position_embeddings=position_embeddings
                        )

            logs.append(
                WindowLog(
                    window_start=layer_indices[0],
                    layers=layer_indices,
                    finalized=finalize_indices,
                    hard_init_loss=hard_init_loss,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    layer_stats=layer_stats,
                )
            )
            logger.info(
                "window %d/%d done: finalized %s (hard-init %.6f, epoch loss %.6f -> %.6f)",
                w + 1, len(schedule), finalize_indices or "none", hard_init_loss, initial_loss, final_loss,
            )

        assert not self.wrapped, "all layers should be finalized"
        return logs
