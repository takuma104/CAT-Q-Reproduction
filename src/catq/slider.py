"""Sliding-layer ternarization optimization (paper Section 2.4).

Interpretation notes (details the paper defers to SliderQuant, unspecified there):
  - Window of `window_size` decoder layers, stride 1. After optimizing a window
    the head layer is finalized (hard-ternarized and baked); trailing layers keep
    their learned factors as warm start for the next window.
  - Window input X is the output of the already-quantized upstream layers, and
    the target is F(W, X) with the window's original FP weights on the same X
    (Eq. 7 uses a single X for both terms).
  - The 60 calibration epochs run per window position, each traversing the full
    ST schedule t in (0, 1].
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
    # Hard-ternarization loss with freshly initialized factors (delta_alpha =
    # delta_delta = 1, delta_mu = 0), i.e. an absmean-style static baseline.
    hard_init_loss: float
    initial_loss: float
    final_loss: float
    layer_stats: dict[str, dict[str, float]] = field(default_factory=dict)


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
                linear, group_size=cfg.group_size, delta0=cfg.delta0, s0=cfg.s0, gamma=cfg.gamma
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
        """FP window outputs on the same X: run wrapped layers at t=0 (identity)."""
        self._set_time(layer_indices, 0.0)
        target = torch.empty_like(hidden)
        for i in range(0, hidden.shape[0], self.config.batch_size):
            batch = hidden[i : i + self.config.batch_size]
            target[i : i + batch.shape[0]] = self._forward_window(
                layer_indices, batch, position_embeddings
            )
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
            total += torch.nn.functional.mse_loss(
                out.float(), target[i : i + batch.shape[0]].float()
            ).item() * batch.shape[0]
        return total / hidden.shape[0]

    # ----------------------------------------------------------------- run

    def run(self, input_ids: torch.Tensor) -> list[WindowLog]:
        cfg = self.config
        n_layers = len(self.layers)
        hidden, position_embeddings = self._build_inputs(input_ids)
        num_samples = hidden.shape[0]
        logs: list[WindowLog] = []

        for start in range(n_layers):
            layer_indices = list(range(start, min(start + cfg.window_size, n_layers)))
            for idx in layer_indices:
                self._wrap_layer(idx)

            target = self._compute_targets(layer_indices, hidden, position_embeddings)
            hard_init_loss = self._window_loss(
                layer_indices, hidden, target, position_embeddings, t=1.0
            )

            params = [
                p for idx in layer_indices for m in self.wrapped[idx].values() for p in m.parameters()
            ]
            optimizer = torch.optim.AdamW(params, lr=cfg.lr)
            steps_per_epoch = (num_samples + cfg.batch_size - 1) // cfg.batch_size
            total_steps = cfg.epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 1.0 - step / total_steps
            )

            generator = torch.Generator().manual_seed(cfg.seed + start)
            initial_loss = final_loss = float("nan")
            for epoch in range(1, cfg.epochs + 1):
                self._set_time(layer_indices, epoch / cfg.epochs)
                perm = torch.randperm(num_samples, generator=generator)
                epoch_loss = 0.0
                for i in range(0, num_samples, cfg.batch_size):
                    idx = perm[i : i + cfg.batch_size]
                    out = self._forward_window(layer_indices, hidden[idx], position_embeddings)
                    loss = torch.nn.functional.mse_loss(
                        out.float(), target[idx].float()
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item() * idx.shape[0]
                epoch_loss /= num_samples
                if epoch == 1:
                    initial_loss = epoch_loss
                final_loss = epoch_loss
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        "window %d/%d layers=%s epoch %d/%d loss=%.6f",
                        start + 1, n_layers, layer_indices, epoch, cfg.epochs, epoch_loss,
                    )

            layer_stats = self._finalize_layer(start)
            with torch.no_grad():
                for i in range(0, num_samples, cfg.batch_size):
                    batch = hidden[i : i + cfg.batch_size]
                    hidden[i : i + batch.shape[0]] = self.layers[start](
                        batch, attention_mask=None, position_embeddings=position_embeddings
                    )

            logs.append(
                WindowLog(
                    window_start=start,
                    layers=layer_indices,
                    hard_init_loss=hard_init_loss,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    layer_stats=layer_stats,
                )
            )
            logger.info(
                "finalized layer %d (hard-init %.6f, epoch loss %.6f -> %.6f)",
                start, hard_init_loss, initial_loss, final_loss,
            )

        assert not self.wrapped, "all layers should be finalized"
        return logs
