"""Optimiser, parameter groups and the learning-rate schedule.

Two things here are easy to get quietly wrong, so both are explicit and both
are tested.

Weight decay grouping
---------------------
Decay is applied to the transformer's weight matrices only. LayerNorm
parameters, biases and the embeddings are excluded: decaying a LayerNorm gain
toward zero fights the normalisation it exists to perform, and decaying
embeddings shrinks rare-token vectors that are already undertrained. Because
the LM head is tied to the token embedding, excluding embeddings also excludes
the output projection -- deliberate, and logged so it is auditable.

Schedule
--------
Linear warmup over the first `warmup_fraction` of steps, then cosine decay to
`min_lr_fraction` of the peak. Warmup starts at step 1, not step 0, so the very
first update is not a no-op.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NO_DECAY_MARKERS = ("ln_", "layernorm", "emb")


def split_parameter_groups(model: nn.Module, weight_decay: float) -> tuple[list[dict], dict]:
    """Return (param_groups, summary) with decay and no-decay separated."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))

        lowered = name.lower()
        is_bias = name.endswith(".bias")
        is_norm_or_emb = any(marker in lowered for marker in NO_DECAY_MARKERS)

        if is_bias or is_norm_or_emb or param.ndim < 2:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)
            decay_names.append(name)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    summary = {
        "decay_tensors": len(decay),
        "decay_parameters": sum(p.numel() for p in decay),
        "no_decay_tensors": len(no_decay),
        "no_decay_parameters": sum(p.numel() for p in no_decay),
        "decay_names": decay_names,
        "no_decay_names": no_decay_names,
    }
    return groups, summary


def build_optimizer(model: nn.Module, cfg) -> tuple[torch.optim.Optimizer, dict]:
    groups, summary = split_parameter_groups(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
    )
    return optimizer, summary


class CosineWithWarmup:
    """Learning-rate schedule, stepped manually so its state is one integer.

    A torch LRScheduler would work too, but keeping the rule explicit means the
    checkpoint stores a step counter rather than an opaque object, and the
    schedule can be recomputed for any step without replaying the run.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        max_steps: int,
        warmup_fraction: float = 0.05,
        min_lr_fraction: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.max_steps = max_steps
        self.warmup_steps = max(1, int(max_steps * warmup_fraction))
        self.min_lr = peak_lr * min_lr_fraction
        self.last_lr = 0.0

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.peak_lr - self.min_lr)

    def set_step(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def state_dict(self) -> dict:
        return {
            "peak_lr": self.peak_lr,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.peak_lr = state["peak_lr"]
        self.max_steps = state["max_steps"]
        self.warmup_steps = state["warmup_steps"]
        self.min_lr = state["min_lr"]


def make_grad_scaler(device: str, enabled: bool):
    """fp16 GradScaler, tolerant of both the old and new torch APIs.

    The T4 is Turing, so bf16 is unavailable and fp16 is the only mixed-precision
    option. fp16 gradients underflow to zero without loss scaling, so the scaler
    is mandatory rather than an optimisation.
    """
    use = enabled and device.startswith("cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use)


def autocast_context(device: str, enabled: bool):
    if not (enabled and device.startswith("cuda")):
        from contextlib import nullcontext

        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)
