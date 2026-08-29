"""Random initialisation -- and the tools that prove it happened.

Every weight in MAX-LM is sampled here, from a normal distribution, at the
moment the model is constructed. Nothing is loaded from disk and nothing is
downloaded. The three functions below produce the evidence for that claim:

  init_weights()      the sampling itself
  weight_statistics() mean/std/min/max per parameter group
  expected_init_loss()  ln(vocab_size) -- what a model that knows nothing scores
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

RESIDUAL_PROJECTIONS = ("attn.proj.weight", "mlp.proj.weight")


def init_weights(model: nn.Module, std: float = 0.02, n_layers: int = 1,
                 scale_residual: bool = True) -> None:
    """Initialise every parameter. The GPT-2 recipe.

    Linear and Embedding weights are drawn from N(0, std); biases are zeroed;
    LayerNorm starts as the identity (weight 1, bias 0).

    Residual projections -- the last matrix in each attention and each
    feed-forward block -- are scaled by 1/sqrt(2 * n_layers). Each block adds
    two contributions to the residual stream, so without this the variance of
    the stream grows with depth and deep models start unstable.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    if scale_residual:
        scaled_std = std / math.sqrt(2 * n_layers)
        for name, param in model.named_parameters():
            if name.endswith(RESIDUAL_PROJECTIONS):
                nn.init.normal_(param, mean=0.0, std=scaled_std)


@torch.no_grad()
def weight_statistics(model: nn.Module) -> dict[str, dict[str, float]]:
    """Per-group weight statistics, for the Review 1 histogram.

    Groups rather than individual tensors, because "embeddings: mean -0.0001,
    std 0.0200" is a claim an examiner can check at a glance.
    """
    groups: dict[str, list[torch.Tensor]] = {
        "embeddings": [],
        "attention": [],
        "feedforward": [],
        "layernorm": [],
        "residual_projections": [],
    }

    for name, param in model.named_parameters():
        flat = param.detach().float().reshape(-1)
        if name.endswith(RESIDUAL_PROJECTIONS):
            groups["residual_projections"].append(flat)
        elif "emb" in name:
            groups["embeddings"].append(flat)
        elif "ln_" in name or "ln_f" in name:
            groups["layernorm"].append(flat)
        elif "attn" in name:
            groups["attention"].append(flat)
        elif "mlp" in name:
            groups["feedforward"].append(flat)

    stats: dict[str, dict[str, float]] = {}
    for group, tensors in groups.items():
        if not tensors:
            continue
        values = torch.cat(tensors)
        stats[group] = {
            "count": int(values.numel()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    allw = torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])
    stats["ALL"] = {
        "count": int(allw.numel()),
        "mean": float(allw.mean()),
        "std": float(allw.std()),
        "min": float(allw.min()),
        "max": float(allw.max()),
    }
    return stats


def expected_init_loss(vocab_size: int) -> float:
    """Cross-entropy of a uniform distribution over `vocab_size` tokens.

    THE RANDOM-INITIALISATION PROOF.

    A model that has learned nothing spreads probability evenly across the
    vocabulary. Cross-entropy under a uniform distribution over V outcomes is
    exactly -ln(1/V) = ln(V).

        ln(4096) = 8.3178
        ln(8192) = 9.0109

    A model carrying pretrained weights starts far below this, because it
    already knows which tokens are likely. Measuring step-0 loss and finding it
    at ln(V) is therefore direct evidence that the weights were sampled here and
    nowhere else -- and it costs a single forward pass to produce.
    """
    return math.log(vocab_size)


@torch.no_grad()
def histogram_data(model: nn.Module, bins: int = 80,
                   limit: int = 2_000_000) -> tuple[list[float], list[int]]:
    """Bin edges and counts for the initial-weight histogram."""
    values = torch.cat([
        p.detach().float().reshape(-1)
        for name, p in model.named_parameters()
        if "ln_" not in name  # LayerNorm starts at exactly 1.0 and swamps the plot
    ])
    if values.numel() > limit:
        idx = torch.randperm(values.numel())[:limit]
        values = values[idx]
    counts = torch.histc(values, bins=bins, min=float(values.min()), max=float(values.max()))
    edges = torch.linspace(float(values.min()), float(values.max()), bins + 1)
    return edges.tolist(), [int(c) for c in counts]
