"""Causal multi-head self-attention, written by us.

The explicit path computes the attention math in full:

    scores = Q Kᵀ / sqrt(head_dim)
    scores = scores.masked_fill(future positions, -inf)      <- the causal mask
    attn   = softmax(scores)
    out    = attn V

`use_sdpa=True` swaps in PyTorch's fused kernel, which is faster but hides the
maths. Both paths must agree -- tests/test_model.py asserts they do to within
floating-point tolerance, which is what lets us use the fast path for a long
training run and still demonstrate the explicit one at the review.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    Parameters (bias off, per config): 4 * d_model^2
        qkv projection   d_model x 3*d_model
        output projection d_model x d_model
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        bias: bool = False,
        dropout: float = 0.0,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_sdpa = use_sdpa

        # One fused projection producing Q, K and V. Mathematically identical to
        # three separate d_model x d_model matrices -- see split_qkv() -- but a
        # single matmul instead of three. At this model size the GPU is
        # launch-bound, so that difference is worth having.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular mask: position t may attend to positions <= t only.
        # Registered as a buffer so it moves with .to(device) but is not a parameter.
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length),
                             persistent=False)

    def split_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three separate projection matrices, for inspection at the review."""
        w_q, w_k, w_v = self.qkv.weight.split(self.d_model, dim=0)
        return w_q, w_k, w_v

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        batch, seq_len, d_model = x.shape
        if seq_len > self.causal_mask.shape[-1]:
            raise ValueError(
                f"sequence length {seq_len} exceeds context length "
                f"{self.causal_mask.shape[-1]}"
            )

        # project once, then split into query / key / value
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        # (batch, seq, d_model) -> (batch, heads, seq, head_dim)
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        attn_weights = None
        if self.use_sdpa and not return_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # --- the explicit attention computation ---
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            out = self.attn_dropout(attn_weights) @ v

        # (batch, heads, seq, head_dim) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.resid_dropout(self.proj(out))

        if return_attn:
            return out, attn_weights
        return out
