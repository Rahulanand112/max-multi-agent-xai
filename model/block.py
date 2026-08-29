"""The transformer block: pre-LN attention and feed-forward, both residual.

    x = x + attention(LayerNorm(x))
    x = x + feedforward(LayerNorm(x))

Pre-LN rather than post-LN. Post-LN needs a carefully tuned warmup to stay
stable at depth; pre-LN trains reliably out of the box. On a two-day schedule
we cannot afford a diverging run.

Parameters per block at d_model=128, d_ff=512:
    attention   4 * 128^2                          =  65,536
    2 layernorms 2 * (2 * 128)                     =     512
    feedforward 128*512 + 512 + 512*128 + 128      = 131,712
    ------------------------------------------------------
                                                     197,760
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: d_model -> d_ff -> d_model.

    GELU rather than SwiGLU. SwiGLU would add a third matrix and change the
    parameter arithmetic; it is a Version 2 experiment, not a Version 1
    dependency.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        context_length: int,
        attn_bias: bool = False,
        ffn_bias: bool = True,
        dropout: float = 0.0,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            bias=attn_bias,
            dropout=dropout,
            use_sdpa=use_sdpa,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, bias=ffn_bias, dropout=dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
