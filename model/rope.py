"""Rotary positional embeddings (RoPE), written by us.

Config A used a learned absolute position table: `context_length x d_model`
parameters that the model has to learn from scratch, and which mean nothing
beyond the longest sequence ever seen in training. Config B replaces it.

The idea
--------
Instead of *adding* a position vector to the token embedding, rotate each
query and key by an angle proportional to its position. Take the head
dimension in pairs; rotate pair `i` at position `m` by angle `m * theta_i`,
where `theta_i = base ** (-2i / head_dim)`.

The payoff is in the dot product. Rotating q at position m and k at position n
by those angles makes `q_m . k_n` depend on `m - n` and not on m and n
separately: attention becomes a function of *relative* distance for free. Two
consequences we care about:

  * **zero parameters.** At Config B's size the learned table would have cost
    256 x 384 = 98,304 parameters, 0.8% of the model, spent on something RoPE
    gets right analytically.
  * **extrapolation.** The rotation is defined at every position, including
    ones past the training context, so the model degrades rather than falling
    off a cliff. That is what makes the depth-generalisation test in the
    evaluation plan meaningful: we train on short reasoning chains and ask
    about longer ones.

Convention
----------
We use the half-split (GPT-NeoX) layout: the first half of the head dimension
pairs with the second half. Any consistent convention works, but it must be
the SAME one at training and at inference, so it is stated here and asserted
in tests/test_rope.py rather than left implicit.

Reference: Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary
Position Embedding*. Implemented from the paper, not copied from a library.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1), splitting the last dimension in half.

    This is the imaginary-part trick: treating the two halves as the real and
    imaginary components of a complex number, this is multiplication by i.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for every position up to `max_positions`.

    Stored as non-persistent buffers: they move with `.to(device)` but are
    never written into a checkpoint, because they are a deterministic function
    of (head_dim, max_positions, base). Saving them would let a checkpoint and
    the code that loads it disagree silently, which is the class of bug this
    project guards against everywhere else.
    """

    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE needs an even head dimension; got {head_dim}. "
                "Check d_model / n_heads in the config."
            )
        self.head_dim = head_dim
        self.base = float(base)
        self.max_positions = max_positions

        # theta_i = base ** (-2i / head_dim), one angle per DIMENSION PAIR
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (self.base ** exponent)          # (head_dim/2,)

        self._build_tables(max_positions, inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_tables(self, n_positions: int, inv_freq: torch.Tensor) -> None:
        positions = torch.arange(n_positions, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)         # (pos, head_dim/2)
        # duplicate so the table lines up with the half-split layout
        angles = torch.cat((angles, angles), dim=-1)      # (pos, head_dim)
        self.register_buffer("cos_table", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_table", angles.sin()[None, None, :, :], persistent=False)

    def _tables_for(self, seq_len: int, offset: int, device, dtype):
        needed = seq_len + offset
        if needed > self.cos_table.shape[-2]:
            # extrapolating past the training context: extend the table rather
            # than raising. The rotation is well defined at any position.
            self._build_tables(needed, self.inv_freq.to("cpu"))
            self.cos_table = self.cos_table.to(device)
            self.sin_table = self.sin_table.to(device)
        cos = self.cos_table[:, :, offset:needed, :].to(device=device, dtype=dtype)
        sin = self.sin_table[:, :, offset:needed, :].to(device=device, dtype=dtype)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys in place of adding a position embedding.

        q, k: (batch, heads, seq, head_dim)
        offset: position of the first token, for cached incremental decoding.
        """
        seq_len = q.shape[-2]
        cos, sin = self._tables_for(seq_len, offset, q.device, q.dtype)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

    def extra_repr(self) -> str:
        return (f"head_dim={self.head_dim}, max_positions={self.max_positions}, "
                f"base={self.base:g}, parameters=0")


@torch.no_grad()
def relative_dot_product_check(rope: RotaryEmbedding, head_dim: int,
                               m: int, n: int, shift: int) -> tuple[float, float]:
    """Evidence that RoPE really is relative, for the review.

    Rotating a fixed q, k pair at positions (m, n) and again at (m+s, n+s)
    must give the same dot product, because both depend only on m - n. Returns
    the two dot products so a test — and a demo — can compare them.
    """
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def dot(pos_q: int, pos_k: int) -> float:
        q_r, _ = rope(q, q, offset=pos_q)
        _, k_r = rope(k, k, offset=pos_k)
        return float((q_r * k_r).sum())

    return dot(m, n), dot(m + shift, n + shift)
