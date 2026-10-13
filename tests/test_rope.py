"""Tests for rotary positional embeddings.

The claim RoPE makes is specific and checkable: after rotation, the
query-key dot product depends on the DIFFERENCE of the two positions and not
on the positions themselves. That is the property worth testing, because it is
the property the architecture choice was made for -- and because an
implementation can look right, train fine, and still be subtly absolute.
"""

from __future__ import annotations

import math

import pytest
import torch

from model.rope import RotaryEmbedding, relative_dot_product_check, rotate_half

HEAD_DIM = 64
CTX = 128


@pytest.fixture
def rope():
    return RotaryEmbedding(HEAD_DIM, CTX)


# --------------------------------------------------------------- the property

def test_dot_product_depends_only_on_relative_distance(rope):
    """(m, n) and (m+s, n+s) must give the same score. This is the whole point."""
    a, b = relative_dot_product_check(rope, HEAD_DIM, m=10, n=3, shift=50)
    assert a == pytest.approx(b, abs=1e-3)


def test_different_distances_give_different_scores(rope):
    """The converse: if everything scored the same, positions would be ignored."""
    same, _ = relative_dot_product_check(rope, HEAD_DIM, m=10, n=3, shift=0)
    other, _ = relative_dot_product_check(rope, HEAD_DIM, m=10, n=9, shift=0)
    assert abs(same - other) > 1e-4


def test_position_zero_is_the_identity(rope):
    """At position 0 every angle is 0, so the rotation must leave q and k alone."""
    q = torch.randn(1, 2, 1, HEAD_DIM)
    k = torch.randn(1, 2, 1, HEAD_DIM)
    q_r, k_r = rope(q, k, offset=0)
    assert torch.allclose(q_r, q, atol=1e-6)
    assert torch.allclose(k_r, k, atol=1e-6)


def test_rotation_preserves_vector_norm(rope):
    """A rotation changes direction, never length. A norm drift means a bug."""
    q = torch.randn(2, 4, 16, HEAD_DIM)
    q_r, _ = rope(q, q)
    assert torch.allclose(q_r.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


# ------------------------------------------------------------------ mechanics

def test_rotate_half_is_multiplication_by_i():
    """(x1, x2) -> (-x2, x1), applied twice, must negate the input."""
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    once = rotate_half(x)
    twice = rotate_half(once)
    assert torch.allclose(twice, -x)


def test_odd_head_dimension_is_rejected():
    with pytest.raises(ValueError, match="even head dimension"):
        RotaryEmbedding(63, CTX)


def test_tables_extend_past_the_training_context(rope):
    """RoPE is defined at every position; past the context it extends, not crashes.

    This is what makes the length-generalisation experiment possible: train on
    short reasoning chains, ask about longer ones.
    """
    q = torch.randn(1, 2, 8, HEAD_DIM)
    q_r, _ = rope(q, q, offset=CTX + 40)
    assert q_r.shape == q.shape
    assert torch.isfinite(q_r).all()


def test_rope_contributes_no_parameters(rope):
    assert sum(p.numel() for p in rope.parameters()) == 0


def test_tables_are_not_persisted(rope):
    """cos/sin are a pure function of the config, so they must not be saved.

    A checkpoint carrying its own tables could disagree with the code that
    loads it and nothing would raise -- the same silent-failure class as a
    tokenizer mismatch.
    """
    assert rope.state_dict() == {}


# ------------------------------------------------------------ in the model

def test_angles_follow_the_stated_formula():
    """theta_i = base ** (-2i / head_dim), asserted against the formula directly."""
    head_dim, base = 8, 10000.0
    rope = RotaryEmbedding(head_dim, 16, base=base)
    expected = [1.0 / (base ** (2 * i / head_dim)) for i in range(head_dim // 2)]
    assert rope.inv_freq.tolist() == pytest.approx(expected, rel=1e-6)
    # and the table is cos of position x angle
    assert float(rope.cos_table[0, 0, 3, 1]) == pytest.approx(math.cos(3 * expected[1]), abs=1e-6)
