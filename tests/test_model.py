"""MAX-LM correctness tests.

Two of these carry the Review 1 claim and should be run live:

  test_step0_loss_equals_ln_vocab   -- the model was randomly initialised
  test_attention_is_causal          -- the model cannot see the future

Everything else guards the plumbing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.attention import CausalSelfAttention        # noqa: E402
from model.init import expected_init_loss              # noqa: E402
from model.transformer import MAXTransformer, build_model  # noqa: E402
from utils.config import load_config                   # noqa: E402
from utils.seeding import set_seed                     # noqa: E402

CONFIG_A = ROOT / "configs" / "model_a.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_A)


@pytest.fixture(scope="module")
def model(cfg):
    set_seed(1337)
    return build_model(cfg)


def tiny(**overrides) -> MAXTransformer:
    kwargs = dict(
        vocab_size=128, d_model=32, n_layers=2, n_heads=4,
        d_ff=128, context_length=16,
    )
    kwargs.update(overrides)
    return MAXTransformer(**kwargs)


# ------------------------------------------------------------ parameter count

def test_parameter_count_is_exactly_as_documented(model, cfg):
    assert model.num_parameters() == 1_331_968 == cfg.expected.n_parameters


def test_parameter_breakdown_sums_to_the_total(model):
    rows = dict(model.parameter_breakdown())
    total = (
        rows["token embedding"]
        + rows["positional embedding"]
        + rows["all 4 blocks"]
        + rows["final layernorm"]
        + rows["LM head (tied)"]
    )
    assert total == rows["TOTAL"] == model.num_parameters()


def test_component_counts_match_the_arithmetic(model, cfg):
    m = cfg.model
    rows = dict(model.parameter_breakdown())
    assert rows["token embedding"] == m.vocab_size * m.d_model == 524_288
    assert rows["positional embedding"] == m.context_length * m.d_model == 16_384
    assert rows["transformer block (x1 of 4)"] == 197_760
    assert rows["final layernorm"] == 2 * m.d_model == 256


def test_tied_head_shares_one_tensor_and_is_counted_once(model):
    assert model.lm_head.weight is model.tok_emb.weight
    untied = build_model(load_config(CONFIG_A))
    untied.lm_head.weight = torch.nn.Parameter(untied.lm_head.weight.clone())
    assert untied.num_parameters() == model.num_parameters() + 524_288


# --------------------------------------------- THE RANDOM-INITIALISATION PROOF

@pytest.mark.parametrize("seed", [0, 1337, 424242])
def test_step0_loss_equals_ln_vocab(cfg, seed):
    """A freshly initialised model must score ln(V) -- it knows nothing."""
    set_seed(seed)
    m = build_model(cfg)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.model.vocab_size, (8, cfg.model.context_length), generator=gen)
    y = torch.randint(0, cfg.model.vocab_size, (8, cfg.model.context_length), generator=gen)

    m.eval()
    with torch.no_grad():
        _, loss = m(x, y)

    assert abs(float(loss) - math.log(cfg.model.vocab_size)) < 0.05


def test_expected_init_loss_helper():
    assert expected_init_loss(4096) == pytest.approx(8.3178, abs=1e-3)
    assert expected_init_loss(8192) == pytest.approx(9.0109, abs=1e-3)


def test_initial_logits_are_near_uniform(model, cfg):
    """The mechanism behind the ln(V) result: logits start flat."""
    x = torch.zeros((2, 8), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    spread = float(logits.std())
    assert spread < 0.5, f"initial logits are not flat (std={spread:.3f})"


def test_weight_statistics_match_the_init_recipe(model, cfg):
    from model.init import weight_statistics

    stats = weight_statistics(model)
    assert abs(stats["embeddings"]["mean"]) < 0.002
    assert abs(stats["embeddings"]["std"] - 0.02) < 0.002
    # residual projections are scaled by 1/sqrt(2L) = 1/sqrt(8)
    want = 0.02 / math.sqrt(2 * cfg.model.n_layers)
    assert abs(stats["residual_projections"]["std"] - want) < 0.0015
    # LayerNorm starts at the identity
    assert stats["layernorm"]["max"] == pytest.approx(1.0)


# ------------------------------------------------------------------ causality

def test_attention_is_causal():
    """Changing a LATER token must not change an EARLIER position's output.

    If this fails the model is reading the future, every loss number is
    meaningless, and the whole project is measuring nothing.
    """
    set_seed(0)
    m = tiny().eval()
    x = torch.randint(0, 128, (1, 16))

    with torch.no_grad():
        base, _ = m(x)
        altered = x.clone()
        altered[0, -1] = (altered[0, -1] + 7) % 128
        after, _ = m(altered)

    # positions before the edit are untouched
    assert torch.allclose(base[0, :-1], after[0, :-1], atol=1e-6)
    # the edited position itself does change
    assert not torch.allclose(base[0, -1], after[0, -1], atol=1e-6)


def test_causal_mask_is_lower_triangular():
    attn = CausalSelfAttention(d_model=32, n_heads=4, context_length=8)
    mask = attn.causal_mask[0, 0]
    assert bool(mask[0, 0]) and not bool(mask[0, 7])
    assert torch.equal(mask, torch.tril(torch.ones(8, 8, dtype=torch.bool)))


def test_attention_weights_sum_to_one_and_ignore_the_future():
    set_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=4, context_length=8).eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        _, weights = attn(x, return_attn=True)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)
    assert float(weights[0, 0, 0, 1:].abs().max()) == 0.0


def test_explicit_and_fused_attention_agree(cfg):
    """Lets us train with the fast kernel and demo the explicit maths."""
    set_seed(1337)
    slow = build_model(cfg, use_sdpa=False).eval()
    fast = build_model(cfg, use_sdpa=True).eval()
    fast.load_state_dict(slow.state_dict())

    x = torch.randint(0, cfg.model.vocab_size, (4, 32))
    with torch.no_grad():
        a, _ = slow(x)
        b, _ = fast(x)
    assert torch.allclose(a, b, atol=1e-4)


# -------------------------------------------------------------------- shapes

def test_forward_shapes(model, cfg):
    x = torch.randint(0, cfg.model.vocab_size, (3, 64))
    logits, loss = model(x)
    assert logits.shape == (3, 64, cfg.model.vocab_size)
    assert loss is None
    logits, loss = model(x, x)
    assert loss is not None and loss.ndim == 0


def test_return_hidden_gives_the_coordinator_its_input(model, cfg):
    """V5 pools these hidden states; the shape contract starts here."""
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    logits, _, hidden = model(x, return_hidden=True)
    assert hidden.shape == (2, 16, cfg.model.d_model)


def test_sequence_longer_than_context_is_rejected(model, cfg):
    x = torch.zeros((1, cfg.model.context_length + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="context length"):
        model(x)


def test_indivisible_head_dimension_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        tiny(d_model=32, n_heads=5)


def test_unimplemented_positional_scheme_is_rejected():
    with pytest.raises(NotImplementedError, match="rope"):
        tiny(positional="rope")


# ---------------------------------------------------------------- generation

def test_generate_extends_the_sequence(model):
    out = model.generate(torch.zeros((2, 3), dtype=torch.long), max_new_tokens=5, temperature=0.8)
    assert out.shape == (2, 8)


def test_greedy_generation_is_deterministic(model):
    prompt = torch.zeros((1, 4), dtype=torch.long)
    a = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)


def test_generation_respects_context_window(cfg):
    """Prompts longer than the context must not crash -- they get cropped."""
    set_seed(0)
    m = build_model(cfg)
    prompt = torch.zeros((1, cfg.model.context_length), dtype=torch.long)
    out = m.generate(prompt, max_new_tokens=4, temperature=0.5)
    assert out.shape == (1, cfg.model.context_length + 4)


# --------------------------------------------------------------- optimisation

def test_every_parameter_receives_a_gradient(model, cfg):
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randint(0, cfg.model.vocab_size, (4, 32))
    _, loss = model(x, x)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"parameters with no gradient: {dead}"


def test_construction_is_deterministic_under_a_seed(cfg):
    set_seed(1337)
    a = build_model(cfg)
    set_seed(1337)
    b = build_model(cfg)
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), na


def test_different_seeds_give_different_weights(cfg):
    set_seed(1)
    a = build_model(cfg)
    set_seed(2)
    b = build_model(cfg)
    assert not torch.equal(a.tok_emb.weight, b.tok_emb.weight)
