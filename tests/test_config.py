"""Config loading, inheritance and hashing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import Config, load_config, save_config  # noqa: E402
from utils.seeding import set_seed                         # noqa: E402


def test_attribute_access_is_recursive() -> None:
    cfg = Config({"model": {"d_model": 128, "layers": {"n": 4}}})
    assert cfg.model.d_model == 128
    assert cfg.model.layers.n == 4
    assert cfg["model"]["d_model"] == 128


def test_missing_key_names_the_alternatives() -> None:
    cfg = Config({"a": 1, "b": 2})
    with pytest.raises(AttributeError) as exc:
        _ = cfg.nope
    assert "a" in str(exc.value) and "b" in str(exc.value)


def test_hash_is_stable_and_order_independent() -> None:
    a = Config({"x": 1, "y": {"z": 2}})
    b = Config({"y": {"z": 2}, "x": 1})
    assert a.hash() == b.hash()
    assert len(a.hash()) == 12


def test_hash_changes_when_a_value_changes() -> None:
    a = Config({"lr": 1e-3})
    b = Config({"lr": 3e-3})
    assert a.hash() != b.hash()


def test_base_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"model": {"d_model": 128, "n_layers": 4}, "seed": 1337})
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"_base_": "base.yaml", "model": {"n_layers": 6}})
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.model.d_model == 128   # inherited
    assert cfg.model.n_layers == 6    # overridden
    assert cfg.seed == 1337


def test_save_then_load_is_identical(tmp_path: Path) -> None:
    cfg = Config({"model": {"d_model": 128}, "lr": 0.001})
    save_config(cfg, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml").hash() == cfg.hash()


# ------------------------------------------------------- the shipped configs

@pytest.mark.parametrize("name", [
    "data.yaml", "tokenizer_a.yaml", "model_a.yaml", "train_a.yaml",
])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(ROOT / "configs" / name)
    assert isinstance(cfg, Config) and len(cfg) > 0


def test_config_a_matches_the_documented_parameter_count() -> None:
    """The number on the slide must be the number the config implies."""
    cfg = load_config(ROOT / "configs" / "model_a.yaml")
    m = cfg.model
    V, d, L, ff, ctx = m.vocab_size, m.d_model, m.n_layers, m.d_ff, m.context_length

    token_emb = V * d
    pos_emb = ctx * d if m.positional == "learned" else 0
    attn = 4 * d * d + (4 * d if m.attn_bias else 0)
    norms = 2 * (2 * d)
    ffn = d * ff + ff * d + ((ff + d) if m.ffn_bias else 0)
    block = attn + norms + ffn
    head = 0 if m.tie_embeddings else V * d
    total = token_emb + pos_emb + L * block + 2 * d + head

    assert total == cfg.expected.n_parameters == 1_331_968


def test_expected_init_loss_is_ln_vocab() -> None:
    """The random-initialisation proof, checked against the config."""
    import math

    cfg = load_config(ROOT / "configs" / "model_a.yaml")
    assert math.isclose(
        cfg.expected.init_loss, math.log(cfg.model.vocab_size), abs_tol=1e-3
    )


def test_head_dimension_divides_evenly() -> None:
    for name in ("model_a.yaml",):
        m = load_config(ROOT / "configs" / name).model
        assert m.d_model % m.n_heads == 0


def test_seeding_is_reproducible() -> None:
    import random

    set_seed(1337)
    a = [random.random() for _ in range(5)]
    set_seed(1337)
    b = [random.random() for _ in range(5)]
    assert a == b
