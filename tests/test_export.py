"""The weights export must still load, and must still refuse a wrong tokenizer.

A smaller artefact is only useful if it is not a weaker one. These tests check
that stripping the optimiser state keeps everything the loader needs — and in
particular keeps the fingerprint guard working, because an export that loads
against any tokenizer would be worse than no export at all.

This caught a real bug: the first version of the export dropped the `format`
key, which the loader checks first, producing "unrecognised checkpoint format:
None" on a file that was otherwise complete.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.load import load_for_inference          # noqa: E402
from model.transformer import build_model              # noqa: E402
from training.checkpoint import CheckpointMismatch, save_checkpoint  # noqa: E402
from utils.config import Config                        # noqa: E402

TOKENIZER = ROOT / "tokenizer" / "artifacts" / "max_bpe_a"

CONFIG = Config({
    "model": {"vocab_size": 4096, "d_model": 32, "n_layers": 1, "n_heads": 2,
              "d_ff": 64, "context_length": 32, "positional": "rope",
              "tie_embeddings": True},
    "init": {"std": 0.02, "scale_residual": True},
})


@pytest.fixture
def fingerprint():
    from tokenizer.bpe import MAXTokenizer
    if not TOKENIZER.exists():
        pytest.skip("trained tokenizer not present")
    return MAXTokenizer.load(TOKENIZER).fingerprint()


@pytest.fixture
def full_checkpoint(tmp_path, fingerprint):
    model = build_model(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randint(0, 4096, (2, 8)),
                 torch.randint(0, 4096, (2, 8)))[1]
    loss.backward()
    optimizer.step()

    from training.optim import CosineWithWarmup, make_grad_scaler
    scheduler = CosineWithWarmup(optimizer, peak_lr=1e-3, max_steps=100,
                                 warmup_fraction=0.05, min_lr_fraction=0.1)

    path = tmp_path / "ckpt_B_final.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=make_grad_scaler("cpu", False),
        step=42, tokens_seen=1234, config=CONFIG.to_dict(),
        config_hash=CONFIG.hash(), tokenizer_fingerprint=fingerprint,
        metrics={"val_loss": 1.5},
    )
    return path


def export(source: Path, target: Path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "17_export_weights.py"),
         "--checkpoint", str(source), "--out", str(target)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_export_loads_for_inference(tmp_path, full_checkpoint):
    target = tmp_path / "weights.pt"
    export(full_checkpoint, target)
    model, _, header = load_for_inference(target, TOKENIZER, device="cpu", root=ROOT)
    assert header["step"] == 42
    assert model.num_parameters() > 0


def test_export_is_smaller_but_keeps_the_weights(tmp_path, full_checkpoint):
    target = tmp_path / "weights.pt"
    export(full_checkpoint, target)
    assert target.stat().st_size < full_checkpoint.stat().st_size

    original = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
    exported = torch.load(target, map_location="cpu", weights_only=False)
    assert exported["model"].keys() == original["model"].keys()
    for key, tensor in original["model"].items():
        assert torch.equal(tensor, exported["model"][key])


def test_export_drops_the_optimiser_and_says_so(tmp_path, full_checkpoint):
    target = tmp_path / "weights.pt"
    export(full_checkpoint, target)
    exported = torch.load(target, map_location="cpu", weights_only=False)
    assert "optimizer" not in exported
    assert exported["resumable"] is False


def test_export_keeps_the_fingerprint_guard(tmp_path, full_checkpoint):
    """An export that loaded against any tokenizer would be worse than none."""
    target = tmp_path / "weights.pt"
    export(full_checkpoint, target)
    with pytest.raises(CheckpointMismatch):
        from training.checkpoint import load_checkpoint
        load_checkpoint(target, expected_tokenizer_fingerprint="000000000000")


def test_export_keeps_the_format_key(tmp_path, full_checkpoint):
    """The loader checks this first. Dropping it was a real bug."""
    target = tmp_path / "weights.pt"
    export(full_checkpoint, target)
    assert torch.load(target, map_location="cpu", weights_only=False).get("format")


def test_frozen_for_inference(tmp_path, full_checkpoint):
    """No agent may train the language model; loading must freeze it."""
    target = tmp_path / "weights.pt"
    export(full_checkpoint, target)
    model, _, _ = load_for_inference(target, TOKENIZER, device="cpu", root=ROOT)
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
