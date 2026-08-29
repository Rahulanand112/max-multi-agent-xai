"""Checkpoints that can actually resume -- and that refuse to load wrongly.

A checkpoint holding only weights forces a restart from step zero. On a free
Colab session that preempts at hour eleven, that ends the schedule. So every
checkpoint carries:

    model / optimizer / scheduler state
    RNG states for Python, NumPy, Torch and CUDA
    the step counter and total tokens seen
    the full config, its hash, and the tokenizer fingerprint

The fingerprint guard is the important one. Loading a checkpoint against a
re-fitted tokenizer produces a model that generates fluent nonsense with no
error message and no traceback -- the failure is completely silent. So the
loader compares fingerprints and raises rather than warning.

Writes are atomic: save to `.tmp`, fsync, then rename. Google Drive interrupts
mid-write often enough that a half-written checkpoint is a real outcome, and a
rename is the only cheap way to make the file appear all at once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from utils.seeding import load_rng_state, rng_state


class CheckpointMismatch(RuntimeError):
    """Raised when a checkpoint does not belong with the current setup."""


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    tokens_seen: int,
    config: dict,
    config_hash: str,
    tokenizer_fingerprint: str,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "max-ckpt-v1",
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state(),
        "config": config,
        "config_hash": config_hash,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "metrics": metrics or {},
        "extra": extra or {},
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    expected_tokenizer_fingerprint: str | None = None,
    expected_config_hash: str | None = None,
    restore_rng: bool = True,
    strict_config: bool = False,
    map_location: str = "cpu",
) -> dict:
    """Load a checkpoint, refusing on any identity mismatch.

    `strict_config=False` by default: a config-hash change is usually a benign
    edit (more steps, different eval cadence) and only warrants a warning. A
    tokenizer mismatch is never benign, so it always raises.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "max-ckpt-v1":
        raise CheckpointMismatch(
            f"unrecognised checkpoint format: {payload.get('format')!r}"
        )

    # ---- the guard that prevents silent fluent nonsense -------------------
    found = payload.get("tokenizer_fingerprint")
    if expected_tokenizer_fingerprint is not None and found != expected_tokenizer_fingerprint:
        raise CheckpointMismatch(
            "TOKENIZER MISMATCH -- refusing to load.\n"
            f"  checkpoint was trained with : {found}\n"
            f"  tokenizer currently loaded  : {expected_tokenizer_fingerprint}\n"
            "  These vocabularies differ, so every token id means something "
            "different to this model. Loading anyway would produce fluent "
            "nonsense with no error. Re-encode the corpus with the checkpoint's "
            "tokenizer, or retrain against the current one."
        )

    if expected_config_hash is not None and payload.get("config_hash") != expected_config_hash:
        message = (
            f"config hash differs: checkpoint {payload.get('config_hash')} "
            f"vs current {expected_config_hash}"
        )
        if strict_config:
            raise CheckpointMismatch(message)
        print(f"  WARNING: {message}")

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng"):
        load_rng_state(payload["rng"])

    return payload


def rotate_checkpoints(directory: str | Path, pattern: str, keep_last: int) -> list[Path]:
    """Delete all but the newest `keep_last` step checkpoints.

    Never touches files outside `pattern`, so ckpt_A_final.pt and best.pt
    survive rotation.
    """
    directory = Path(directory)
    if keep_last <= 0 or not directory.exists():
        return []
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    removed: list[Path] = []
    for stale in files[:-keep_last]:
        stale.unlink()
        removed.append(stale)
    return removed


def find_latest(directory: str | Path, pattern: str = "ckpt_step_*.pt") -> Path | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def describe(path: str | Path) -> dict:
    """Read a checkpoint's metadata without loading the weights onto a device."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        "format": payload.get("format"),
        "step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "config_hash": payload.get("config_hash"),
        "tokenizer_fingerprint": payload.get("tokenizer_fingerprint"),
        "metrics": payload.get("metrics", {}),
        "n_model_tensors": len(payload.get("model", {})),
        "has_optimizer": payload.get("optimizer") is not None,
        "has_rng": bool(payload.get("rng")),
    }
