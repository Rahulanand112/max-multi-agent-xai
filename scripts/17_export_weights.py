#!/usr/bin/env python3
"""Strip a checkpoint down to weights, for release.

    python scripts/17_export_weights.py --checkpoint <drive>/ckpt_B_sft_final.pt

A training checkpoint carries the optimiser moments, the scheduler position
and the RNG states, because without them a preempted Colab run cannot resume
bit-exactly. That is roughly three times the size of the weights: Config B
comes to about 140 MB, and GitHub refuses any single file over 100 MB.

This writes the same model with the optimiser state removed — about 49 MB for
Config B — keeping everything needed to LOAD and RUN it:

    the weights, the config, the config hash, the tokenizer fingerprint, the
    step it was saved at, and the metrics recorded with it

What it loses is the ability to resume training from it. Keep the full
checkpoint on Drive; release the export.

The fingerprint travels with the export, so the loader's refusal-on-mismatch
still works exactly as it does for a full checkpoint — an export is not a
weaker artefact, only a smaller one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.checkpoint import describe                           # noqa: E402
from utils.logging_utils import get_logger                         # noqa: E402

log = get_logger("export")

# "format" is what the loader checks first; dropping it makes the export
# unloadable with a confusing error, so it is kept deliberately and the
# round-trip is asserted by tests/test_export.py.
KEEP = ("format", "model", "config", "config_hash", "tokenizer_fingerprint",
        "step", "tokens_seen", "metrics", "extra")
GITHUB_LIMIT_MB = 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None,
                        help="default: alongside the input, with _weights appended")
    args = parser.parse_args()

    source = Path(args.checkpoint)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise SystemExit(f"checkpoint not found: {source}")

    target = Path(args.out) if args.out else source.with_name(
        source.stem + "_weights" + source.suffix)
    if not target.is_absolute():
        target = ROOT / target

    header = describe(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    dropped = sorted(k for k in payload if k not in KEEP)
    export = {k: v for k, v in payload.items() if k in KEEP}
    export["exported_from"] = str(source)
    export["resumable"] = False

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, target)

    before = source.stat().st_size / 1e6
    after = target.stat().st_size / 1e6

    log.info("=" * 70)
    log.info("WEIGHTS EXPORT")
    log.info("=" * 70)
    log.info("source        : %s", source.name)
    log.info("step          : %s   tokens seen %s",
             f"{header['step']:,}", f"{header['tokens_seen']:,}")
    log.info("fingerprint   : %s  (carried into the export)",
             header["tokenizer_fingerprint"])
    log.info("dropped       : %s", ", ".join(dropped) or "nothing")
    log.info("size          : %.1f MB  ->  %.1f MB  (%.0f%% smaller)",
             before, after, 100 * (1 - after / before))
    log.info("written       : %s", target)

    if after > GITHUB_LIMIT_MB:
        log.warning("")
        log.warning("STILL over GitHub's %.0f MB limit. Options, in order of preference:",
                    GITHUB_LIMIT_MB)
        log.warning("  1. attach it to a GitHub Release rather than committing it")
        log.warning("  2. use Git LFS")
        log.warning("  3. leave it out and say in the README where it can be downloaded")
        log.warning("Do NOT commit it and hope -- the push is rejected after the upload.")
    else:
        log.info("")
        log.info("under GitHub's %.0f MB limit: safe to commit.", GITHUB_LIMIT_MB)
        log.info("it loads for inference; it cannot resume training.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
