#!/usr/bin/env python3
"""Collect the artefacts a reader needs, into results/ — ready to commit.

    python scripts/10_collect_results.py

Copies from experiments/ into results/, and brings in the trained tokenizer and
the final checkpoint. Deliberately leaves behind everything a reader does not
need and git should not carry: the token .bin files (regenerate with scripts
01-03), and the intermediate step checkpoints.

Run this after training, before `git add`. It prints exactly what it copied and
what it skipped, so nothing lands in the repository by accident.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# what to copy out of experiments/ — everything except heavyweight blobs
SKIP_SUFFIXES = {".bin", ".npy", ".pt", ".tmp"}
MAX_FILE_MB = 25.0


def rel(path: Path) -> str:
    """Display a path relative to the repo when possible, absolute otherwise.

    --experiments can point outside the repository (a Drive folder, say), so
    relative_to() is not always valid.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def copy_tree(src: Path, dst: Path, log: list) -> None:
    if not src.exists():
        log.append(("skip", f"{rel(src)} (not found)"))
        return
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        size_mb = path.stat().st_size / 1e6
        if path.suffix in SKIP_SUFFIXES or size_mb > MAX_FILE_MB:
            log.append(("skip", f"{rel(path)}  ({size_mb:.1f} MB)"))
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        log.append(("copy", f"{rel(target)}  ({size_mb:.2f} MB)"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--checkpoint", default=None,
                        help="path to ckpt_A_final.pt (e.g. on Google Drive)")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts/max_bpe_a/tokenizer.json")
    args = parser.parse_args()

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    log: list = []

    # 1. every run's manifests, metrics, samples and plots
    copy_tree(Path(args.experiments) if Path(args.experiments).is_absolute()
              else ROOT / args.experiments, results, log)

    # 2. the trained tokenizer — small, essential, and the fingerprint anchor
    tok = Path(args.tokenizer)
    if not tok.is_absolute():
        tok = ROOT / tok
    if tok.exists():
        target = ROOT / "tokenizer" / "artifacts" / "max_bpe_a" / "tokenizer.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if tok.resolve() != target.resolve():
            shutil.copy2(tok, target)
        log.append(("copy", f"{rel(target)}  ({tok.stat().st_size/1e6:.2f} MB)"))
    else:
        log.append(("skip", f"{tok} (not found) — pass --tokenizer"))

    # 3. the final checkpoint — the artefact the milestone is judged on
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1e6
            target = ROOT / "checkpoints" / "ckpt_A_final.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt, target)
            log.append(("copy", f"{rel(target)}  ({size_mb:.1f} MB)"))
            if size_mb > 90:
                log.append(("warn", "checkpoint over 90 MB — use Git LFS or leave it out"))
        else:
            log.append(("skip", f"{ckpt} (not found)"))

    # ---- report -----------------------------------------------------------
    copied = [m for k, m in log if k == "copy"]
    skipped = [m for k, m in log if k == "skip"]
    warned = [m for k, m in log if k == "warn"]

    print("=" * 66)
    print(f"COPIED INTO results/  ({len(copied)} files)")
    print("=" * 66)
    for m in copied:
        print("  +", m)
    if skipped:
        print()
        print(f"SKIPPED  ({len(skipped)}) — regenerate these with scripts 01-03")
        for m in skipped:
            print("  -", m)
    for m in warned:
        print("  ! ", m)

    size = sum(f.stat().st_size for f in (ROOT / "results").rglob("*") if f.is_file())
    print()
    print(f"results/ is now {size/1e6:.2f} MB")
    print("next:  git add -A && git commit -m 'Review 1: results' && git push")


if __name__ == "__main__":
    main()
