#!/usr/bin/env python3
"""Encode each split into a flat uint16 token stream on disk.

    python scripts/03_encode_corpus.py --config configs/tokenizer_a.yaml

Documents are concatenated, separated by <|eos|>, and written as a raw uint16
memmap. Two decisions worth defending:

  * PACK, DO NOT PAD. Padding a 128-token context would waste a large fraction
    of an already small compute budget. We concatenate and slice instead.
  * uint16. Vocabularies of 4,096 and 8,192 both fit in 16 bits, which halves
    disk usage and lets the training loop memory-map the file for zero-copy
    random access -- Colab's disk is slow enough for this to matter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer.bpe import MAXTokenizer                    # noqa: E402
from tokenizer.special_tokens import EOS_ID               # noqa: E402
from utils.config import load_config                      # noqa: E402
from utils.logging_utils import get_logger, write_manifest  # noqa: E402

log = get_logger("encode")

SPLITS = ("train", "val", "test")


def iter_documents(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n").replace("\\n", "\n").replace("\\\\", "\\")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = load_config(ROOT / args.data_config)

    tok_dir = ROOT / cfg.paths.out_dir
    tok = MAXTokenizer.load(tok_dir)
    if tok.vocab_size > 65536:
        raise ValueError("vocab exceeds uint16 range; switch the dtype")

    processed = ROOT / data_cfg.paths.processed_dir

    log.info("=" * 64)
    log.info("encoding corpus  |  %r", tok)
    log.info("fingerprint %s", tok.fingerprint())
    log.info("=" * 64)

    stats: dict[str, dict] = {}
    started = time.time()

    for split in SPLITS:
        src = processed / f"{split}.txt"
        if not src.exists():
            log.warning("skipping %s (not found)", split)
            continue

        out = processed / f"{split}.bin"
        buffer: list[int] = []
        n_docs = 0
        n_chars = 0

        with out.open("wb") as handle:
            for doc in iter_documents(src):
                n_docs += 1
                n_chars += len(doc)
                buffer.extend(tok.encode(doc))
                buffer.append(EOS_ID)
                if len(buffer) >= 4_000_000:
                    np.asarray(buffer, dtype=np.uint16).tofile(handle)
                    buffer.clear()
                if n_docs % 20000 == 0:
                    log.info("  %s: %s docs encoded", split, f"{n_docs:,}")
            if buffer:
                np.asarray(buffer, dtype=np.uint16).tofile(handle)

        n_tokens = out.stat().st_size // 2
        stats[split] = {
            "documents": n_docs,
            "characters": n_chars,
            "tokens": n_tokens,
            "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
            "bytes_on_disk": out.stat().st_size,
        }
        log.info("  %-5s -> %14s tokens  (%.2f chars/token)  %s",
                 split, f"{n_tokens:,}", stats[split]["chars_per_token"], out.name)

    elapsed = time.time() - started
    total = sum(s["tokens"] for s in stats.values())
    log.info("-" * 64)
    log.info("total %s tokens in %.1fs", f"{total:,}", elapsed)

    # sanity: decode the first 60 tokens of train and eyeball them
    train_bin = processed / "train.bin"
    if train_bin.exists():
        head = np.memmap(train_bin, dtype=np.uint16, mode="r")[:60]
        log.info("first 60 train tokens decode to:")
        log.info("  %r", tok.decode(head.tolist()))

    write_manifest(
        ROOT / "experiments" / f"encode_{cfg.name}",
        stage="corpus_encoding",
        tokenizer_name=cfg.name,
        tokenizer_fingerprint=tok.fingerprint(),
        vocab_size=tok.vocab_size,
        dtype="uint16",
        splits=stats,
        total_tokens=total,
        elapsed_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    main()
