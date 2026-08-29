#!/usr/bin/env python3
"""Train the MAX BPE tokenizer on the TRAIN SPLIT ONLY.

    python scripts/02_train_tokenizer.py --config configs/tokenizer_a.yaml

Fitting the vocabulary on the full corpus would leak validation statistics into
the tokenizer. Small effect, free to avoid, and an examiner may well ask -- so
this script refuses to read anything but train.txt.

Writes tokenizer/artifacts/<name>/tokenizer.json plus a manifest with the
measured compression ratio, UNK rate and round-trip result.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer.bpe import MAXTokenizer                    # noqa: E402
from tokenizer.special_tokens import ROLE_TOKENS          # noqa: E402
from utils.config import load_config                      # noqa: E402
from utils.logging_utils import get_logger, write_manifest  # noqa: E402
from utils.seeding import set_seed                        # noqa: E402

log = get_logger("tokenizer")


def read_documents(path: Path, limit: int | None = None) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            docs.append(line.rstrip("\n").replace("\\n", "\n").replace("\\\\", "\\"))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = load_config(ROOT / args.data_config)
    set_seed(cfg.training.seed)

    processed = ROOT / data_cfg.paths.processed_dir
    train_path = processed / "train.txt"
    val_path = processed / "val.txt"
    if not train_path.exists():
        log.error("%s not found. Run scripts/01_prepare_corpus.py first.", train_path)
        sys.exit(1)

    log.info("=" * 64)
    log.info("MAX BPE tokenizer  |  target vocab = %s", f"{cfg.vocab_size:,}")
    log.info("=" * 64)

    limit = cfg.training.get("max_train_documents")
    log.info("reading train split (limit=%s)...", limit)
    train_docs = read_documents(train_path, limit)
    log.info("  %s training documents", f"{len(train_docs):,}")

    started = time.time()
    tok = MAXTokenizer.train(
        train_docs,
        vocab_size=cfg.vocab_size,
        min_char_freq=cfg.training.min_char_freq,
        min_pair_freq=cfg.training.min_pair_freq,
        verbose=True,
    )
    elapsed = time.time() - started
    log.info("trained in %.1fs -> %r", elapsed, tok)

    # ---- verification -------------------------------------------------
    log.info("-" * 64)
    log.info("verification")

    rng = random.Random(cfg.training.seed)
    val_docs = read_documents(val_path) if val_path.exists() else []
    sample = rng.sample(val_docs, min(1000, len(val_docs))) if val_docs else train_docs[:1000]

    ok = sum(1 for d in sample if tok.roundtrip_ok(d))
    roundtrip_rate = ok / len(sample)
    log.info("  round-trip exact     : %d / %d  (%.2f%%)",
             ok, len(sample), 100 * roundtrip_rate)

    unk = tok.unk_rate(sample)
    log.info("  <|unk|> rate         : %.4f%%", 100 * unk)

    total_chars = sum(len(d) for d in sample)
    total_tokens = sum(len(tok.encode(d)) for d in sample)
    compression = total_chars / max(total_tokens, 1)
    log.info("  compression          : %.2f chars/token", compression)

    # role tokens must be single ids -- this is what makes agent prompting cheap
    for role in ROLE_TOKENS:
        ids = tok.encode(role)
        assert len(ids) == 1, f"role token {role} did not encode to a single id: {ids}"
    log.info("  role tokens          : all %d encode to a single id", len(ROLE_TOKENS))

    probe = "Once upon a time, there was a little robot named Max."
    log.info("  sample text          : %s", probe)
    log.info("  sample ids           : %s", tok.encode(probe))
    log.info("  sample decoded       : %s", tok.decode(tok.encode(probe)))

    # ---- save ---------------------------------------------------------
    out_dir = ROOT / cfg.paths.out_dir
    path = tok.save(out_dir)
    log.info("-" * 64)
    log.info("saved -> %s", path)
    log.info("fingerprint          : %s", tok.fingerprint())

    write_manifest(
        ROOT / "experiments" / f"tokenizer_{cfg.name}",
        stage="tokenizer_training",
        tokenizer_name=cfg.name,
        vocab_size=tok.vocab_size,
        n_merges=len(tok.merges),
        alphabet_size=len(tok.alphabet),
        fingerprint=tok.fingerprint(),
        train_documents=len(train_docs),
        config=cfg.to_dict(),
        config_hash=cfg.hash(),
        elapsed_seconds=round(elapsed, 2),
        measured={
            "roundtrip_exact_rate": round(roundtrip_rate, 6),
            "unk_rate": round(unk, 8),
            "chars_per_token": round(compression, 4),
            "eval_documents": len(sample),
        },
    )


if __name__ == "__main__":
    main()
