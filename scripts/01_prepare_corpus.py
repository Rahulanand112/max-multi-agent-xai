#!/usr/bin/env python3
"""Stage 1 data preparation: download -> clean -> dedup -> split.

    python scripts/01_prepare_corpus.py --config configs/data.yaml
    python scripts/01_prepare_corpus.py --config configs/data.yaml --source smoke

Writes data/processed/{train,val,test}.txt (one document per line, with literal
newlines escaped) plus a manifest recording every count at every stage.

Order matters: we deduplicate BEFORE splitting, so a near-duplicate cannot
straddle train and validation and inflate the score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import load_config                      # noqa: E402
from utils.logging_utils import get_logger, write_manifest  # noqa: E402
from utils.seeding import set_seed                        # noqa: E402

log = get_logger("prepare")

CONTROL_OK = {"\n", "\t"}
MARKER_CHARS = {"Ġ", "Ċ", "ĉ", "Ř"}


# --------------------------------------------------------------------- source

def load_tinystories(cfg, raw_dir: Path) -> list[str]:
    """Load TinyStories via HuggingFace Datasets.

    Downloads data only. No model and no pretrained weights are fetched.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("`datasets` is not installed. pip install datasets")
        raise

    limit = cfg.tinystories.get("max_documents")
    log.info("loading %s (split=%s, limit=%s)",
             cfg.tinystories.hf_dataset, cfg.tinystories.hf_split, limit)

    if limit:
        # streaming avoids pulling all 7.6 GB when we only need a slice
        stream = load_dataset(
            cfg.tinystories.hf_dataset,
            split=cfg.tinystories.hf_split,
            streaming=True,
        )
        docs = []
        for i, row in enumerate(stream):
            if i >= limit:
                break
            docs.append(row["text"])
    else:
        ds = load_dataset(cfg.tinystories.hf_dataset, split=cfg.tinystories.hf_split)
        docs = list(ds["text"])

    log.info("loaded %s documents", f"{len(docs):,}")
    return docs


def load_smoke(cfg, raw_dir: Path) -> list[str]:
    """Fetch a ~1 MB public-domain text and slice it into pseudo-documents.

    This exists so the entire pipeline can be exercised in seconds on any
    machine, including ones without HuggingFace access. It is a smoke test,
    never a corpus for a reported result.
    """
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = raw_dir / "smoke_corpus.txt"

    if not cached.exists():
        log.info("downloading smoke corpus from %s", cfg.smoke.url)
        with urllib.request.urlopen(cfg.smoke.url, timeout=60) as response:
            cached.write_bytes(response.read())
    else:
        log.info("using cached smoke corpus at %s", cached)

    text = cached.read_text(encoding="utf-8", errors="replace")
    size = int(cfg.smoke.chars_per_document)
    docs = [text[i : i + size] for i in range(0, len(text), size)]
    log.info("sliced %s chars into %s pseudo-documents", f"{len(text):,}", f"{len(docs):,}")
    return docs


# -------------------------------------------------------------------- cleaning

def clean_document(text: str, cfg) -> str | None:
    """Normalise one document. Returns None if it should be dropped."""
    if cfg.unicode_normalise:
        text = unicodedata.normalize(cfg.unicode_normalise, text)

    if cfg.drop_marker_chars:
        # the tokenizer uses these to make whitespace visible; if they occur
        # literally in the corpus, decoding becomes ambiguous
        for marker in MARKER_CHARS:
            text = text.replace(marker, "")

    if cfg.strip_control_chars:
        text = "".join(
            ch for ch in text
            if ch in CONTROL_OK or unicodedata.category(ch)[0] != "C"
        )

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.collapse_blank_lines:
        lines = [ln.rstrip() for ln in text.split("\n")]
        out: list[str] = []
        blank = 0
        for line in lines:
            if line:
                blank = 0
                out.append(line)
            else:
                blank += 1
                if blank == 1:
                    out.append("")
        text = "\n".join(out)

    text = text.strip()

    if len(text) < cfg.min_chars or len(text) > cfg.max_chars:
        return None
    return text


def normalised_hash(text: str) -> str:
    key = " ".join(text.lower().split())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--source", default=None, choices=["tinystories", "smoke"])
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    source = args.source or cfg.source
    set_seed(cfg.split.seed)

    raw_dir = ROOT / cfg.paths.raw_dir
    out_dir = ROOT / cfg.paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    log.info("=" * 64)
    log.info("MAX corpus preparation  |  source = %s", source)
    log.info("=" * 64)

    # 1. load ------------------------------------------------------------
    docs = load_tinystories(cfg, raw_dir) if source == "tinystories" else load_smoke(cfg, raw_dir)
    n_loaded = len(docs)

    # 2. clean -----------------------------------------------------------
    log.info("cleaning...")
    cleaned = []
    for doc in docs:
        result = clean_document(doc, cfg.cleaning)
        if result is not None:
            cleaned.append(result)
    n_cleaned = len(cleaned)
    log.info("  kept %s / %s after cleaning (%.1f%% dropped)",
             f"{n_cleaned:,}", f"{n_loaded:,}",
             100 * (1 - n_cleaned / max(n_loaded, 1)))

    # 3. deduplicate -----------------------------------------------------
    if cfg.cleaning.dedup:
        log.info("deduplicating...")
        seen: set[str] = set()
        deduped = []
        for doc in cleaned:
            key = normalised_hash(doc)
            if key not in seen:
                seen.add(key)
                deduped.append(doc)
        log.info("  removed %s duplicates", f"{n_cleaned - len(deduped):,}")
        cleaned = deduped
    n_deduped = len(cleaned)

    # 4. shuffle + split -------------------------------------------------
    rng = random.Random(cfg.split.seed)
    rng.shuffle(cleaned)

    n = len(cleaned)
    n_train = int(n * cfg.split.train)
    n_val = int(n * cfg.split.val)
    splits = {
        "train": cleaned[:n_train],
        "val": cleaned[n_train : n_train + n_val],
        "test": cleaned[n_train + n_val :],
    }

    # 5. write -----------------------------------------------------------
    counts: dict[str, dict[str, int]] = {}
    for name, documents in splits.items():
        path = out_dir / f"{name}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for doc in documents:
                handle.write(doc.replace("\\", "\\\\").replace("\n", "\\n") + "\n")
        chars = sum(len(d) for d in documents)
        counts[name] = {"documents": len(documents), "characters": chars}
        log.info("  %-5s -> %6s docs, %12s chars  (%s)",
                 name, f"{len(documents):,}", f"{chars:,}", path.name)

    elapsed = time.time() - started
    total_chars = sum(v["characters"] for v in counts.values())
    log.info("-" * 64)
    log.info("total %s documents, %s characters in %.1fs",
             f"{n:,}", f"{total_chars:,}", elapsed)
    log.info("estimated tokens at ~4 chars/token: ~%s", f"{total_chars // 4:,}")

    # 6. manifest --------------------------------------------------------
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "experiments" / "data_prep"
    write_manifest(
        run_dir,
        stage="corpus_preparation",
        source=source,
        source_config=cfg[source].to_dict(),
        license=cfg[source].get("license"),
        cleaning=cfg.cleaning.to_dict(),
        split=cfg.split.to_dict(),
        seed=cfg.split.seed,
        counts={
            "loaded": n_loaded,
            "after_cleaning": n_cleaned,
            "after_dedup": n_deduped,
            "splits": counts,
        },
        elapsed_seconds=round(elapsed, 2),
        config_hash=cfg.hash(),
    )
    log.info("manifest -> %s", run_dir / "manifest.json")


if __name__ == "__main__":
    main()
