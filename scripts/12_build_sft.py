#!/usr/bin/env python3
"""Encode MAX-Reason into training arrays: Stage 2 SFT rows and Stage 1 text.

    python scripts/12_build_sft.py --max-problems 40000

Writes, into data/processed/:

  sft_train.npz / sft_val.npz   rows of `context_length` tokens plus a loss
                                mask, for Stage 2 reasoning SFT
  reason_train.bin / reason_val.bin
                                plain narrative token streams with NO role
                                tokens, for mixing into Stage 1 pretraining

Two decisions worth stating.

**Row packing, not window sampling.** The pretraining dataset draws random
windows from one flat stream, which is right for next-token prediction. For
SFT it is wrong: a window starting mid-example would train the model on a
completion whose question it cannot see. So examples are packed whole into
fixed-length rows and a row is padded rather than split.

**The loss mask.** Loss is computed on completion tokens only. Training on the
prompt as well would spend most of the gradient teaching the model to
reproduce questions it will always be given, and would let a model score well
by memorising question text. The mask is built here and asserted downstream.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasoning.problem import load_jsonl                            # noqa: E402
from reasoning.render import render_all, render_narrative           # noqa: E402
from tokenizer.bpe import MAXTokenizer                              # noqa: E402
from tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID         # noqa: E402
from utils.config import load_config                                # noqa: E402
from utils.logging_utils import get_logger, write_manifest          # noqa: E402
from utils.seeding import set_seed                                  # noqa: E402

log = get_logger("build_sft")


def build_rows(problems, tokenizer, context_length: int, seed: int):
    """Pack rendered examples into fixed-length rows with a loss mask."""
    rng = random.Random(seed)
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    by_role: collections.Counter = collections.Counter()
    lengths: list[int] = []
    dropped = 0

    current = np.full(context_length, PAD_ID, dtype=np.uint16)
    current_mask = np.zeros(context_length, dtype=np.uint8)
    cursor = 0

    def flush() -> None:
        nonlocal current, current_mask, cursor
        if cursor > 0:
            rows.append(current)
            masks.append(current_mask)
            current = np.full(context_length, PAD_ID, dtype=np.uint16)
            current_mask = np.zeros(context_length, dtype=np.uint8)
            cursor = 0

    for problem in problems:
        for role, prompt, completion in render_all(problem, rng):
            prompt_ids = [BOS_ID] + tokenizer.encode(prompt)
            completion_ids = tokenizer.encode(completion) + [EOS_ID]
            length = len(prompt_ids) + len(completion_ids)
            lengths.append(length)

            if length > context_length:
                # a truncated target teaches a cut-off answer; drop it instead
                dropped += 1
                continue
            if cursor + length > context_length:
                flush()

            tokens = np.asarray(prompt_ids + completion_ids, dtype=np.uint16)
            mask = np.concatenate([
                np.zeros(len(prompt_ids), dtype=np.uint8),
                np.ones(len(completion_ids), dtype=np.uint8),
            ])
            current[cursor:cursor + length] = tokens
            current_mask[cursor:cursor + length] = mask
            cursor += length
            by_role[role] += 1

    flush()
    return (np.stack(rows), np.stack(masks), by_role, lengths, dropped)


def build_narrative(problems, tokenizer) -> np.ndarray:
    """Plain-text token stream for the Stage 1 mixture. No role tokens."""
    stream: list[int] = []
    for problem in problems:
        stream.append(BOS_ID)
        stream.extend(tokenizer.encode(render_narrative(problem)))
        stream.append(EOS_ID)
    return np.asarray(stream, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_b.yaml")
    parser.add_argument("--synth-dir", default="data/synth")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts/max_bpe_a")
    parser.add_argument("--max-problems", type=int, default=40_000,
                        help="cap on TRAIN problems rendered for SFT (val is uncapped)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(ROOT / args.config)
    context_length = cfg.model.context_length
    tokenizer = MAXTokenizer.load(ROOT / args.tokenizer)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 66)
    log.info("BUILDING STAGE 2 SFT DATA")
    log.info("=" * 66)
    log.info("tokenizer            : vocab %d, fingerprint %s",
             tokenizer.vocab_size, tokenizer.fingerprint())
    log.info("context length       : %d", context_length)

    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != config vocab {cfg.model.vocab_size}"
        )

    stats: dict = {}
    for split in ("train", "val"):
        path = ROOT / args.synth_dir / f"{split}.jsonl"
        all_problems = load_jsonl(path)
        problems = all_problems
        if split == "train" and args.max_problems and len(problems) > args.max_problems:
            problems = problems[:args.max_problems]

        rows, masks, by_role, lengths, dropped = build_rows(
            problems, tokenizer, context_length, args.seed)

        np.savez(out_dir / f"sft_{split}.npz", tokens=rows, mask=masks)
        # The narrative stream is built from EVERY problem, not the SFT cap.
        # It costs nothing extra to encode and a larger pool means Stage 1
        # repeats the reasoning text fewer times, which is what keeps the
        # mixture from turning into memorisation of a few thousand sums.
        narrative = build_narrative(all_problems, tokenizer)
        narrative.tofile(out_dir / f"reason_{split}.bin")

        trained = int(masks.sum())
        total = int(masks.size)
        unk = tokenizer.unk_rate([p.question for p in problems[:2000]])
        stats[split] = {
            "problems": len(problems),
            "sequences": int(sum(by_role.values())),
            "rows": int(rows.shape[0]),
            "row_length": context_length,
            "tokens_total": total,
            "tokens_with_loss": trained,
            "loss_fraction": round(trained / total, 4),
            "padding_fraction": round(1 - sum(lengths) / total, 4),
            "mean_example_tokens": round(sum(lengths) / max(len(lengths), 1), 1),
            "max_example_tokens": max(lengths) if lengths else 0,
            "dropped_too_long": dropped,
            "by_role": {k: int(v) for k, v in by_role.items()},
            "narrative_tokens": int(narrative.size),
            "unk_rate": round(unk, 6),
        }

        log.info("")
        log.info("%s", split.upper())
        log.info("  problems           : %s", f"{len(problems):,}")
        log.info("  rendered sequences : %s", f"{sum(by_role.values()):,}")
        log.info("  packed rows        : %s x %d", f"{rows.shape[0]:,}", context_length)
        log.info("  tokens with loss   : %s of %s  (%.1f%%)",
                 f"{trained:,}", f"{total:,}", 100 * trained / total)
        log.info("  padding            : %.1f%%", 100 * stats[split]["padding_fraction"])
        log.info("  mean example       : %.1f tokens (max %d, dropped %d)",
                 stats[split]["mean_example_tokens"], stats[split]["max_example_tokens"], dropped)
        log.info("  UNK rate           : %.4f%%", unk * 100)
        log.info("  narrative stream   : %s tokens -> reason_%s.bin",
                 f"{narrative.size:,}", split)

    if stats["train"]["unk_rate"] > 0.001:
        log.warning("UNK rate above 0.1%% -- the tokenizer is losing characters "
                    "the reasoning data needs")

    write_manifest(
        ROOT / "experiments" / "sft_data",
        script="scripts/12_build_sft.py",
        seed=args.seed,
        config=cfg.to_dict(),
        config_hash=cfg.hash(),
        tokenizer_fingerprint=tokenizer.fingerprint(),
        context_length=context_length,
        max_problems=args.max_problems,
        splits=stats,
    )
    log.info("")
    log.info("manifest             : experiments/sft_data/manifest.json")
    log.info("next: python scripts/13_pretrain_b.py")


if __name__ == "__main__":
    main()
