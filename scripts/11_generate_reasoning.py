#!/usr/bin/env python3
"""Generate MAX-Reason: the project's own reasoning benchmark.

    python scripts/11_generate_reasoning.py --n 120000

Produces data/synth/{train,val,test}.jsonl plus a manifest. Every problem is
produced by code in reasoning/generators/, so every label is exact and no item
can have appeared in any model's pretraining corpus.

Three things this script does that matter more than the generation itself:

  1. DEDUPLICATES on an exact fingerprint of the question text, and splits by
     fingerprint, so one question can never appear in two splits. A benchmark
     whose test items also sit in training measures memorisation.
  2. REPORTS the duplicate rate it removed. If that number is large, the
     generator's combinatorial space is too small for the dataset size asked
     for, and the honest fix is a bigger space, not a quiet dedup.
  3. ASSERTS zero overlap between splits before writing anything.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reasoning.generators import arithmetic, logic                 # noqa: E402
from reasoning.problem import Problem, write_jsonl                 # noqa: E402
from utils.logging_utils import get_logger, write_manifest         # noqa: E402
from utils.seeding import set_seed                                 # noqa: E402

log = get_logger("generate_reasoning")


def deduplicate(problems: list[Problem]) -> tuple[list[Problem], int]:
    seen: set[str] = set()
    kept: list[Problem] = []
    for problem in problems:
        fingerprint = problem.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(problem)
    return kept, len(problems) - len(kept)


def split(problems: list[Problem], fractions: tuple[float, float, float],
          seed: int) -> dict[str, list[Problem]]:
    """Split by fingerprint hash, not by shuffling.

    Hashing the question means the split a problem lands in is a property of
    the problem, so regenerating with more data never moves an existing item
    from test into train.
    """
    train_cut, val_cut = fractions[0], fractions[0] + fractions[1]
    buckets: dict[str, list[Problem]] = {"train": [], "val": [], "test": []}
    for problem in problems:
        # stable 0..1 from the fingerprint, salted with the split seed
        digest = int(problem.fingerprint(), 16)
        position = ((digest * 2654435761 + seed) % 1_000_003) / 1_000_003
        name = "train" if position < train_cut else ("val" if position < val_cut else "test")
        buckets[name].append(problem)
    return buckets


def describe(problems: list[Problem]) -> dict:
    return {
        "n": len(problems),
        "by_family": dict(collections.Counter(p.family for p in problems)),
        "by_template": dict(collections.Counter(p.template for p in problems)),
        "by_difficulty": dict(sorted(collections.Counter(p.difficulty for p in problems).items())),
        "mean_distractors": round(
            sum(len(p.distractors) for p in problems) / max(len(problems), 1), 3),
        "mean_question_chars": round(
            sum(len(p.question) for p in problems) / max(len(problems), 1), 1),
        "mean_steps": round(
            sum(len(p.steps) for p in problems) / max(len(problems), 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120_000,
                        help="problems to generate before deduplication")
    parser.add_argument("--arithmetic-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--train", type=float, default=0.90)
    parser.add_argument("--val", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ROOT / args.out
    test_fraction = 1.0 - args.train - args.val
    if test_fraction <= 0:
        raise SystemExit("--train and --val leave no room for a test split")

    n_arith = int(args.n * args.arithmetic_fraction)
    n_logic = args.n - n_arith

    log.info("=" * 66)
    log.info("MAX-REASON GENERATION")
    log.info("=" * 66)
    log.info("requested            : %s arithmetic + %s logic", f"{n_arith:,}", f"{n_logic:,}")
    log.info("seed                 : %d", args.seed)

    problems = (arithmetic.generate(n_arith, seed=args.seed)
                + logic.generate(n_logic, seed=args.seed + 1))

    problems, removed = deduplicate(problems)
    rate = removed / max(args.n, 1)
    log.info("after deduplication  : %s  (removed %s, %.2f%%)",
             f"{len(problems):,}", f"{removed:,}", rate * 100)
    if rate > 0.10:
        log.warning("  duplicate rate above 10%% -- the generator's space is too small")
        log.warning("  for %s items. Reduce --n or widen the template pools.", f"{args.n:,}")

    buckets = split(problems, (args.train, args.val, test_fraction), args.seed)

    # the guarantee, asserted rather than assumed
    fingerprints = {name: {p.fingerprint() for p in items} for name, items in buckets.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = fingerprints[a] & fingerprints[b]
        if overlap:
            raise SystemExit(f"FATAL: {len(overlap)} problems appear in both {a} and {b}")
    log.info("split overlap        : 0 problems shared between any two splits")

    log.info("")
    stats: dict = {}
    for name in ("train", "val", "test"):
        items = buckets[name]
        stats[name] = describe(items)
        write_jsonl(items, out_dir / f"{name}.jsonl")
        log.info("%-6s %8s problems  ->  %s",
                 name, f"{len(items):,}", f"{args.out}/{name}.jsonl")

    log.info("")
    log.info("TRAIN COMPOSITION")
    for key in ("by_family", "by_template", "by_difficulty"):
        log.info("  %-14s %s", key.replace("by_", ""), stats["train"][key])
    log.info("  %-14s %s", "mean steps", stats["train"]["mean_steps"])
    log.info("  %-14s %s", "distractors", stats["train"]["mean_distractors"])

    run_dir = ROOT / "experiments" / "reason_data"
    write_manifest(
        run_dir,
        script="scripts/11_generate_reasoning.py",
        seed=args.seed,
        requested=args.n,
        arithmetic_fraction=args.arithmetic_fraction,
        kept=len(problems),
        duplicates_removed=removed,
        duplicate_rate=round(rate, 6),
        split_fractions={"train": args.train, "val": args.val, "test": round(test_fraction, 4)},
        split_overlap=0,
        splits=stats,
        contamination_note=(
            "Every problem is produced by reasoning/generators/*.py. No item is "
            "drawn from a public corpus, so none can appear in any pretrained "
            "model's training data."
        ),
    )
    log.info("")
    log.info("manifest             : experiments/reason_data/manifest.json")
    log.info("next: python scripts/12_build_sft.py")


if __name__ == "__main__":
    main()
