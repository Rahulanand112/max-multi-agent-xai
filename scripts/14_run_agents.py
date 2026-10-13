#!/usr/bin/env python3
"""Run the five agents over a split and cache every generation.

    python scripts/14_run_agents.py --split test --limit 1500 \
        --checkpoint checkpoints/ckpt_B_sft_final.pt

This is the only expensive step in the evaluation, and it is deliberately
separated from the analysis. Generations are produced ONCE and written to
disk; the coordinator training and every aggregation method then read the same
cached bundles. Three consequences, all of them the point:

  * every method is scored on identical generations, so the comparison
    measures aggregation and not sampling luck;
  * re-running the analysis after a bug fix costs seconds instead of GPU
    minutes;
  * the cached file is itself evidence — a reviewer can recompute every
    number in the results table from it without a GPU.

Run it once on the validation split (to train the coordinator) and once on
test (to report).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.roster import build_roster, run_bundle                 # noqa: E402
from collaboration.aggregate import pairwise_disagreement          # noqa: E402
from inference.load import load_for_inference                      # noqa: E402
from reasoning.problem import load_jsonl                           # noqa: E402
from utils.logging_utils import get_logger, hardware_info, write_manifest  # noqa: E402
from utils.seeding import set_seed                                 # noqa: E402

log = get_logger("run_agents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_B_sft_final.pt")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts/max_bpe_a")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--synth-dir", default="data/synth")
    parser.add_argument("--limit", type=int, default=1500,
                        help="problems to run; the whole split if 0")
    parser.add_argument("--self-consistency", type=int, default=4,
                        help="extra Solver samples for the M2 baseline")
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer, header = load_for_inference(
        args.checkpoint, args.tokenizer, root=ROOT)
    device = header["device"]

    problems = load_jsonl(ROOT / args.synth_dir / f"{args.split}.jsonl")
    if args.limit:
        problems = problems[:args.limit]

    log.info("=" * 74)
    log.info("RUNNING THE FIVE AGENTS  |  split %s  |  device %s", args.split, device)
    log.info("=" * 74)
    log.info("checkpoint    : %s  (step %s)", Path(header["path"]).name,
             f"{header['step']:,}")
    log.info("fingerprint   : %s  (matched)", header["tokenizer_fingerprint"])
    log.info("parameters    : %s  -- ONE model, five role tokens, zero extra weights",
             f"{header['n_parameters']:,}")
    log.info("problems      : %s", f"{len(problems):,}")
    log.info("self-consistency samples: %d (the M2 budget)", args.self_consistency)

    roster = build_roster(model, tokenizer, device, max_new_tokens=args.max_new_tokens)
    for agent in roster.values():
        log.info("  %s", agent)
    log.info("-" * 74)

    bundles = []
    started = time.time()
    for index, problem in enumerate(problems):
        bundle = run_bundle(
            roster, problem.question, gold=problem.answer,
            seed=args.seed + index * 17,
            self_consistency=args.self_consistency,
        )
        bundles.append(bundle)

        if (index + 1) % 50 == 0 or index + 1 == len(problems):
            elapsed = time.time() - started
            rate = (index + 1) / max(elapsed, 1e-9)
            log.info("  %5d/%d problems | %.1f problems/s | eta %4.1f min",
                     index + 1, len(problems), rate,
                     (len(problems) - index - 1) / max(rate, 1e-9) / 60)

    elapsed = time.time() - started
    out_path = Path(args.out) if args.out else ROOT / "experiments" / "agents" / \
        f"bundles_{args.split}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bundles": bundles,
                "problem_ids": [p.id for p in problems],
                "split": args.split,
                "checkpoint": header["path"],
                "tokenizer_fingerprint": header["tokenizer_fingerprint"]}, out_path)

    disagreement = sum(pairwise_disagreement(b) for b in bundles) / max(len(bundles), 1)
    compliance = sum(b.format_compliance() for b in bundles) / max(len(bundles), 1)
    generations = sum(b.generations + len(b.samples) for b in bundles)

    log.info("-" * 74)
    log.info("  problems run        %s", f"{len(bundles):,}")
    log.info("  generations         %s  (%.1f per problem)",
             f"{generations:,}", generations / max(len(bundles), 1))
    log.info("  wall clock          %.1f min", elapsed / 60)
    log.info("  format compliance   %.1f%%", 100 * compliance)
    log.info("  pairwise disagreement %.1f%%", 100 * disagreement)
    if disagreement < 0.15:
        log.warning("  disagreement is below the 15%% floor: the agents are producing")
        log.warning("  near-identical answers, so aggregation methods will score alike")
    log.info("  cached to           %s", out_path.relative_to(ROOT))

    write_manifest(
        ROOT / "experiments" / f"agents_{args.split}",
        script="scripts/14_run_agents.py",
        seed=args.seed, split=args.split, device=device,
        hardware=hardware_info(),
        checkpoint=header["path"], checkpoint_step=header["step"],
        tokenizer_fingerprint=header["tokenizer_fingerprint"],
        n_parameters=header["n_parameters"],
        problems=len(bundles), generations=generations,
        self_consistency=args.self_consistency,
        max_new_tokens=args.max_new_tokens,
        elapsed_seconds=round(elapsed, 2),
        mean_format_compliance=round(compliance, 6),
        mean_pairwise_disagreement=round(disagreement, 6),
        cache=str(out_path),
    )
    log.info("=" * 74)
    log.info("next: python scripts/15_train_coordinator.py   (on the val bundles)")


if __name__ == "__main__":
    main()
