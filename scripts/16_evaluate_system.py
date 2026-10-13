#!/usr/bin/env python3
"""Level 2 evaluation: score every aggregation method on the test bundles.

    python scripts/16_evaluate_system.py

Reads cached generations, applies all five methods to the SAME generations,
and writes a results table with confidence intervals, calibration, per-family
and per-difficulty breakdowns, ablations, and paired significance tests
against two baselines.

Nothing here runs the language model, so it is fast, deterministic, and can be
re-run by anyone with the cache and no GPU.

What this script will not do
----------------------------
It will not drop unanswerable problems, will not report the best of several
seeds, and will not present M5 without saying when it fell back to M4. If the
agents did not disagree enough for the comparison to mean anything, the
disagreement warning appears above the table rather than in a footnote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collaboration.aggregate import run_all                        # noqa: E402
from collaboration.fusion import FusionCoordinator                 # noqa: E402
from evaluation.level2 import evaluate_methods, format_table       # noqa: E402
from explainability.decision_record import render_bundle           # noqa: E402
from reasoning.problem import load_jsonl                           # noqa: E402
from utils.logging_utils import get_logger, write_manifest         # noqa: E402
from utils.seeding import set_seed                                 # noqa: E402

log = get_logger("evaluate")


def load_coordinator(path: Path):
    if not path.exists():
        return None, None
    blob = torch.load(path, weights_only=False)
    coordinator = FusionCoordinator(blob["hidden_dim"], width=blob["width"])
    coordinator.load_state_dict(blob["model"])
    coordinator.eval()
    return coordinator, blob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", default="experiments/agents/bundles_test.pt")
    parser.add_argument("--coordinator", default="checkpoints/coordinator.pt")
    parser.add_argument("--synth-dir", default="data/synth")
    parser.add_argument("--self-consistency-k", type=int, default=4)
    parser.add_argument("--out", default="experiments/level2")
    parser.add_argument("--records", type=int, default=6,
                        help="decision records to write out in full")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    bundle_path = Path(args.bundles)
    if not bundle_path.is_absolute():
        bundle_path = ROOT / bundle_path
    blob = torch.load(bundle_path, weights_only=False)
    bundles = blob["bundles"]

    by_id = {p.id: p for p in load_jsonl(ROOT / args.synth_dir / f"{blob['split']}.jsonl")}
    problems = [by_id[pid] for pid in blob["problem_ids"]]

    coordinator_path = Path(args.coordinator)
    if not coordinator_path.is_absolute():
        coordinator_path = ROOT / coordinator_path
    coordinator, coordinator_blob = load_coordinator(coordinator_path)

    log.info("=" * 74)
    log.info("LEVEL 2 EVALUATION  |  the multi-agent SYSTEM, not the language model")
    log.info("=" * 74)
    log.info("bundles       : %s problems from %s", f"{len(bundles):,}", bundle_path.name)
    log.info("checkpoint    : %s", Path(blob["checkpoint"]).name)
    if coordinator is None:
        log.warning("no trained coordinator at %s", coordinator_path)
        log.warning("M5 will fall back to M4 and the table will say so")
    else:
        log.info("coordinator   : %s parameters, holdout selection accuracy %.4f",
                 f"{coordinator.num_parameters():,}", coordinator_blob["holdout_accuracy"])

    per_method = [run_all(b, coordinator, k=args.self_consistency_k) for b in bundles]
    report = evaluate_methods(problems, bundles, per_method)

    log.info("-" * 74)
    for line in format_table(report).splitlines():
        log.info("%s", line)

    health = report["health"]
    if not health["disagreement_ok"]:
        log.warning("")
        for line in health["warning"].split(". "):
            log.warning("  %s", line.strip())

    # ---- ablations: what each mechanism contributes ----------------------
    log.info("-" * 74)
    log.info("ABLATIONS (accuracy)")
    ablations: dict = {}
    from collaboration.aggregate import m2_self_consistency, m4_confidence_weighted
    from agents.roster import is_correct

    def score(answers) -> float:
        return sum(1 for a, p in zip(answers, problems) if is_correct(a, p.answer)) / len(problems)

    ablations["M4_without_verifier_veto"] = score(
        [m4_confidence_weighted(b, veto_weight=1.0)[0] for b in bundles])
    ablations["M4_with_veto"] = report["methods"]["M4_confidence_weighted"]["accuracy"]
    for k in (1, 2, 4, 8):
        if any(len(b.samples) >= k for b in bundles):
            ablations[f"M2_k{k}"] = score([m2_self_consistency(b, k)[0] for b in bundles])
    for name, value in ablations.items():
        log.info("  %-34s %.4f", name, value)
    report["ablations"] = {k: round(v, 6) for k, v in ablations.items()}

    # ---- significance ----------------------------------------------------
    log.info("-" * 74)
    log.info("PAIRED TESTS vs M2 self-consistency (the honest baseline)")
    for method, result in report["comparisons"]["vs_M2_self_consistency"].items():
        boot = result["bootstrap"]
        mcn = result["mcnemar"]
        verdict = "SIGNIFICANT" if boot["significant"] else "not distinguishable"
        log.info("  %-24s %+0.4f  CI [%+0.4f, %+0.4f]  p=%.4f  %s",
                 method, boot["difference"], boot["low"], boot["high"],
                 mcn["p_value"], verdict)

    log.info("-" * 74)
    agreement = report["confidence_agreement"]
    log.info("STATED vs MEASURED confidence (solver): correlation %s over %d outputs",
             agreement["correlation"], agreement["n"])

    # ---- write everything out -------------------------------------------
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "level2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "results_table.txt").write_text(format_table(report), encoding="utf-8")

    records = []
    for bundle, problem, methods in list(zip(bundles, problems, per_method))[:args.records]:
        records.append(render_bundle(bundle, methods, gold=problem.answer))
    (out_dir / "decision_records.txt").write_text("\n\n".join(records), encoding="utf-8")

    write_manifest(
        out_dir, script="scripts/16_evaluate_system.py", seed=args.seed,
        bundles=str(bundle_path), checkpoint=blob["checkpoint"],
        tokenizer_fingerprint=blob["tokenizer_fingerprint"],
        coordinator=str(coordinator_path) if coordinator else None,
        coordinator_parameters=coordinator.num_parameters() if coordinator else 0,
        self_consistency_k=args.self_consistency_k,
        n_problems=len(problems), report=report,
    )
    log.info("results       : %s", (out_dir / "level2_report.json").relative_to(ROOT))
    log.info("records       : %s", (out_dir / "decision_records.txt").relative_to(ROOT))
    log.info("=" * 74)


if __name__ == "__main__":
    main()
