"""Level 2 evaluation: the multi-agent system, not the language model.

Level 1 asks whether the LM is a good language model — perplexity, next-token
accuracy. Those live in evaluation/level1.py and say nothing about reasoning.
Level 2 asks whether the SYSTEM answers questions correctly, and whether
collaboration helped. Keeping the two apart is deliberate: a better
perplexity does not imply a better answer, and conflating them is how a
project claims a reasoning result it never measured.

What this module reports for each aggregation method:

  accuracy              exact match against generated ground truth
  by family, template, difficulty
                        where the system works and where it fails, which is
                        more useful than one number
  coverage              fraction of problems that produced any parseable
                        answer at all. An unparseable answer is scored WRONG,
                        never skipped — dropping them would flatter a method
                        that often fails to produce output
  calibration           ECE, Brier, overconfidence, reliability bins
  disagreement          mean pairwise disagreement between agents. Below ~15%
                        the comparison between methods is close to vacuous
                        and the report says so rather than burying it
  format compliance     how often agent output matched the schema
  generations           the inference budget each method actually spent
"""

from __future__ import annotations

import collections

from agents.roster import Bundle, is_correct
from collaboration.aggregate import METHODS, pairwise_disagreement
from evaluation import calibration as cal
from evaluation import stats as st

DISAGREEMENT_FLOOR = 0.15


def _breakdown(correct: list[bool], keys: list) -> dict:
    groups: dict = collections.defaultdict(list)
    for key, hit in zip(keys, correct):
        groups[key].append(hit)
    return {
        str(key): {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}
        for key, hits in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def evaluate_methods(problems, bundles: list[Bundle],
                     per_method: list[dict[str, dict]]) -> dict:
    """Score every aggregation method over one set of problems.

    `per_method[i]` is collaboration.aggregate.run_all(bundles[i]).
    """
    if not (len(problems) == len(bundles) == len(per_method)):
        raise ValueError("problems, bundles and results must be the same length")

    families = [p.family for p in problems]
    templates = [p.template for p in problems]
    difficulties = [p.difficulty for p in problems]

    report: dict = {"n_problems": len(problems), "methods": {}}

    correctness: dict[str, list[bool]] = {}
    for method in METHODS:
        answers = [row[method]["answer"] for row in per_method]
        confidences = [row[method]["confidence"] for row in per_method]
        correct = [is_correct(a, p.answer) for a, p in zip(answers, problems)]
        correctness[method] = correct

        generations = [row[method].get("generations", 0) for row in per_method]
        answered = sum(1 for a in answers if a is not None)

        entry = {
            "accuracy": round(st.accuracy(correct), 6),
            "accuracy_ci": st.bootstrap_ci(correct),
            "coverage": round(answered / len(answers), 6),
            "mean_generations": round(sum(generations) / len(generations), 3),
            "calibration": cal.summarise(confidences, correct),
            "by_family": _breakdown(correct, families),
            "by_difficulty": _breakdown(correct, difficulties),
            "by_template": _breakdown(correct, templates),
        }
        fallback = {row[method].get("fallback") for row in per_method} - {None}
        if fallback:
            entry["fallback"] = sorted(fallback)[0]
        report["methods"][method] = entry

    # ---- the health checks that decide whether any of this means anything
    disagreements = [pairwise_disagreement(b) for b in bundles]
    mean_disagreement = sum(disagreements) / len(disagreements) if disagreements else 0.0
    compliance = [b.format_compliance() for b in bundles]
    verifier_verdicts = collections.Counter(b.verifier_verdict() for b in bundles)

    report["health"] = {
        "mean_pairwise_disagreement": round(mean_disagreement, 6),
        "disagreement_floor": DISAGREEMENT_FLOOR,
        "disagreement_ok": bool(mean_disagreement >= DISAGREEMENT_FLOOR),
        "mean_format_compliance": round(sum(compliance) / len(compliance), 6)
        if compliance else 0.0,
        "verifier_verdicts": {str(k): v for k, v in verifier_verdicts.items()},
        "problems_with_a_disagreement": sum(1 for b in bundles if b.disagreement()),
    }
    if mean_disagreement < DISAGREEMENT_FLOOR:
        report["health"]["warning"] = (
            f"agents disagreed on only {mean_disagreement:.1%} of pairs, below the "
            f"{DISAGREEMENT_FLOOR:.0%} floor. Aggregation methods cannot differ much "
            "when the things being aggregated are nearly identical, so the "
            "comparison below is weak evidence either way."
        )

    # ---- is any difference real
    report["comparisons"] = {
        "vs_M1_single": st.compare_methods(correctness, "M1_single"),
        "vs_M2_self_consistency": st.compare_methods(correctness, "M2_self_consistency"),
    }

    # ---- the stated-vs-measured confidence question
    verbal, measured = [], []
    for bundle in bundles:
        if bundle.solver is not None:
            verbal.append(bundle.solver.parsed.self_confidence)
            measured.append(bundle.solver.confidence)
    report["confidence_agreement"] = cal.verbal_vs_measured(verbal, measured)

    return report


def format_table(report: dict) -> str:
    """The results table, as a reviewer would want to read it."""
    lines = [
        f"{'method':<24}{'acc':>8}{'95% CI':>18}{'cov':>7}{'ECE':>8}{'gen':>7}",
        "-" * 72,
    ]
    for method, entry in report["methods"].items():
        ci = entry["accuracy_ci"]
        interval = f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci["low"] is not None else "-"
        lines.append(
            f"{method:<24}{entry['accuracy']:>8.4f}{interval:>18}"
            f"{entry['coverage']:>7.2f}{entry['calibration']['ece']:>8.4f}"
            f"{entry['mean_generations']:>7.1f}"
        )
    lines.append("-" * 72)
    health = report["health"]
    lines.append(f"pairwise disagreement   {health['mean_pairwise_disagreement']:.1%}"
                 f"   (floor {health['disagreement_floor']:.0%})"
                 f"   {'OK' if health['disagreement_ok'] else 'BELOW FLOOR'}")
    lines.append(f"format compliance       {health['mean_format_compliance']:.1%}")
    return "\n".join(lines)
