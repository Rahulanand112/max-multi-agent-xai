"""Statistics for claims about differences between methods.

Two methods scoring 61.2% and 63.8% on 2,000 problems may or may not differ.
Reporting the two numbers and letting a reader infer "M5 is better" is the
single most common way a small evaluation overstates itself. These functions
exist so the project never does that.

  bootstrap_ci        confidence interval on one method's accuracy
  paired_bootstrap    interval on the DIFFERENCE between two methods, resampled
                      on the same problems — the right test here, because
                      every method is scored on the same bundles
  mcnemar             exact test on the disagreements between two methods

The paired tests matter precisely because of how the harness is built: all
five methods see identical generations, so their errors are correlated, and an
unpaired test would be far too conservative. Pairing uses that structure
instead of throwing it away.

No SciPy — the exact binomial tail for McNemar is a few lines, and keeping the
dependency list short matters more than the convenience.
"""

from __future__ import annotations

import math
import random


def accuracy(correct: list[bool]) -> float:
    return sum(1 for c in correct if c) / len(correct) if correct else float("nan")


def bootstrap_ci(correct: list[bool], n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 1337) -> dict:
    """Percentile bootstrap interval for one method's accuracy."""
    if not correct:
        return {"accuracy": None, "low": None, "high": None, "n": 0}

    rng = random.Random(seed)
    n = len(correct)
    draws = []
    for _ in range(n_resamples):
        total = sum(1 for _ in range(n) if correct[rng.randrange(n)])
        draws.append(total / n)
    draws.sort()

    return {
        "accuracy": round(accuracy(correct), 6),
        "low": round(draws[int(alpha / 2 * n_resamples)], 6),
        "high": round(draws[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))], 6),
        "n": n,
        "resamples": n_resamples,
    }


def paired_bootstrap(a: list[bool], b: list[bool], n_resamples: int = 2000,
                     alpha: float = 0.05, seed: int = 1337) -> dict:
    """Interval on accuracy(a) - accuracy(b), resampling problems jointly.

    If the interval contains zero, the two methods are not distinguishable at
    this sample size. That sentence belongs in the results table, not in a
    footnote.
    """
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length result lists")
    if not a:
        return {"difference": None, "low": None, "high": None, "n": 0}

    rng = random.Random(seed)
    n = len(a)
    differences = []
    for _ in range(n_resamples):
        hits_a = hits_b = 0
        for _ in range(n):
            index = rng.randrange(n)
            hits_a += a[index]
            hits_b += b[index]
        differences.append((hits_a - hits_b) / n)
    differences.sort()

    low = differences[int(alpha / 2 * n_resamples)]
    high = differences[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return {
        "difference": round(accuracy(a) - accuracy(b), 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "n": n,
        "significant": bool(low > 0 or high < 0),
    }


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar test on the cases where the two methods disagree.

    Only the discordant pairs carry information: problems both got right, or
    both got wrong, say nothing about which method is better.
    """
    if len(a) != len(b):
        raise ValueError("McNemar needs equal-length result lists")

    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    discordant = a_only + b_only

    if discordant == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0,
                "note": "the two methods never disagreed"}

    # exact two-sided binomial test at p = 0.5
    smaller = min(a_only, b_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    p_value = min(1.0, 2 * tail)

    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": round(p_value, 6),
        "significant_at_05": bool(p_value < 0.05),
    }


def compare_methods(results: dict[str, list[bool]], baseline: str,
                    seed: int = 1337) -> dict[str, dict]:
    """Every method against one baseline, paired, with both tests.

    The baseline should be M1 (a single agent) for the headline claim and M2
    (self-consistency) for the honest one — beating a single generation is
    easy when you are allowed four, and the project says so in its results.
    """
    if baseline not in results:
        raise KeyError(f"baseline {baseline!r} not among {sorted(results)}")

    out: dict[str, dict] = {}
    for name, correct in results.items():
        if name == baseline:
            continue
        out[name] = {
            "vs": baseline,
            "bootstrap": paired_bootstrap(correct, results[baseline], seed=seed),
            "mcnemar": mcnemar(correct, results[baseline]),
        }
    return out
