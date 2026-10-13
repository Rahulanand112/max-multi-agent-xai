"""Calibration: does a stated confidence mean what it says?

A system that is 70% accurate and reports 0.95 confidence on everything is
worse than useless in a decision pipeline — it is confidently wrong and gives
no signal about when to escalate. Accuracy alone cannot see that; these
measures can.

  ECE   expected calibration error: bin predictions by confidence, and
        average |accuracy - confidence| over bins, weighted by bin size.
        Zero is perfect. It is the headline calibration number.
  MCE   maximum calibration error: the worst single bin. ECE can look fine
        while one bin is badly broken, which is exactly the bin a user would
        have trusted most.
  Brier mean squared error between confidence and correctness. Unlike ECE it
        is a proper scoring rule, so it cannot be gamed by a model that
        reports the base rate for everything.

The reliability table these produce is what the calibration plot draws, and
it is reported for every aggregation method — a method that buys accuracy by
destroying calibration has not obviously improved anything.
"""

from __future__ import annotations

import math


def bin_predictions(confidences: list[float], correct: list[bool], n_bins: int = 10
                    ) -> list[dict]:
    """Equal-width confidence bins with their accuracy. The reliability table."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")

    bins: list[dict] = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        members = [
            (c, k) for c, k in zip(confidences, correct)
            if (c >= low and c < high) or (index == n_bins - 1 and c == 1.0)
        ]
        if members:
            mean_confidence = sum(c for c, _ in members) / len(members)
            accuracy = sum(1 for _, k in members if k) / len(members)
        else:
            mean_confidence = accuracy = 0.0
        bins.append({
            "lower": round(low, 3),
            "upper": round(high, 3),
            "count": len(members),
            "mean_confidence": round(mean_confidence, 4),
            "accuracy": round(accuracy, 4),
            "gap": round(accuracy - mean_confidence, 4),
        })
    return bins


def expected_calibration_error(confidences: list[float], correct: list[bool],
                               n_bins: int = 10) -> float:
    if not confidences:
        return float("nan")
    total = len(confidences)
    error = 0.0
    for row in bin_predictions(confidences, correct, n_bins):
        if row["count"]:
            error += (row["count"] / total) * abs(row["accuracy"] - row["mean_confidence"])
    return error


def maximum_calibration_error(confidences: list[float], correct: list[bool],
                              n_bins: int = 10) -> float:
    if not confidences:
        return float("nan")
    gaps = [abs(row["accuracy"] - row["mean_confidence"])
            for row in bin_predictions(confidences, correct, n_bins) if row["count"]]
    return max(gaps) if gaps else float("nan")


def brier_score(confidences: list[float], correct: list[bool]) -> float:
    if not confidences:
        return float("nan")
    return sum((c - (1.0 if k else 0.0)) ** 2 for c, k in zip(confidences, correct)) / len(confidences)


def overconfidence(confidences: list[float], correct: list[bool]) -> float:
    """Mean confidence minus accuracy. Positive means systematically overconfident.

    Reported alongside ECE because ECE is unsigned: a model that is
    under-confident everywhere and one that is over-confident everywhere can
    score the same, and they are very different problems to have.
    """
    if not confidences:
        return float("nan")
    accuracy = sum(1 for k in correct if k) / len(correct)
    return sum(confidences) / len(confidences) - accuracy


def summarise(confidences: list[float], correct: list[bool], n_bins: int = 10) -> dict:
    accuracy = sum(1 for k in correct if k) / len(correct) if correct else float("nan")
    return {
        "n": len(confidences),
        "accuracy": round(accuracy, 6),
        "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
        "ece": round(expected_calibration_error(confidences, correct, n_bins), 6),
        "mce": round(maximum_calibration_error(confidences, correct, n_bins), 6),
        "brier": round(brier_score(confidences, correct), 6),
        "overconfidence": round(overconfidence(confidences, correct), 6),
        "bins": bin_predictions(confidences, correct, n_bins),
    }


def verbal_vs_measured(verbal: list[float | None], measured: list[float]) -> dict:
    """Do the model's STATED confidences track the ones we measured?

    `<|conf|>` is a token the model was trained to emit; `exp(mean token
    log-probability)` is computed from its distribution. They are different
    objects and this project never conflates them. Whether they agree is a
    finding in its own right: a model whose stated confidence is uncorrelated
    with its measured confidence has learned the format of self-report
    without the substance.
    """
    pairs = [(v, m) for v, m in zip(verbal, measured) if v is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "correlation": None}

    verbal_values = [v for v, _ in pairs]
    measured_values = [m for _, m in pairs]
    n = len(pairs)
    mean_v = sum(verbal_values) / n
    mean_m = sum(measured_values) / n
    covariance = sum((v - mean_v) * (m - mean_m) for v, m in pairs)
    var_v = sum((v - mean_v) ** 2 for v in verbal_values)
    var_m = sum((m - mean_m) ** 2 for m in measured_values)
    denominator = math.sqrt(var_v * var_m)

    return {
        "n": n,
        "mean_verbal": round(mean_v, 6),
        "mean_measured": round(mean_m, 6),
        "correlation": round(covariance / denominator, 6) if denominator > 0 else None,
    }
