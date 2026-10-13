"""The five aggregation methods, as pure functions over one Bundle.

M1  single agent          the Solver alone                       1 generation
M2  self-consistency      k Solver samples, majority vote        k generations
M3  multi-agent vote      Solver + Alternative + Coordinator     4 generations
M4  confidence-weighted   the same votes, weighted by MEASURED   4 generations
                          confidence, and vetoed by the Verifier
M5  neural fusion         a trained coordinator over the agents' 4 generations
                          hidden states and confidences

Why these five and not others: M1 is the baseline any multi-agent system has
to beat to justify its cost. M2 is the honest competitor — it spends the same
number of generations on one role instead of four, and a great deal of
published multi-agent benefit disappears once you compare against it. M3 and
M4 are the aggregation-at-the-text-level methods. M5 aggregates at the level
of learned representations, which is the actual research question.

Every function returns the same shape, so the evaluation harness treats them
interchangeably:

    (answer, confidence, detail)

`confidence` is always in [0, 1] and is the number fed to the calibration
analysis. None of these confidences is invented: each is computed from the
model's own token probabilities or from agreement among generations.
"""

from __future__ import annotations

from collections import Counter

from agents.roster import Bundle

METHODS = ("M1_single", "M2_self_consistency", "M3_majority",
           "M4_confidence_weighted", "M5_neural_fusion")


def m1_single(bundle: Bundle) -> tuple[str | None, float, dict]:
    """The Solver, alone. One generation; everything else must beat this."""
    solver = bundle.solver
    if solver is None:
        return None, 0.0, {"generations": 0}
    return solver.answer, solver.confidence, {
        "generations": 1,
        "source": "solver",
    }


def m2_self_consistency(bundle: Bundle, k: int | None = None) -> tuple[str | None, float, dict]:
    """Sample the Solver k times, take the most common answer.

    Confidence is the vote share -- the fraction of samples that agreed --
    which is the standard and honest confidence for this method.
    """
    samples = bundle.samples if k is None else bundle.samples[:k]
    answers = [s.answer for s in samples if s.answer is not None]
    if not answers:
        return m1_single(bundle)[0], 0.0, {"generations": len(samples), "agreed": 0}

    counts = Counter(answers)
    answer, votes = counts.most_common(1)[0]
    return answer, votes / len(samples), {
        "generations": len(samples),
        "agreed": votes,
        "distinct": len(counts),
    }


def m3_majority(bundle: Bundle) -> tuple[str | None, float, dict]:
    """Unweighted majority over the answering agents. The text-level baseline."""
    answers = [o.answer for o in bundle.answering() if o.answer is not None]
    if not answers:
        return None, 0.0, {"generations": bundle.generations, "voters": 0}

    counts = Counter(answers)
    answer, votes = counts.most_common(1)[0]
    return answer, votes / len(answers), {
        "generations": bundle.generations,
        "voters": len(answers),
        "agreed": votes,
        "distinct": len(counts),
    }


def m4_confidence_weighted(bundle: Bundle, veto_weight: float = 0.5
                           ) -> tuple[str | None, float, dict]:
    """Votes weighted by measured confidence, with a Verifier veto.

    Two mechanisms, both reported separately so their contributions can be
    told apart in the ablation:

      weighting   each agent's vote counts for exp(mean token log-prob), the
                  confidence it actually had rather than one it asserted
      veto        if the Verifier says "incorrect", the candidate it was shown
                  loses half its weight. Not all of it: a verifier at this
                  model scale is itself unreliable, and a full veto would let
                  one noisy generation overrule three.
    """
    answering = [o for o in bundle.answering() if o.answer is not None]
    if not answering:
        return None, 0.0, {"generations": bundle.generations, "voters": 0}

    verdict = bundle.verifier_verdict()
    vetoed = bundle.solver.answer if (verdict == "incorrect" and bundle.solver) else None

    weights: Counter = Counter()
    for output in answering:
        weight = max(output.confidence, 1e-6)
        if vetoed is not None and output.answer == vetoed:
            weight *= veto_weight
        weights[output.answer] += weight

    total = sum(weights.values())
    answer, weight = weights.most_common(1)[0]
    return answer, weight / total, {
        "generations": bundle.generations,
        "voters": len(answering),
        "vetoed_candidate": vetoed,
        "verifier_verdict": verdict,
        "critic_objected": bundle.critic_found_issue(),
    }


def m5_neural_fusion(bundle: Bundle, coordinator=None) -> tuple[str | None, float, dict]:
    """A trained coordinator over agent hidden states, confidences and signals.

    With no trained coordinator this falls back to M4 and SAYS SO in the
    detail dict, so a result table can never silently report M4's numbers
    under M5's name.
    """
    if coordinator is None:
        answer, confidence, detail = m4_confidence_weighted(bundle)
        detail = dict(detail)
        detail["fallback"] = "no trained coordinator -- this row is M4"
        return answer, confidence, detail

    answer, confidence, detail = coordinator.decide(bundle)
    detail = dict(detail)
    detail["generations"] = bundle.generations
    return answer, confidence, detail


def run_all(bundle: Bundle, coordinator=None, k: int | None = None) -> dict[str, dict]:
    """Every method on one bundle. The evaluation harness calls only this."""
    results: dict[str, dict] = {}
    for name, (answer, confidence, detail) in {
        "M1_single": m1_single(bundle),
        "M2_self_consistency": m2_self_consistency(bundle, k),
        "M3_majority": m3_majority(bundle),
        "M4_confidence_weighted": m4_confidence_weighted(bundle),
        "M5_neural_fusion": m5_neural_fusion(bundle, coordinator),
    }.items():
        results[name] = {
            "answer": answer,
            "confidence": float(min(max(confidence, 0.0), 1.0)),
            **detail,
        }
    return results


def pairwise_disagreement(bundle: Bundle) -> float:
    """Fraction of agent PAIRS that gave different answers.

    The blueprint calls a low value here the quiet killer: if every agent says
    the same thing, every aggregation method scores identically and the whole
    comparison is vacuous. Below roughly 15% the right response is to fix
    decoding diversity and role training, not to report the comparison.
    """
    answers = [o.answer for o in bundle.answering() if o.answer is not None]
    if len(answers) < 2:
        return 0.0
    pairs = differing = 0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            pairs += 1
            differing += answers[i] != answers[j]
    return differing / pairs
