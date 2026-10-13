"""Tests for aggregation, fusion, calibration and the statistics.

These are the functions that turn generations into the claims in the results
table, so a bug here does not crash — it produces a number that looks fine and
is wrong. Each test below pins one behaviour that a results table depends on.

The bundles are built by hand rather than by running a model: the aggregation
logic is what is under test, and synthetic agents let a test state exactly
what the agents said.
"""

from __future__ import annotations

import math

import pytest
import torch

from agents.base import AgentOutput
from agents.roster import Bundle, is_correct
from collaboration import aggregate as agg
from collaboration.fusion import (FusionCoordinator, N_SCALAR_FEATURES,
                                  extract_candidates, fusion_loss)
from evaluation import calibration as cal
from evaluation import stats as st
from reasoning.schema import parse

HIDDEN = 16


def make_output(name: str, role: str, answer: str | None, confidence: float,
                text: str | None = None, temperature: float = 0.0) -> AgentOutput:
    body = text if text is not None else f" steps.<|answer|> {answer}<|conf|> 0.90"
    return AgentOutput(
        role=role, name=name, text=body, parsed=parse(role, body),
        answer=answer, confidence=confidence,
        mean_logprob=math.log(max(confidence, 1e-9)), entropy=1.0,
        hidden=torch.ones(HIDDEN) * confidence, n_tokens=8,
        seed=1, temperature=temperature,
    )


def make_bundle(solver: str | None, alternative: str | None, gold: str = "6",
                solver_conf: float = 0.9, alt_conf: float = 0.4,
                verdict: str = "correct", samples: list[str] | None = None) -> Bundle:
    verifier_text = f" evidence<|verdict|> {verdict}<|conf|> 0.90"
    bundle = Bundle(question="q", gold=gold)
    bundle.solver = make_output("solver", "<|solver|>", solver, solver_conf)
    bundle.alternative = make_output("alternative", "<|alternative|>", alternative,
                                     alt_conf, temperature=0.9)
    bundle.critic = make_output("critic", "<|critic|>", None, 0.5,
                                text=" no error found<|conf|> 0.80")
    bundle.verifier = make_output("verifier", "<|verifier|>", None, 0.8,
                                  text=verifier_text)
    bundle.generations = 4
    for index, answer in enumerate(samples or []):
        bundle.samples.append(make_output(f"s{index}", "<|solver|>", answer, 0.5))
    return bundle


# -------------------------------------------------------------- scoring

def test_exact_match_after_normalisation():
    assert is_correct("6", "6")
    assert is_correct(" 6 ", "6")
    assert is_correct("the answer is 6", "6")
    assert not is_correct("16", "6")
    assert not is_correct(None, "6")


def test_unparseable_answers_count_as_wrong_not_skipped():
    """Dropping them would flatter a method that often produces no answer."""
    bundle = make_bundle(None, None)
    answer, _, _ = agg.m1_single(bundle)
    assert answer is None
    assert not is_correct(answer, "6")


# ------------------------------------------------------------ the methods

def test_m1_is_the_solver_alone_at_one_generation():
    bundle = make_bundle("6", "7")
    answer, confidence, detail = agg.m1_single(bundle)
    assert answer == "6"
    assert confidence == pytest.approx(0.9)
    assert detail["generations"] == 1


def test_m2_majority_over_samples_and_vote_share_confidence():
    bundle = make_bundle("6", "7", samples=["6", "6", "7", "6"])
    answer, confidence, detail = agg.m2_self_consistency(bundle)
    assert answer == "6"
    assert confidence == pytest.approx(0.75)
    assert detail["generations"] == 4


def test_m2_respects_the_budget_k():
    """A budget-matched comparison must use exactly k samples, not all of them."""
    bundle = make_bundle("6", "7", samples=["6", "7", "7", "7"])
    assert agg.m2_self_consistency(bundle, k=1)[2]["generations"] == 1
    assert agg.m2_self_consistency(bundle, k=4)[0] == "7"


def test_m3_is_unweighted_majority():
    bundle = make_bundle("6", "7", solver_conf=0.99, alt_conf=0.01)
    _, confidence, detail = agg.m3_majority(bundle)
    assert detail["voters"] == 2
    assert confidence == pytest.approx(0.5)      # a tie, confidence must say so


def test_m4_lets_a_confident_agent_outvote_an_unconfident_one():
    bundle = make_bundle("6", "7", solver_conf=0.9, alt_conf=0.1)
    answer, _, _ = agg.m4_confidence_weighted(bundle)
    assert answer == "6"
    flipped = make_bundle("6", "7", solver_conf=0.1, alt_conf=0.9)
    assert agg.m4_confidence_weighted(flipped)[0] == "7"


def test_verifier_veto_is_partial_not_absolute():
    """A full veto would let one noisy generation overrule everything else."""
    bundle = make_bundle("6", "7", solver_conf=0.9, alt_conf=0.5, verdict="incorrect")
    answer, _, detail = agg.m4_confidence_weighted(bundle)
    assert detail["vetoed_candidate"] == "6"
    assert answer == "7"                       # 0.9 * 0.5 = 0.45 < 0.5
    strong = make_bundle("6", "7", solver_conf=0.9, alt_conf=0.3, verdict="incorrect")
    assert agg.m4_confidence_weighted(strong)[0] == "6"   # 0.45 > 0.3, not overruled


def test_m5_without_a_coordinator_says_it_fell_back():
    """A results table must never show M4's numbers under M5's name."""
    _, _, detail = agg.m5_neural_fusion(make_bundle("6", "7"), coordinator=None)
    assert "fallback" in detail
    assert "M4" in detail["fallback"]


def test_run_all_returns_every_method_with_bounded_confidence():
    results = agg.run_all(make_bundle("6", "7", samples=["6", "6"]))
    assert set(results) == set(agg.METHODS)
    for row in results.values():
        assert 0.0 <= row["confidence"] <= 1.0


def test_pairwise_disagreement():
    assert agg.pairwise_disagreement(make_bundle("6", "6")) == 0.0
    assert agg.pairwise_disagreement(make_bundle("6", "7")) == 1.0


# -------------------------------------------------------------- the fusion

def test_candidate_extraction_is_per_distinct_answer():
    rows = extract_candidates(make_bundle("6", "6"))
    assert len(rows) == 1
    assert rows[0].scalars.shape == (N_SCALAR_FEATURES,)
    assert len(extract_candidates(make_bundle("6", "7"))) == 2


def test_agreement_is_carried_as_a_feature():
    unanimous = extract_candidates(make_bundle("6", "6"))[0]
    split = extract_candidates(make_bundle("6", "7"))[0]
    assert float(unanimous.scalars[3]) > float(split.scalars[3])


def test_coordinator_is_small_and_picks_a_candidate():
    coordinator = FusionCoordinator(HIDDEN, width=16)
    assert coordinator.num_parameters() < 300_000
    answer, confidence, detail = coordinator.decide(make_bundle("6", "7"))
    assert answer in ("6", "7")
    assert 0.0 <= confidence <= 1.0
    assert detail["candidates"] == 2


def test_coordinator_on_a_unanimous_bundle_does_not_invent_a_choice():
    _, _, detail = FusionCoordinator(HIDDEN, width=16).decide(make_bundle("6", "6"))
    assert detail["unanimous"]


def test_fusion_can_learn_a_separable_signal():
    """Train on a signal that IS in the features; it must get better than chance.

    Not a claim about real data -- a check that gradients reach the module and
    the loss is wired the right way round.
    """
    torch.manual_seed(0)
    coordinator = FusionCoordinator(HIDDEN, width=32, dropout=0.0)
    optimizer = torch.optim.AdamW(coordinator.parameters(), lr=5e-3)

    rows = []
    for index in range(80):
        target = index % 2
        hidden = torch.randn(2, HIDDEN)
        scalars = torch.rand(2, N_SCALAR_FEATURES)
        scalars[target, 0] = 0.95          # the correct one is the confident one
        scalars[1 - target, 0] = 0.05
        rows.append((hidden, scalars, target))

    for _ in range(25):
        for hidden, scalars, target in rows:
            optimizer.zero_grad()
            loss, _ = fusion_loss(coordinator(hidden, scalars), target)
            loss.backward()
            optimizer.step()

    coordinator.eval()
    with torch.no_grad():
        hits = sum(1 for h, s, t in rows if int(coordinator(h, s).argmax()) == t)
    assert hits / len(rows) > 0.75


def test_calibration_term_punishes_confident_mistakes():
    logits = torch.tensor([5.0, 0.0])
    confident_wrong, _ = fusion_loss(logits, target=1, calibration_weight=1.0)
    plain_wrong, _ = fusion_loss(logits, target=1, calibration_weight=0.0)
    assert float(confident_wrong) > float(plain_wrong)


# ---------------------------------------------------------- calibration

def test_perfect_calibration_scores_zero_ece():
    confidences = [0.0] * 50 + [1.0] * 50
    correct = [False] * 50 + [True] * 50
    assert cal.expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-9)


def test_confident_and_wrong_is_caught():
    confidences = [0.95] * 100
    correct = [False] * 100
    assert cal.expected_calibration_error(confidences, correct) == pytest.approx(0.95, abs=1e-6)
    assert cal.overconfidence(confidences, correct) == pytest.approx(0.95, abs=1e-6)


def test_overconfidence_is_signed():
    under = cal.overconfidence([0.1] * 10, [True] * 10)
    over = cal.overconfidence([0.9] * 10, [False] * 10)
    assert under < 0 < over


def test_bins_cover_every_prediction():
    confidences = [i / 100 for i in range(101)]
    correct = [i % 2 == 0 for i in range(101)]
    assert sum(b["count"] for b in cal.bin_predictions(confidences, correct)) == 101


def test_verbal_and_measured_correlation():
    agreement = cal.verbal_vs_measured([0.1, 0.5, 0.9], [0.2, 0.5, 0.8])
    assert agreement["correlation"] > 0.9
    assert cal.verbal_vs_measured([None, None], [0.5, 0.5])["correlation"] is None


# ------------------------------------------------------------- statistics

def test_bootstrap_interval_brackets_the_accuracy():
    correct = [True] * 70 + [False] * 30
    result = st.bootstrap_ci(correct, n_resamples=500)
    assert result["low"] < result["accuracy"] < result["high"]


def test_identical_methods_are_not_distinguishable():
    correct = [True, False] * 50
    result = st.paired_bootstrap(correct, list(correct), n_resamples=500)
    assert result["difference"] == 0.0
    assert not result["significant"]


def test_a_clear_difference_is_detected():
    better = [True] * 90 + [False] * 10
    worse = [False] * 100
    result = st.paired_bootstrap(better, worse, n_resamples=500)
    assert result["significant"]
    assert st.mcnemar(better, worse)["significant_at_05"]


def test_mcnemar_only_counts_disagreements():
    a = [True, True, False, False]
    b = [True, False, True, False]
    result = st.mcnemar(a, b)
    assert result["a_only"] == 1 and result["b_only"] == 1
    assert result["discordant"] == 2


def test_mcnemar_handles_perfect_agreement():
    result = st.mcnemar([True] * 10, [True] * 10)
    assert result["p_value"] == 1.0


def test_compare_methods_rejects_an_unknown_baseline():
    with pytest.raises(KeyError):
        st.compare_methods({"a": [True]}, baseline="missing")
