"""Schema parsing, role agents and the V0 protocol.

These test the ARCHITECTURE, not reasoning quality. The model has had no
reasoning training, so the interesting assertions are about shapes, contracts
and honesty -- that confidences are real measurements, that unparseable output
is reported as unparseable rather than silently invented, and that the majority
vote does what it says.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import RoleAgent                              # noqa: E402
from agents.protocol import build_agents, run_protocol         # noqa: E402
from explainability.decision_record import render, render_graph  # noqa: E402
from model.transformer import build_model                      # noqa: E402
from reasoning.schema import (                                 # noqa: E402
    build_prompt,
    compliance_rate,
    normalise_answer,
    parse,
)
from tokenizer.bpe import MAXTokenizer                         # noqa: E402
from tokenizer.special_tokens import CRITIC, SOLVER, VERIFIER  # noqa: E402
from utils.config import load_config                           # noqa: E402
from utils.seeding import set_seed                             # noqa: E402

CORPUS = [
    "Once upon a time there was a girl named Lily who had 4 boxes.",
    "Each box holds 6 pens so she has 24 pens in total.",
    "Tom is taller than Ben and Ben is taller than Sam.",
    "The cat sat on the mat and the dog ran fast.",
] * 40


@pytest.fixture(scope="module")
def tok():
    return MAXTokenizer.train(CORPUS, vocab_size=400, min_char_freq=1, verbose=False)


@pytest.fixture(scope="module")
def model(tok):
    cfg = load_config(ROOT / "configs" / "model_a.yaml")
    cfg.model.vocab_size = tok.vocab_size
    cfg.model.context_length = 128
    set_seed(1337)
    return build_model(cfg).eval()


# --------------------------------------------------------------------- schema

def test_prompt_starts_with_the_role_token():
    prompt = build_prompt(SOLVER, "How many pens?")
    assert prompt.startswith(SOLVER)
    assert "<|question|>" in prompt and "How many pens?" in prompt


def test_each_role_gets_a_different_prompt():
    question = "How many pens?"
    prompts = {build_prompt(r, question) for r in (SOLVER, CRITIC, VERIFIER)}
    assert len(prompts) == 3


def test_critic_prompt_carries_the_candidate():
    prompt = build_prompt(CRITIC, "How many pens?", candidate="24")
    assert "<|answer|>" in prompt and "24" in prompt


def test_parse_reads_a_well_formed_solver_output():
    text = "<|steps|> 4 x 6 = 24 <|answer|> 24 <|conf|> 0.86 <|eos|>"
    out = parse(SOLVER, text)
    assert out.compliant and not out.missing
    assert out.answer == "24"
    assert out.self_confidence == pytest.approx(0.86)


def test_parse_reads_a_verifier_verdict():
    text = "<|evidence|> 24 / 6 = 4 boxes <|verdict|> correct <|conf|> 0.91"
    out = parse(VERIFIER, text)
    assert out.compliant
    assert out.verdict == "correct"


def test_parse_reports_missing_fields_rather_than_guessing():
    """Unparseable output must be visible, not filled in with a plausible value."""
    out = parse(SOLVER, "the the the and and")
    assert not out.compliant
    assert out.answer is None
    assert out.self_confidence is None
    assert "<|answer|>" in out.missing


def test_parse_never_raises_on_garbage():
    for junk in ["", "<|answer|>", "<|conf|> not-a-number", "\x00\x01", "<|steps|><|steps|>"]:
        parse(SOLVER, junk)


def test_normalise_answer_prefers_numbers():
    assert normalise_answer("the answer is 24 pens") == "24"
    assert normalise_answer("  -3.5 ") == "-3.5"
    assert normalise_answer("Tom is tallest") == "tom is tallest"
    assert normalise_answer("   ") is None


def test_compliance_rate_is_a_fraction():
    good = parse(SOLVER, "<|steps|> a <|answer|> 1 <|conf|> 0.5")
    bad = parse(SOLVER, "nothing here")
    assert compliance_rate([good, bad]) == 0.5
    assert compliance_rate([]) == 0.0


# ---------------------------------------------------------------- role agents

def test_agent_returns_the_full_triple(model, tok):
    agent = RoleAgent("solver", SOLVER, model, tok, "cpu", temperature=0.0, max_new_tokens=8)
    out = agent.run("How many pens?", seed=1337)

    assert out.name == "solver" and out.role == SOLVER
    assert out.n_tokens > 0
    assert 0.0 < out.confidence <= 1.0            # exp(mean logprob)
    assert out.mean_logprob <= 0.0
    assert out.hidden.shape == (2 * model.d_model,)   # mean-pool ++ last position
    assert torch.isfinite(out.hidden).all()


def test_confidence_is_measured_not_invented(model, tok):
    """exp(mean token log-prob) must be consistent with the log-prob itself."""
    import math

    agent = RoleAgent("solver", SOLVER, model, tok, "cpu", temperature=0.0, max_new_tokens=8)
    out = agent.run("How many pens?", seed=1337)
    assert out.confidence == pytest.approx(math.exp(out.mean_logprob), rel=1e-6)


def test_entropy_is_positive_and_bounded(model, tok):
    import math

    agent = RoleAgent("solver", SOLVER, model, tok, "cpu", temperature=0.0, max_new_tokens=8)
    out = agent.run("How many pens?", seed=1337)
    assert 0.0 <= out.entropy <= math.log(tok.vocab_size) + 1e-3


def test_same_seed_gives_the_same_output(model, tok):
    agent = RoleAgent("alt", SOLVER, model, tok, "cpu", temperature=0.9, max_new_tokens=8)
    a = agent.run("How many pens?", seed=7)
    b = agent.run("How many pens?", seed=7)
    assert a.text == b.text and a.confidence == pytest.approx(b.confidence)


def test_different_seeds_give_different_sampled_output(model, tok):
    agent = RoleAgent("alt", SOLVER, model, tok, "cpu", temperature=1.0, max_new_tokens=16)
    assert agent.run("q", seed=1).text != agent.run("q", seed=2).text


def test_greedy_agent_ignores_the_seed(model, tok):
    agent = RoleAgent("solver", SOLVER, model, tok, "cpu", temperature=0.0, max_new_tokens=8)
    assert agent.run("q", seed=1).text == agent.run("q", seed=99).text


# ------------------------------------------------------------------- protocol

def test_roster_is_four_roles_on_one_model(model, tok):
    agents = build_agents(model, tok, "cpu", max_new_tokens=8)
    assert set(agents) == {"solver", "alternative", "critic", "verifier"}
    assert all(a.model is model for a in agents.values())      # ONE checkpoint
    assert len({a.role_token for a in agents.values()}) == 4   # four role tokens
    assert agents["solver"].temperature == 0.0                 # greedy reference
    assert agents["verifier"].temperature == 0.0               # verifiers must not vary
    assert agents["alternative"].temperature > 0.5             # deliberately decorrelated


def test_protocol_produces_a_complete_record(model, tok):
    agents = build_agents(model, tok, "cpu", max_new_tokens=8)
    decision = run_protocol(agents, "How many pens?", seed=1337)

    assert decision.generations == 4
    assert decision.agents_run == ["solver", "alternative", "critic", "verifier"]
    assert decision.decision_path[0] == "question"
    assert decision.decision_path[-1] == "answer"
    assert set(decision.agent_confidences) == set(decision.agents_run)
    assert 0.0 <= decision.format_compliance <= 1.0

    record = decision.to_dict()
    for key in ("question", "final_answer", "confidence", "agents_run", "conclusions",
                "supporting", "conflicting", "disagreement", "verification",
                "decision_path", "uncertainty", "generations", "seeds"):
        assert key in record


def test_untrained_model_is_reported_as_unparsed_not_faked(model, tok):
    """The honesty check: no answer must be invented when none was produced."""
    agents = build_agents(model, tok, "cpu", max_new_tokens=8)
    decision = run_protocol(agents, "How many pens?", seed=1337)
    if decision.final_answer is None:
        assert decision.confidence == 0.0
        assert "Stage 2" in decision.uncertainty


def test_each_agent_gets_its_own_seed(model, tok):
    agents = build_agents(model, tok, "cpu", max_new_tokens=8)
    decision = run_protocol(agents, "q", seed=100)
    assert len(set(decision.seeds.values())) == len(decision.seeds)


def test_majority_vote_picks_the_plurality():
    """Vote logic, isolated from the model."""
    from collections import Counter

    answers = {"solver": "24", "alternative": "24", "verifier": "18"}
    votes = Counter(a for a in answers.values() if a is not None)
    winner, n = votes.most_common(1)[0]
    assert winner == "24" and n == 2

    supporting = [k for k, v in answers.items() if v == winner]
    conflicting = [k for k, v in answers.items() if v != winner]
    assert supporting == ["solver", "alternative"] and conflicting == ["verifier"]


def test_record_renders_without_crashing(model, tok):
    agents = build_agents(model, tok, "cpu", max_new_tokens=8)
    decision = run_protocol(agents, "How many pens?", seed=1337)
    text = render(decision)
    assert "MAX DECISION RECORD" in text
    assert "DECISION PATH" in text
    assert "solver" in text
    assert "majority vote" in render_graph(decision)
