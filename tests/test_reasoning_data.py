"""Tests for the generated benchmark and its role renderings.

The generators produce both the questions and the labels, so a bug here does
not announce itself as a crash — it announces itself as a model that scores
60% on a benchmark where the answers are wrong. These tests check the labels
against the arithmetic and the deductions directly.
"""

from __future__ import annotations

import random
import re

import pytest

from reasoning.generators import arithmetic, logic
from reasoning.problem import Problem, write_jsonl, load_jsonl
from reasoning.render import (confidence_target, render_all, render_narrative,
                              render_critic, render_verifier)
from reasoning.schema import ROLE_FIELDS, compliance_rate, normalise_answer, parse
from tokenizer.special_tokens import ROLE_TOKENS


# ------------------------------------------------------------ ground truth

def test_arithmetic_answers_follow_from_the_steps():
    """The final step must state the answer. If it does not, the label drifted."""
    for problem in arithmetic.generate(300, seed=5):
        last = problem.steps[-1]
        numbers = re.findall(r"-?\d+", last)
        assert numbers, f"no number in final step: {last}"
        assert numbers[-1] == problem.answer, (
            f"final step {last!r} does not end at the answer {problem.answer}")


def test_arithmetic_answers_are_never_negative():
    """"How many are left" has no negative answers; one means a template is wrong."""
    for problem in arithmetic.generate(500, seed=6):
        assert int(problem.answer) >= 0


def test_distractors_are_wrong_and_plausible():
    for problem in arithmetic.generate(300, seed=7) + logic.generate(300, seed=7):
        for distractor in problem.distractors:
            assert distractor.answer != problem.answer, "a distractor equals the answer"
            assert distractor.error, "a distractor carries no error description"
            assert distractor.kind, "a distractor carries no error class"


def test_transitive_chains_are_consistent():
    """Re-derive the ordering from the premises and check the stated extreme."""
    problems = [p for p in logic.generate(400, seed=8) if p.template == "transitive_order"]
    assert problems
    for problem in problems:
        pairs = [tuple(re.match(r"(\w+) is \w+ than (\w+)\.", f).groups())
                 for f in problem.facts]
        more = {a for a, _ in pairs}
        less = {b for _, b in pairs}
        top = (more - less).pop()
        bottom = (less - more).pop()
        assert problem.answer in (top, bottom)


def test_category_chains_answer_yes_or_no():
    problems = [p for p in logic.generate(400, seed=9) if p.template == "category_chain"]
    assert problems
    assert {p.answer for p in problems} == {"yes", "no"}
    # roughly balanced: a model that always says yes must not score well
    yes = sum(1 for p in problems if p.answer == "yes")
    assert 0.35 < yes / len(problems) < 0.65


def test_constraint_puzzles_have_a_unique_solution():
    """The generator discards non-unique puzzles; make sure none slips through."""
    problems = [p for p in logic.generate(300, seed=10) if p.template == "constraint_grid"]
    assert problems
    for problem in problems:
        pinned = [f for f in problem.facts if " has the " in f]
        assert len(pinned) == 1, "a constraint puzzle needs exactly one positive clue"


def test_generation_is_deterministic_in_the_seed():
    a = arithmetic.generate(50, seed=11)
    b = arithmetic.generate(50, seed=11)
    assert [p.question for p in a] == [p.question for p in b]
    assert [p.answer for p in a] == [p.answer for p in b]


def test_different_seeds_give_different_problems():
    a = {p.question for p in arithmetic.generate(100, seed=12)}
    b = {p.question for p in arithmetic.generate(100, seed=13)}
    assert len(a & b) < len(a) * 0.5


def test_duplicate_rate_stays_low_at_scale():
    """A high duplicate rate turns a reasoning benchmark into a memorisation one."""
    for generator in (arithmetic, logic):
        problems = generator.generate(5000, seed=14)
        unique = len({p.fingerprint() for p in problems})
        assert unique / 5000 > 0.95, f"{generator.__name__} duplicate rate too high"


def test_unknown_template_is_rejected():
    with pytest.raises(KeyError):
        arithmetic.generate(5, seed=1, templates=["no_such_template"])


# -------------------------------------------------------------- round trip

def test_problems_survive_a_jsonl_round_trip(tmp_path):
    original = arithmetic.generate(20, seed=15) + logic.generate(20, seed=15)
    path = tmp_path / "problems.jsonl"
    write_jsonl(original, path)
    restored = load_jsonl(path)
    assert len(restored) == len(original)
    assert [p.question for p in restored] == [p.question for p in original]
    assert [d.kind for p in restored for d in p.distractors] == \
           [d.kind for p in original for d in p.distractors]


# ----------------------------------------------------------------- renders

def test_every_rendered_target_parses_into_its_schema():
    """Our own supervision must satisfy the contract we score the model against.

    If the targets themselves do not parse, format compliance measures the
    renderer rather than the model.
    """
    rng = random.Random(0)
    outputs = []
    for problem in arithmetic.generate(60, seed=16) + logic.generate(60, seed=16):
        for role, _, completion in render_all(problem, rng):
            outputs.append(parse(role, completion))
    assert compliance_rate(outputs) == 1.0


def test_rendered_prompts_end_on_the_first_expected_field():
    rng = random.Random(1)
    for problem in arithmetic.generate(20, seed=17):
        for role, prompt, _ in render_all(problem, rng):
            assert prompt.endswith(ROLE_FIELDS[role][0])


def test_every_role_token_is_exercised():
    rng = random.Random(2)
    roles = {role for p in arithmetic.generate(10, seed=18)
             for role, _, _ in render_all(p, rng)}
    assert roles == set(ROLE_TOKENS)


def test_critic_and_verifier_see_both_right_and_wrong_candidates():
    """A critic trained only on wrong answers learns to object to everything."""
    rng = random.Random(3)
    verdicts = []
    issues = []
    for problem in arithmetic.generate(80, seed=19):
        for role, _, completion in render_all(problem, rng):
            parsed = parse(role, completion)
            if role == "<|verifier|>":
                verdicts.append(parsed.verdict)
            if role == "<|critic|>":
                issues.append("no error" in completion)
    assert set(verdicts) == {"correct", "incorrect"}
    assert 0.3 < sum(issues) / len(issues) < 0.7


def test_alternative_wording_differs_from_the_solver():
    """Identical candidates would make every aggregation method score the same."""
    rng = random.Random(4)
    differing = 0
    problems = arithmetic.generate(60, seed=20)
    for problem in problems:
        rendered = dict((role, completion) for role, _, completion in render_all(problem, rng))
        differing += rendered["<|solver|>"] != rendered["<|alternative|>"]
    assert differing == len(problems)


def test_solver_and_alternative_agree_on_the_answer():
    """Different words, same arithmetic. A different ANSWER would be a bug."""
    rng = random.Random(5)
    for problem in arithmetic.generate(40, seed=21):
        answers = {parse(role, completion).answer
                   for role, _, completion in render_all(problem, rng)
                   if role in ("<|solver|>", "<|alternative|>")}
        assert answers == {normalise_answer(problem.answer)}


def test_confidence_target_falls_with_difficulty():
    values = [confidence_target(d) for d in (1, 2, 3, 4)]
    assert values == sorted(values, reverse=True)
    assert all(0.5 <= v <= 0.95 for v in values)


def test_narrative_contains_no_role_tokens():
    """Role tokens in pretraining text would have to be unlearned in Stage 2."""
    for problem in arithmetic.generate(30, seed=22) + logic.generate(30, seed=22):
        text = render_narrative(problem)
        assert "<|" not in text
        assert problem.answer in text


def test_verifier_verdict_matches_the_candidate():
    problem = arithmetic.generate(1, seed=23)[0]
    _, right = render_verifier(problem, problem.answer, correct=True)
    _, wrong = render_verifier(problem, "999999", correct=False)
    assert parse("<|verifier|>", right).verdict == "correct"
    assert parse("<|verifier|>", wrong).verdict == "incorrect"


def test_critic_reports_no_error_on_a_correct_candidate():
    problem = arithmetic.generate(1, seed=24)[0]
    _, completion = render_critic(problem, problem.answer, None)
    assert "no error" in completion


# ------------------------------------------------------------ the parser

def test_parser_credits_the_implied_leading_field():
    """The prompt ends with the first marker, so the model never emits it.

    Counting it as missing capped format compliance at zero regardless of
    what the model produced.
    """
    parsed = parse("<|solver|>", " 2 times 3 is 6.<|answer|> 6<|conf|> 0.95")
    assert parsed.compliant
    assert parsed.answer == "6"
    assert parsed.self_confidence == pytest.approx(0.95)


def test_parser_reports_what_is_missing_rather_than_raising():
    parsed = parse("<|solver|>", " some steps but nothing else")
    assert not parsed.compliant
    assert "<|answer|>" in parsed.missing
    assert parsed.answer is None


def test_parser_never_raises_on_garbage():
    for text in ("", "<|answer|>", "<|<|>>", "\x00\x01", "<|conf|> not a number"):
        parse("<|solver|>", text)


# --------------------------------------------- training vs inference prompts

def test_inference_prompts_match_training_prompts():
    """The silent accuracy loss this project nearly shipped.

    reasoning/render.py builds the prompts the model is TRAINED on;
    reasoning/schema.build_prompt builds the ones it is ASKED at inference. If
    they differ by so much as a marker, the model is given a context it has
    never seen, nothing raises, and accuracy quietly falls. Two real
    mismatches were found this way: the Verifier was trained without the
    Critic's objection but given one at inference, and the Coordinator was
    trained on one <|answer|> marker per candidate but asked with them
    space-joined under a single marker.
    """
    from reasoning.render import NO_ERROR, render_all
    from reasoning.schema import build_prompt
    from tokenizer.special_tokens import (ALTERNATIVE, COORDINATOR, CRITIC,
                                          SOLVER, VERIFIER)

    rng = random.Random(0)
    for problem in arithmetic.generate(6, seed=30) + logic.generate(6, seed=30):
        rendered = render_all(problem, rng)
        _ = problem.distractors[0].answer if problem.distractors else None

        by_role: dict[str, list[str]] = {}
        for role, prompt, _ in rendered:
            by_role.setdefault(role, []).append(prompt)

        assert build_prompt(SOLVER, problem.question) in by_role[SOLVER]
        assert build_prompt(ALTERNATIVE, problem.question) in by_role[ALTERNATIVE]
        assert build_prompt(CRITIC, problem.question,
                            candidate=problem.answer) in by_role[CRITIC]
        assert build_prompt(VERIFIER, problem.question, candidate=problem.answer,
                            issues=NO_ERROR) in by_role[VERIFIER]
        # the Coordinator's candidates are drawn at random, so rebuild from the
        # prompt itself: every rendered prompt must be reproducible by
        # build_prompt given the candidates it contains
        for prompt in by_role[COORDINATOR]:
            body = prompt.split("<|question|>", 1)[1].rsplit("<|verdict|>", 1)[0]
            candidates = [c.strip() for c in body.split("<|answer|>")[1:]]
            assert build_prompt(COORDINATOR, problem.question,
                                candidates=candidates) == prompt


def test_build_prompt_emits_one_marker_per_candidate():
    from reasoning.schema import build_prompt
    from tokenizer.special_tokens import COORDINATOR
    prompt = build_prompt(COORDINATOR, "q?", candidates=["6", "8"])
    assert prompt.count("<|answer|>") == 2
    assert " 6" in prompt and " 8" in prompt


def test_verifier_prompt_carries_the_critics_objection():
    from reasoning.render import render_verifier
    problem = arithmetic.generate(1, seed=31)[0]
    prompt, _ = render_verifier(problem, "999", correct=False, issue="added instead")
    assert "<|issue|> added instead" in prompt
    bare, _ = render_verifier(problem, "999", correct=False)
    assert "<|issue|>" not in bare
