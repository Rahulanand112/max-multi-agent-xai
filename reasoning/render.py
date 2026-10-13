"""Turning one Problem into training text for each of the five roles.

Role specialisation in MAX does not come from five models. It comes from one
model trained on the same problems presented five ways, each introduced by its
own control token. This file is where that happens, and it is the honest
centre of the project: if the renderings here do not genuinely differ, the
agents will not genuinely differ, every aggregation method will score the
same, and the research question becomes unanswerable.

So each role gets supervision that only it could have:

  solver        the derivation, step by step, then the answer
  alternative   the same arithmetic in a condensed, differently-phrased route,
                so the two candidates are not textually identical
  critic        a candidate answer and the NAME of the error that produced it
                -- available only because we generated the wrong answer on
                purpose and recorded the mistake that reaches it
  verifier      a candidate, a recomputed check, and a verdict
  coordinator   several candidates and the one to choose

On the confidence field
-----------------------
`<|conf|>` is a TRAINED token, not a measurement. Its target is a stated
function of difficulty -- 0.95 falling by 0.08 per reasoning step -- so the
model learns to claim less certainty on longer chains. It is reported
separately from, and never confused with, the MEASURED confidence
`exp(mean token log-probability)` that agents/base.py computes from the
model's own distribution. Whether the two agree is itself one of the
calibration results, which is the reason both exist.
"""

from __future__ import annotations

import random

from reasoning.problem import Problem
from tokenizer.special_tokens import (
    ALTERNATIVE,
    ANSWER,
    CONF,
    COORDINATOR,
    CRITIC,
    EVIDENCE,
    ISSUE,
    QUESTION,
    SOLVER,
    STEPS,
    VERDICT,
    VERIFIER,
)

NO_ERROR = "no error found, the steps follow from the question"


def confidence_target(difficulty: int, correct: bool = True) -> float:
    """The value the model is trained to WRITE after <|conf|>.

    A declared rule, not a measurement: start at 0.95 and lose 0.08 per
    reasoning step. An item needing four steps has more places to go wrong
    than one needing one, and the target says so. Judgements about a wrong
    candidate are given a flat 0.90 -- spotting that an answer is wrong does
    not get harder with chain length in the same way.
    """
    if not correct:
        return 0.90
    return round(max(0.5, 0.95 - 0.08 * (max(1, difficulty) - 1)), 2)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def condense(steps: list[str]) -> str:
    """The Alternative Reasoner's surface form: one line, not a list.

    Same arithmetic, different words. This is what keeps the Solver and the
    Alternative from collapsing into the same string, which would make the
    whole multi-agent comparison vacuous. It does NOT claim to be a different
    method -- it is a different presentation, and the evaluation reports
    pairwise disagreement so the real degree of independence is measured
    rather than assumed.
    """
    if not steps:
        return ""
    body = steps[:-1] if len(steps) > 1 else steps
    tail = steps[-1]
    joined = ", then ".join(s.replace(" is ", " gives ") for s in body)
    if len(steps) == 1:
        return f"{tail.replace(' is ', ' gives ')}"
    return f"{joined}, which leaves {tail.split(' is ')[-1]}"


# ------------------------------------------------------------------- roles

def render_solver(problem: Problem) -> tuple[str, str]:
    """(prompt, completion). Loss is computed on the completion only."""
    prompt = f"{SOLVER}{QUESTION} {problem.question}{STEPS}"
    steps = ". ".join(problem.steps)
    completion = (f" {steps}.{ANSWER} {problem.answer}"
                  f"{CONF} {_fmt(confidence_target(problem.difficulty))}")
    return prompt, completion


def render_alternative(problem: Problem) -> tuple[str, str]:
    prompt = f"{ALTERNATIVE}{QUESTION} {problem.question}{STEPS}"
    completion = (f" {condense(problem.steps)}.{ANSWER} {problem.answer}"
                  f"{CONF} {_fmt(confidence_target(problem.difficulty))}")
    return prompt, completion


def render_critic(problem: Problem, candidate: str, error: str | None) -> tuple[str, str]:
    """Criticise a candidate. `error` is None when the candidate is correct."""
    prompt = f"{CRITIC}{QUESTION} {problem.question}{ANSWER} {candidate}{ISSUE}"
    issue = error if error else NO_ERROR
    completion = f" {issue}{CONF} {_fmt(confidence_target(problem.difficulty, error is None))}"
    return prompt, completion


def render_verifier(problem: Problem, candidate: str, correct: bool,
                    issue: str | None = None) -> tuple[str, str]:
    """Check a candidate against the derivation and give a verdict.

    The Verifier is shown the Critic's objection, because that is the order the
    protocol runs in: Critic, then Verifier informed by it. Training it without
    the objection and then supplying one at inference would hand the model a
    context it had never seen — so the issue section is part of the prompt in
    both places, and `build_prompt` produces the same shape.
    """
    issue_part = f"{ISSUE} {issue.strip()}" if issue else ""
    prompt = (f"{VERIFIER}{QUESTION} {problem.question}{ANSWER} {candidate}"
              f"{issue_part}{EVIDENCE}")
    check = problem.steps[-1] if problem.steps else f"the answer is {problem.answer}"
    evidence = (f"{check}, which matches the candidate" if correct
                else f"{check}, but the candidate says {candidate}")
    verdict = "correct" if correct else "incorrect"
    completion = (f" {evidence}{VERDICT} {verdict}"
                  f"{CONF} {_fmt(confidence_target(problem.difficulty, correct))}")
    return prompt, completion


def render_coordinator(problem: Problem, candidates: list[str]) -> tuple[str, str]:
    """Choose between candidate answers.

    The text coordinator is the baseline that the Version 5 neural fusion
    module is compared against. Keeping both means the comparison is between
    two trained things, not between a trained thing and a heuristic.
    """
    listed = "".join(f"{ANSWER} {c}" for c in candidates)   # one marker each
    prompt = f"{COORDINATOR}{QUESTION} {problem.question}{listed}{VERDICT}"
    completion = (f" the candidates disagree{ANSWER} {problem.answer}"
                  if len(set(candidates)) > 1 else
                  f" the candidates agree{ANSWER} {problem.answer}")
    completion += f"{CONF} {_fmt(confidence_target(problem.difficulty))}"
    return prompt, completion


# ------------------------------------------------------- the full rendering

def render_all(problem: Problem, rng: random.Random) -> list[tuple[str, str, str]]:
    """Every training sequence for one problem, as (role, prompt, completion).

    The Critic and Verifier each get one correct and one incorrect candidate,
    so neither can score well by always saying the same thing. That balance is
    deliberate: a critic trained only on wrong answers learns to object to
    everything, which is worse than useless inside a voting protocol.
    """
    out: list[tuple[str, str, str]] = []

    prompt, completion = render_solver(problem)
    out.append((SOLVER, prompt, completion))

    prompt, completion = render_alternative(problem)
    out.append((ALTERNATIVE, prompt, completion))

    wrong = rng.choice(problem.distractors) if problem.distractors else None

    # critic: once on the right answer, once on a wrong one
    prompt, completion = render_critic(problem, problem.answer, None)
    out.append((CRITIC, prompt, completion))
    if wrong is not None:
        prompt, completion = render_critic(problem, wrong.answer, wrong.error)
        out.append((CRITIC, prompt, completion))

    # verifier: the same balance, each shown the objection the Critic would
    # have raised about that candidate -- the protocol's actual running order
    prompt, completion = render_verifier(problem, problem.answer, correct=True,
                                         issue=NO_ERROR)
    out.append((VERIFIER, prompt, completion))
    if wrong is not None:
        prompt, completion = render_verifier(problem, wrong.answer, correct=False,
                                             issue=wrong.error)
        out.append((VERIFIER, prompt, completion))

    # coordinator: an agreeing pair and a disagreeing pair
    if wrong is not None:
        candidates = [problem.answer, wrong.answer]
        rng.shuffle(candidates)
        prompt, completion = render_coordinator(problem, candidates)
        out.append((COORDINATOR, prompt, completion))
    else:
        prompt, completion = render_coordinator(problem, [problem.answer, problem.answer])
        out.append((COORDINATOR, prompt, completion))

    return out


def render_narrative(problem: Problem) -> str:
    """A plain-text version for Stage 1 pretraining -- no role tokens at all.

    Mixing this into pretraining is what gives a 12M model any chance on
    arithmetic: it meets digits, counting and "times"/"minus" in ordinary
    prose before it is ever asked to follow the reasoning schema. Keeping the
    role tokens OUT here matters just as much -- if the control tokens appear
    during pretraining in an unstructured way, Stage 2 has to unlearn that
    before it can teach the schema.
    """
    steps = ". ".join(problem.steps)
    return f"{problem.question} {steps}. The answer is {problem.answer}."
