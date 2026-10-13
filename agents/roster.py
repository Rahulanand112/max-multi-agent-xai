"""The five agents, and one run of the protocol over a single problem.

Everything downstream — every aggregation method, the calibration study, the
decision record, the demo — consumes one `Bundle`. That is the central design
decision in the evaluation, and it is a fairness decision rather than an
engineering one:

    all five methods are scored on the SAME generations.

If each method re-sampled the model, a method could win by getting luckier
draws, and the comparison would measure sampling noise. Generating once and
aggregating many ways means the only thing that differs between M1 and M5 is
the aggregation, which is the thing being compared.

The inference budget is counted in generations, not in wall-clock, and it is
recorded on every Bundle. M1 costs 1 generation. Self-consistency at k=4 costs
4. The multi-agent protocol costs 4 (solver, alternative, critic, verifier).
Those are the numbers that make "at a matched inference budget" a checkable
claim rather than a hopeful phrase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.base import AgentOutput, RoleAgent
from reasoning.schema import normalise_answer
from tokenizer.special_tokens import ALTERNATIVE, COORDINATOR, CRITIC, SOLVER, VERIFIER

# (name, role token, temperature, top_k) -- the reasons for each temperature
# are in the docstring of build_roster.
ROSTER_SPEC = [
    ("solver", SOLVER, 0.0, 50),
    ("alternative", ALTERNATIVE, 0.9, 40),
    ("critic", CRITIC, 0.7, 50),
    ("verifier", VERIFIER, 0.0, 50),
    ("coordinator", COORDINATOR, 0.0, 50),
]


@dataclass
class Bundle:
    """Every generation produced for one problem, plus what they claimed."""
    question: str
    gold: str | None = None
    solver: AgentOutput | None = None
    alternative: AgentOutput | None = None
    critic: AgentOutput | None = None
    verifier: AgentOutput | None = None
    coordinator: AgentOutput | None = None
    samples: list[AgentOutput] = field(default_factory=list)   # self-consistency draws
    generations: int = 0
    seeds: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------- helpers

    def agents(self) -> list[AgentOutput]:
        out = [self.solver, self.alternative, self.critic, self.verifier, self.coordinator]
        return [o for o in out if o is not None]

    def answering(self) -> list[AgentOutput]:
        """Agents that propose an answer. The Critic does not; it objects."""
        out = [self.solver, self.alternative, self.coordinator]
        return [o for o in out if o is not None and o.answer is not None]

    def candidates(self) -> list[str]:
        return [o.answer for o in self.answering() if o.answer is not None]

    def verifier_verdict(self) -> str | None:
        return self.verifier.parsed.verdict if self.verifier is not None else None

    def critic_found_issue(self) -> bool:
        """Did the Critic object to anything, or did it clear the candidate?"""
        if self.critic is None:
            return False
        from reasoning.render import NO_ERROR
        issue = (self.critic.parsed.fields.get("<|issue|>") or self.critic.text or "").lower()
        if not issue.strip():
            return False
        return "no error" not in issue and NO_ERROR.split(",")[0] not in issue

    def format_compliance(self) -> float:
        outputs = self.agents()
        return sum(1 for o in outputs if o.parsed.compliant) / len(outputs) if outputs else 0.0

    def disagreement(self) -> bool:
        distinct = {c for c in self.candidates()}
        return len(distinct) > 1

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "gold": self.gold,
            "generations": self.generations,
            "candidates": self.candidates(),
            "verifier_verdict": self.verifier_verdict(),
            "critic_objected": self.critic_found_issue(),
            "disagreement": self.disagreement(),
            "format_compliance": round(self.format_compliance(), 4),
            "agents": {
                o.name: {
                    "answer": o.answer,
                    "confidence": round(o.confidence, 6),
                    "entropy": round(o.entropy, 6),
                    "verbal_confidence": o.parsed.self_confidence,
                    "compliant": o.parsed.compliant,
                    "temperature": o.temperature,
                    "seed": o.seed,
                    "text": o.text[:200],
                }
                for o in self.agents()
            },
            "self_consistency_answers": [s.answer for s in self.samples],
            "seeds": self.seeds,
        }


def build_roster(model, tokenizer, device, max_new_tokens: int = 72,
                 temperatures: dict[str, float] | None = None) -> dict[str, RoleAgent]:
    """The five role views of one frozen checkpoint.

    Temperatures are not decoration. The Solver is greedy because it is the
    reference attempt and must be reproducible. The Alternative runs hot so the
    two candidates are actually two — at temperature 0 it would reproduce the
    Solver and every aggregation method would score identically, which is the
    failure mode the blueprint flags as "the quiet killer". The Critic is
    slightly varied because a deterministic critic repeats one objection
    forever. The Verifier and Coordinator are greedy, because a verifier that
    changes its verdict between runs is not a verifier.
    """
    overrides = temperatures or {}
    return {
        name: RoleAgent(
            name, token, model, tokenizer, device,
            temperature=overrides.get(name, temp),
            top_k=top_k, max_new_tokens=max_new_tokens,
        )
        for name, token, temp, top_k in ROSTER_SPEC
    }


def run_bundle(
    roster: dict[str, RoleAgent],
    question: str,
    gold: str | None = None,
    seed: int = 1337,
    self_consistency: int = 0,
    sample_temperature: float = 0.8,
) -> Bundle:
    """Run the protocol once and collect everything it produced.

    Order matters and is the V1 protocol:
      1. Solver and Alternative answer INDEPENDENTLY. The Alternative must not
         see the Solver's answer, or it anchors and stops being a second
         opinion.
      2. The Critic is shown the leading candidate and asked for an objection.
      3. The Verifier is shown the candidate AND the critique, and gives a
         verdict. It sees the critique because a verifier working from the
         critic's findings is the point of having both.
      4. The Coordinator is shown every candidate.
    """
    bundle = Bundle(question=question, gold=gold)

    bundle.solver = roster["solver"].run(question, seed=seed)
    bundle.alternative = roster["alternative"].run(question, seed=seed + 1)
    bundle.seeds.update({"solver": seed, "alternative": seed + 1})
    bundle.generations += 2

    candidates = [o.answer for o in (bundle.solver, bundle.alternative) if o.answer]
    leading = candidates[0] if candidates else ""

    bundle.critic = roster["critic"].run(question, seed=seed + 2, candidate=leading)
    bundle.seeds["critic"] = seed + 2
    bundle.generations += 1

    bundle.verifier = roster["verifier"].run(
        question, seed=seed + 3, candidate=leading, issues=bundle.critic.text[:120])
    bundle.seeds["verifier"] = seed + 3
    bundle.generations += 1

    if "coordinator" in roster:
        # one <|answer|> marker per candidate -- the shape render_coordinator
        # trains on. Space-joining them under a single marker would be a
        # prompt the model has never seen.
        distinct = list(dict.fromkeys(candidates)) or [""]
        bundle.coordinator = roster["coordinator"].run(
            question, seed=seed + 4, candidates=distinct)
        bundle.seeds["coordinator"] = seed + 4
        bundle.generations += 1

    # extra Solver draws for the self-consistency baseline. Counted separately
    # so a budget-matched comparison can use exactly k of them.
    if self_consistency > 0:
        sampler = roster["solver"]
        original = sampler.temperature
        sampler.temperature = sample_temperature
        try:
            for i in range(self_consistency):
                bundle.samples.append(sampler.run(question, seed=seed + 100 + i))
        finally:
            sampler.temperature = original

    return bundle


def is_correct(predicted: str | None, gold: str | None) -> bool:
    """Exact match after the schema's normalisation. No partial credit.

    MAX-Reason answers are numbers, yes/no, or short names, so exact match
    after normalisation is a fair and fully automatic scoring rule — there is
    no judgement call anywhere in the evaluation.
    """
    if predicted is None or gold is None:
        return False
    return normalise_answer(predicted) == normalise_answer(gold)
