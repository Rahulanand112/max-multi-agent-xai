"""V0 protocol: three roles, a majority-vote coordinator, a decision record.

This is the architectural skeleton, wired to the real trained checkpoint. It is
deliberately the SIMPLEST coordinator in blueprint section J -- majority voting,
method M3 -- so that the neural fusion of Version 5 has something honest to be
compared against later.

What it does NOT do yet, and must not be presented as doing:
  * the model has had no reasoning training, so the agents produce noise
  * the coordinator has no learned parameters
  * there is no verification of correctness against ground truth

What it DOES demonstrate: five roles addressing one frozen checkpoint through
role-control tokens, each returning an answer, a measured confidence and a
pooled hidden state, aggregated into a machine-readable decision record. Every
later version replaces the coordinator and keeps this interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from agents.base import AgentOutput, RoleAgent
from tokenizer.special_tokens import ALTERNATIVE, CRITIC, SOLVER, VERIFIER


@dataclass
class Decision:
    question: str
    final_answer: str | None
    confidence: float
    method: str
    agents_run: list[str] = field(default_factory=list)
    conclusions: dict[str, str | None] = field(default_factory=dict)
    supporting: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)
    disagreement: bool = False
    escalated: bool = False
    verification: dict = field(default_factory=dict)
    agent_confidences: dict[str, float] = field(default_factory=dict)
    format_compliance: float = 0.0
    decision_path: list[str] = field(default_factory=list)
    uncertainty: str = ""
    generations: int = 0
    seeds: dict[str, int] = field(default_factory=dict)
    outputs: list[AgentOutput] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "agents_run": self.agents_run,
            "conclusions": self.conclusions,
            "supporting": self.supporting,
            "conflicting": self.conflicting,
            "disagreement": self.disagreement,
            "escalated": self.escalated,
            "verification": self.verification,
            "agent_confidences": {k: round(v, 4) for k, v in self.agent_confidences.items()},
            "format_compliance": round(self.format_compliance, 4),
            "decision_path": self.decision_path,
            "uncertainty": self.uncertainty,
            "generations": self.generations,
            "seeds": self.seeds,
        }


def build_agents(model, tokenizer, device, max_new_tokens: int = 60) -> dict[str, RoleAgent]:
    """The V0 roster. Temperatures follow blueprint section G.

    The Solver is greedy (the reference attempt), the Alternative deliberately
    decorrelated, the Critic slightly varied because a deterministic critic
    repeats one objection, and the Verifier greedy because a verifier that
    varies run to run is not a verifier.
    """
    spec = [
        ("solver", SOLVER, 0.0, 50),
        ("alternative", ALTERNATIVE, 0.9, 40),
        ("critic", CRITIC, 0.7, 50),
        ("verifier", VERIFIER, 0.0, 50),
    ]
    return {
        name: RoleAgent(name, token, model, tokenizer, device,
                        temperature=temp, top_k=top_k, max_new_tokens=max_new_tokens)
        for name, token, temp, top_k in spec
    }


def run_protocol(agents: dict[str, RoleAgent], question: str, seed: int = 1337) -> Decision:
    """Solver + Alternative in parallel -> Critic -> Verifier -> majority vote."""
    path: list[str] = ["question"]
    outputs: list[AgentOutput] = []
    seeds: dict[str, int] = {}

    # Round 1: two independent attempts. The Alternative must NOT see the
    # Solver's answer, or it anchors and the two candidates stop being two.
    solver = agents["solver"].run(question, seed=seed)
    alternative = agents["alternative"].run(question, seed=seed + 1)
    outputs += [solver, alternative]
    seeds.update({"solver": seed, "alternative": seed + 1})
    path.append("solver+alternative")

    candidates = [o.answer for o in (solver, alternative) if o.answer]
    leading = Counter(candidates).most_common(1)[0][0] if candidates else None

    # Round 2: criticism of the leading candidate
    critic = agents["critic"].run(question, seed=seed + 2, candidate=leading or "")
    outputs.append(critic)
    seeds["critic"] = seed + 2
    path.append("critic")

    # Round 3: independent verification, informed by the critique
    verifier = agents["verifier"].run(
        question, seed=seed + 3,
        candidate=leading or "", issues=critic.text[:120],
    )
    outputs.append(verifier)
    seeds["verifier"] = seed + 3
    path.append("verifier")

    # Round 4: agreement check
    answers = {o.name: o.answer for o in outputs if o.name != "critic"}
    distinct = {a for a in answers.values() if a is not None}
    disagreement = len(distinct) > 1 or not distinct
    path.append("disagree" if disagreement else "agree")

    # Round 5: majority vote weighted by nothing -- this is M3, the baseline
    votes = Counter(a for a in answers.values() if a is not None)
    final_answer, n_votes = votes.most_common(1)[0] if votes else (None, 0)
    path += ["majority_vote", "answer"]

    supporting = [n for n, a in answers.items() if a is not None and a == final_answer]
    conflicting = [n for n, a in answers.items() if a is not None and a != final_answer]

    confidences = {o.name: o.confidence for o in outputs}
    agreement_fraction = n_votes / max(len(answers), 1)
    supporting_conf = [confidences[n] for n in supporting] or [0.0]
    confidence = (sum(supporting_conf) / len(supporting_conf)) * agreement_fraction

    compliant = sum(1 for o in outputs if o.parsed.compliant) / len(outputs)

    if final_answer is None:
        uncertainty = ("no agent produced a parseable answer -- expected before "
                       "Stage 2 reasoning training")
    elif disagreement:
        uncertainty = f"agents disagreed; {n_votes}/{len(answers)} supported the final answer"
    else:
        uncertainty = "all answering agents agreed"

    return Decision(
        question=question,
        final_answer=final_answer,
        confidence=confidence,
        method="M3 multi-agent majority vote (V0 stub)",
        agents_run=[o.name for o in outputs],
        conclusions={o.name: o.answer for o in outputs},
        supporting=supporting,
        conflicting=conflicting,
        disagreement=disagreement,
        escalated=False,
        verification={
            "verdict": verifier.parsed.verdict,
            "evidence": verifier.text[:160].replace("\n", " "),
            "compliant": verifier.parsed.compliant,
        },
        agent_confidences=confidences,
        format_compliance=compliant,
        decision_path=path,
        uncertainty=uncertainty,
        generations=len(outputs),
        seeds=seeds,
        outputs=outputs,
    )
