"""The structured reasoning schema, and a tolerant parser for it.

Stage 2 will train the model to emit these fields. Until then the parser is
still worth having: it defines the contract the SFT data must satisfy, and its
FAILURE rate on an untrained model is itself a measurement -- the "format
compliance" metric of blueprint section J.

The schema produces reasoning SUMMARIES, not exposed chain-of-thought. What the
model writes after <|steps|> is a short, structured justification we can show a
user, not a claim to have revealed the model's internal deliberation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tokenizer.special_tokens import (
    ALTERNATIVE,
    ANSWER,
    CONF,
    CRITIC,
    EVIDENCE,
    ISSUE,
    QUESTION,
    SOLVER,
    STEPS,
    VERDICT,
    VERIFIER,
)

ROLE_FIELDS: dict[str, list[str]] = {
    SOLVER: [STEPS, ANSWER, CONF],
    ALTERNATIVE: [STEPS, ANSWER, CONF],
    CRITIC: [ISSUE, CONF],
    VERIFIER: [EVIDENCE, VERDICT, CONF],
}

_FIELD_RE = re.compile(r"(<\|[a-z_]+\|>)")


@dataclass
class ParsedOutput:
    """What a role agent claimed, as far as we could read it."""
    role: str
    fields: dict[str, str] = field(default_factory=dict)
    answer: str | None = None
    verdict: str | None = None
    self_confidence: float | None = None
    compliant: bool = False
    missing: list[str] = field(default_factory=list)
    raw: str = ""


def build_prompt(role: str, question: str, candidate: str | None = None,
                 issues: str | None = None) -> str:
    """Assemble the prompt for one role. Roles differ only by these tokens."""
    parts = [role, QUESTION, " " + question.strip()]
    if candidate is not None:
        parts += [ANSWER, " " + candidate.strip()]
    if issues is not None:
        parts += [ISSUE, " " + issues.strip()]
    parts.append(ROLE_FIELDS.get(role, [STEPS])[0])
    return "".join(parts)


def parse(role: str, text: str) -> ParsedOutput:
    """Read a role's output. Never raises -- unparseable output is a result."""
    out = ParsedOutput(role=role, raw=text)

    # split on field markers, keeping them
    chunks = _FIELD_RE.split(text)
    current: str | None = None
    for chunk in chunks:
        if _FIELD_RE.fullmatch(chunk):
            current = chunk
        elif current is not None:
            out.fields.setdefault(current, chunk.strip())

    expected = ROLE_FIELDS.get(role, [])
    out.missing = [f for f in expected if f not in out.fields]
    out.compliant = not out.missing

    if ANSWER in out.fields:
        out.answer = normalise_answer(out.fields[ANSWER])
    if VERDICT in out.fields:
        verdict = out.fields[VERDICT].lower()
        out.verdict = "correct" if "correct" in verdict and "in" not in verdict[:2] else \
                      ("incorrect" if "incorrect" in verdict else verdict.split()[0] if verdict else None)
    if CONF in out.fields:
        match = re.search(r"[01]?\.\d+|[01]\b", out.fields[CONF])
        if match:
            try:
                value = float(match.group())
                out.self_confidence = min(max(value, 0.0), 1.0)
            except ValueError:
                pass
    return out


def normalise_answer(text: str) -> str | None:
    """Reduce an answer span to a comparable token.

    Numeric answers win: MAX-Reason answers are numbers, labels or short
    strings, so exact match after normalisation is a fair scoring rule.
    """
    text = text.strip()
    if not text:
        return None
    number = re.search(r"-?\d+(?:\.\d+)?", text)
    if number:
        return number.group()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:6]) if words else None


def compliance_rate(outputs: list[ParsedOutput]) -> float:
    """Fraction of outputs that parsed into the full schema.

    Below ~90% the multi-agent evaluation is measuring a parser, not a model.
    """
    return sum(1 for o in outputs if o.compliant) / len(outputs) if outputs else 0.0
