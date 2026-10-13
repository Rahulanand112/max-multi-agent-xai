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

ROLE_FIELDS: dict[str, list[str]] = {
    SOLVER: [STEPS, ANSWER, CONF],
    ALTERNATIVE: [STEPS, ANSWER, CONF],
    CRITIC: [ISSUE, CONF],
    VERIFIER: [EVIDENCE, VERDICT, CONF],
    # The text coordinator. It is given candidates and emits a verdict on
    # whether they agree, then the answer it selects. Its neural counterpart
    # lives in collaboration/fusion.py; both are evaluated, so the comparison
    # is between two trained aggregators rather than one and a heuristic.
    COORDINATOR: [VERDICT, ANSWER, CONF],
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
                 issues: str | None = None,
                 candidates: list[str] | None = None) -> str:
    """Assemble the prompt for one role. Roles differ only by these tokens.

    This function and reasoning/render.py must produce IDENTICAL prompt shapes.
    A prompt shape the model never saw in training is a silent accuracy loss:
    nothing raises, the model simply continues from a context it does not
    recognise. `tests/test_reasoning_data.py::test_inference_prompts_match_
    training_prompts` asserts the two agree for every role.

    `candidates` emits one <|answer|> marker per candidate, which is how the
    Coordinator sees them in training. Passing them space-joined under a single
    marker would be exactly that silent mismatch.
    """
    parts = [role, QUESTION, " " + question.strip()]
    if candidates is not None:
        for item in candidates:
            parts += [ANSWER, " " + item.strip()]
    elif candidate is not None:
        parts += [ANSWER, " " + candidate.strip()]
    if issues is not None:
        parts += [ISSUE, " " + issues.strip()]
    parts.append(ROLE_FIELDS.get(role, [STEPS])[0])
    return "".join(parts)


def parse(role: str, text: str) -> ParsedOutput:
    """Read a role's output. Never raises -- unparseable output is a result.

    The leading field is IMPLIED, not emitted. build_prompt() ends the prompt
    with the role's first field marker, so the model continues from after it
    and never writes it. An earlier version of this function counted that
    first field as missing, which capped format compliance at zero no matter
    what the model produced -- the V0 stub's reported 0% was partly this
    parser and not only an untrained model. The first chunk of generated text,
    before any marker, is therefore credited to the role's first field.
    """
    out = ParsedOutput(role=role, raw=text)
    expected_fields = ROLE_FIELDS.get(role, [])

    # split on field markers, keeping them
    chunks = _FIELD_RE.split(text)
    current: str | None = None
    for index, chunk in enumerate(chunks):
        if _FIELD_RE.fullmatch(chunk):
            current = chunk
        elif current is not None:
            out.fields.setdefault(current, chunk.strip())
        elif index == 0 and expected_fields and chunk.strip():
            # text before any marker continues the field the prompt ended on
            out.fields.setdefault(expected_fields[0], chunk.strip())

    out.missing = [f for f in expected_fields if f not in out.fields]
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
