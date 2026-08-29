"""Rendering a Decision as a human-readable record.

Blueprint section I: the record reports which agents ran, what each concluded,
how confident each was, where they disagreed, what the verifier found, and how
the decision was reached. Every one of those is a fact about the system's
execution -- fully supported, no interpretation required.

Nothing here claims to explain why the *model* produced a token. That
distinction between system-level explanation and model-level analysis is the
one an examiner will press on.
"""

from __future__ import annotations

from agents.protocol import Decision

WIDTH = 74


def render(decision: Decision, show_text: bool = True, text_chars: int = 110) -> str:
    lines: list[str] = [
        "=" * WIDTH,
        "MAX DECISION RECORD",
        "=" * WIDTH,
        "",
        f"QUESTION        {decision.question}",
        "",
        f"FINAL ANSWER    {decision.final_answer if decision.final_answer else '(none parsed)'}",
        f"CONFIDENCE      {decision.confidence:.4f}",
        f"METHOD          {decision.method}",
        f"GENERATIONS     {decision.generations}",
        "",
        "-" * WIDTH,
        "AGENTS",
        "-" * WIDTH,
        f"  {'agent':<13}{'answer':<14}{'conf':>7}{'schema':>9}{'T':>7}{'seed':>7}",
    ]

    for out in decision.outputs:
        answer = out.answer if out.answer else "-"
        lines.append(
            f"  {out.name:<13}{str(answer)[:13]:<14}{out.confidence:>7.4f}"
            f"{('ok' if out.parsed.compliant else 'partial'):>9}"
            f"{out.temperature:>7.1f}{out.seed:>7}"
        )

    lines += [
        "",
        f"SUPPORTING      {', '.join(decision.supporting) or '(none)'}",
        f"CONFLICTING     {', '.join(decision.conflicting) or '(none)'}",
        f"DISAGREEMENT    {'yes' if decision.disagreement else 'no'}",
        f"ESCALATED       {'yes' if decision.escalated else 'no'}",
        "",
        f"VERIFICATION    verdict: {decision.verification.get('verdict') or '(unparsed)'}",
        f"                evidence: {decision.verification.get('evidence', '')[:60]}",
        "",
        f"FORMAT COMPLY   {decision.format_compliance:.0%} of agent outputs matched the schema",
        f"UNCERTAINTY     {decision.uncertainty}",
        "",
        "DECISION PATH   " + " -> ".join(decision.decision_path),
    ]

    if show_text:
        lines += ["", "-" * WIDTH, "RAW AGENT OUTPUT", "-" * WIDTH]
        for out in decision.outputs:
            snippet = out.text.replace("\n", " ")[:text_chars].strip()
            lines.append(f"  {out.name:<13}{snippet or '(empty)'}")

    lines += ["", "=" * WIDTH]
    return "\n".join(lines)


def render_graph(decision: Decision) -> str:
    """An ASCII decision graph. Version 6 replaces this with NetworkX + Plotly."""
    answers = decision.conclusions
    return "\n".join([
        "                        question",
        "                           |",
        "              +------------+------------+",
        "              |                         |",
        f"         solver [{str(answers.get('solver') or '-')[:6]:^6}]      "
        f"alternative [{str(answers.get('alternative') or '-')[:6]:^6}]",
        "              |                         |",
        "              +------------+------------+",
        "                           |",
        "                        critic",
        "                           |",
        f"                       verifier [{str(decision.verification.get('verdict') or '-')[:9]:^9}]",
        "                           |",
        f"                  {'DISAGREEMENT' if decision.disagreement else 'agreement'}",
        "                           |",
        "                    majority vote          <- V5 replaces this with neural fusion",
        "                           |",
        f"                 answer: {decision.final_answer or '(none)'}  "
        f"(confidence {decision.confidence:.3f})",
    ])
