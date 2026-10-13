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


def render_bundle(bundle, methods: dict[str, dict], gold: str | None = None) -> str:
    """The V1 record: one problem, every agent, and what each method concluded.

    The V0 renderer below reports a single aggregation. This one reports all
    five side by side, because the interesting explanation at this stage is
    not only "what did the system answer" but "where did the methods part
    company, and which agent caused it".

    Correctness is shown against ground truth we generated, so every tick and
    cross here is a fact rather than a judgement.
    """
    from agents.roster import is_correct

    lines = [
        "=" * WIDTH,
        "MAX DECISION RECORD",
        "=" * WIDTH,
        "",
        f"QUESTION        {bundle.question}",
        f"GROUND TRUTH    {gold if gold is not None else '(not supplied)'}",
        "",
        "-" * WIDTH,
        "AGENTS      one checkpoint, five role tokens, zero extra weights",
        "-" * WIDTH,
        f"  {'agent':<13}{'answer':<14}{'conf':>7}{'stated':>8}{'schema':>9}{'T':>6}",
    ]
    for output in bundle.agents():
        stated = output.parsed.self_confidence
        lines.append(
            f"  {output.name:<13}{str(output.answer or '-')[:13]:<14}"
            f"{output.confidence:>7.4f}"
            f"{(f'{stated:.2f}' if stated is not None else '-'):>8}"
            f"{('ok' if output.parsed.compliant else 'partial'):>9}"
            f"{output.temperature:>6.1f}"
        )

    if bundle.samples:
        answers = [s.answer or "-" for s in bundle.samples]
        lines += ["", f"  self-consistency draws: {', '.join(answers)}"]

    lines += [
        "",
        f"CRITIC          {'objected' if bundle.critic_found_issue() else 'found no error'}",
        f"VERIFIER        verdict: {bundle.verifier_verdict() or '(unparsed)'}",
        f"DISAGREEMENT    {'yes' if bundle.disagreement() else 'no'}"
        f"   candidates: {', '.join(bundle.candidates()) or '(none)'}",
        f"FORMAT COMPLY   {bundle.format_compliance():.0%} of agent outputs matched the schema",
        f"GENERATIONS     {bundle.generations} protocol + {len(bundle.samples)} self-consistency",
        "",
        "-" * WIDTH,
        "AGGREGATION     every method scored on these same generations",
        "-" * WIDTH,
        f"  {'method':<24}{'answer':<14}{'conf':>7}{'correct':>9}{'gen':>6}",
    ]
    for name, result in methods.items():
        correct = is_correct(result["answer"], gold) if gold is not None else None
        mark = "-" if correct is None else ("yes" if correct else "no")
        lines.append(
            f"  {name:<24}{str(result['answer'] or '-')[:13]:<14}"
            f"{result['confidence']:>7.4f}{mark:>9}{result.get('generations', 0):>6}"
        )
        if result.get("fallback"):
            lines.append(f"      note: {result['fallback']}")

    lines += ["", "=" * WIDTH]
    return "\n".join(lines)


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
