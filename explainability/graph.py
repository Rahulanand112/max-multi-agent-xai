"""The decision graph: what happened, as a picture.

A careful distinction, and the one an examiner is most likely to press on.
This is a SYSTEM-level explanation. Every node and edge is a fact about
execution — which agent ran, what it concluded, how confident it measurably
was, which candidate the coordinator selected. None of it claims to explain
why the model produced a particular token.

That second thing — model-level interpretability — is a different and much
harder problem. `attention_attribution` below goes as far as this project
honestly can: it reports which input positions the attention heads actually
weighted, which is a measurement of the computation, labelled as attention
weight and not as "the reason for the answer". Attention weight is not
explanation, and the function's docstring says so where someone reading the
output will see it.

Two renderings: `as_text` for terminals and logs, `as_dot` for a real graph.
Both are built from the same facts.
"""

from __future__ import annotations

import torch

ARROW = "   |"


def _cell(value, width: int = 9) -> str:
    text = "-" if value is None else str(value)
    return text[:width].center(width)


def as_text(bundle, decision: dict | None = None) -> str:
    """ASCII decision graph for one bundle."""
    answers = {o.name: o.answer for o in bundle.agents()}
    confidences = {o.name: o.confidence for o in bundle.agents()}
    chosen = decision["answer"] if decision else None
    confidence = decision["confidence"] if decision else None
    method = decision.get("method", "") if decision else ""

    lines = [
        "                          question",
        "                              |",
        "              +---------------+---------------+",
        "              |                               |",
        f"       solver [{_cell(answers.get('solver'))}]        "
        f"alternative [{_cell(answers.get('alternative'))}]",
        f"       conf   {confidences.get('solver', 0):.3f}                  "
        f"conf   {confidences.get('alternative', 0):.3f}",
        "              |                               |",
        "              +---------------+---------------+",
        "                              |",
        f"                          critic  [{'objected' if bundle.critic_found_issue() else 'no issue'}]",
        "                              |",
        f"                         verifier [{_cell(bundle.verifier_verdict())}]",
        "                              |",
        f"                     {'DISAGREEMENT' if bundle.disagreement() else '  agreement '}",
        "                              |",
    ]
    if bundle.samples:
        draws = ", ".join(str(s.answer or "-") for s in bundle.samples)
        lines.insert(0, f"   self-consistency draws: {draws}")
    lines += [
        f"                   {method or 'aggregation'}",
        "                              |",
        f"                     answer: {chosen if chosen is not None else '(none)'}"
        + (f"   (confidence {confidence:.3f})" if confidence is not None else ""),
    ]
    return "\n".join(lines)


def as_dot(bundle, decision: dict | None = None) -> str:
    """Graphviz DOT. Render with `dot -Tpng record.dot -o record.png`.

    Kept as DOT text rather than a rendered image so the graph is a
    committable artefact that diffs, and so nothing in the evaluation path
    depends on a drawing library being installed.
    """
    rows = ['digraph decision {', '  rankdir=TB;',
            '  node [shape=box, style=rounded, fontname="Helvetica"];',
            f'  q [label="question", shape=ellipse];']

    for output in bundle.agents():
        label = (f"{output.name}\\nanswer: {output.answer or '-'}\\n"
                 f"measured conf: {output.confidence:.3f}")
        rows.append(f'  {output.name} [label="{label}"];')

    for name in ("solver", "alternative"):
        if name in {o.name for o in bundle.agents()}:
            rows.append(f'  q -> {name};')
    names = {o.name for o in bundle.agents()}
    if "critic" in names:
        for name in ("solver", "alternative"):
            if name in names:
                rows.append(f'  {name} -> critic;')
    if "verifier" in names and "critic" in names:
        rows.append('  critic -> verifier;')

    if decision:
        label = (f"{decision.get('method', 'aggregation')}\\n"
                 f"answer: {decision['answer']}\\nconfidence: {decision['confidence']:.3f}")
        rows.append(f'  final [label="{label}", shape=box, style="rounded,bold"];')
        for name in names:
            rows.append(f'  {name} -> final [style=dashed];')

    rows.append('}')
    return "\n".join(rows)


@torch.no_grad()
def attention_attribution(model, tokenizer, text: str, layer: int = -1,
                          device: str = "cpu", top_k: int = 8) -> list[dict]:
    """Which input positions did the attention heads weight most?

    READ THE LABEL. This reports attention weight, which is a measurement of
    the model's computation. It is NOT an explanation of the answer, and it
    is not evidence that the model "used" those tokens: attention weight and
    causal importance are known to come apart, and a high-attention token can
    be irrelevant to the output. It is included because it is a real,
    reproducible measurement of the forward pass, and excluded from any claim
    about why the system answered as it did.
    """
    ids = tokenizer.encode(text)[: model.context_length]
    if not ids:
        return []
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.tok_emb(tokens)
    if model.pos_emb is not None:
        x = x + model.pos_emb(model.position_ids[:, :tokens.shape[1]])

    weights = None
    blocks = list(model.blocks)
    target = blocks[layer] if layer >= 0 else blocks[len(blocks) + layer]
    for block in blocks:
        if block is target:
            x, weights = block(x, return_attn=True)
        else:
            x = block(x)

    if weights is None:
        return []

    # average over heads, then read the final query position's distribution
    final_position = weights[0].mean(0)[-1]
    order = torch.argsort(final_position, descending=True)[:top_k]
    return [
        {
            "position": int(index),
            "token": tokenizer.decode([ids[int(index)]], skip_special=False),
            "attention": round(float(final_position[int(index)]), 6),
        }
        for index in order
    ]
