"""MAX demo: ask a question, watch five agents deliberate.

    streamlit run app/streamlit_app.py -- --checkpoint checkpoints/ckpt_B_sft_final.pt

What this shows, and what it does not
-------------------------------------
It shows the real system: one checkpoint this project trained from random
initialisation, addressed through five role tokens, with every number on
screen measured at the moment you press the button. Confidences are
exp(mean token log-probability) from the model's own distribution. The
decision graph is a record of what executed.

It does not show a model that reliably answers questions. At 12M parameters
trained for well under an hour, many answers will be wrong and some will not
parse. The demo displays failures exactly as it displays successes — a demo
that hides them would be worth less than no demo, because the first question
a reviewer asks is "what happens when it gets one wrong".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.roster import build_roster, is_correct, run_bundle      # noqa: E402
from collaboration.aggregate import pairwise_disagreement, run_all  # noqa: E402
from collaboration.fusion import FusionCoordinator                  # noqa: E402
from explainability.graph import as_text                            # noqa: E402
from inference.load import load_for_inference                       # noqa: E402
from reasoning.problem import load_jsonl                            # noqa: E402

EXAMPLES = [
    "Maya has 4 boxes. Each box holds 6 pens. She gives away 7 pens. "
    "How many pens does she have left?",
    "Ana is taller than Ben. Ben is taller than Cara. Who is the tallest?",
    "All roses are flowers. Fin has a rose. Does Fin have a flower?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_B_sft_final.pt")
    parser.add_argument("--coordinator", default="checkpoints/coordinator.pt")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts/max_bpe_a")
    known, _ = parser.parse_known_args()
    return known


@st.cache_resource(show_spinner="Loading the checkpoint...")
def load_everything(checkpoint: str, tokenizer_dir: str, coordinator_path: str):
    import torch
    model, tokenizer, header = load_for_inference(checkpoint, tokenizer_dir, root=ROOT)
    roster = build_roster(model, tokenizer, header["device"], max_new_tokens=72)

    coordinator = None
    path = ROOT / coordinator_path
    if path.exists():
        blob = torch.load(path, weights_only=False)
        coordinator = FusionCoordinator(blob["hidden_dim"], width=blob["width"])
        coordinator.load_state_dict(blob["model"])
        coordinator.eval()
    return roster, header, coordinator


@st.cache_data(show_spinner=False)
def load_test_problems() -> list:
    path = ROOT / "data" / "synth" / "test.jsonl"
    return load_jsonl(path)[:200] if path.exists() else []


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="MAX — Multi-Agent Explainable Reasoning",
                       layout="wide")

    st.title("MAX — Multi-Agent Explainable Reasoning")
    st.caption("One language model, built and trained from random initialisation. "
               "Five agents, addressed by role tokens. No pretrained weights at any stage.")

    try:
        roster, header, coordinator = load_everything(
            args.checkpoint, args.tokenizer, args.coordinator)
    except FileNotFoundError as error:
        st.error(f"{error}\n\nTrain a checkpoint first, or pass --checkpoint.")
        return

    with st.sidebar:
        st.subheader("Model")
        st.metric("Parameters", f"{header['n_parameters']:,}")
        st.metric("Training step", f"{header['step']:,}")
        st.write(f"**Tokenizer fingerprint** `{header['tokenizer_fingerprint']}`")
        st.write(f"**Device** {header['device']}")
        st.write("**Coordinator** " +
                 ("trained neural fusion" if coordinator else "not trained — M5 falls back to M4"))
        st.divider()
        temperature = st.slider("Alternative Reasoner temperature", 0.0, 1.5, 0.9, 0.1,
                                help="At 0 it reproduces the Solver and the two "
                                     "candidates stop being two.")
        k = st.slider("Self-consistency samples (M2 budget)", 0, 8, 4)
        seed = st.number_input("Seed", value=1337, step=1)

    problems = load_test_problems()
    choice = st.selectbox(
        "Pick a held-out test problem, or write your own below",
        ["(write my own)"] + [p.question for p in problems[:40]],
    )
    default = "" if choice == "(write my own)" else choice
    question = st.text_area("Question", value=default or EXAMPLES[0], height=90)
    gold = next((p.answer for p in problems if p.question == question), None)

    if not st.button("Run the five agents", type="primary"):
        st.info("Questions from the test split come with ground truth, so the demo "
                "can show whether each method was right — not just what it said.")
        return

    roster["alternative"].temperature = temperature
    with st.spinner("Generating..."):
        bundle = run_bundle(roster, question, gold=gold, seed=int(seed),
                            self_consistency=k)
        methods = run_all(bundle, coordinator, k=k or None)

    top = st.columns(4)
    final = methods["M5_neural_fusion"]
    top[0].metric("Final answer", final["answer"] or "—")
    top[1].metric("Confidence", f"{final['confidence']:.3f}")
    top[2].metric("Generations", bundle.generations + len(bundle.samples))
    top[3].metric("Agents disagreed", "yes" if bundle.disagreement() else "no")

    if gold is not None:
        correct = is_correct(final["answer"], gold)
        (st.success if correct else st.error)(
            f"Ground truth: **{gold}** — the system was {'right' if correct else 'wrong'}.")

    st.subheader("What each agent said")
    rows = []
    for output in bundle.agents():
        rows.append({
            "agent": output.name,
            "answer": output.answer or "—",
            "measured confidence": round(output.confidence, 4),
            "stated <|conf|>": output.parsed.self_confidence,
            "schema": "ok" if output.parsed.compliant else "partial",
            "temperature": output.temperature,
            "output": output.text[:160].replace("\n", " "),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Measured confidence is exp(mean token log-probability) from the model's "
               "own distribution. The stated value is a token it was trained to emit. "
               "They are different things and this project never conflates them.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Aggregation methods")
        table = []
        for name, result in methods.items():
            table.append({
                "method": name,
                "answer": result["answer"] or "—",
                "confidence": round(result["confidence"], 4),
                "generations": result.get("generations", 0),
                "correct": "—" if gold is None else
                           ("yes" if is_correct(result["answer"], gold) else "no"),
                "note": result.get("fallback", ""),
            })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(f"Pairwise disagreement on this problem: "
                   f"{pairwise_disagreement(bundle):.0%}. All five methods are scored "
                   "on the same generations, so the only thing differing is aggregation.")

    with right:
        st.subheader("Decision graph")
        st.code(as_text(bundle, {"answer": final["answer"],
                                 "confidence": final["confidence"],
                                 "method": "M5 neural fusion" if coordinator
                                           else "M4 confidence-weighted"}),
                language="text")

    with st.expander("Raw agent output"):
        for output in bundle.agents():
            st.markdown(f"**{output.name}** `T={output.temperature}` "
                        f"`seed={output.seed}`")
            st.code(output.text or "(empty)", language="text")


if __name__ == "__main__":
    main()
