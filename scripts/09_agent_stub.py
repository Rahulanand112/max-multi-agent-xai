#!/usr/bin/env python3
"""V0 multi-agent stub, running on the real trained checkpoint.

    python scripts/09_agent_stub.py \
        --checkpoint /content/drive/MyDrive/max_v0/checkpoints/ckpt_A_final.pt

Review 1 demo step 7. It demonstrates the ARCHITECTURE -- four role-control
tokens addressing one frozen checkpoint, each returning an answer, a measured
confidence and a pooled hidden state, aggregated into a decision record.

READ THIS BEFORE PRESENTING IT
------------------------------
The model has had NO reasoning training. Stage 2 has not happened. The role
tokens exist in the vocabulary and were reserved before pretraining began, but
the model has never seen an example of what should follow <|solver|>. So the
agents will produce noise, most outputs will not parse, and the format
compliance figure will be low.

That is the correct and expected result, and saying so first is the difference
between a modest demonstration and a disappointing one. What this proves is
that the plumbing is real and connected to weights we trained -- not that the
system reasons. Version 3 trains the reasoning behaviour; Version 5 replaces
the majority vote with the learned fusion.

The confidences are honest: each is exp(mean token log-probability) computed
from the model's own distribution over the tokens it emitted. Nothing here is
a placeholder number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.protocol import build_agents, run_protocol            # noqa: E402
from explainability.decision_record import render, render_graph   # noqa: E402
from model.transformer import build_model                         # noqa: E402
from reasoning.schema import compliance_rate                      # noqa: E402
from tokenizer.bpe import MAXTokenizer                            # noqa: E402
from training.checkpoint import describe, load_checkpoint         # noqa: E402
from utils.config import Config, load_config                      # noqa: E402
from utils.logging_utils import get_logger, write_manifest        # noqa: E402
from utils.seeding import set_seed                                # noqa: E402

log = get_logger("agents")

QUESTIONS = [
    "Maya has 4 boxes. Each box holds 6 pens. How many pens does she have?",
    "Tom is taller than Ben. Ben is taller than Sam. Who is tallest?",
    "The cat sat on the mat. Where did the cat sit?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_A_final.pt")
    parser.add_argument("--config", default="configs/train_a.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--question", default=None, help="run a single question")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--no-raw", action="store_true", help="hide raw agent text")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    tok_cfg = load_config(ROOT / args.tokenizer_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    log.info("=" * 74)
    log.info("MAX V0 MULTI-AGENT STUB  |  device %s", device)
    log.info("=" * 74)

    tok = MAXTokenizer.load(ROOT / tok_cfg.paths.out_dir)
    header = describe(ckpt_path)
    payload = load_checkpoint(
        ckpt_path,
        expected_tokenizer_fingerprint=tok.fingerprint(),
        restore_rng=False, map_location=device,
    )
    ckpt_cfg = Config(payload["config"])
    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    log.info("checkpoint    : %s (step %s, %s tokens)",
             ckpt_path.name, f"{header['step']:,}", f"{header['tokens_seen']:,}")
    log.info("fingerprint   : %s  (matched)", header["tokenizer_fingerprint"])
    log.info("parameters    : %s  -- ONE model, four role tokens, zero extra weights",
             f"{model.num_parameters():,}")

    agents = build_agents(model, tok, device, max_new_tokens=args.max_new_tokens)
    for agent in agents.values():
        log.info("  %s", agent)

    log.info("")
    log.info("NOTE: this model has had no reasoning training (Stage 2 is Version 3).")
    log.info("      Agent outputs will be noise and most will not parse. The point")
    log.info("      is that the architecture runs on weights we trained ourselves.")
    log.info("")

    questions = [args.question] if args.question else QUESTIONS
    records: list[dict] = []
    all_outputs = []

    for i, question in enumerate(questions):
        decision = run_protocol(agents, question, seed=args.seed + i * 10)
        all_outputs += [o.parsed for o in decision.outputs]
        print()
        print(render(decision, show_text=not args.no_raw))
        if i == 0:
            print()
            print(render_graph(decision))
            print()
        records.append(decision.to_dict())

    overall = compliance_rate(all_outputs)
    hidden_dim = decision.outputs[0].hidden.shape[0]

    log.info("")
    log.info("-" * 74)
    log.info("SUMMARY OVER %d QUESTION(S)", len(questions))
    log.info("  format compliance     : %.0f%%  (expected to be low before Stage 2)",
             100 * overall)
    log.info("  generations per answer: %d", records[0]["generations"])
    log.info("  pooled hidden state   : %d dims per agent -- this is what the",
             hidden_dim)
    log.info("                          Version 5 coordinator will fuse, instead of text")
    log.info("-" * 74)

    run_dir = ROOT / "experiments" / "agent_stub_v0"
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "decision_records.json").write_text(json.dumps(records, indent=2))

    write_manifest(
        run_dir,
        stage="multi_agent_v0_stub",
        checkpoint=str(ckpt_path),
        checkpoint_step=header["step"],
        tokenizer_fingerprint=header["tokenizer_fingerprint"],
        n_parameters=model.num_parameters(),
        device=device,
        seed=args.seed,
        method="M3 majority vote",
        agents={n: {"role_token": a.role_token, "temperature": a.temperature,
                    "top_k": a.top_k} for n, a in agents.items()},
        questions=questions,
        measured={
            "format_compliance": round(overall, 4),
            "generations_per_answer": records[0]["generations"],
            "pooled_hidden_dim": int(hidden_dim),
        },
        caveat=("The model has had no reasoning training. Agent outputs are "
                "expected to be unparseable noise. This stub demonstrates the "
                "architecture and its connection to our own trained weights, "
                "not reasoning capability."),
        records=records,
    )
    log.info("records -> %s", run_dir / "decision_records.json")


if __name__ == "__main__":
    main()
