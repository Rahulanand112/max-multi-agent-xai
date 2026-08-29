#!/usr/bin/env python3
"""Evaluate ckpt_A_final.pt in a fresh process -- the portability criterion.

    python scripts/08_evaluate.py \
        --checkpoint /content/drive/MyDrive/max_v0/checkpoints/ckpt_A_final.pt

Four checks, which together are Review 1 demo step 6:

  1. The checkpoint loads in a process that trained nothing, and REFUSES if the
     tokenizer fingerprint does not match.
  2. Validation loss reproduces the number the training run reported. If it does
     not, the checkpoint is not the artefact we think it is.
  3. The model generates text from fixed prompts at three temperatures, with
     every seed recorded -> samples.txt
  4. Train and validation loss are plotted on one axis -> loss_curve.png

The expected validation loss is read from the checkpoint's own metrics, so this
script cannot be accidentally pointed at the wrong target number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.level1 import evaluate_split, text_statistics    # noqa: E402
from evaluation.plots import plot_loss_curve                     # noqa: E402
from model.transformer import build_model                        # noqa: E402
from tokenizer.bpe import MAXTokenizer                           # noqa: E402
from tokenizer.special_tokens import BOS_ID, EOS_ID              # noqa: E402
from training.checkpoint import describe, load_checkpoint        # noqa: E402
from training.data import TokenDataset                           # noqa: E402
from utils.config import Config, load_config                     # noqa: E402
from utils.logging_utils import get_logger, write_manifest       # noqa: E402
from utils.seeding import set_seed                               # noqa: E402

log = get_logger("eval")

PROMPTS = [
    "",                                   # unconditional, from <|bos|> alone
    "Once upon a time",
    "The little girl",
    "Tom and Lily went to the",
    "One day, a big dog",
]
TEMPERATURES = [0.2, 0.7, 1.0]
BASE_SEED = 1337


def generate_samples(model, tok, device, max_new_tokens=80, top_k=50) -> list[dict]:
    rows: list[dict] = []
    for p_index, prompt in enumerate(PROMPTS):
        ids = [BOS_ID] + (tok.encode(prompt) if prompt else [])
        for t_index, temperature in enumerate(TEMPERATURES):
            seed = BASE_SEED + p_index * 100 + t_index
            set_seed(seed)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(
                idx, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, eos_id=EOS_ID,
            )
            text = tok.decode(out[0].tolist())
            rows.append({
                "prompt": prompt or "(unconditional)",
                "temperature": temperature,
                "top_k": top_k,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "tokens_generated": int(out.shape[1] - len(ids)),
                "text": text,
            })
    return rows


def write_samples(rows: list[dict], path: Path, header: dict) -> None:
    lines = [
        "=" * 78,
        "MAX-1M generation samples",
        "=" * 78,
        "",
    ]
    for key, value in header.items():
        lines.append(f"{key:24s}: {value}")
    lines += [
        "",
        "Every sample below is reproducible: same checkpoint + same seed +",
        "same temperature gives the same text. A 1.33M-parameter model trained",
        "on ~50M tokens produces simple, often incoherent English. That is the",
        "expected result at this scale, not a disappointment.",
        "",
    ]
    for row in rows:
        lines += [
            "-" * 78,
            f"prompt      : {row['prompt']}",
            f"temperature : {row['temperature']}   top_k: {row['top_k']}   "
            f"seed: {row['seed']}   tokens: {row['tokens_generated']}",
            "-" * 78,
            row["text"],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_A_final.pt")
    parser.add_argument("--config", default="configs/train_a.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--expect-val-loss", type=float, default=None,
                        help="default: read from the checkpoint's own metrics")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = load_config(ROOT / args.data_config)
    tok_cfg = load_config(ROOT / args.tokenizer_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures: list[str] = []

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    log.info("=" * 74)
    log.info("MAX-1M EVALUATION  |  fresh process  |  device %s", device)
    log.info("=" * 74)

    # ------------------------------------------------- 1. load the checkpoint
    log.info("")
    log.info("[1] CHECKPOINT LOAD")
    log.info("  file        : %s", ckpt_path)
    if not ckpt_path.exists():
        log.error("  not found. Pass --checkpoint <path to ckpt_A_final.pt>")
        sys.exit(1)
    log.info("  size        : %.1f MB", ckpt_path.stat().st_size / 1e6)

    header = describe(ckpt_path)
    log.info("  step        : %s", f"{header['step']:,}")
    log.info("  tokens seen : %s", f"{header['tokens_seen']:,}")
    log.info("  fingerprint : %s", header["tokenizer_fingerprint"])

    tok = MAXTokenizer.load(ROOT / tok_cfg.paths.out_dir)
    log.info("  tokenizer   : %r, fingerprint %s", tok, tok.fingerprint())

    # this is the guard -- it raises rather than warns
    payload = load_checkpoint(
        ckpt_path,
        expected_tokenizer_fingerprint=tok.fingerprint(),
        restore_rng=False,
        map_location=device,
    )
    log.info("  PASS  fingerprints match; the checkpoint belongs with this tokenizer")

    # rebuild from the checkpoint's own config, not from the file on disk --
    # the config may have been edited since the run
    ckpt_cfg = Config(payload["config"])
    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(payload["model"])
    log.info("  parameters  : %s", f"{model.num_parameters():,}")
    log.info("  config hash : %s", payload.get("config_hash"))

    # --------------------------------------------------- 2. reproduce val loss
    log.info("")
    log.info("[2] VALIDATION LOSS REPRODUCTION")
    processed = ROOT / data_cfg.paths.processed_dir
    ctx = ckpt_cfg.model.context_length
    val = TokenDataset(processed / "val.bin", ctx)
    starts = val.fixed_starts(
        n_batches=ckpt_cfg.eval.batches,
        batch_size=ckpt_cfg.train.batch_size,
        seed=ckpt_cfg.seed,
    )
    use_fp16 = ckpt_cfg.train.get("precision") == "fp16" and device.startswith("cuda")

    started = time.time()
    stats = evaluate_split(model, val, starts, device, use_fp16)
    log.info("  val tokens        : %s over %d fixed batches",
             f"{stats['tokens_evaluated']:,}", stats["batches"])
    log.info("  val loss          : %.6f", stats["loss"])
    log.info("  perplexity        : %.4f", stats["perplexity"])
    log.info("  bits per token    : %.4f", stats["bits_per_token"])
    log.info("  next-token acc@1  : %.4f  (%.2f%%)",
             stats["accuracy_at_1"], 100 * stats["accuracy_at_1"])

    expected = args.expect_val_loss
    source = "--expect-val-loss"
    if expected is None:
        expected = payload.get("metrics", {}).get("val_loss")
        source = "checkpoint metrics"
    if expected is None:
        log.warning("  no expected value available; skipping the reproduction check")
    else:
        delta = stats["loss"] - expected
        log.info("  expected (%s): %.6f", source, expected)
        log.info("  delta             : %+.6f", delta)
        if abs(delta) <= args.tolerance:
            log.info("  PASS  reproduces within +/-%.3f. The checkpoint is portable:",
                     args.tolerance)
            log.info("        a process that trained nothing gets the training run's number.")
        else:
            failures.append(
                f"val loss {stats['loss']:.6f} differs from {expected:.6f} by {delta:+.6f}"
            )
            log.error("  FAIL  outside tolerance +/-%.3f", args.tolerance)
            log.error("        Check: same device? same precision? same val.bin?")

    # ------------------------------------------------------- 3. generate text
    log.info("")
    log.info("[3] GENERATION")
    rows = generate_samples(model, tok, device, max_new_tokens=args.max_new_tokens)
    text_stats = text_statistics([r["text"] for r in rows])
    log.info("  %d samples: %d prompts x %d temperatures",
             len(rows), len(PROMPTS), len(TEMPERATURES))
    log.info("  distinct-1        : %.4f", text_stats["distinct_1"])
    log.info("  distinct-2        : %.4f", text_stats["distinct_2"])
    log.info("  repetition rate   : %.4f  (immediate word repeats)",
             text_stats["repetition_rate"])

    run_dir = ROOT / cfg.logging.experiments_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.txt"
    write_samples(rows, samples_path, {
        "checkpoint": str(ckpt_path),
        "step": header["step"],
        "tokens seen": f"{header['tokens_seen']:,}",
        "parameters": f"{model.num_parameters():,}",
        "tokenizer fingerprint": header["tokenizer_fingerprint"],
        "validation loss": f"{stats['loss']:.6f}",
        "perplexity": f"{stats['perplexity']:.4f}",
        "device": device,
        "top_k": 50,
    })
    log.info("  samples -> %s", samples_path)

    log.info("")
    for row in rows[:3]:
        preview = row["text"].replace("\n", " ")[:150]
        log.info("  [T=%.1f] %s", row["temperature"], preview)

    # ----------------------------------------------------------- 4. the plot
    log.info("")
    log.info("[4] LOSS CURVE")
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else run_dir / "metrics.csv"
    plot_path = run_dir / "loss_curve.png"
    if not metrics_csv.exists():
        failures.append(f"metrics.csv not found at {metrics_csv}")
        log.error("  not found: %s", metrics_csv)
    else:
        plot_loss_curve(
            metrics_csv, plot_path,
            vocab_size=ckpt_cfg.model.vocab_size,
            final_val=payload.get("metrics", {}).get("val_loss"),
            final_step=header["step"],
            title=f"MAX-1M pretraining  |  {header['tokens_seen']:,} tokens  |  "
                  f"{model.num_parameters():,} parameters",
        )
        log.info("  plot -> %s", plot_path)

    # -------------------------------------------------------------- manifest
    elapsed = time.time() - started
    write_manifest(
        run_dir / "evaluation",
        stage="level1_evaluation",
        checkpoint=str(ckpt_path),
        checkpoint_step=header["step"],
        tokens_seen=header["tokens_seen"],
        tokenizer_fingerprint=header["tokenizer_fingerprint"],
        config_hash=payload.get("config_hash"),
        device=device,
        precision="fp16" if use_fp16 else "fp32",
        n_parameters=model.num_parameters(),
        measured={
            "val_loss": round(stats["loss"], 6),
            "val_perplexity": round(stats["perplexity"], 4),
            "bits_per_token": round(stats["bits_per_token"], 4),
            "next_token_accuracy_at_1": round(stats["accuracy_at_1"], 6),
            "tokens_evaluated": stats["tokens_evaluated"],
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in text_stats.items() if k != "most_common"},
        },
        reproduction={
            "expected_val_loss": expected,
            "expected_source": source,
            "delta": round(stats["loss"] - expected, 6) if expected else None,
            "tolerance": args.tolerance,
        },
        generation={
            "prompts": PROMPTS, "temperatures": TEMPERATURES,
            "base_seed": BASE_SEED, "top_k": 50,
            "max_new_tokens": args.max_new_tokens,
        },
        artefacts={"samples": str(samples_path), "loss_curve": str(plot_path)},
        elapsed_seconds=round(elapsed, 2),
        failures=failures,
        passed=not failures,
    )
    (run_dir / "evaluation" / "samples.json").write_text(json.dumps(rows, indent=2))

    log.info("")
    log.info("=" * 74)
    if failures:
        log.error("EVALUATION FAILED (%d)", len(failures))
        for item in failures:
            log.error("  - %s", item)
        sys.exit(1)
    log.info("ALL CHECKS PASSED")
    log.info("  val loss     %.6f  (perplexity %.2f, %.4f bits/token)",
             stats["loss"], stats["perplexity"], stats["bits_per_token"])
    log.info("  acc@1        %.2f%%", 100 * stats["accuracy_at_1"])
    log.info("  samples      %s", samples_path)
    log.info("  loss curve   %s", plot_path)
    log.info("=" * 74)


if __name__ == "__main__":
    main()
