#!/usr/bin/env python3
"""THE RUN. Pretrain MAX-1M from random initialisation.

    python scripts/07_train.py
    python scripts/07_train.py --resume                      # after a preemption
    python scripts/07_train.py --checkpoint-dir /content/drive/MyDrive/max_v0/checkpoints

Produces ckpt_A_final.pt -- the single artefact Review 1 needs.

Before running this, both gates must have passed:
    python scripts/04_verify_model.py     step-0 loss == ln(4096)
    python scripts/05_overfit_batch.py    loss < 0.1 on a fixed batch

Colab preempts without warning. Point --checkpoint-dir at Drive and the run
resumes from the last 500-step checkpoint with the optimiser moments, the
schedule position, the RNG states and the data order all intact -- the resumed
run sees the same batches the original would have.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import build_model                        # noqa: E402
from tokenizer.bpe import MAXTokenizer                           # noqa: E402
from training.checkpoint import find_latest, load_checkpoint     # noqa: E402
from training.data import TokenDataset                           # noqa: E402
from training.trainer import Trainer                             # noqa: E402
from utils.config import load_config                             # noqa: E402
from utils.logging_utils import get_logger, hardware_info, write_manifest  # noqa: E402
from utils.seeding import set_seed                               # noqa: E402

log = get_logger("run")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_a.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="point at Drive on Colab, e.g. /content/drive/MyDrive/max_v0/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="override, e.g. from the LR probe")
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = load_config(ROOT / args.data_config)
    tok_cfg = load_config(ROOT / args.tokenizer_config)

    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.checkpoint_every is not None:
        cfg.checkpoint.every_steps = args.checkpoint_every
    if args.eval_every is not None:
        cfg.eval.every_steps = args.eval_every

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.train.get("precision") == "fp16" and device == "cpu":
        log.warning("fp16 requested but no GPU present -- running in fp32 on CPU")
        cfg.train.precision = "fp32"

    # ---- tokenizer identity, carried into every checkpoint --------------
    tok = MAXTokenizer.load(ROOT / tok_cfg.paths.out_dir)
    fingerprint = tok.fingerprint()
    if tok.vocab_size != cfg.model.vocab_size:
        log.error("tokenizer vocab %d != model vocab %d", tok.vocab_size, cfg.model.vocab_size)
        sys.exit(1)

    # ---- data ------------------------------------------------------------
    processed = ROOT / data_cfg.paths.processed_dir
    ctx = cfg.model.context_length
    train_data = TokenDataset(processed / "train.bin", ctx)
    val_path = processed / "val.bin"
    val_data = TokenDataset(val_path, ctx) if val_path.exists() else None

    tokens_step = cfg.train.batch_size * ctx * cfg.train.get("grad_accum_steps", 1)
    epoch_equiv = train_data.n_tokens / tokens_step

    hw = hardware_info()
    log.info("=" * 74)
    log.info("MAX-1M PRETRAINING  |  run %s  |  seed %d", cfg.run_name, cfg.seed)
    log.info("=" * 74)
    log.info("device        : %s%s", device,
             f"  ({hw.get('gpu')}, {hw.get('gpu_total_mem_gb')} GB)" if hw.get("gpu") else "")
    if hw.get("cuda_available") and not hw.get("bf16_supported"):
        log.info("precision     : fp16 + GradScaler (this GPU predates bf16)")
    log.info("tokenizer     : vocab %d, fingerprint %s", tok.vocab_size, fingerprint)
    log.info("train tokens  : %s", f"{train_data.n_tokens:,}")
    log.info("val tokens    : %s", f"{val_data.n_tokens:,}" if val_data else "none")
    log.info("tokens/step   : %s  (batch %d x ctx %d)", f"{tokens_step:,}",
             cfg.train.batch_size, ctx)
    log.info("max steps     : %s  = %.2f epoch-equivalents of tokens",
             f"{cfg.train.max_steps:,}", cfg.train.max_steps / epoch_equiv)
    log.info("peak lr       : %.2e -> %.2e cosine, %.0f%% warmup",
             cfg.optim.lr, cfg.optim.lr * cfg.schedule.min_lr_fraction,
             100 * cfg.schedule.warmup_fraction)

    # ---- model -----------------------------------------------------------
    model = build_model(cfg).to(device)
    log.info("parameters    : %s", f"{model.num_parameters():,}")
    log.info("expected init loss: %.4f = ln(%d)", model.expected_init_loss(), cfg.model.vocab_size)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / cfg.checkpoint.dir
    run_dir = ROOT / cfg.logging.experiments_dir / cfg.run_name

    trainer = Trainer(
        model=model, cfg=cfg, train_data=train_data, val_data=val_data,
        device=device, tokenizer_fingerprint=fingerprint,
        run_dir=run_dir, checkpoint_dir=ckpt_dir,
    )
    log.info("weight decay  : %s params decayed, %s excluded (LayerNorm, biases, embeddings)",
             f"{trainer.param_summary['decay_parameters']:,}",
             f"{trainer.param_summary['no_decay_parameters']:,}")

    # ---- resume ----------------------------------------------------------
    start_step = 0
    if args.resume:
        latest = find_latest(ckpt_dir)
        if latest is None:
            log.warning("--resume given but no checkpoint found in %s; starting fresh", ckpt_dir)
        else:
            payload = load_checkpoint(
                latest,
                model=model, optimizer=trainer.optimizer,
                scheduler=trainer.scheduler, scaler=trainer.scaler,
                expected_tokenizer_fingerprint=fingerprint,
                expected_config_hash=cfg.hash(),
                map_location=device,
            )
            start_step = payload["step"]
            trainer.tokens_seen = payload["tokens_seen"]
            trainer.best_val = payload.get("extra", {}).get("best_val_loss", float("inf"))
            log.info("resumed from %s at step %s (%s tokens seen)",
                     latest.name, f"{start_step:,}", f"{trainer.tokens_seen:,}")

    # ---- the step-0 measurement, recorded in the run's own log ----------
    if start_step == 0:
        model.eval()
        with torch.no_grad():
            x, y = train_data.batch(0, cfg.train.batch_size, cfg.seed, device)
            _, loss0 = model(x, y)
        step0 = float(loss0)
        log.info("-" * 74)
        log.info("STEP-0 LOSS   : %.4f   vs ln(%d) = %.4f   (delta %+.4f)",
                 step0, cfg.model.vocab_size, model.expected_init_loss(),
                 step0 - model.expected_init_loss())
        log.info("                the model is uniform over the vocabulary: it knows nothing")
        trainer.metrics.log(
            step=0, train_loss=round(step0, 6),
            train_perplexity=round(math.exp(min(step0, 20.0)), 4),
            lr=0.0, grad_norm=0.0, tokens_seen=0, tokens_per_sec="",
            elapsed_s=0.0, val_loss="", val_perplexity="",
        )
    else:
        step0 = None

    log.info("-" * 74)
    summary = trainer.fit(start_step=start_step)

    log.info("=" * 74)
    log.info("TRAINING COMPLETE")
    log.info("  steps            %s", f"{summary['steps']:,}")
    log.info("  tokens seen      %s", f"{summary['tokens_seen']:,}")
    log.info("  final train loss %.4f", summary["final_train_loss"])
    if summary.get("val_loss") is not None:
        log.info("  final val loss   %.4f   (perplexity %.1f)",
                 summary["val_loss"], summary["val_perplexity"])
    log.info("  best val loss    %s", summary["best_val_loss"])
    log.info("  wall clock       %.1f min at %s tokens/s",
             summary["elapsed_seconds"] / 60, f"{summary['tokens_per_second']:,.0f}")
    if "peak_vram_gb" in summary:
        log.info("  peak VRAM        %.2f GB", summary["peak_vram_gb"])
    log.info("  checkpoint       %s", ckpt_dir / "ckpt_A_final.pt")

    write_manifest(
        run_dir, stage="pretraining", run_name=cfg.run_name, seed=cfg.seed,
        device=device, config=cfg.to_dict(), config_hash=cfg.hash(),
        tokenizer_fingerprint=fingerprint, tokenizer_vocab_size=tok.vocab_size,
        dataset={
            "train_tokens": train_data.n_tokens,
            "val_tokens": val_data.n_tokens if val_data else None,
            "processed_dir": str(processed),
        },
        n_parameters=model.num_parameters(),
        step0_loss=round(step0, 6) if step0 is not None else None,
        expected_step0_loss=round(model.expected_init_loss(), 6),
        parameter_groups={
            k: v for k, v in trainer.param_summary.items() if not k.endswith("names")
        },
        resumed_from_step=start_step if args.resume else None,
        results=summary,
        checkpoint_dir=str(ckpt_dir),
    )
    log.info("  manifest         %s", run_dir / "manifest.json")
    log.info("=" * 74)


if __name__ == "__main__":
    main()
