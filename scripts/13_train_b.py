#!/usr/bin/env python3
"""Stage 1 and Stage 2 for Config B, in one script.

    # Stage 1 -- pretrain MAX-12M from random init on the mixture corpus
    python scripts/13_train_b.py --stage pretrain

    # Stage 2 -- reasoning SFT, starting from the Stage 1 checkpoint
    python scripts/13_train_b.py --stage sft --init-from checkpoints/ckpt_B_final.pt

    # after a Colab preemption, either stage
    python scripts/13_train_b.py --stage pretrain --resume

Both stages use the SAME training loop as Review 1 (training/trainer.py). The
only difference between them is which dataset object is handed to it:

  pretrain  MixtureDataset   TinyStories + reasoning narratives, no masking
  sft       SFTDataset       packed role sequences, loss on completions only

That is deliberate. The loop whose resume is bit-exact, whose clipping happens
after unscaling, and whose checkpoints refuse a tokenizer mismatch is already
tested; re-implementing it for a second stage would be the easiest way to
reintroduce a bug the test suite no longer watches for.

--init-from is NOT a pretrained-weights loophole. It loads a checkpoint this
project produced, two stages earlier, from weights this project sampled. The
no-pretrained rule is about never starting from weights someone else trained.
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
from training.sft_data import MixtureDataset, SFTDataset         # noqa: E402
from training.trainer import Trainer                             # noqa: E402
from utils.config import load_config                             # noqa: E402
from utils.logging_utils import get_logger, hardware_info, write_manifest  # noqa: E402
from utils.seeding import set_seed                               # noqa: E402

log = get_logger("train_b")

DEFAULT_CONFIG = {"pretrain": "configs/train_b.yaml", "sft": "configs/sft_b.yaml"}


def build_datasets(stage: str, cfg, processed: Path):
    """The one thing that differs between the two stages."""
    ctx = cfg.model.context_length
    if stage == "pretrain":
        story_train = TokenDataset(processed / "train.bin", ctx)
        reason_train = TokenDataset(processed / "reason_train.bin", ctx)
        fraction = cfg.data.get("reasoning_fraction", 0.25)
        train = MixtureDataset(story_train, reason_train, fraction,
                               names=("tinystories", "reasoning"))
        val = None
        if (processed / "val.bin").exists() and (processed / "reason_val.bin").exists():
            val = MixtureDataset(
                TokenDataset(processed / "val.bin", ctx),
                TokenDataset(processed / "reason_val.bin", ctx),
                fraction, names=("tinystories", "reasoning"),
            )
        return train, val, {
            "kind": "mixture",
            "tinystories_tokens": story_train.n_tokens,
            "reasoning_tokens": reason_train.n_tokens,
            "reasoning_fraction": fraction,
        }

    train = SFTDataset(processed / "sft_train.npz")
    val_path = processed / "sft_val.npz"
    val = SFTDataset(val_path) if val_path.exists() else None
    return train, val, {
        "kind": "sft",
        "rows": train.n_rows,
        "supervised_tokens": train.n_supervised,
        "supervision_fraction": round(train.supervision_fraction(), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--init-from", default=None,
                        help="checkpoint produced by an earlier stage of THIS project")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reasoning-fraction", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / (args.config or DEFAULT_CONFIG[args.stage]))
    for key, value in (("max_steps", args.max_steps), ("batch_size", args.batch_size)):
        if value is not None:
            cfg.train[key] = value
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.reasoning_fraction is not None:
        cfg.data.reasoning_fraction = args.reasoning_fraction

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.train.get("precision") == "fp16" and device == "cpu":
        log.warning("fp16 requested but no GPU present -- running in fp32 on CPU")
        cfg.train.precision = "fp32"

    tok = MAXTokenizer.load(ROOT / cfg.data.tokenizer_dir)
    fingerprint = tok.fingerprint()
    if tok.vocab_size != cfg.model.vocab_size:
        raise SystemExit(f"tokenizer vocab {tok.vocab_size} != config {cfg.model.vocab_size}")

    processed = ROOT / cfg.data.processed_dir
    train_data, val_data, data_info = build_datasets(args.stage, cfg, processed)

    ctx = cfg.model.context_length
    tokens_step = cfg.train.batch_size * ctx * cfg.train.get("grad_accum_steps", 1)
    hw = hardware_info()

    log.info("=" * 74)
    log.info("MAX-12M  STAGE %s  |  run %s  |  seed %d",
             "1 PRETRAIN" if args.stage == "pretrain" else "2 REASONING SFT",
             cfg.run_name, cfg.seed)
    log.info("=" * 74)
    log.info("device        : %s%s", device,
             f"  ({hw.get('gpu')}, {hw.get('gpu_total_mem_gb')} GB)" if hw.get("gpu") else "")
    log.info("tokenizer     : vocab %d, fingerprint %s", tok.vocab_size, fingerprint)
    log.info("data          : %s", train_data)
    for key, value in data_info.items():
        if key != "kind":
            log.info("   %-22s %s", key, f"{value:,}" if isinstance(value, int) else value)
    log.info("tokens/step   : %s  (batch %d x ctx %d)",
             f"{tokens_step:,}", cfg.train.batch_size, ctx)
    log.info("max steps     : %s  = %s tokens",
             f"{cfg.train.max_steps:,}", f"{cfg.train.max_steps * tokens_step:,}")
    log.info("peak lr       : %.2e", cfg.optim.lr)

    model = build_model(cfg).to(device)
    log.info("parameters    : %s", f"{model.num_parameters():,}")

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / cfg.checkpoint.dir
    run_dir = ROOT / cfg.logging.experiments_dir / cfg.run_name

    trainer = Trainer(
        model=model, cfg=cfg, train_data=train_data, val_data=val_data,
        device=device, tokenizer_fingerprint=fingerprint,
        run_dir=run_dir, checkpoint_dir=ckpt_dir,
    )

    # ---- start from an earlier stage of this project ----------------------
    init_source = None
    if args.init_from and not args.resume:
        source = Path(args.init_from)
        if not source.is_absolute():
            source = ROOT / source
        payload = load_checkpoint(
            source, model=model,
            expected_tokenizer_fingerprint=fingerprint,
            map_location=device,
            # A NEW STAGE gets a fresh RNG, seeded from this config. Inheriting
            # the previous stage's generator state would make Stage 2's data
            # order depend on exactly where Stage 1 happened to stop, which is
            # the opposite of reproducible. Resume (below) does restore it,
            # because there the point IS to continue the same stream.
            restore_rng=False,
        )
        init_source = str(source)
        log.info("-" * 74)
        log.info("initialised from %s (step %s of the previous stage)",
                 source.name, f"{payload['step']:,}")
        log.info("weights produced by THIS project; the config hash is not required to")
        log.info("match because the optimiser state is deliberately NOT carried over")

    # ---- resume ----------------------------------------------------------
    start_step = 0
    if args.resume:
        latest = find_latest(ckpt_dir, pattern=f"{cfg.checkpoint.get('prefix','ckpt_A')}_step_*.pt")
        if latest is None:
            log.warning("--resume given but no checkpoint found in %s; starting fresh", ckpt_dir)
        else:
            payload = load_checkpoint(
                latest, model=model, optimizer=trainer.optimizer,
                scheduler=trainer.scheduler, scaler=trainer.scaler,
                expected_tokenizer_fingerprint=fingerprint,
                expected_config_hash=cfg.hash(), map_location=device,
            )
            start_step = payload["step"]
            trainer.tokens_seen = payload["tokens_seen"]
            trainer.best_val = payload.get("extra", {}).get("best_val_loss", float("inf"))
            log.info("resumed from %s at step %s", latest.name, f"{start_step:,}")

    # ---- the step-0 measurement ------------------------------------------
    step0 = None
    if start_step == 0:
        model.eval()
        with torch.no_grad():
            x, y = train_data.batch(0, cfg.train.batch_size, cfg.seed, device)
            _, loss0 = model(x, y)
        step0 = float(loss0.detach())
        log.info("-" * 74)
        if init_source is None:
            log.info("STEP-0 LOSS   : %.4f   vs ln(%d) = %.4f   (delta %+.4f)",
                     step0, cfg.model.vocab_size, model.expected_init_loss(),
                     step0 - model.expected_init_loss())
            log.info("                random initialisation: uniform over the vocabulary")
        else:
            log.info("STEP-0 LOSS   : %.4f   -- far below ln(%d) = %.4f, as it must be:",
                     step0, cfg.model.vocab_size, model.expected_init_loss())
            log.info("                these weights were trained by Stage 1 of this project")

    log.info("-" * 74)
    summary = trainer.fit(start_step=start_step)

    log.info("=" * 74)
    log.info("STAGE COMPLETE")
    log.info("  steps            %s", f"{summary['steps']:,}")
    log.info("  tokens seen      %s", f"{summary['tokens_seen']:,}")
    log.info("  final train loss %.4f", summary["final_train_loss"])
    if summary.get("val_loss") is not None:
        log.info("  final val loss   %.4f   (perplexity %.1f)",
                 summary["val_loss"], summary["val_perplexity"])
    log.info("  wall clock       %.1f min at %s tokens/s",
             summary["elapsed_seconds"] / 60, f"{summary['tokens_per_second']:,.0f}")
    if "peak_vram_gb" in summary:
        log.info("  peak VRAM        %.2f GB", summary["peak_vram_gb"])
    final_name = f"{cfg.checkpoint.get('prefix', 'ckpt_A')}_final.pt"
    log.info("  checkpoint       %s", ckpt_dir / final_name)

    write_manifest(
        run_dir, stage=args.stage, run_name=cfg.run_name, seed=cfg.seed,
        device=device, config=cfg.to_dict(), config_hash=cfg.hash(),
        tokenizer_fingerprint=fingerprint, tokenizer_vocab_size=tok.vocab_size,
        dataset=data_info, n_parameters=model.num_parameters(),
        step0_loss=round(step0, 6) if step0 is not None else None,
        expected_step0_loss=round(model.expected_init_loss(), 6),
        initialised_from=init_source,
        resumed_from_step=start_step if args.resume else None,
        results=summary, checkpoint_dir=str(ckpt_dir),
    )
    log.info("  manifest         %s", run_dir / "manifest.json")
    log.info("=" * 74)
    nxt = ("python scripts/13_train_b.py --stage sft --init-from <dir>/ckpt_B_final.pt"
           if args.stage == "pretrain" else
           "python scripts/15_eval_reasoning.py --checkpoint <dir>/ckpt_B_sft_final.pt")
    log.info("next: %s", nxt)


if __name__ == "__main__":
    main()
