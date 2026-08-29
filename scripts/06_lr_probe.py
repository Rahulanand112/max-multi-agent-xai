#!/usr/bin/env python3
"""Learning-rate probe: three short runs, pick the winner.

    python scripts/06_lr_probe.py

Runs 200 steps at each of 5e-4, 1e-3 and 3e-3 from an identical random
initialisation, and reports the mean loss over the final 20 steps plus a
stability verdict.

This costs about three minutes and prevents the failure that would cost the
schedule: a main run that diverges at step 4,000, discovered an hour later. A
learning rate that is too high often looks fine for a hundred steps and then
comes apart, so the probe checks for non-finite losses and for a loss that is
climbing at the end, not just the final number.

Selection rule: the lowest final loss among the runs that stayed stable. A
lower loss from an unstable run is not a winner.
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

from model.transformer import build_model                     # noqa: E402
from tokenizer.bpe import MAXTokenizer                        # noqa: E402
from training.data import TokenDataset                        # noqa: E402
from training.trainer import Trainer                          # noqa: E402
from utils.config import Config, load_config                  # noqa: E402
from utils.logging_utils import get_logger, write_manifest    # noqa: E402
from utils.seeding import set_seed                            # noqa: E402

log = get_logger("lrprobe")


def probe(cfg, lr: float, steps: int, train, val, device: str,
          fingerprint: str, run_dir: Path) -> dict:
    set_seed(cfg.seed)  # identical init for every candidate -- only the LR varies

    trial = Config(cfg.to_dict())
    trial.optim.lr = lr
    trial.train.max_steps = steps
    trial.eval.every_steps = 0
    trial.checkpoint.every_steps = 0
    trial.logging.log_every_steps = 50

    model = build_model(trial).to(device)
    trainer = Trainer(
        model=model, cfg=trial, train_data=train, val_data=val, device=device,
        tokenizer_fingerprint=fingerprint,
        run_dir=run_dir / f"lr_{lr:.0e}", checkpoint_dir=run_dir / f"lr_{lr:.0e}" / "ckpt",
    )

    losses: list[float] = []
    started = time.time()
    diverged = False

    for step in range(steps):
        loss, grad_norm, _ = trainer.train_step()
        trainer.step += 1
        losses.append(loss)
        if not math.isfinite(loss):
            log.error("  lr %.0e | step %3d | loss is %s -- DIVERGED", lr, step, loss)
            diverged = True
            break
        if step % 50 == 0:
            log.info("  lr %.0e | step %3d | loss %7.4f | gn %6.3f", lr, step, loss, grad_norm)

    elapsed = time.time() - started
    trainer.metrics.close()

    val_stats = {} if diverged else trainer.evaluate()

    final = sum(losses[-20:]) / len(losses[-20:]) if losses else float("nan")
    early = sum(losses[-40:-20]) / 20 if len(losses) >= 40 else final
    climbing = final > early + 0.02
    stable = not diverged and math.isfinite(final) and not climbing

    log.info("  lr %.0e | final(mean last 20) %7.4f | val %s | %s | %.0fs",
             lr, final,
             f"{val_stats.get('val_loss', float('nan')):.4f}" if val_stats else "n/a",
             "stable" if stable else ("DIVERGED" if diverged else "CLIMBING"),
             elapsed)

    return {
        "lr": lr,
        "steps_run": len(losses),
        "first_loss": round(losses[0], 6) if losses else None,
        "final_loss_mean_last20": round(final, 6) if math.isfinite(final) else None,
        "prev_window_mean": round(early, 6) if math.isfinite(early) else None,
        "val_loss": round(val_stats["val_loss"], 6) if val_stats else None,
        "diverged": diverged,
        "climbing": bool(climbing),
        "stable": bool(stable),
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_a.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-4, 1e-3, 3e-3])
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = load_config(ROOT / args.data_config)
    tok_cfg = load_config(ROOT / args.tokenizer_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fingerprint = MAXTokenizer.load(ROOT / tok_cfg.paths.out_dir).fingerprint()
    processed = ROOT / data_cfg.paths.processed_dir
    ctx = cfg.model.context_length
    train = TokenDataset(processed / "train.bin", ctx)
    val_path = processed / "val.bin"
    val = TokenDataset(val_path, ctx) if val_path.exists() else None

    log.info("=" * 70)
    log.info("LEARNING-RATE PROBE  |  %d steps each  |  device %s", args.steps, device)
    log.info("candidates: %s", ", ".join(f"{lr:.0e}" for lr in args.lrs))
    log.info("tokenizer fingerprint %s  |  %s", fingerprint, train)
    log.info("=" * 70)

    run_dir = ROOT / "experiments" / "lr_probe"
    results = [
        probe(cfg, lr, args.steps, train, val, device, fingerprint, run_dir)
        for lr in args.lrs
    ]

    stable = [r for r in results if r["stable"]]
    winner = min(stable, key=lambda r: r["final_loss_mean_last20"]) if stable else None

    log.info("-" * 70)
    log.info("%-10s %14s %12s %10s", "lr", "final loss", "val loss", "verdict")
    for r in results:
        verdict = "stable" if r["stable"] else ("DIVERGED" if r["diverged"] else "climbing")
        mark = "  <- pick" if winner and r["lr"] == winner["lr"] else ""
        log.info("%-10.0e %14s %12s %10s%s", r["lr"],
                 f"{r['final_loss_mean_last20']:.4f}" if r["final_loss_mean_last20"] else "nan",
                 f"{r['val_loss']:.4f}" if r["val_loss"] else "n/a", verdict, mark)

    log.info("-" * 70)
    if winner is None:
        log.error("every candidate was unstable. Try lower rates, e.g. 1e-4 and 3e-4.")
    else:
        log.info("SELECTED lr = %.0e", winner["lr"])
        log.info("Set optim.lr in configs/train_a.yaml to this value, then run")
        log.info("  python scripts/07_train.py")

    write_manifest(
        run_dir, stage="lr_probe", device=device, seed=cfg.seed,
        steps_per_candidate=args.steps, candidates=args.lrs,
        tokenizer_fingerprint=fingerprint, config_hash=cfg.hash(),
        results=results,
        selected_lr=winner["lr"] if winner else None,
    )
    (run_dir / "selected_lr.json").write_text(
        json.dumps({"selected_lr": winner["lr"] if winner else None,
                    "results": results}, indent=2)
    )
    log.info("manifest -> %s", run_dir / "manifest.json")


if __name__ == "__main__":
    main()
