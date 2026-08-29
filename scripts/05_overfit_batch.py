#!/usr/bin/env python3
"""THE GATE: can the training loop memorise 8 sequences?

    python scripts/05_overfit_batch.py

A correct training loop, given the same tiny batch over and over and no
regularisation, must drive the loss to near zero. A 1.33M-parameter model
memorising 8 x 128 = 1,024 tokens is not learning anything interesting -- that
is the point. It is a plumbing test.

If loss does NOT fall below 0.1, something is wrong in a way that more training
will never fix. The usual culprits, in order of how often they happen:

    targets misaligned by one position (predicting the input, not the next token)
    gradients clipped before unscaling under fp16, so clipping never fires
    optimizer.zero_grad() called after backward instead of before
    the causal mask leaking, or not being applied at all
    a learning rate so low nothing moves

Run this before the LR probe and before the real run. It takes under a minute
and it is the cheapest bug-catch in the project.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import build_model                     # noqa: E402
from training.data import TokenDataset                        # noqa: E402
from training.trainer import Trainer                          # noqa: E402
from utils.config import Config, load_config                  # noqa: E402
from utils.logging_utils import get_logger, write_manifest    # noqa: E402
from utils.seeding import set_seed                            # noqa: E402

log = get_logger("overfit")


class FixedBatch:
    """A dataset that returns the same batch forever.

    Deliberately duck-types TokenDataset so the Trainer runs unmodified -- this
    tests the real training step, including autocast, loss scaling, unscaling
    and clipping, not a simplified copy of it.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y
        self.n_tokens = int(x.numel())

    def batch(self, step, batch_size, seed, device):
        return self.x.to(device), self.y.to(device)

    def gather(self, starts, device="cpu"):
        return self.x.to(device), self.y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_a.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = load_config(ROOT / args.data_config)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=" * 70)
    log.info("OVERFIT-ONE-BATCH TEST  |  %d sequences x %d tokens  |  device %s",
             args.sequences, cfg.model.context_length, device)
    log.info("=" * 70)

    # --- one real batch from the corpus ---------------------------------
    processed = ROOT / data_cfg.paths.processed_dir
    train = TokenDataset(processed / "train.bin", cfg.model.context_length)
    x, y = train.batch(step=0, batch_size=args.sequences, seed=cfg.seed, device="cpu")
    log.info("batch: %s inputs, %s targets, %d target tokens total",
             tuple(x.shape), tuple(y.shape), int(y.numel()))

    # sanity: targets must be inputs shifted left by one
    shifted_ok = bool(torch.equal(x[:, 1:], y[:, :-1]))
    log.info("targets are inputs shifted by one: %s", shifted_ok)
    if not shifted_ok:
        log.error("TARGET ALIGNMENT IS WRONG -- fix training/data.py before anything else")
        sys.exit(1)

    # --- a config tuned to memorise, not to generalise -------------------
    over = Config(cfg.to_dict())
    over.train.batch_size = args.sequences
    over.train.grad_accum_steps = 1
    over.train.max_steps = args.max_steps
    over.optim.lr = args.lr
    over.optim.weight_decay = 0.0          # decay fights memorisation
    over.schedule.warmup_fraction = 0.02
    over.schedule.min_lr_fraction = 1.0    # flat LR: no decay to confound the test
    over.eval.every_steps = 0
    over.checkpoint.every_steps = 0
    over.logging.log_every_steps = 25

    model = build_model(over).to(device)
    log.info("model: %s parameters, dropout %.2f",
             f"{model.num_parameters():,}", over.model.get("dropout", 0.0))

    run_dir = ROOT / "experiments" / "overfit_batch"
    trainer = Trainer(
        model=model, cfg=over, train_data=FixedBatch(x, y), val_data=None,
        device=device, tokenizer_fingerprint="overfit-test",
        run_dir=run_dir, checkpoint_dir=run_dir / "ckpt",
    )

    log.info("-" * 70)
    started = time.time()
    losses: list[float] = []
    first_loss = None
    reached_at = None

    for step in range(args.max_steps):
        loss, grad_norm, lr = trainer.train_step()
        trainer.step += 1
        losses.append(loss)
        if first_loss is None:
            first_loss = loss
        if step % 25 == 0 or loss < args.threshold:
            log.info("  step %4d | loss %8.5f | ppl %10.2f | gn %6.3f | lr %.2e",
                     step, loss, math.exp(min(loss, 20.0)), grad_norm, lr)
        if loss < args.threshold:
            reached_at = step
            break

    elapsed = time.time() - started
    trainer.metrics.close()

    log.info("-" * 70)
    log.info("first loss    : %.5f   (~ln(%d) = %.4f, as expected at init)",
             first_loss, over.model.vocab_size, math.log(over.model.vocab_size))
    log.info("final loss    : %.5f", losses[-1])
    log.info("steps run     : %d in %.1fs", len(losses), elapsed)

    passed = reached_at is not None
    if passed:
        log.info("threshold     : reached %.2f at step %d", args.threshold, reached_at)
        log.info("")
        log.info("PASS -- the loop learns. Gradients flow, targets align, clipping")
        log.info("        behaves under fp16. Safe to proceed to the LR probe.")
    else:
        log.error("threshold     : NOT reached (%.2f) in %d steps", args.threshold, args.max_steps)
        log.error("")
        log.error("FAIL -- do not start the real run. See the header of this file")
        log.error("        for the five usual causes, in order of likelihood.")

    write_manifest(
        run_dir,
        stage="overfit_one_batch",
        device=device,
        seed=cfg.seed,
        sequences=args.sequences,
        context_length=cfg.model.context_length,
        target_tokens=int(y.numel()),
        lr=args.lr,
        threshold=args.threshold,
        config_hash=over.hash(),
        results={
            "first_loss": round(first_loss, 6),
            "final_loss": round(losses[-1], 6),
            "steps_run": len(losses),
            "reached_threshold_at_step": reached_at,
            "elapsed_seconds": round(elapsed, 2),
            "targets_shifted_correctly": shifted_ok,
        },
        passed=passed,
    )
    log.info("manifest -> %s", run_dir / "manifest.json")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
