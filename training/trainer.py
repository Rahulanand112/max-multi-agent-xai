"""The training loop.

Next-token prediction, AdamW, cosine schedule, fp16 with loss scaling, gradient
clipping, periodic validation, resumable checkpoints.

One ordering detail is easy to get wrong and silently ruins clipping: with a
GradScaler the gradients are scaled by a large factor, so clipping them
directly would compare a scaled norm against an unscaled threshold and clip
essentially never. `scaler.unscale_(optimizer)` must come first. That sequence
is asserted in tests/test_training.py.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from training.checkpoint import rotate_checkpoints, save_checkpoint
from training.data import TokenDataset, tokens_per_step
from training.optim import (
    CosineWithWarmup,
    autocast_context,
    build_optimizer,
    make_grad_scaler,
)
from utils.logging_utils import MetricsLogger, get_logger

log = get_logger("train")


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        train_data: TokenDataset,
        val_data: TokenDataset | None,
        device: str,
        tokenizer_fingerprint: str,
        run_dir: Path,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.seed = cfg.seed
        self.batch_size = cfg.train.batch_size
        self.grad_accum = cfg.train.get("grad_accum_steps", 1)
        self.max_steps = cfg.train.max_steps
        self.grad_clip = cfg.optim.grad_clip
        self.use_fp16 = cfg.train.get("precision", "fp32") == "fp16"

        self.optimizer, self.param_summary = build_optimizer(model, cfg.optim)
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            peak_lr=cfg.optim.lr,
            max_steps=self.max_steps,
            warmup_fraction=cfg.schedule.warmup_fraction,
            min_lr_fraction=cfg.schedule.min_lr_fraction,
        )
        self.scaler = make_grad_scaler(device, self.use_fp16)

        self.step = 0
        self.tokens_seen = 0
        self.best_val = float("inf")
        self.tokens_per_step = tokens_per_step(
            self.batch_size, cfg.model.context_length, self.grad_accum
        )

        self.val_starts = None
        if val_data is not None:
            self.val_starts = val_data.fixed_starts(
                n_batches=cfg.eval.batches, batch_size=self.batch_size, seed=self.seed
            )

        self.metrics = MetricsLogger(self.run_dir / "metrics.csv")

    # ------------------------------------------------------------------ steps

    def train_step(self) -> tuple[float, float, float]:
        """One optimiser step. Returns (loss, grad_norm, lr)."""
        self.model.train()
        lr = self.scheduler.set_step(self.step)
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for micro in range(self.grad_accum):
            x, y = self.train_data.batch(
                step=self.step * self.grad_accum + micro,
                batch_size=self.batch_size,
                seed=self.seed,
                device=self.device,
            )
            with autocast_context(self.device, self.use_fp16):
                _, loss = self.model(x, y)
                loss = loss / self.grad_accum
            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * self.grad_accum

        # unscale BEFORE clipping, or the threshold is meaningless
        self.scaler.unscale_(self.optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.tokens_seen += self.tokens_per_step
        return total_loss / self.grad_accum, grad_norm, lr

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Validation on fixed batches -- identical data at every evaluation."""
        if self.val_data is None or not self.val_starts:
            return {}
        self.model.eval()
        losses = []
        for starts in self.val_starts:
            x, y = self.val_data.gather(starts, self.device)
            with autocast_context(self.device, self.use_fp16):
                _, loss = self.model(x, y)
            losses.append(float(loss))
        self.model.train()
        mean = sum(losses) / len(losses)
        return {"val_loss": mean, "val_perplexity": math.exp(min(mean, 20.0))}

    # ------------------------------------------------------------------- loop

    def fit(self, start_step: int = 0) -> dict:
        self.step = start_step
        started = time.time()
        last_log = started
        log_every = self.cfg.logging.log_every_steps
        eval_every = self.cfg.eval.every_steps
        ckpt_every = self.cfg.checkpoint.every_steps

        log.info("training %d -> %d steps  |  %s tokens/step  |  fp16=%s",
                 start_step, self.max_steps, f"{self.tokens_per_step:,}", self.use_fp16)

        while self.step < self.max_steps:
            loss, grad_norm, lr = self.train_step()

            if not math.isfinite(loss):
                log.error("loss is %s at step %d -- stopping", loss, self.step)
                self._checkpoint(tag="diverged", metrics={"train_loss": loss})
                raise RuntimeError(f"non-finite loss at step {self.step}")

            self.step += 1

            if self.step % log_every == 0 or self.step == 1:
                now = time.time()
                tps = (log_every * self.tokens_per_step) / max(now - last_log, 1e-9)
                last_log = now
                remaining = (self.max_steps - self.step) * self.tokens_per_step / max(tps, 1)
                log.info(
                    "step %6d/%d | loss %7.4f | ppl %9.1f | lr %.2e | gn %5.2f | "
                    "%7.0f tok/s | eta %4.1fm",
                    self.step, self.max_steps, loss, math.exp(min(loss, 20.0)),
                    lr, grad_norm, tps, remaining / 60,
                )
                self.metrics.log(
                    step=self.step, train_loss=round(loss, 6),
                    train_perplexity=round(math.exp(min(loss, 20.0)), 4),
                    lr=lr, grad_norm=round(grad_norm, 4),
                    tokens_seen=self.tokens_seen, tokens_per_sec=round(tps, 1),
                    elapsed_s=round(now - started, 2), val_loss="", val_perplexity="",
                )

            if eval_every and self.step % eval_every == 0:
                stats = self.evaluate()
                if stats:
                    log.info("           validation | loss %7.4f | ppl %9.1f%s",
                             stats["val_loss"], stats["val_perplexity"],
                             "  <- best" if stats["val_loss"] < self.best_val else "")
                    self.metrics.log(
                        step=self.step, train_loss=round(loss, 6),
                        train_perplexity=round(math.exp(min(loss, 20.0)), 4),
                        lr=lr, grad_norm=round(grad_norm, 4),
                        tokens_seen=self.tokens_seen, tokens_per_sec="",
                        elapsed_s=round(time.time() - started, 2),
                        val_loss=round(stats["val_loss"], 6),
                        val_perplexity=round(stats["val_perplexity"], 4),
                    )
                    if stats["val_loss"] < self.best_val:
                        self.best_val = stats["val_loss"]
                        self._checkpoint(tag="best", metrics=stats, rotate=False)

            if ckpt_every and self.step % ckpt_every == 0:
                self._checkpoint(metrics={"train_loss": loss})

        elapsed = time.time() - started
        final = self.evaluate()
        if final:
            # the final evaluation happens after the loop, so log it explicitly
            # rather than letting metrics.csv stop short of the reported result
            self.metrics.log(
                step=self.step, train_loss=round(loss, 6),
                train_perplexity=round(math.exp(min(loss, 20.0)), 4),
                lr=lr, grad_norm=round(grad_norm, 4),
                tokens_seen=self.tokens_seen, tokens_per_sec="",
                elapsed_s=round(elapsed, 2),
                val_loss=round(final["val_loss"], 6),
                val_perplexity=round(final["val_perplexity"], 4),
            )
        self._checkpoint(tag="final", metrics=final, rotate=False)
        self.metrics.close()

        summary = {
            "steps": self.step,
            "tokens_seen": self.tokens_seen,
            "final_train_loss": round(loss, 6),
            "best_val_loss": round(self.best_val, 6) if self.best_val < float("inf") else None,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(self.tokens_seen / max(elapsed, 1e-9), 1),
        }
        summary.update({k: round(v, 6) for k, v in final.items()})
        if self.device.startswith("cuda"):
            summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        return summary

    # ------------------------------------------------------------ checkpoints

    def _checkpoint(self, tag: str | None = None, metrics=None, rotate: bool = True) -> Path:
        name = f"ckpt_A_{tag}.pt" if tag else f"ckpt_step_{self.step:06d}.pt"
        path = save_checkpoint(
            self.checkpoint_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            tokens_seen=self.tokens_seen,
            config=self.cfg.to_dict(),
            config_hash=self.cfg.hash(),
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            metrics=metrics or {},
            extra={"best_val_loss": self.best_val, "seed": self.seed},
        )
        if rotate:
            removed = rotate_checkpoints(
                self.checkpoint_dir, "ckpt_step_*.pt", self.cfg.checkpoint.keep_last
            )
            if removed:
                log.info("           rotated out %d old checkpoint(s)", len(removed))
        log.info("           saved %s", path.name)
        return path
