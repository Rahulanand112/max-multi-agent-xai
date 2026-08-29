"""Batching from a uint16 memmap, with resume that costs nothing.

The corpus is one flat token stream on disk. A batch is `batch_size` random
windows of `context_length + 1` tokens; inputs are the first context_length,
targets are the same window shifted by one -- next-token prediction.

Resumability
------------
The window start positions for step N are drawn from a generator seeded by
(seed, N). Two consequences that matter on Colab:

  * resuming at step 4,000 costs one line, not four thousand batches of
    fast-forwarding, and produces the *same* batch the original run saw;
  * two runs with the same seed see identical data in identical order, so a
    loss curve is reproducible.

Sampling is with replacement, which is standard for LM pretraining. 6,300 steps
at 8,192 tokens is one epoch-*equivalent* of tokens, not a clean pass -- some
windows repeat and some are never drawn. The logs say "tokens seen", never
"epochs", because the second would be a slightly false claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: str | Path, context_length: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run scripts/03_encode_corpus.py first."
            )
        self.context_length = context_length
        # memory-mapped: Colab's disk is slow, and this avoids loading 100 MB
        # into RAM per worker
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = int(self.tokens.shape[0])
        if self.n_tokens < context_length + 1:
            raise ValueError(
                f"{self.path} holds {self.n_tokens} tokens, "
                f"fewer than context_length + 1 = {context_length + 1}"
            )
        self.max_start = self.n_tokens - context_length - 1

    def __len__(self) -> int:
        return self.n_tokens

    def __repr__(self) -> str:
        return f"TokenDataset({self.path.name}, {self.n_tokens:,} tokens)"

    # ---------------------------------------------------------------- batches

    def starts_for_step(self, step: int, batch_size: int, seed: int) -> np.ndarray:
        """Window starts for a given step. Deterministic in (seed, step)."""
        rng = np.random.default_rng([seed, step])
        return rng.integers(0, self.max_start, size=batch_size)

    def fixed_starts(self, n_batches: int, batch_size: int, seed: int) -> list[np.ndarray]:
        """Validation batches, drawn once and reused at every evaluation.

        Evaluating on different data each time would make the validation curve
        noisy for a reason that has nothing to do with the model.
        """
        # 999_983 is an arbitrary fixed offset that keeps validation windows in
        # a different part of the generator's stream from any training step
        rng = np.random.default_rng([seed, 999_983])
        return [rng.integers(0, self.max_start, size=batch_size) for _ in range(n_batches)]

    def gather(self, starts: np.ndarray, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_length
        block = np.stack([
            np.asarray(self.tokens[s : s + ctx + 1], dtype=np.int64) for s in starts
        ])
        x = torch.from_numpy(block[:, :-1])
        y = torch.from_numpy(block[:, 1:])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def batch(
        self,
        step: int,
        batch_size: int,
        seed: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gather(self.starts_for_step(step, batch_size, seed), device)


def tokens_per_step(batch_size: int, context_length: int, grad_accum: int = 1) -> int:
    return batch_size * context_length * grad_accum
