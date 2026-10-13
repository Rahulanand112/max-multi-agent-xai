"""Two datasets for the Config B stages, both speaking TokenDataset's interface.

`Trainer` asks a dataset for three things: `batch(step, ...)`, `fixed_starts()`
and `gather(starts)`. Anything providing those can be trained by the existing
loop, so neither of these classes requires a change to training/trainer.py —
which matters, because that loop is the one whose resume behaviour and
clip-after-unscale ordering are already tested.

MixtureDataset — Stage 1
    Draws each batch from TinyStories and from the reasoning narratives in a
    fixed proportion. A 12M-parameter model has no chance of learning
    multi-step arithmetic from the SFT stage alone; it needs to have met
    digits, "times", "minus" and counting in ordinary prose first. Mixing at
    pretraining time is cheaper and works better than a longer SFT.

SFTDataset — Stage 2
    Reads the packed rows built by scripts/12_build_sft.py and applies the
    loss mask by setting masked targets to -100, which the model's
    cross-entropy already ignores. The mask is the whole point: loss is
    computed on completions, never on the prompts the model will always be
    given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import TokenDataset

IGNORE_INDEX = -100


class MixtureDataset:
    """Sample from two token streams in a fixed proportion.

    The split within a batch is deterministic in (seed, step), so a resumed
    run sees the same mixture the original would have — the same guarantee
    TokenDataset gives for window positions, extended to source choice.
    """

    def __init__(
        self,
        primary: TokenDataset,
        secondary: TokenDataset,
        secondary_fraction: float,
        names: tuple[str, str] = ("primary", "secondary"),
    ) -> None:
        if not 0.0 <= secondary_fraction <= 1.0:
            raise ValueError(f"secondary_fraction must be in [0, 1], got {secondary_fraction}")
        self.primary = primary
        self.secondary = secondary
        self.secondary_fraction = secondary_fraction
        self.names = names
        self.context_length = primary.context_length
        if secondary.context_length != primary.context_length:
            raise ValueError("both sources must use the same context length")

    def __repr__(self) -> str:
        return (f"MixtureDataset({self.names[0]}={self.primary.n_tokens:,}, "
                f"{self.names[1]}={self.secondary.n_tokens:,}, "
                f"fraction={self.secondary_fraction:.2f})")

    @property
    def n_tokens(self) -> int:
        return self.primary.n_tokens + self.secondary.n_tokens

    def _split(self, batch_size: int, step: int, seed: int) -> int:
        """How many sequences in this batch come from the secondary source."""
        rng = np.random.default_rng([seed, step, 7919])
        draws = rng.random(batch_size)
        return int((draws < self.secondary_fraction).sum())

    def batch(self, step: int, batch_size: int, seed: int, device: str = "cpu"):
        n_secondary = self._split(batch_size, step, seed)
        n_primary = batch_size - n_secondary

        parts_x, parts_y = [], []
        if n_primary:
            x, y = self.primary.batch(step, n_primary, seed, device="cpu")
            parts_x.append(x)
            parts_y.append(y)
        if n_secondary:
            x, y = self.secondary.batch(step, n_secondary, seed + 1, device="cpu")
            parts_x.append(x)
            parts_y.append(y)

        x = torch.cat(parts_x, dim=0)
        y = torch.cat(parts_y, dim=0)
        return _to_device(x, y, device)

    def fixed_starts(self, n_batches: int, batch_size: int, seed: int):
        """Validation batches: half from each source, fixed for the whole run."""
        half = max(1, batch_size // 2)
        primary = self.primary.fixed_starts(n_batches, batch_size - half, seed)
        secondary = self.secondary.fixed_starts(n_batches, half, seed)
        return list(zip(primary, secondary))

    def gather(self, starts, device: str = "cpu"):
        primary_starts, secondary_starts = starts
        x1, y1 = self.primary.gather(primary_starts, device="cpu")
        x2, y2 = self.secondary.gather(secondary_starts, device="cpu")
        return _to_device(torch.cat([x1, x2]), torch.cat([y1, y2]), device)


class SFTDataset:
    """Packed prompt+completion rows with a completion-only loss mask."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run scripts/12_build_sft.py first."
            )
        with np.load(self.path) as blob:
            self.tokens = blob["tokens"]
            self.mask = blob["mask"]
        if self.tokens.shape != self.mask.shape:
            raise ValueError("tokens and mask have different shapes")
        self.n_rows, self.row_length = self.tokens.shape
        self.context_length = self.row_length - 1
        self.n_tokens = int(self.tokens.size)
        self.n_supervised = int(self.mask.sum())

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return (f"SFTDataset({self.path.name}, {self.n_rows:,} rows x {self.row_length}, "
                f"{self.n_supervised:,} supervised tokens)")

    def starts_for_step(self, step: int, batch_size: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng([seed, step])
        return rng.integers(0, self.n_rows, size=batch_size)

    def fixed_starts(self, n_batches: int, batch_size: int, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng([seed, 999_983])
        return [rng.integers(0, self.n_rows, size=batch_size) for _ in range(n_batches)]

    def gather(self, starts, device: str = "cpu"):
        rows = self.tokens[starts].astype(np.int64)
        mask = self.mask[starts]

        x = torch.from_numpy(rows[:, :-1])
        y = torch.from_numpy(rows[:, 1:]).clone()

        # the mask belongs to the TARGET position, so it shifts with y
        supervised = torch.from_numpy(mask[:, 1:].astype(np.bool_))
        y[~supervised] = IGNORE_INDEX
        return _to_device(x, y, device)

    def batch(self, step: int, batch_size: int, seed: int, device: str = "cpu"):
        return self.gather(self.starts_for_step(step, batch_size, seed), device)

    def supervision_fraction(self) -> float:
        return self.n_supervised / self.n_tokens


def _to_device(x: torch.Tensor, y: torch.Tensor, device: str):
    if device.startswith("cuda"):
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)
