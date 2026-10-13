"""Tests for the Stage 1 mixture and the Stage 2 masked-loss dataset.

The loss mask is the single most silently-breakable thing in this project. If
it is off by one, the model is trained to predict prompts, loss still falls,
no test fails, and the reasoning results come out mysteriously poor. These
tests pin the alignment directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.data import TokenDataset
from training.sft_data import IGNORE_INDEX, MixtureDataset, SFTDataset

CTX = 16


@pytest.fixture
def stream(tmp_path):
    def make(name: str, value: int, n: int = 4000):
        path = tmp_path / name
        np.full(n, value, dtype=np.uint16).tofile(path)
        return TokenDataset(path, CTX)
    return make


@pytest.fixture
def sft_file(tmp_path):
    """Two rows: tokens 0..15, with the first 5 positions masked out."""
    rows = np.arange(32, dtype=np.uint16).reshape(2, 16)
    mask = np.zeros((2, 16), dtype=np.uint8)
    mask[:, 5:] = 1
    path = tmp_path / "sft.npz"
    np.savez(path, tokens=rows, mask=mask)
    return path


# ------------------------------------------------------------- the mixture

def test_mixture_draws_from_both_sources(stream):
    mixture = MixtureDataset(stream("a.bin", 1), stream("b.bin", 2), 0.5)
    x, _ = mixture.batch(step=0, batch_size=64, seed=1337)
    values = set(x.flatten().tolist())
    assert values == {1, 2}


def test_mixture_fraction_is_respected(stream):
    mixture = MixtureDataset(stream("a.bin", 1), stream("b.bin", 2), 0.25)
    counts = []
    for step in range(40):
        x, _ = mixture.batch(step=step, batch_size=64, seed=1337)
        counts.append(float((x[:, 0] == 2).float().mean()))
    assert 0.18 < sum(counts) / len(counts) < 0.32


def test_mixture_zero_fraction_uses_only_the_primary(stream):
    mixture = MixtureDataset(stream("a.bin", 1), stream("b.bin", 2), 0.0)
    x, _ = mixture.batch(step=3, batch_size=32, seed=1337)
    assert set(x.flatten().tolist()) == {1}


def test_mixture_is_deterministic_in_seed_and_step(stream):
    mixture = MixtureDataset(stream("a.bin", 1), stream("b.bin", 2), 0.5)
    first, _ = mixture.batch(step=11, batch_size=32, seed=1337)
    second, _ = mixture.batch(step=11, batch_size=32, seed=1337)
    assert torch.equal(first, second)
    other, _ = mixture.batch(step=12, batch_size=32, seed=1337)
    assert not torch.equal(first, other)


def test_mixture_batch_size_is_exact(stream):
    mixture = MixtureDataset(stream("a.bin", 1), stream("b.bin", 2), 0.37)
    for step in range(10):
        x, y = mixture.batch(step=step, batch_size=48, seed=1337)
        assert x.shape == (48, CTX) and y.shape == (48, CTX)


def test_mixture_rejects_an_out_of_range_fraction(stream):
    with pytest.raises(ValueError):
        MixtureDataset(stream("a.bin", 1), stream("b.bin", 2), 1.5)


def test_mixture_rejects_mismatched_context_lengths(tmp_path):
    for name, value in (("a.bin", 1), ("b.bin", 2)):
        np.full(4000, value, dtype=np.uint16).tofile(tmp_path / name)
    with pytest.raises(ValueError, match="context length"):
        MixtureDataset(TokenDataset(tmp_path / "a.bin", 16),
                       TokenDataset(tmp_path / "b.bin", 8), 0.5)


# ------------------------------------------------------------- the SFT mask

def test_inputs_and_targets_are_shifted_by_one(sft_file):
    x, y = SFTDataset(sft_file).gather(np.array([0]))
    assert x.shape == y.shape == (1, 15)
    assert x[0].tolist() == list(range(15))
    supervised = y[0] != IGNORE_INDEX
    assert y[0][supervised].tolist() == list(range(5, 16))


def test_prompt_positions_are_ignored_by_the_loss(sft_file):
    """Positions 0..4 are prompt; their targets must not contribute gradient."""
    _, y = SFTDataset(sft_file).gather(np.array([0]))
    assert (y[0, :4] == IGNORE_INDEX).all()
    assert (y[0, 4:] != IGNORE_INDEX).all()


def test_the_mask_shifts_with_the_targets_not_the_inputs(sft_file):
    """The off-by-one that would quietly train the model on its own prompts."""
    dataset = SFTDataset(sft_file)
    _, y = dataset.gather(np.array([0]))
    first_supervised = int((y[0] != IGNORE_INDEX).nonzero()[0])
    # mask[5] is the first supervised TOKEN, which is target index 4
    assert first_supervised == 4


def test_supervision_fraction_is_reported(sft_file):
    dataset = SFTDataset(sft_file)
    assert dataset.n_supervised == 22
    assert dataset.supervision_fraction() == pytest.approx(22 / 32)


def test_every_batch_has_something_to_learn_from(sft_file):
    """An all-masked batch makes cross-entropy NaN with no traceback."""
    dataset = SFTDataset(sft_file)
    for step in range(20):
        _, y = dataset.batch(step=step, batch_size=4, seed=1337)
        assert (y != IGNORE_INDEX).any()


def test_batches_are_deterministic_in_seed_and_step(sft_file):
    dataset = SFTDataset(sft_file)
    first, _ = dataset.batch(step=5, batch_size=2, seed=1337)
    second, _ = dataset.batch(step=5, batch_size=2, seed=1337)
    assert torch.equal(first, second)


def test_validation_batches_are_fixed(sft_file):
    dataset = SFTDataset(sft_file)
    a = dataset.fixed_starts(3, 2, seed=1337)
    b = dataset.fixed_starts(3, 2, seed=1337)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]


def test_missing_file_names_the_script_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="12_build_sft"):
        SFTDataset(tmp_path / "absent.npz")


def test_masked_loss_differs_from_unmasked(sft_file):
    """If masking made no difference, it would not be doing anything."""
    import torch.nn.functional as F
    dataset = SFTDataset(sft_file)
    x, y = dataset.gather(np.array([0, 1]))

    torch.manual_seed(0)
    logits = torch.randn(*x.shape, 32)
    masked = F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1),
                             ignore_index=IGNORE_INDEX)

    rows = dataset.tokens[np.array([0, 1])].astype(np.int64)
    unmasked_targets = torch.from_numpy(rows[:, 1:])
    unmasked = F.cross_entropy(logits.reshape(-1, 32), unmasked_targets.reshape(-1))
    assert abs(float(masked) - float(unmasked)) > 1e-4
