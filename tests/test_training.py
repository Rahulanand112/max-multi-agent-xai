"""Training-loop tests.

Three of these guard failures that are silent -- they produce a run that looks
fine and results that are wrong:

  test_targets_are_inputs_shifted_by_one   predicting the input, not the next token
  test_clip_happens_after_unscale          fp16 clipping that never fires
  test_load_refuses_on_tokenizer_mismatch  fluent nonsense with no error
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import build_model                      # noqa: E402
from training.checkpoint import (                              # noqa: E402
    CheckpointMismatch,
    describe,
    find_latest,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from training.data import TokenDataset, tokens_per_step        # noqa: E402
from training.optim import (                                   # noqa: E402
    CosineWithWarmup,
    build_optimizer,
    split_parameter_groups,
)
from utils.config import Config, load_config                   # noqa: E402
from utils.seeding import set_seed                             # noqa: E402

CTX = 16


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "toy.bin"
    rng = np.random.default_rng(0)
    rng.integers(0, 4096, size=20_000, dtype=np.uint16).tofile(path)
    return path


@pytest.fixture(scope="module")
def dataset(corpus) -> TokenDataset:
    return TokenDataset(corpus, CTX)


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "train_a.yaml")


# ----------------------------------------------------------------- dataloader

def test_targets_are_inputs_shifted_by_one(dataset):
    """The entire training objective depends on this alignment."""
    x, y = dataset.batch(step=0, batch_size=4, seed=1337)
    assert x.shape == y.shape == (4, CTX)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_same_step_and_seed_give_the_same_batch(dataset):
    a = dataset.batch(step=7, batch_size=4, seed=1337)
    b = dataset.batch(step=7, batch_size=4, seed=1337)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_different_steps_give_different_batches(dataset):
    a, _ = dataset.batch(step=7, batch_size=4, seed=1337)
    b, _ = dataset.batch(step=8, batch_size=4, seed=1337)
    assert not torch.equal(a, b)


def test_resume_sees_the_batch_the_original_run_would_have(dataset):
    """O(1) resume: step 500 is step 500 regardless of how you got there."""
    original = dataset.batch(step=500, batch_size=4, seed=1337)
    after_restart = dataset.batch(step=500, batch_size=4, seed=1337)
    assert torch.equal(original[0], after_restart[0])


def test_validation_batches_are_fixed(dataset):
    """Evaluating on different data each time makes the curve noisy for no reason."""
    a = dataset.fixed_starts(3, 4, seed=1337)
    b = dataset.fixed_starts(3, 4, seed=1337)
    for left, right in zip(a, b):
        assert np.array_equal(left, right)


def test_windows_stay_inside_the_corpus(dataset):
    starts = dataset.starts_for_step(0, 64, seed=1337)
    assert starts.min() >= 0
    assert starts.max() + CTX + 1 <= dataset.n_tokens


def test_missing_corpus_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="03_encode_corpus"):
        TokenDataset(tmp_path / "nope.bin", CTX)


def test_tokens_per_step_arithmetic():
    assert tokens_per_step(64, 128) == 8192
    assert tokens_per_step(64, 128, grad_accum=2) == 16384


# ------------------------------------------------------------ parameter groups

def test_layernorm_biases_and_embeddings_are_excluded_from_decay(cfg):
    set_seed(1337)
    model = build_model(cfg)
    _, summary = split_parameter_groups(model, weight_decay=0.1)

    for name in summary["no_decay_names"]:
        assert ("ln_" in name) or name.endswith(".bias") or ("emb" in name), name
    for name in summary["decay_names"]:
        assert "ln_" not in name and "emb" not in name and not name.endswith(".bias")

    assert any("tok_emb" in n for n in summary["no_decay_names"])
    assert any("qkv" in n for n in summary["decay_names"])


def test_every_parameter_lands_in_exactly_one_group(cfg):
    set_seed(1337)
    model = build_model(cfg)
    _, summary = split_parameter_groups(model, 0.1)
    assert summary["decay_parameters"] + summary["no_decay_parameters"] == 1_331_968
    overlap = set(summary["decay_names"]) & set(summary["no_decay_names"])
    assert not overlap


def test_optimizer_carries_the_two_decay_settings(cfg):
    set_seed(1337)
    opt, _ = build_optimizer(build_model(cfg), cfg.optim)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    assert opt.defaults["betas"] == (0.9, 0.95)


# ----------------------------------------------------------------- scheduler

def test_schedule_warms_up_then_decays_to_the_floor():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1.0)
    sched = CosineWithWarmup(opt, peak_lr=1e-3, max_steps=1000,
                             warmup_fraction=0.05, min_lr_fraction=0.1)
    assert sched.warmup_steps == 50
    assert sched.lr_at(0) == pytest.approx(1e-3 / 50)      # not zero: step 0 still updates
    assert sched.lr_at(49) == pytest.approx(1e-3)          # peak at end of warmup
    assert sched.lr_at(999) == pytest.approx(1e-4, rel=0.02)
    assert sched.lr_at(2000) == pytest.approx(1e-4)        # clamped past the end


def test_schedule_is_monotonic_after_warmup():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1.0)
    sched = CosineWithWarmup(opt, 1e-3, 1000, 0.05, 0.1)
    after = [sched.lr_at(s) for s in range(50, 1000, 10)]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:]))


def test_set_step_writes_the_lr_into_the_optimizer():
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([param], lr=1.0)
    sched = CosineWithWarmup(opt, 1e-3, 100, 0.1, 0.1)
    lr = sched.set_step(10)
    assert opt.param_groups[0]["lr"] == lr == sched.lr_at(10)


# ---------------------------------------------------------------- checkpoints

def make_state(cfg, tmp_path, fingerprint="d6080bac0a9a", step=500):
    set_seed(1337)
    model = build_model(cfg)
    opt, _ = build_optimizer(model, cfg.optim)
    sched = CosineWithWarmup(opt, cfg.optim.lr, 6100, 0.05, 0.1)
    path = save_checkpoint(
        tmp_path / f"ckpt_step_{step:06d}.pt",
        model=model, optimizer=opt, scheduler=sched, scaler=None,
        step=step, tokens_seen=step * 8192,
        config=cfg.to_dict(), config_hash=cfg.hash(),
        tokenizer_fingerprint=fingerprint,
        metrics={"train_loss": 4.2},
    )
    return model, opt, sched, path


def test_checkpoint_carries_everything_needed_to_resume(cfg, tmp_path):
    _, _, _, path = make_state(cfg, tmp_path)
    info = describe(path)
    assert info["step"] == 500
    assert info["tokens_seen"] == 500 * 8192
    assert info["tokenizer_fingerprint"] == "d6080bac0a9a"
    assert info["has_optimizer"] and info["has_rng"]
    assert info["n_model_tensors"] > 0


def test_reload_restores_the_exact_weights(cfg, tmp_path):
    model, opt, sched, path = make_state(cfg, tmp_path)
    set_seed(999)
    fresh = build_model(cfg)
    assert not torch.equal(fresh.tok_emb.weight, model.tok_emb.weight)

    load_checkpoint(path, model=fresh, expected_tokenizer_fingerprint="d6080bac0a9a")
    for (n, a), (_, b) in zip(fresh.named_parameters(), model.named_parameters()):
        assert torch.equal(a, b), n


def test_load_refuses_on_tokenizer_mismatch(cfg, tmp_path):
    """The guard against fluent nonsense with no error message."""
    _, _, _, path = make_state(cfg, tmp_path, fingerprint="d6080bac0a9a")
    set_seed(0)
    fresh = build_model(cfg)
    with pytest.raises(CheckpointMismatch, match="TOKENIZER MISMATCH"):
        load_checkpoint(path, model=fresh, expected_tokenizer_fingerprint="deadbeef0000")


def test_config_hash_change_warns_but_loads(cfg, tmp_path, capsys):
    _, _, _, path = make_state(cfg, tmp_path)
    set_seed(0)
    fresh = build_model(cfg)
    load_checkpoint(path, model=fresh, expected_config_hash="0000deadbeef")
    assert "config hash differs" in capsys.readouterr().out


def test_config_hash_change_can_be_made_fatal(cfg, tmp_path):
    _, _, _, path = make_state(cfg, tmp_path)
    with pytest.raises(CheckpointMismatch, match="config hash"):
        load_checkpoint(path, expected_config_hash="0000deadbeef", strict_config=True)


def test_rotation_keeps_the_newest_and_spares_named_checkpoints(cfg, tmp_path):
    for step in (100, 200, 300, 400):
        make_state(cfg, tmp_path, step=step)
    (tmp_path / "ckpt_A_final.pt").write_bytes(b"x")
    (tmp_path / "ckpt_A_best.pt").write_bytes(b"x")

    removed = rotate_checkpoints(tmp_path, "ckpt_step_*.pt", keep_last=2)
    assert len(removed) == 2
    survivors = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert "ckpt_A_final.pt" in survivors and "ckpt_A_best.pt" in survivors
    assert "ckpt_step_000300.pt" in survivors and "ckpt_step_000400.pt" in survivors


def test_find_latest_picks_the_newest(cfg, tmp_path):
    import time
    for step in (100, 200):
        make_state(cfg, tmp_path, step=step)
        time.sleep(0.01)
    assert find_latest(tmp_path).name == "ckpt_step_000200.pt"


def test_atomic_write_leaves_no_tmp_file(cfg, tmp_path):
    make_state(cfg, tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------- the fp16 clip ordering

def test_clip_happens_after_unscale(cfg, monkeypatch, tmp_path):
    """Clipping a scaled gradient compares a scaled norm to an unscaled
    threshold, so it effectively never fires. Order matters; assert it."""
    from training.data import TokenDataset as _TD  # noqa: F401
    from training.trainer import Trainer

    calls: list[str] = []

    class FakeData:
        n_tokens = 1000

        def batch(self, step, batch_size, seed, device):
            gen = torch.Generator().manual_seed(step)
            w = torch.randint(0, cfg.model.vocab_size, (2, 9), generator=gen)
            return w[:, :-1], w[:, 1:]

    small = Config(cfg.to_dict())
    small.train.batch_size = 2
    small.train.max_steps = 2
    small.train.precision = "fp32"
    small.eval.every_steps = 0
    small.checkpoint.every_steps = 0

    set_seed(1337)
    trainer = Trainer(
        model=build_model(small), cfg=small, train_data=FakeData(), val_data=None,
        device="cpu", tokenizer_fingerprint="test",
        run_dir=tmp_path / "run", checkpoint_dir=tmp_path / "ckpt",
    )

    real_unscale = trainer.scaler.unscale_
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_unscale(opt):
        calls.append("unscale")
        return real_unscale(opt)

    def spy_clip(params, max_norm, *a, **kw):
        calls.append("clip")
        return real_clip(params, max_norm, *a, **kw)

    monkeypatch.setattr(trainer.scaler, "unscale_", spy_unscale)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    trainer.train_step()
    trainer.metrics.close()

    assert calls == ["unscale", "clip"], f"wrong order: {calls}"


def test_train_step_reduces_loss_on_a_repeated_batch(cfg, tmp_path):
    """A miniature of scripts/05_overfit_batch.py, run in CI."""
    from training.trainer import Trainer

    window = torch.randint(0, cfg.model.vocab_size, (4, 17))
    fixed_x, fixed_y = window[:, :-1], window[:, 1:]   # properly shifted

    class OneBatch:
        n_tokens = 1000

        def batch(self, step, batch_size, seed, device):
            return fixed_x, fixed_y

    small = Config(cfg.to_dict())
    small.train.batch_size = 4
    small.train.max_steps = 60
    small.train.precision = "fp32"
    small.optim.lr = 3e-3
    small.optim.weight_decay = 0.0
    small.schedule.min_lr_fraction = 1.0
    small.eval.every_steps = 0
    small.checkpoint.every_steps = 0

    set_seed(1337)
    trainer = Trainer(
        model=build_model(small), cfg=small, train_data=OneBatch(), val_data=None,
        device="cpu", tokenizer_fingerprint="test",
        run_dir=tmp_path / "run", checkpoint_dir=tmp_path / "ckpt",
    )

    first, _, _ = trainer.train_step()
    trainer.step += 1
    for _ in range(59):
        last, _, _ = trainer.train_step()
        trainer.step += 1
    trainer.metrics.close()

    assert first == pytest.approx(math.log(cfg.model.vocab_size), abs=0.3)
    assert last < first / 2, f"loss barely moved: {first:.3f} -> {last:.3f}"


def test_misaligned_targets_do_not_score_ln_vocab(cfg):
    """Why the ln(V) check is a real test and not a formality.

    The LM head is tied to the token embedding, so the residual stream carries
    each token's own embedding into a head that scores it against itself. Feed
    targets == inputs -- the classic off-by-one -- and step-0 loss lands near
    7.30 instead of 8.32. The alignment bug is therefore *visible* in the very
    number we present as the random-initialisation proof.
    """
    set_seed(1337)
    model = build_model(cfg).eval()
    window = torch.randint(0, cfg.model.vocab_size, (4, 17))

    with torch.no_grad():
        _, misaligned = model(window[:, :-1], window[:, :-1])
        _, correct = model(window[:, :-1], window[:, 1:])

    ln_v = math.log(cfg.model.vocab_size)
    assert abs(float(correct) - ln_v) < 0.1
    assert float(misaligned) < ln_v - 0.5


def test_resume_is_bit_exact(cfg, tmp_path):
    """30 + checkpoint + resume + 30 must equal 60 uninterrupted, exactly.

    This is the Colab insurance policy. If resume drifts, a preempted run
    silently becomes a different experiment from the one you started.
    """
    from training.trainer import Trainer

    small = Config(cfg.to_dict())
    small.train.batch_size = 4
    small.train.max_steps = 20
    small.train.precision = "fp32"
    small.eval.every_steps = 0
    small.checkpoint.every_steps = 0

    window = torch.randint(0, small.model.vocab_size, (4, 17))

    class Fixed:
        n_tokens = 1000

        def batch(self, step, batch_size, seed, device):
            gen = torch.Generator().manual_seed(step)
            offset = torch.randint(0, 100, (1,), generator=gen).item()
            return (window[:, :-1] + offset) % small.model.vocab_size, window[:, 1:]

    def fresh(tag):
        set_seed(1337)
        model = build_model(small)
        return model, Trainer(
            model=model, cfg=small, train_data=Fixed(), val_data=None, device="cpu",
            tokenizer_fingerprint="fp-test",
            run_dir=tmp_path / tag, checkpoint_dir=tmp_path / tag / "ck",
        )

    # A: uninterrupted
    model_a, trainer_a = fresh("a")
    for _ in range(20):
        trainer_a.train_step()
        trainer_a.step += 1
    trainer_a.metrics.close()

    # B: half, then checkpoint
    model_b, trainer_b = fresh("b")
    for _ in range(10):
        trainer_b.train_step()
        trainer_b.step += 1
    path = save_checkpoint(
        tmp_path / "mid.pt", model=model_b, optimizer=trainer_b.optimizer,
        scheduler=trainer_b.scheduler, scaler=trainer_b.scaler,
        step=trainer_b.step, tokens_seen=trainer_b.tokens_seen,
        config=small.to_dict(), config_hash=small.hash(),
        tokenizer_fingerprint="fp-test",
    )
    trainer_b.metrics.close()

    # C: a "new process" with a deliberately wrong seed, resumed from B
    model_c, trainer_c = fresh("c")
    set_seed(999)
    payload = load_checkpoint(
        path, model=model_c, optimizer=trainer_c.optimizer,
        scheduler=trainer_c.scheduler, scaler=trainer_c.scaler,
        expected_tokenizer_fingerprint="fp-test",
    )
    trainer_c.step = payload["step"]
    trainer_c.tokens_seen = payload["tokens_seen"]
    for _ in range(10):
        trainer_c.train_step()
        trainer_c.step += 1
    trainer_c.metrics.close()

    for (name, a), (_, c) in zip(model_a.named_parameters(), model_c.named_parameters()):
        assert torch.equal(a, c), f"{name} drifted after resume"
    assert trainer_a.tokens_seen == trainer_c.tokens_seen
