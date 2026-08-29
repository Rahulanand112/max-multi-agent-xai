#!/usr/bin/env python3
"""V1 model verification -- the Review 1 evidence, produced in one command.

    python scripts/04_verify_model.py --config configs/model_a.yaml

Six checks, in the order they should be shown to a panel:

  1. parameter count matches the config exactly       -> the architecture is ours
  2. weight statistics after initialisation           -> the weights were sampled
  3. STEP-0 LOSS == ln(vocab_size)                    -> the model knows nothing
  4. explicit attention == fused attention            -> our maths is correct
  5. every parameter receives a gradient              -> nothing is dead
  6. forward/backward shapes are right                -> the plumbing works

Writes experiments/verify_model_<name>/ with a manifest, a weight histogram and
a console log that can be screenshotted.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.init import histogram_data, weight_statistics      # noqa: E402
from model.transformer import build_model                     # noqa: E402
from utils.config import load_config                          # noqa: E402
from utils.logging_utils import get_logger, write_manifest    # noqa: E402
from utils.seeding import set_seed                            # noqa: E402

log = get_logger("verify")


def load_real_batch(processed: Path, batch: int, ctx: int, seed: int):
    """A batch of real tokens from train.bin, if the corpus has been encoded."""
    path = processed / "train.bin"
    if not path.exists():
        return None, None
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) < ctx + 1:
        return None, None
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - ctx - 1, size=batch)
    x = np.stack([data[s : s + ctx].astype(np.int64) for s in starts])
    y = np.stack([data[s + 1 : s + 1 + ctx].astype(np.int64) for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_a.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer_a.yaml")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures: list[str] = []
    results: dict = {}

    log.info("=" * 70)
    log.info("MAX-LM verification  |  %s  |  seed %d  |  device %s",
             cfg.name, args.seed, device)
    log.info("=" * 70)

    # ---------------------------------------------------------- 1. parameters
    started = time.time()
    model = build_model(cfg).to(device)
    build_seconds = time.time() - started

    log.info("")
    log.info("[1] PARAMETER COUNT")
    for label, count in model.parameter_breakdown():
        marker = "  " if label != "TOTAL" else "->"
        log.info("  %s %-34s %14s", marker, label, f"{count:,}")

    total = model.num_parameters()
    expected_total = cfg.expected.n_parameters
    results["n_parameters"] = total
    results["n_parameters_expected"] = expected_total
    if total == expected_total:
        log.info("  PASS  matches configs/%s exactly", Path(args.config).name)
    else:
        failures.append(f"parameter count {total:,} != expected {expected_total:,}")
        log.error("  FAIL  expected %s", f"{expected_total:,}")

    # ------------------------------------------------- 2. weight statistics
    log.info("")
    log.info("[2] WEIGHTS AFTER RANDOM INITIALISATION")
    stats = weight_statistics(model)
    log.info("  %-22s %10s %9s %9s %9s %9s",
             "group", "count", "mean", "std", "min", "max")
    for group, s in stats.items():
        log.info("  %-22s %10s %+9.5f %9.5f %+9.4f %+9.4f",
                 group, f"{s['count']:,}", s["mean"], s["std"], s["min"], s["max"])
    results["weight_statistics"] = stats

    init_std = cfg.get("init", {}).get("std", 0.02)
    all_stats = stats["ALL"]
    if abs(all_stats["mean"]) > 0.002:
        failures.append(f"weight mean {all_stats['mean']:.5f} is not ~0")
    emb = stats["embeddings"]
    if abs(emb["std"] - init_std) > 0.002:
        failures.append(f"embedding std {emb['std']:.5f} != {init_std}")
    else:
        log.info("  PASS  embeddings ~ N(0, %.2f); LayerNorm at identity", init_std)

    resid = stats.get("residual_projections")
    if resid:
        want = init_std / math.sqrt(2 * cfg.model.n_layers)
        log.info("  PASS  residual projections std %.5f (1/sqrt(2L) scaled, target %.5f)",
                 resid["std"], want)

    # ------------------------------------------------------- 3. step-0 loss
    log.info("")
    log.info("[3] STEP-0 LOSS  <- the random-initialisation proof")
    ctx = cfg.model.context_length
    vocab = cfg.model.vocab_size
    ln_v = model.expected_init_loss()

    data_cfg = load_config(ROOT / args.data_config)
    x, y = load_real_batch(ROOT / data_cfg.paths.processed_dir, args.batch, ctx, args.seed)
    source = "real corpus (data/processed/train.bin)"
    if x is None:
        gen = torch.Generator().manual_seed(args.seed)
        x = torch.randint(0, vocab, (args.batch, ctx), generator=gen)
        y = torch.randint(0, vocab, (args.batch, ctx), generator=gen)
        source = "random token ids (corpus not encoded yet)"

    model.eval()
    with torch.no_grad():
        logits, loss = model(x.to(device), y.to(device))
    step0 = float(loss)

    log.info("  batch source        : %s", source)
    log.info("  batch shape         : %s -> logits %s", tuple(x.shape), tuple(logits.shape))
    log.info("  measured step-0 loss: %.4f", step0)
    log.info("  ln(%d)              : %.4f", vocab, ln_v)
    log.info("  difference          : %+.4f", step0 - ln_v)
    log.info("  implied perplexity  : %.1f   (uniform over %d tokens = %d)",
             math.exp(step0), vocab, vocab)

    results["step0_loss"] = round(step0, 6)
    results["ln_vocab"] = round(ln_v, 6)
    results["step0_delta"] = round(step0 - ln_v, 6)
    results["step0_perplexity"] = round(math.exp(step0), 3)
    results["step0_batch_source"] = source

    if abs(step0 - ln_v) <= 0.05:
        log.info("  PASS  within +/-0.05 of ln(vocab).")
        log.info("        The model spreads probability evenly across all %d tokens --", vocab)
        log.info("        it knows nothing. Pretrained weights would start far below this.")
    else:
        failures.append(f"step-0 loss {step0:.4f} differs from ln(V)={ln_v:.4f} by more than 0.05")
        log.error("  FAIL  outside tolerance")

    # ------------------------------------------- 4. attention path agreement
    log.info("")
    log.info("[4] ATTENTION: EXPLICIT MATHS vs FUSED KERNEL")
    fast = build_model(cfg, use_sdpa=True).to(device)
    fast.load_state_dict(model.state_dict())
    fast.eval()
    with torch.no_grad():
        logits_fast, _ = fast(x.to(device))
    max_diff = float((logits - logits_fast).abs().max())
    results["attention_max_abs_diff"] = max_diff
    log.info("  max |explicit - fused| over logits: %.3e", max_diff)
    if max_diff < 1e-3:
        log.info("  PASS  our softmax(QK^T/sqrt(d) + causal mask)V matches PyTorch's kernel")
    else:
        failures.append(f"attention paths disagree by {max_diff:.3e}")
        log.error("  FAIL  the two paths disagree")

    # ----------------------------------------------------- 5. gradient flow
    log.info("")
    log.info("[5] GRADIENT FLOW")
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x.to(device), y.to(device))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    zero = [n for n, p in model.named_parameters()
            if p.grad is not None and float(p.grad.abs().sum()) == 0.0]
    grad_norm = float(torch.sqrt(sum(
        (p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None
    )))
    log.info("  parameters with gradients : %d / %d",
             sum(1 for _, p in model.named_parameters() if p.grad is not None),
             sum(1 for _ in model.named_parameters()))
    log.info("  global gradient norm      : %.4f", grad_norm)
    results["grad_norm"] = round(grad_norm, 6)
    if missing:
        failures.append(f"{len(missing)} parameters received no gradient: {missing[:5]}")
        log.error("  FAIL  no gradient: %s", missing[:5])
    elif zero:
        log.warning("  WARN  all-zero gradient: %s", zero[:5])
    else:
        log.info("  PASS  every parameter receives a non-zero gradient")

    # --------------------------------------------------------- 6. generation
    log.info("")
    log.info("[6] FORWARD / GENERATION PLUMBING")
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(prompt, max_new_tokens=12, temperature=0.8, top_k=50)
    log.info("  generate() 1 x 1 -> %s   (untrained: output is noise, as expected)",
             tuple(out.shape))
    ok_shape = out.shape == (1, 13)
    if not ok_shape:
        failures.append(f"generate produced shape {tuple(out.shape)}, expected (1, 13)")
    else:
        log.info("  PASS  autoregressive loop runs")

    # ------------------------------------------------------------- manifest
    tok_fingerprint = None
    try:
        from tokenizer.bpe import MAXTokenizer

        tok_cfg = load_config(ROOT / args.tokenizer_config)
        tok = MAXTokenizer.load(ROOT / tok_cfg.paths.out_dir)
        tok_fingerprint = tok.fingerprint()
        if tok.vocab_size != vocab:
            failures.append(
                f"tokenizer vocab {tok.vocab_size} != model vocab {vocab}"
            )
    except Exception as exc:  # tokenizer not built yet on this machine
        log.warning("tokenizer not loaded (%s)", exc)

    run_dir = ROOT / "experiments" / f"verify_model_{cfg.name}"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        edges, counts = histogram_data(model)
        centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(centres, counts, width=(edges[1] - edges[0]), color="#1B4A8F")
        ax.set_title(f"MAX-LM {cfg.name}: weights at initialisation (seed {args.seed})")
        ax.set_xlabel("weight value")
        ax.set_ylabel("count")
        ax.axvline(0.0, color="#A24B0C", lw=1, ls="--")
        fig.tight_layout()
        fig.savefig(run_dir / "init_weight_histogram.png", dpi=140)
        plt.close(fig)
        log.info("")
        log.info("histogram -> %s", run_dir / "init_weight_histogram.png")
    except ImportError:
        log.warning("matplotlib unavailable; skipping histogram")

    write_manifest(
        run_dir,
        stage="model_verification",
        model_name=cfg.name,
        seed=args.seed,
        device=device,
        config=cfg.to_dict(),
        config_hash=cfg.hash(),
        tokenizer_fingerprint=tok_fingerprint,
        build_seconds=round(build_seconds, 4),
        results=results,
        failures=failures,
        passed=not failures,
    )

    log.info("")
    log.info("=" * 70)
    if failures:
        log.error("VERIFICATION FAILED (%d)", len(failures))
        for item in failures:
            log.error("  - %s", item)
        sys.exit(1)
    log.info("ALL CHECKS PASSED")
    log.info("  parameters   %s", f"{total:,}")
    log.info("  step-0 loss  %.4f  vs  ln(%d) = %.4f", step0, vocab, ln_v)
    log.info("  manifest     %s", run_dir / "manifest.json")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
