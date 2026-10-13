#!/usr/bin/env python3
"""Stage 3: train the neural fusion coordinator on cached VALIDATION bundles.

    python scripts/15_train_coordinator.py

The language model is frozen; only the coordinator learns. It is trained on
validation bundles and reported on test bundles, which is the whole reason the
two were generated separately — a coordinator trained on test data would make
the M5 row meaningless.

Only problems where the agents produced at least two DISTINCT candidates, one
of which is correct, can teach anything: if every agent said the same thing
there is no choice to make, and if none is right no choice helps. The script
reports how many problems that leaves, because a coordinator trained on a few
hundred examples is a different claim from one trained on thousands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.roster import is_correct                               # noqa: E402
from collaboration.fusion import (FusionCoordinator, extract_candidates,  # noqa: E402
                                  fusion_loss)
from utils.logging_utils import get_logger, write_manifest          # noqa: E402
from utils.seeding import set_seed                                  # noqa: E402

log = get_logger("coordinator")


def build_training_set(bundles) -> tuple[list, dict]:
    """(rows, stats). A row is (hidden, scalars, target_index)."""
    rows = []
    unanimous = no_correct = unusable = 0

    for bundle in bundles:
        candidates = extract_candidates(bundle)
        if len(candidates) < 2:
            unanimous += 1
            continue
        targets = [i for i, c in enumerate(candidates) if is_correct(c.answer, bundle.gold)]
        if not targets:
            no_correct += 1
            continue
        if len(targets) > 1:
            unusable += 1          # two candidates normalise to the same answer
            continue
        rows.append((
            torch.stack([c.hidden for c in candidates]),
            torch.stack([c.scalars for c in candidates]),
            targets[0],
        ))

    return rows, {
        "usable": len(rows),
        "skipped_unanimous": unanimous,
        "skipped_no_correct_candidate": no_correct,
        "skipped_ambiguous": unusable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", default="experiments/agents/bundles_val.pt")
    parser.add_argument("--out", default="checkpoints/coordinator.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--calibration-weight", type=float, default=0.3)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    bundle_path = ROOT / args.bundles if not Path(args.bundles).is_absolute() \
        else Path(args.bundles)
    blob = torch.load(bundle_path, weights_only=False)
    bundles = blob["bundles"]

    log.info("=" * 74)
    log.info("STAGE 3: NEURAL FUSION COORDINATOR  |  the language model stays frozen")
    log.info("=" * 74)
    log.info("bundles       : %s from %s", f"{len(bundles):,}", bundle_path.name)

    rows, stats = build_training_set(bundles)
    log.info("usable rows   : %s", f"{stats['usable']:,}")
    log.info("  skipped, all agents agreed          %s",
             f"{stats['skipped_unanimous']:,}")
    log.info("  skipped, no candidate was correct   %s",
             f"{stats['skipped_no_correct_candidate']:,}")
    log.info("  skipped, candidates were ambiguous  %s", f"{stats['skipped_ambiguous']:,}")

    if stats["usable"] < 50:
        log.error("fewer than 50 usable rows -- there is not enough disagreement in the")
        log.error("bundles to train a coordinator. M5 will fall back to M4 and say so.")
        log.error("The fix is better role separation in Stage 2, not more epochs here.")
        write_manifest(ROOT / "experiments" / "coordinator",
                       script="scripts/15_train_coordinator.py", trained=False,
                       reason="insufficient disagreement", **stats)
        return

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    split_at = int(len(rows) * (1 - args.holdout))
    train_rows = [rows[i] for i in order[:split_at]]
    held_rows = [rows[i] for i in order[split_at:]]
    log.info("train / holdout: %s / %s", f"{len(train_rows):,}", f"{len(held_rows):,}")

    hidden_dim = rows[0][0].shape[-1]
    coordinator = FusionCoordinator(hidden_dim, width=args.width)
    log.info("coordinator   : %s parameters (hidden dim %d)",
             f"{coordinator.num_parameters():,}", hidden_dim)
    optimizer = torch.optim.AdamW(coordinator.parameters(), lr=args.lr, weight_decay=0.01)

    def accuracy_on(subset) -> float:
        coordinator.eval()
        with torch.no_grad():
            hits = sum(
                1 for hidden, scalars, target in subset
                if int(coordinator(hidden, scalars).argmax()) == target
            )
        return hits / max(len(subset), 1)

    log.info("-" * 74)
    log.info("baseline: always pick the first candidate -> %.4f on the holdout",
             sum(1 for _, _, t in held_rows if t == 0) / max(len(held_rows), 1))

    best = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        coordinator.train()
        epoch_order = torch.randperm(len(train_rows), generator=generator).tolist()
        total_loss = total_brier = 0.0
        for index in epoch_order:
            hidden, scalars, target = train_rows[index]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = fusion_loss(
                coordinator(hidden, scalars), target, args.calibration_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(coordinator.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_brier += parts["brier"]

        held = accuracy_on(held_rows)
        train_accuracy = accuracy_on(train_rows)
        history.append({"epoch": epoch, "loss": round(total_loss / len(train_rows), 6),
                        "train_accuracy": round(train_accuracy, 6),
                        "holdout_accuracy": round(held, 6)})
        log.info("epoch %2d | loss %.4f | brier %.4f | train %.4f | holdout %.4f%s",
                 epoch, total_loss / len(train_rows), total_brier / len(train_rows),
                 train_accuracy, held, "  <- best" if held > best else "")
        if held > best:
            best = held
            best_state = {k: v.clone() for k, v in coordinator.state_dict().items()}

    if best_state is not None:
        coordinator.load_state_dict(best_state)

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": coordinator.state_dict(),
        "hidden_dim": hidden_dim,
        "width": args.width,
        "calibration_weight": args.calibration_weight,
        "holdout_accuracy": best,
        "trained_on": str(bundle_path),
    }, out_path)

    log.info("-" * 74)
    log.info("best holdout selection accuracy : %.4f", best)
    log.info("saved                            : %s", out_path.relative_to(ROOT))

    write_manifest(
        ROOT / "experiments" / "coordinator",
        script="scripts/15_train_coordinator.py", trained=True, seed=args.seed,
        bundles=str(bundle_path), n_bundles=len(bundles),
        coordinator_parameters=coordinator.num_parameters(),
        hidden_dim=hidden_dim, width=args.width, lr=args.lr, epochs=args.epochs,
        calibration_weight=args.calibration_weight,
        holdout_accuracy=round(best, 6), history=history,
        language_model_frozen=True, **stats,
    )
    log.info("=" * 74)
    log.info("next: python scripts/16_evaluate_system.py")


if __name__ == "__main__":
    main()
