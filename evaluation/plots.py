"""The loss curve.

Train and validation on one axis, with ln(vocab) drawn as a reference line so
the whole story is visible in one picture: the model starts exactly where a
model that knows nothing starts, and falls from there.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def read_metrics(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Split metrics.csv into training rows and validation rows."""
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            if row.get("train_loss") not in ("", None):
                train_rows.append({"step": step, "loss": float(row["train_loss"])})
            if row.get("val_loss") not in ("", None):
                val_rows.append({"step": step, "loss": float(row["val_loss"])})
    return train_rows, val_rows


def plot_loss_curve(
    metrics_csv: str | Path,
    out_path: str | Path,
    vocab_size: int,
    final_val: float | None = None,
    final_step: int | None = None,
    title: str = "MAX-1M pretraining",
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_rows, val_rows = read_metrics(metrics_csv)
    if not train_rows:
        raise ValueError(f"no training rows found in {metrics_csv}")

    # The final evaluation happens after the loop ends, so it may not be in the
    # CSV. Add it rather than leaving the curve stopping short of the result we
    # actually report.
    if final_val is not None and final_step is not None:
        if not any(r["step"] == final_step for r in val_rows):
            val_rows.append({"step": final_step, "loss": final_val})
            val_rows.sort(key=lambda r: r["step"])

    ln_v = math.log(vocab_size)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.axhline(ln_v, color="#A24B0C", lw=1.2, ls="--", zorder=1,
               label=f"ln({vocab_size}) = {ln_v:.4f}  (random init)")
    ax.plot([r["step"] for r in train_rows], [r["loss"] for r in train_rows],
            color="#1B4A8F", lw=1.4, label="training loss", zorder=3)
    if val_rows:
        ax.plot([r["step"] for r in val_rows], [r["loss"] for r in val_rows],
                color="#146049", lw=1.8, marker="o", ms=3.5,
                label="validation loss", zorder=4)

    first = train_rows[0]
    last_val = val_rows[-1] if val_rows else None
    max_step = max(r["step"] for r in train_rows)
    ax.annotate(f"step 0: {first['loss']:.4f}",
                xy=(first["step"], first["loss"]),
                xytext=(max_step * 0.04, first["loss"] - 0.5),
                fontsize=9, color="#101C31")
    if last_val:
        ax.annotate(f"final val: {last_val['loss']:.4f}\nppl {math.exp(last_val['loss']):.1f}",
                    xy=(last_val["step"], last_val["loss"]),
                    xytext=(last_val["step"] * 0.62, last_val["loss"] + 0.85),
                    fontsize=9, color="#146049",
                    arrowprops=dict(arrowstyle="->", color="#146049", lw=1))

    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title(title)
    ax.grid(alpha=0.18, lw=0.7)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
