"""Console + CSV logging, and the run manifest.

Two rules this module exists to enforce:
  1. Every metric that appears on a slide is also a row in a CSV on disk.
  2. Every run writes a manifest recording exactly how it was produced.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "max", logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(stream)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class MetricsLogger:
    """Append-only CSV writer. Header is fixed on the first row written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None

    def log(self, **row: Any) -> None:
        if self._writer is None:
            self._fields = list(row.keys())
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fields)
            if self.path.stat().st_size == 0:
                self._writer.writeheader()
        assert self._fields is not None
        self._writer.writerow({k: row.get(k, "") for k in self._fields})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
            # bf16 needs Ampere (8.0) or newer; the T4 is Turing (7.5).
            info["bf16_supported"] = props.major >= 8
    except ImportError:
        info["torch"] = None
    return info


def write_manifest(run_dir: str | Path, **extra: Any) -> Path:
    """Write experiments/<run>/manifest.json.

    This is the file that makes a result reproducible three months later, when
    the details have gone. Never skip it.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "hardware": hardware_info(),
        "argv": sys.argv,
    }
    manifest.update(extra)

    path = run_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return path
