"""Reproducible randomness.

Every script calls set_seed() exactly once, at the top, before anything else
touches a random number generator. The seed is recorded in the run manifest.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy and (if available) torch + CUDA.

    Returns the seed so callers can log it without repeating themselves.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cudnn autotuning picks different algorithms run to run, which makes
        # results irreproducible at the last decimal place. Off by default.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def rng_state() -> dict[str, Any]:
    """Capture every RNG state, for a checkpoint that can resume bit-identically."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by rng_state()."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except ImportError:
        pass
