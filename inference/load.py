"""Loading a trained checkpoint for inference, with the guards kept on.

Three scripts and the demo app all need the same four lines, and the
temptation in each of them is to skip the fingerprint check "just for a quick
look". That check is the one standing between this project and a model
generating fluent nonsense with no error message, so it lives here and every
caller gets it whether they remember to ask or not.

The config travels INSIDE the checkpoint. Rebuilding the model from a YAML on
disk would mean a config edited after training silently produces a different
architecture, and `load_state_dict` would then fail with a shape error if you
are lucky and succeed if you are not.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model.transformer import build_model
from tokenizer.bpe import MAXTokenizer
from training.checkpoint import describe, load_checkpoint
from utils.config import Config


def load_for_inference(
    checkpoint: str | Path,
    tokenizer_dir: str | Path,
    device: str | None = None,
    root: Path | None = None,
):
    """Return (model, tokenizer, header). The model is in eval mode and frozen."""
    root = root or Path(__file__).resolve().parents[1]
    path = Path(checkpoint)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = Path(tokenizer_dir)
    if not tokenizer_path.is_absolute():
        tokenizer_path = root / tokenizer_path
    tokenizer = MAXTokenizer.load(tokenizer_path)

    header = describe(path)
    payload = load_checkpoint(
        path,
        expected_tokenizer_fingerprint=tokenizer.fingerprint(),
        restore_rng=False,
        map_location=device,
    )

    model = build_model(Config(payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        # every agent reads the SAME frozen weights; nothing downstream of
        # this point is allowed to train the language model
        parameter.requires_grad_(False)

    header = dict(header)
    header["device"] = device
    header["path"] = str(path)
    header["n_parameters"] = model.num_parameters()
    return model, tokenizer, header
