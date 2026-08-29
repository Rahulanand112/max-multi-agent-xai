"""The from-scratch guarantee, enforced by a test rather than by a promise.

This is the machine-checkable half of the Review 1 claim. The other half is the
step-0 loss equalling ln(vocab_size), which arrives with the model in V1.

Run it live during the review:  pytest tests/test_no_pretrained.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Any of these appearing in our source would mean weights came from somewhere
# other than our own training loop.
FORBIDDEN = [
    r"\bfrom_pretrained\b",
    r"\bAutoModel\w*\b",
    r"\bAutoTokenizer\b",
    r"\bhf_hub_download\b",
    r"\bsnapshot_download\b",
    r"\btimm\.create_model\b",
    r"\btorch\.hub\.load\b",
    r"\bGPT2\w*\b",
    r"\bLlama\w*For\w*\b",
]

# Directories that are not our source code.
# Note: "data" itself is NOT skipped -- data/synth/ holds the MAX-Reason
# generator, which is our source and must be scanned. Only the download and
# artefact directories are excluded.
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "venv", ".venv", "node_modules",
    "raw", "interim", "processed", "checkpoints", "experiments", "artifacts",
}

# This file necessarily contains the forbidden strings in order to search for them.
SKIP_FILES = {"test_no_pretrained.py"}


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in {".py", ".yaml", ".yml", ".ipynb"}:
            files.append(path)
    return files


def test_no_pretrained_model_loading() -> None:
    patterns = [re.compile(p) for p in FORBIDDEN]
    hits: list[str] = []

    for path in source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            # a comment saying "we do not use from_pretrained" is not a violation
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            for pattern in patterns:
                if pattern.search(line):
                    rel = path.relative_to(ROOT)
                    hits.append(f"{rel}:{line_no}: {stripped[:100]}")

    assert not hits, (
        "Pretrained-model loading detected. The MAX language model must be "
        "randomly initialised and trained by us:\n  " + "\n  ".join(hits)
    )


def test_transformers_is_not_a_dependency() -> None:
    """`datasets` is fine -- it downloads data. `transformers` ships models."""
    req = ROOT / "requirements.txt"
    active = [
        ln.strip() for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    offenders = [ln for ln in active if ln.lower().startswith(("transformers", "timm"))]
    assert not offenders, f"model-bearing library in requirements.txt: {offenders}"


def test_no_checkpoint_files_committed() -> None:
    """A .pt file inside the source tree would be a checkpoint of unknown origin."""
    weights = [
        p.relative_to(ROOT)
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix in {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
        and "checkpoints" not in p.parts
        and "data" not in p.parts
    ]
    assert not weights, f"unexpected weight files in the source tree: {weights}"
