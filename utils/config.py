"""Configuration loading.

Every experiment is defined by a YAML file. Configs are loaded into a plain
dict wrapper that supports attribute access, and every config carries a stable
hash so a checkpoint can refuse to load against a config it was not trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively.

    cfg.model.d_model  ==  cfg["model"]["d_model"]
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key {name!r}. available: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        out: dict = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def hash(self) -> str:
        """Stable 12-char hash of the config contents.

        Stored in every checkpoint. Loading a checkpoint whose config hash does
        not match the config you are loading it with is a hard error, not a
        warning -- that mismatch is how you end up with a model that generates
        confident nonsense and no traceback.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving a single optional `_base_` inheritance key."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_name = raw.pop("_base_", None)
    if base_name is not None:
        base = load_config(path.parent / base_name)
        raw = _deep_merge(base.to_dict(), raw)

    return Config(raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=True)
