"""MAXTokenizer -- our BPE tokenizer, trained on our corpus.

Nothing is downloaded and no pretrained vocabulary is used: the merge table is
learned from the training split of our corpus and nothing else. The learning
algorithm lives in trainer.py; the regex and whitespace handling in
pretokenizer.py. This file is the tokenizer itself: encode, decode, save, load,
and the diagnostics that tell us whether it is any good.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from .pretokenizer import (  # re-exported: scripts and tests import from here
    PRETOKEN_PATTERN,
    mark_whitespace,
    unmark_whitespace,
)
from .special_tokens import (
    BOS_ID,
    EOS_ID,
    N_RESERVED,
    SPECIAL_TOKENS,
    SPECIAL_TO_ID,
    UNK_ID,
)
from .trainer import learn_merges

__all__ = ["MAXTokenizer", "PRETOKEN_PATTERN"]


class MAXTokenizer:
    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
        alphabet: list[str] | None = None,
    ) -> None:
        self.vocab: list[str] = list(vocab)
        self.merges: list[tuple[str, str]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.alphabet: set[str] = set(alphabet or [])

        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token: list[str] = self.vocab
        self.merge_ranks: dict[tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"MAXTokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, alphabet={len(self.alphabet)})"
        )

    # --------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_char_freq: int = 5,
        min_pair_freq: int = 2,
        verbose: bool = True,
        log_every: int = 500,
    ) -> "MAXTokenizer":
        """Learn a merge table from `texts`.

        Only ever call this on the TRAINING split. Fitting the vocabulary on the
        full corpus leaks validation statistics into the tokenizer -- a small
        effect, free to avoid, and an examiner may well ask.
        """
        vocab, merges, alphabet = learn_merges(
            texts,
            vocab_size=vocab_size,
            min_char_freq=min_char_freq,
            min_pair_freq=min_pair_freq,
            verbose=verbose,
            log_every=log_every,
        )
        return cls(vocab=vocab, merges=merges, alphabet=alphabet)

    # --------------------------------------------------------------- encoding

    def _bpe_chunk(self, chunk: str) -> list[int]:
        """Apply the merge table to one pre-tokenized chunk. Memoised."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        symbols: list[str | None] = [c if c in self.alphabet else None for c in chunk]

        while len(symbols) > 1:
            best_rank: int | None = None
            best_index = -1
            for i in range(len(symbols) - 1):
                left, right = symbols[i], symbols[i + 1]
                if left is None or right is None:
                    continue
                rank = self.merge_ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            if best_rank is None:
                break
            merged = str(symbols[best_index]) + str(symbols[best_index + 1])
            symbols[best_index : best_index + 2] = [merged]

        ids = [UNK_ID if s is None else self.token_to_id.get(s, UNK_ID) for s in symbols]
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to token ids. Special-token literals in the text are honoured."""
        ids: list[int] = [BOS_ID] if add_bos else []
        for piece, is_special in _split_on_specials(text):
            if is_special:
                ids.append(SPECIAL_TO_ID[piece])
                continue
            for chunk in PRETOKEN_PATTERN.findall(piece):
                ids.extend(self._bpe_chunk(mark_whitespace(chunk)))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_batch(self, texts: Iterable[str], **kwargs) -> list[list[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    # --------------------------------------------------------------- decoding

    def decode(
        self,
        ids: Iterable[int],
        skip_special: bool = True,
        unk_repr: str = "�",
    ) -> str:
        """Decode token ids back to text.

        <|unk|> renders as U+FFFD by default rather than being dropped. Encoding
        an out-of-alphabet character is inherently lossy, and silently deleting
        it would hide that loss -- you would see a clean round-trip and never
        learn that characters were vanishing. Pass unk_repr="" to drop them.
        """
        pieces: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                continue
            token = self.id_to_token[i]
            if i == UNK_ID:
                pieces.append(unk_repr if skip_special else token)
            elif i < N_RESERVED:
                if skip_special:
                    continue
                pieces.append(token)
            else:
                pieces.append(token)
        return unmark_whitespace("".join(pieces))

    # ------------------------------------------------------------ diagnostics

    def unk_rate(self, texts: Iterable[str]) -> float:
        """Fraction of emitted tokens that are <|unk|>. Report it; do not assume it."""
        total = unks = 0
        for text in texts:
            ids = self.encode(text)
            total += len(ids)
            unks += sum(1 for i in ids if i == UNK_ID)
        return unks / total if total else 0.0

    def roundtrip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text

    # ---------------------------------------------------------- serialisation

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "max-bpe-v1",
            "vocab_size": self.vocab_size,
            "n_reserved": N_RESERVED,
            "alphabet": sorted(self.alphabet),
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }
        path = directory / "tokenizer.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "MAXTokenizer":
        directory = Path(directory)
        path = directory / "tokenizer.json" if directory.is_dir() else directory
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "max-bpe-v1":
            raise ValueError(f"unrecognised tokenizer format: {payload.get('format')}")
        return cls(
            vocab=payload["vocab"],
            merges=[tuple(m) for m in payload["merges"]],
            alphabet=payload["alphabet"],
        )

    def fingerprint(self) -> str:
        """Stable hash of the vocabulary, stored in every checkpoint.

        Loading a checkpoint against a re-fitted tokenizer produces a model that
        generates confident nonsense with no error message. The checkpoint
        loader compares this and refuses on mismatch.
        """
        blob = "\x1f".join(self.vocab).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


def _split_on_specials(text: str) -> Iterator[tuple[str, bool]]:
    for piece in _SPECIAL_RE.split(text):
        if piece:
            yield piece, piece in SPECIAL_TO_ID
