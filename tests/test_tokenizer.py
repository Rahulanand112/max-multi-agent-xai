"""Tokenizer correctness tests.

The round-trip test is the gate for V0: if encode->decode is not exact, every
number produced downstream is measuring the wrong thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer.bpe import PRETOKEN_PATTERN, MAXTokenizer   # noqa: E402
from tokenizer.special_tokens import (                     # noqa: E402
    BOS_ID,
    EOS_ID,
    N_RESERVED,
    ROLE_TOKENS,
    SPECIAL_TOKENS,
    SPECIAL_TO_ID,
)

CORPUS = [
    "Once upon a time, there was a little girl named Lily.",
    "She had a red ball. The ball was very bouncy!",
    "Tom said, \"Let's play in the park today.\"",
    "The dog ran fast. It ran and ran and ran.",
    "Lily and Tom were happy.\nThey played all day long.",
    "A big tree stood near the pond; birds sang in it.",
    "He didn't want to go home, but it was getting dark.",
    "They counted 1, 2, 3 stars in the sky.\nThen they went inside.",
] * 30


@pytest.fixture(scope="module")
def tok() -> MAXTokenizer:
    return MAXTokenizer.train(CORPUS, vocab_size=300, min_char_freq=1, verbose=False)


# ------------------------------------------------------------ pre-tokenizer

@pytest.mark.parametrize("text", [
    "hello world",
    "  leading and   multiple   spaces  ",
    "line one\nline two\n\nline four",
    "punctuation!!! and, commas; plus 123 numbers",
    "don't can't we'll they've I'm",
    "tabs\there\tand\tthere",
    "",
])
def test_pretokenizer_covers_the_whole_string(text: str) -> None:
    """Concatenating the matches must reproduce the input exactly.

    If this fails, characters are silently vanishing before BPE ever sees them.
    """
    assert "".join(PRETOKEN_PATTERN.findall(text)) == text


# ------------------------------------------------------------- special tokens

def test_special_tokens_occupy_reserved_ids() -> None:
    assert len(SPECIAL_TOKENS) == N_RESERVED
    assert len(set(SPECIAL_TOKENS)) == N_RESERVED
    assert SPECIAL_TO_ID["<|pad|>"] == 0
    assert SPECIAL_TO_ID["<|bos|>"] == 1
    assert SPECIAL_TO_ID["<|eos|>"] == 2
    assert SPECIAL_TO_ID["<|unk|>"] == 3


def test_role_tokens_are_single_ids(tok: MAXTokenizer) -> None:
    """Agent prompting is only cheap if a role costs exactly one token."""
    for role in ROLE_TOKENS:
        assert len(tok.encode(role)) == 1, role


def test_specials_survive_inside_text(tok: MAXTokenizer) -> None:
    ids = tok.encode("<|solver|>the ball<|answer|>red")
    assert SPECIAL_TO_ID["<|solver|>"] in ids
    assert SPECIAL_TO_ID["<|answer|>"] in ids


def test_bos_eos_wrapping(tok: MAXTokenizer) -> None:
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


# ------------------------------------------------------------------ roundtrip

@pytest.mark.parametrize("text", [
    "Once upon a time, there was a little girl named Lily.",
    "The dog ran fast. It ran and ran and ran.",
    "hello world",
    "a",
    "   ",
    "line one\nline two",
    "He didn't want to go home, but it was getting dark.",
])
def test_roundtrip_is_exact(tok: MAXTokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_over_whole_corpus(tok: MAXTokenizer) -> None:
    failures = [d for d in CORPUS if tok.decode(tok.encode(d)) != d]
    assert not failures, f"{len(failures)} documents failed round-trip"


def test_empty_string(tok: MAXTokenizer) -> None:
    assert tok.encode("") == []
    assert tok.decode([]) == ""


# ------------------------------------------------------------------ vocabulary

def test_vocab_size_is_respected(tok: MAXTokenizer) -> None:
    assert tok.vocab_size <= 300
    assert tok.vocab_size == len(tok.vocab)


def test_all_ids_are_in_range(tok: MAXTokenizer) -> None:
    for doc in CORPUS[:20]:
        for i in tok.encode(doc):
            assert 0 <= i < tok.vocab_size


def test_merges_compress_below_character_level(tok: MAXTokenizer) -> None:
    """A trained BPE must beat one-token-per-character, or it learned nothing."""
    text = " ".join(CORPUS[:20])
    assert len(tok.encode(text)) < len(text)


def test_vocab_size_too_small_is_rejected() -> None:
    with pytest.raises(ValueError):
        MAXTokenizer.train(CORPUS, vocab_size=N_RESERVED - 1, verbose=False)


# --------------------------------------------------------------- persistence

def test_save_and_load_roundtrip(tok: MAXTokenizer, tmp_path: Path) -> None:
    tok.save(tmp_path)
    reloaded = MAXTokenizer.load(tmp_path)

    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.merges == tok.merges
    assert reloaded.fingerprint() == tok.fingerprint()

    for doc in CORPUS[:20]:
        assert reloaded.encode(doc) == tok.encode(doc)


def test_fingerprint_changes_with_vocabulary(tok: MAXTokenizer) -> None:
    """The loader relies on this to refuse a mismatched checkpoint."""
    other = MAXTokenizer.train(CORPUS, vocab_size=200, min_char_freq=1, verbose=False)
    assert other.fingerprint() != tok.fingerprint()


# --------------------------------------------------------------- determinism

def test_training_is_deterministic() -> None:
    a = MAXTokenizer.train(CORPUS, vocab_size=250, min_char_freq=1, verbose=False)
    b = MAXTokenizer.train(CORPUS, vocab_size=250, min_char_freq=1, verbose=False)
    assert a.merges == b.merges
    assert a.fingerprint() == b.fingerprint()


def test_unknown_characters_become_unk(tok: MAXTokenizer) -> None:
    ids = tok.encode("日本語")
    assert all(i == SPECIAL_TO_ID["<|unk|>"] for i in ids)


def test_unknown_characters_are_visible_on_decode(tok: MAXTokenizer) -> None:
    """Lossy encoding must be visible. Silent deletion hides a real problem."""
    decoded = tok.decode(tok.encode("cat 日本 dog"))
    assert "�" in decoded
    assert "cat" in decoded and "dog" in decoded


def test_rare_characters_are_dropped_from_the_alphabet() -> None:
    """min_char_freq controls the alphabet; report the resulting UNK rate."""
    corpus = CORPUS + ["a rare character: §"]
    strict = MAXTokenizer.train(corpus, vocab_size=250, min_char_freq=5, verbose=False)
    loose = MAXTokenizer.train(corpus, vocab_size=250, min_char_freq=1, verbose=False)
    assert "§" not in strict.alphabet
    assert "§" in loose.alphabet
    assert strict.unk_rate(["§"]) > 0
    assert loose.unk_rate(["§"]) == 0
