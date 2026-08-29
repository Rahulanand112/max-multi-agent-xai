"""Learning the BPE merge table (Sennrich et al., 2016), written by us.

The algorithm is simple: repeatedly merge the most frequent adjacent symbol
pair. The engineering is where the care goes.

A naive implementation recounts every pair after every merge, which is
O(merges x corpus) and unusable past a few megabytes -- learning 4,000 merges
on 200 MB would take hours. Instead we keep:

    pair_counts   pair -> total frequency
    pair_words    pair -> the set of chunks containing it
    heap          a lazy max-heap over pair_counts

and update all three incrementally, touching only the chunks that actually
contain the merged pair. Heap entries go stale as counts change and are
verified on pop rather than being deleted, which is far cheaper than keeping
the heap exact.
"""

from __future__ import annotations

import heapq
from collections import Counter

from .pretokenizer import PRETOKEN_PATTERN, mark_whitespace
from .special_tokens import N_RESERVED, SPECIAL_TOKENS

UNKNOWN = "\x00"  # placeholder for a character outside the alphabet


def count_chunks(texts) -> tuple[Counter, Counter, int]:
    """Frequency of every pre-tokenized chunk, and of every character."""
    word_freqs: Counter[str] = Counter()
    char_freqs: Counter[str] = Counter()
    n_docs = 0
    for text in texts:
        n_docs += 1
        char_freqs.update(mark_whitespace(text))
        for chunk in PRETOKEN_PATTERN.findall(text):
            word_freqs[mark_whitespace(chunk)] += 1
    return word_freqs, char_freqs, n_docs


def learn_merges(
    texts,
    vocab_size: int,
    min_char_freq: int = 5,
    min_pair_freq: int = 2,
    verbose: bool = True,
    log_every: int = 500,
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Return (vocab, merges, alphabet)."""
    if vocab_size <= N_RESERVED:
        raise ValueError(f"vocab_size must exceed {N_RESERVED} reserved ids")

    word_freqs, char_freqs, n_docs = count_chunks(texts)
    if verbose:
        print(f"  documents           : {n_docs:,}")
        print(f"  distinct chunks     : {len(word_freqs):,}")
        print(f"  distinct characters : {len(char_freqs):,}")

    # --- alphabet ---------------------------------------------------------
    alphabet = sorted(c for c, n in char_freqs.items() if n >= min_char_freq)
    n_dropped = sum(1 for c, n in char_freqs.items() if n < min_char_freq)
    if verbose:
        print(f"  alphabet kept       : {len(alphabet):,} "
              f"(dropped {n_dropped:,} rare chars -> <|unk|>)")

    alphabet_set = set(alphabet)
    vocab: list[str] = list(SPECIAL_TOKENS) + alphabet
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than {N_RESERVED} specials "
            f"+ {len(alphabet)} characters"
        )
    if verbose:
        print(f"  merges to learn     : {num_merges:,}")

    # --- initial splits and pair statistics -------------------------------
    splits: list[list[str]] = []
    freqs: list[int] = []
    for word, freq in word_freqs.items():
        symbols = [c if c in alphabet_set else UNKNOWN for c in word]
        if all(s == UNKNOWN for s in symbols):
            continue  # carries no merge signal
        splits.append(symbols)
        freqs.append(freq)

    pair_counts: dict[tuple[str, str], int] = {}
    pair_words: dict[tuple[str, str], set[int]] = {}
    for wid, symbols in enumerate(splits):
        freq = freqs[wid]
        for pair in zip(symbols, symbols[1:]):
            if UNKNOWN in pair:
                continue
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_words.setdefault(pair, set()).add(wid)

    heap: list[tuple[int, tuple[str, str]]] = [
        (-count, pair) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    # --- the merge loop ---------------------------------------------------
    merges: list[tuple[str, str]] = []
    while len(merges) < num_merges:
        best = _pop_best(heap, pair_counts)
        if best is None or pair_counts.get(best, 0) < min_pair_freq:
            if verbose:
                print(f"  stopped early at {len(merges):,} merges "
                      f"(no pair reaches min_pair_freq={min_pair_freq})")
            break

        merged = best[0] + best[1]
        merges.append(best)

        for wid in list(pair_words.get(best, ())):
            symbols = splits[wid]
            freq = freqs[wid]
            _retract(symbols, freq, wid, pair_counts, pair_words)
            symbols = _apply_merge(symbols, best, merged)
            splits[wid] = symbols
            _post(symbols, freq, wid, pair_counts, pair_words, heap)

        pair_counts.pop(best, None)
        pair_words.pop(best, None)
        vocab.append(merged)

        if verbose and log_every and len(merges) % log_every == 0:
            print(f"    merge {len(merges):>5,}/{num_merges:,}  "
                  f"{best[0]!r}+{best[1]!r} -> {merged!r}")

    if verbose:
        print(f"  final vocab size    : {len(vocab):,}")
    return vocab, merges, alphabet


# ----------------------------------------------------------------- internals

def _pop_best(heap, pair_counts):
    """Pop until we find a heap entry whose count is still current."""
    while heap:
        neg_count, pair = heapq.heappop(heap)
        current = pair_counts.get(pair)
        if current is not None and current == -neg_count:
            return pair
    return None


def _retract(symbols, freq, wid, pair_counts, pair_words) -> None:
    for pair in zip(symbols, symbols[1:]):
        if UNKNOWN in pair:
            continue
        remaining = pair_counts.get(pair, 0) - freq
        if remaining <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = remaining
        holders = pair_words.get(pair)
        if holders is not None:
            holders.discard(wid)


def _apply_merge(symbols, best, merged) -> list[str]:
    out: list[str] = []
    i, n = 0, len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def _post(symbols, freq, wid, pair_counts, pair_words, heap) -> None:
    for pair in zip(symbols, symbols[1:]):
        if UNKNOWN in pair:
            continue
        count = pair_counts.get(pair, 0) + freq
        pair_counts[pair] = count
        pair_words.setdefault(pair, set()).add(wid)
        heapq.heappush(heap, (-count, pair))
