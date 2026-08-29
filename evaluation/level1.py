"""Level 1 metrics: the language model alone, with no agents involved.

Kept strictly separate from Level 2 (the multi-agent system) so that the
question "did the improvement come from the model or from the architecture?"
stays answerable. Every Level 2 result must name the Level 1 checkpoint that
produced it.
"""

from __future__ import annotations

import math
from collections import Counter

import torch

from training.optim import autocast_context


@torch.no_grad()
def evaluate_split(model, dataset, starts_list, device: str, use_fp16: bool) -> dict:
    """Loss, perplexity, bits/token and next-token accuracy on fixed batches.

    `starts_list` must be the same fixed windows the trainer validated on, or
    the number will not reproduce -- it would be measuring different data.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for starts in starts_list:
        x, y = dataset.gather(starts, device)
        with autocast_context(device, use_fp16):
            logits, loss = model(x, y)
        total_loss += float(loss) * y.numel()
        total_correct += int((logits.float().argmax(-1) == y).sum())
        total_tokens += int(y.numel())

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "bits_per_token": mean_loss / math.log(2),
        "accuracy_at_1": total_correct / total_tokens,
        "tokens_evaluated": total_tokens,
        "batches": len(starts_list),
    }


def text_statistics(samples: list[str]) -> dict:
    """Degeneracy measures for generated text.

    A tiny model's most common failure is looping -- "and then and then and
    then". distinct-n and the repetition rate quantify that, so "the samples
    look repetitive" becomes a number rather than an impression.
    """
    words: list[str] = []
    for text in samples:
        words.extend(text.split())
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0, "words": 0}

    bigrams = list(zip(words, words[1:]))
    repeated = sum(1 for a, b in zip(words, words[1:]) if a == b)

    return {
        "distinct_1": len(set(words)) / len(words),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "repetition_rate": repeated / max(len(words) - 1, 1),
        "words": len(words),
        "most_common": Counter(words).most_common(5),
    }
