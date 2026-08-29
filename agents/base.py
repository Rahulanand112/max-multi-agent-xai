"""One trained checkpoint, addressed by a role-control token.

Every agent in MAX is the same frozen weights called with a different role
token. Nothing here has its own parameters -- that is what makes five agents
affordable on student hardware, and it is why any measured difference between
agents comes from the role token and the decoding settings, nothing else.

Each call returns three things, matching the (answer, hidden, confidence)
triple that the Version 5 coordinator will consume:

  * the generated text, parsed against the role schema
  * a pooled hidden state from the final layer
  * a confidence that is MEASURED, not invented -- the mean token
    log-probability of the generated span

That last point matters. It would be easy to print a plausible-looking 0.87
here. Instead the number is computed from the model's own distribution over the
tokens it chose, so it means something even before Stage 2 teaches the model to
report confidence in words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from reasoning.schema import ParsedOutput, build_prompt, parse
from tokenizer.special_tokens import BOS_ID, EOS_ID


@dataclass
class AgentOutput:
    role: str
    name: str
    text: str
    parsed: ParsedOutput
    answer: str | None
    confidence: float          # measured: exp(mean token log-prob)
    mean_logprob: float
    entropy: float             # mean predictive entropy over generated positions
    hidden: torch.Tensor       # pooled final-layer state, the V5 coordinator's input
    n_tokens: int
    seed: int
    temperature: float


class RoleAgent:
    """A role-token view of the shared checkpoint."""

    def __init__(
        self,
        name: str,
        role_token: str,
        model,
        tokenizer,
        device: str,
        temperature: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 60,
    ) -> None:
        self.name = name
        self.role_token = role_token
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def __repr__(self) -> str:
        return (f"RoleAgent({self.name}, token={self.role_token}, "
                f"T={self.temperature}, top_k={self.top_k})")

    def run(self, question: str, seed: int, **prompt_kwargs) -> AgentOutput:
        from utils.seeding import set_seed

        prompt = build_prompt(self.role_token, question, **prompt_kwargs)
        prompt_ids = [BOS_ID] + self.tokenizer.encode(prompt)

        set_seed(seed)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_k=self.top_k, eos_id=EOS_ID,
        )

        generated_ids = out[0, len(prompt_ids):].tolist()
        text = self.tokenizer.decode(generated_ids)
        scores = self._score(out, len(prompt_ids))

        return AgentOutput(
            role=self.role_token,
            name=self.name,
            text=text,
            parsed=(p := parse(self.role_token, text)),
            answer=p.answer,
            confidence=scores["confidence"],
            mean_logprob=scores["mean_logprob"],
            entropy=scores["entropy"],
            hidden=scores["hidden"],
            n_tokens=len(generated_ids),
            seed=seed,
            temperature=self.temperature,
        )

    @torch.no_grad()
    def _score(self, sequence: torch.Tensor, prompt_len: int) -> dict:
        """Measure how confident the model was in what it just wrote.

        One forward pass over the whole sequence, then read off the model's own
        probability for each token it actually emitted.
        """
        self.model.eval()
        window = sequence[:, -self.model.context_length:]
        logits, _, hidden = self.model(window, return_hidden=True)

        offset = max(0, prompt_len - (sequence.shape[1] - window.shape[1]))
        log_probs = F.log_softmax(logits.float(), dim=-1)

        chosen: list[float] = []
        entropies: list[float] = []
        for pos in range(offset - 1, window.shape[1] - 1):
            if pos < 0:
                continue
            token = int(window[0, pos + 1])
            chosen.append(float(log_probs[0, pos, token]))
            probs = log_probs[0, pos].exp()
            entropies.append(float(-(probs * log_probs[0, pos]).sum()))

        mean_logprob = sum(chosen) / len(chosen) if chosen else float("-inf")
        entropy = sum(entropies) / len(entropies) if entropies else float("nan")

        # pooled state: mean over the generated span plus the final position,
        # which is the shape the V5 coordinator expects
        span = hidden[0, max(offset - 1, 0):]
        pooled = torch.cat([span.mean(0), hidden[0, -1]]) if span.numel() else hidden[0, -1].repeat(2)

        return {
            "mean_logprob": mean_logprob,
            "confidence": math.exp(mean_logprob) if chosen else 0.0,
            "entropy": entropy,
            "hidden": pooled.detach().cpu(),
        }
