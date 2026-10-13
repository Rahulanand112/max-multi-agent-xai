"""The neural fusion coordinator: Stage 3, and the project's research question.

M3 and M4 aggregate at the level of TEXT — they count answer strings. This
module aggregates at the level of the representations the agents computed on
their way to those strings. The question the whole project exists to answer is
whether that is worth anything.

How it works
------------
Each distinct candidate answer gets a feature vector built from:

  * the pooled final-layer hidden state of the agent that produced it
    (2 x d_model: the mean over the generated span, and the final position)
  * that agent's MEASURED confidence, exp(mean token log-probability)
  * its predictive entropy
  * its stated <|conf|> value, when it emitted one
  * how many agents proposed this same answer
  * whether the Verifier called it correct, and whether the Critic objected

A small MLP scores each candidate; a softmax over candidates picks one. About
0.2M parameters, trained in minutes, with the language model FROZEN
throughout — so nothing here can improve the result by secretly training the
LM further, and the comparison against M3 and M4 stays about aggregation.

Calibration in the loss
-----------------------
Training on cross-entropy alone produces a coordinator that is accurate and
badly overconfident. The loss adds a Brier term on the selected candidate's
probability, which penalises confident mistakes specifically. The weight is a
config value, and the ablation reports accuracy and ECE with the term on and
off — because a calibration term that costs accuracy is a trade, not a free
win, and the honest thing is to show both numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.roster import Bundle

N_SCALAR_FEATURES = 7


@dataclass
class CandidateFeatures:
    answer: str
    hidden: torch.Tensor      # (2 * d_model,)
    scalars: torch.Tensor     # (N_SCALAR_FEATURES,)


def extract_candidates(bundle: Bundle) -> list[CandidateFeatures]:
    """Turn one Bundle into one feature row per DISTINCT candidate answer.

    Distinct, not per agent: the coordinator chooses between answers, and two
    agents proposing the same answer is evidence for that answer rather than
    two things to choose between. The agreement count carries that evidence.
    """
    answering = [o for o in bundle.answering() if o.answer is not None]
    if not answering:
        return []

    counts: dict[str, int] = {}
    for output in answering:
        counts[output.answer] = counts.get(output.answer, 0) + 1

    verdict = bundle.verifier_verdict()
    objected = float(bundle.critic_found_issue())
    verifier_confidence = bundle.verifier.confidence if bundle.verifier else 0.0
    shown_to_verifier = answering[0].answer

    seen: set[str] = set()
    rows: list[CandidateFeatures] = []
    for output in answering:
        if output.answer in seen:
            continue
        seen.add(output.answer)
        is_shown = float(output.answer == shown_to_verifier)
        rows.append(CandidateFeatures(
            answer=output.answer,
            hidden=output.hidden.float(),
            scalars=torch.tensor([
                output.confidence,
                output.entropy if output.entropy == output.entropy else 0.0,  # NaN guard
                output.parsed.self_confidence if output.parsed.self_confidence is not None else 0.5,
                counts[output.answer] / max(len(answering), 1),
                1.0 if (verdict == "correct" and is_shown) else 0.0,
                1.0 if (verdict == "incorrect" and is_shown) else 0.0,
                objected * is_shown,
            ], dtype=torch.float32),
        ))

    # the verifier's own confidence scales the two verdict features, so a
    # hesitant verdict counts for less than a sure one
    for row in rows:
        row.scalars[4] *= verifier_confidence
        row.scalars[5] *= verifier_confidence
    return rows


class FusionCoordinator(nn.Module):
    """Scores candidates from agent representations. The LM stays frozen."""

    def __init__(self, hidden_dim: int, width: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        # a gate over the scalar evidence, so the module can learn to lean on
        # the verifier when the hidden states are uninformative and vice versa
        self.gate = nn.Sequential(
            nn.Linear(N_SCALAR_FEATURES, width),
            nn.Sigmoid(),
        )
        self.score = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width + N_SCALAR_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, hidden: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """hidden (n, hidden_dim), scalars (n, F) -> logits (n,)"""
        projected = self.project(hidden) * self.gate(scalars)
        return self.score(torch.cat([projected, scalars], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def decide(self, bundle: Bundle) -> tuple[str | None, float, dict]:
        """Pick a candidate. Confidence is the softmax probability it assigned."""
        rows = extract_candidates(bundle)
        if not rows:
            return None, 0.0, {"method": "M5", "candidates": 0}
        if len(rows) == 1:
            # nothing to choose between; report the agents' own confidence
            agreement = float(rows[0].scalars[3])
            return rows[0].answer, agreement, {"method": "M5", "candidates": 1,
                                               "unanimous": True}

        self.eval()
        hidden = torch.stack([r.hidden for r in rows])
        scalars = torch.stack([r.scalars for r in rows])
        probabilities = F.softmax(self(hidden, scalars), dim=0)
        best = int(probabilities.argmax())
        return rows[best].answer, float(probabilities[best]), {
            "method": "M5",
            "candidates": len(rows),
            "unanimous": False,
            "probabilities": [round(float(p), 4) for p in probabilities],
        }


def fusion_loss(logits: torch.Tensor, target: int, calibration_weight: float = 0.3
                ) -> tuple[torch.Tensor, dict]:
    """Cross-entropy over candidates, plus a Brier penalty on overconfidence.

    The Brier term is applied to the probability of the candidate the model
    actually selected, against whether that selection was right. A confident
    correct pick is barely penalised; a confident wrong pick is penalised
    hard. That is the behaviour a decision system needs, and plain
    cross-entropy does not produce it on its own.
    """
    probabilities = F.softmax(logits, dim=0)
    cross_entropy = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))

    chosen = int(probabilities.argmax())
    chosen_probability = probabilities[chosen]
    correct = 1.0 if chosen == target else 0.0
    brier = (chosen_probability - correct) ** 2

    total = cross_entropy + calibration_weight * brier
    return total, {
        "cross_entropy": float(cross_entropy.detach()),
        "brier": float(brier.detach()),
        "correct": correct,
    }
