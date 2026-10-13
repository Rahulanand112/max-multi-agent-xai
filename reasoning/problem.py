"""The Problem record: one reasoning item with its ground truth.

Every generator in `reasoning/generators/` produces these, and everything
downstream — SFT rendering, agent evaluation, the demo — consumes them. The
fields are chosen so that *nothing* is ever scored by judgement:

  answer        the exact string a correct response must reduce to
  steps         the derivation, computed alongside the answer, never written
                by hand and never produced by a model
  distractors   wrong answers a plausible mistake would reach, each labelled
                with the mistake that produces it

That last field is what makes a Critic and a Verifier trainable without a
teacher model. A critic has to be shown wrong answers *and told what is wrong
with them*; because we generate the problem, we know both exactly. No API
model, no pretrained teacher, no human annotation — the supervision falls out
of the generator.

Contamination
-------------
Because these problems are produced by code we wrote, no item can have
appeared in anyone's pretraining corpus. That is a guarantee work built on
downloaded models cannot make, and it is the reason the benchmark is
generated rather than borrowed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Distractor:
    """A wrong answer, and the specific error that produces it."""
    answer: str
    error: str                 # short phrase: what went wrong
    kind: str                  # machine-readable error class


@dataclass
class Problem:
    id: str
    family: str                # "arithmetic" | "logic"
    template: str              # which template produced it, for leakage checks
    difficulty: int            # 1..4, number of reasoning steps
    question: str
    answer: str                # the exact ground-truth answer string
    steps: list[str] = field(default_factory=list)
    distractors: list[Distractor] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)   # logic: the premises
    meta: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Hash of the question text, for exact deduplication across splits."""
        return hashlib.sha256(" ".join(self.question.split()).lower().encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["fingerprint"] = self.fingerprint()
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def from_dict(row: dict) -> "Problem":
        row = dict(row)
        row.pop("fingerprint", None)
        row["distractors"] = [Distractor(**d) for d in row.get("distractors", [])]
        return Problem(**row)


def load_jsonl(path) -> list[Problem]:
    from pathlib import Path
    problems: list[Problem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                problems.append(Problem.from_dict(json.loads(line)))
    return problems


def write_jsonl(problems: list[Problem], path) -> int:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(problem.to_json() + "\n")
    return len(problems)
