"""Logical deduction problems, generated with a verified unique answer.

Three families, each with ground truth computed by actually doing the
deduction in code rather than asserting it:

  transitive_order   a chain of comparisons; answer is the element at one end
  category_chain     a chain of "all A are B" rules, asked forwards (valid) or
                     backwards (invalid, answer "no")
  constraint_grid    small assignment puzzle, solved by exhaustive search

The constraint puzzles are what make the Verifier agent meaningful. They are
generated, then SOLVED, then kept only if exactly one assignment satisfies
every clue — a puzzle with two valid solutions is discarded rather than
shipped with one of them labelled correct.

A note on combinatorial space
-----------------------------
The first version of this file produced a 33% duplicate rate at 5,000 items,
because a two-premise syllogism over four category pairs and eight names has
only a few hundred distinct forms. Duplicates matter more here than they look:
a repeated question that lands in both the training and the test split turns
a reasoning benchmark into a memorisation benchmark. The pools below, and the
variable-length category chain that replaced the fixed syllogism, exist to
push the space past a hundred thousand distinct items per template. The
measured duplicate rate is printed by scripts/11_generate_reasoning.py on
every run, and the splits are made by fingerprint so a duplicate can never
straddle them.
"""

from __future__ import annotations

import itertools
import random

from reasoning.problem import Distractor, Problem

PEOPLE = [
    "Ana", "Ben", "Cara", "Dev", "Esha", "Fin", "Gita", "Hugo", "Iris", "Jai",
    "Kira", "Liam", "Mina", "Nile", "Opal", "Pari", "Quin", "Rhea", "Sam", "Tara",
]

ATTRIBUTES = [
    ("taller", "shortest", "tallest"),
    ("faster", "slowest", "fastest"),
    ("older", "youngest", "oldest"),
    ("heavier", "lightest", "heaviest"),
    ("stronger", "weakest", "strongest"),
    ("richer", "poorest", "richest"),
    ("louder", "quietest", "loudest"),
    ("warmer", "coolest", "warmest"),
]

# singular / plural pairs used to build "all A are B" chains
CATEGORIES = [
    ("rose", "roses"), ("flower", "flowers"), ("plant", "plants"),
    ("robin", "robins"), ("bird", "birds"), ("animal", "animals"),
    ("square", "squares"), ("rectangle", "rectangles"), ("shape", "shapes"),
    ("van", "vans"), ("car", "cars"), ("vehicle", "vehicles"),
    ("oak", "oaks"), ("tree", "trees"), ("apple", "apples"), ("fruit", "fruits"),
    ("terrier", "terriers"), ("dog", "dogs"), ("pebble", "pebbles"), ("stone", "stones"),
]

ITEM_POOLS = [
    ("pet", ["cat", "dog", "bird", "fish", "rabbit", "hamster", "turtle"]),
    ("favourite colour", ["red", "blue", "green", "yellow", "purple", "orange", "pink"]),
    ("favourite fruit", ["apple", "pear", "plum", "mango", "peach", "cherry", "melon"]),
    ("hat", ["brown hat", "white hat", "black hat", "grey hat", "straw hat", "wool hat"]),
    ("favourite sport", ["running", "swimming", "cycling", "tennis", "football", "hockey"]),
]


def _article(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


# ---------------------------------------------------------------- templates

def t_transitive_order(rng: random.Random):
    """A chain of comparisons; the answer is the element at one end.

    Built from a shuffled true ordering, so the premises are guaranteed
    consistent and the extreme is guaranteed unique.
    """
    length = rng.choice([3, 3, 4, 4, 5])
    people = rng.sample(PEOPLE, length)
    more, least, most = rng.choice(ATTRIBUTES)

    ordered_pairs = [(people[i], people[i + 1]) for i in range(length - 1)]
    shuffled = list(ordered_pairs)
    rng.shuffle(shuffled)
    facts = [f"{a} is {more} than {b}." for a, b in shuffled]

    ask_most = rng.random() < 0.5
    answer = people[0] if ask_most else people[-1]
    question = " ".join(facts) + f" Who is the {most if ask_most else least}?"

    steps = [f"{a} is {more} than {b}" for a, b in ordered_pairs]
    steps.append(f"so the order is {' then '.join(people)}")
    steps.append(f"the answer is {answer}")

    distractors = [
        Distractor(people[-1] if ask_most else people[0],
                   "read the chain in the wrong direction", "reversed_order"),
        Distractor(people[1] if ask_most else people[-2],
                   "stopped one link early in the chain", "incomplete_chain"),
    ]
    return question, answer, steps, distractors, max(1, length - 1), facts


def t_category_chain(rng: random.Random):
    """A chain of "all A are B" rules, asked in a valid or invalid direction.

    Forwards the inference is sound and the answer is yes. Backwards — being
    a B does not make you an A — it is unsound and the answer is no. The
    invalid half is deliberately half the data: a model that learns to answer
    "yes" to anything shaped like a syllogism scores 50% here, and the
    per-direction breakdown in the evaluation makes that visible.
    """
    # Categories are sampled freely, so chains like "all dogs are plants" occur.
    # That is deliberate, not a bug. If the premises always agreed with world
    # knowledge, a model could score well by recalling facts instead of
    # applying the rules, and we could not tell the two apart. Arbitrary
    # pairings make the premises the only available route to the answer --
    # and make the item impossible to answer from anything memorised.
    depth = rng.choice([2, 2, 3])
    chain = rng.sample(CATEGORIES, depth + 1)
    name = rng.choice(PEOPLE)

    rules = [f"All {chain[i][1]} are {chain[i + 1][1]}."
             for i in range(depth)]
    first, last = chain[0][0], chain[-1][0]
    valid = rng.random() < 0.5

    if valid:
        facts = rules + [f"{name} has {_article(first)}."]
        question = " ".join(facts) + f" Does {name} have {_article(last)}?"
        answer = "yes"
        steps = [r.rstrip(".") for r in rules]
        steps += [f"{name} has {_article(first)}",
                  f"following the rules, {name} has {_article(last)}",
                  "so the answer is yes"]
        distractors = [
            Distractor("no", "did not follow the chain of rules to the end",
                       "incomplete_chain"),
        ]
    else:
        facts = rules + [f"{name} has {_article(last)}."]
        question = " ".join(facts) + f" Does {name} definitely have {_article(first)}?"
        answer = "no"
        steps = [r.rstrip(".") for r in rules]
        steps += [f"the rules say every {first} is {_article(last)}",
                  f"they do not say every {last} is {_article(first)}",
                  "so we cannot be sure, the answer is no"]
        distractors = [
            Distractor("yes", "applied the rule backwards, which it does not support",
                       "converse_error"),
        ]
    return question, answer, steps, distractors, depth + 1, facts


def t_constraint_grid(rng: random.Random):
    """Match people to items under constraints; solved by exhaustive search."""
    size = rng.choice([3, 3, 4])
    people = rng.sample(PEOPLE, size)
    noun, pool = rng.choice(ITEM_POOLS)
    items = rng.sample(pool, size)

    truth = dict(zip(people, rng.sample(items, size)))

    clues: list[str] = []
    constraints: list[tuple[str, str, bool]] = []

    for _ in range(size - 1):
        person = rng.choice(people)
        wrong = rng.choice([i for i in items if i != truth[person]])
        clues.append(f"{person} does not have the {wrong}.")
        constraints.append((person, wrong, False))

    pinned = rng.choice(people)
    clues.append(f"{pinned} has the {truth[pinned]}.")
    constraints.append((pinned, truth[pinned], True))
    rng.shuffle(clues)

    def satisfies(assignment: dict) -> bool:
        for person, item, positive in constraints:
            if positive and assignment[person] != item:
                return False
            if not positive and assignment[person] == item:
                return False
        return True

    solutions = [dict(zip(people, perm)) for perm in itertools.permutations(items)
                 if satisfies(dict(zip(people, perm)))]
    if len(solutions) != 1:
        return None                      # not uniquely determined: discard

    candidates = [p for p in people if p != pinned]
    asked = rng.choice(candidates) if candidates else pinned
    answer = truth[asked]
    question = " ".join(clues) + f" What is {asked}'s {noun}?"

    steps = [c.rstrip(".") for c in clues]
    steps.append(f"only one assignment fits every clue, so {asked} has the {answer}")

    distractors = [
        Distractor(truth[pinned], "repeated the item from the clue that names a person",
                   "copied_premise"),
    ]
    other = [i for i in items if i != answer and i != truth[pinned]]
    if other:
        distractors.append(
            Distractor(rng.choice(other), "ignored one of the clues", "dropped_constraint")
        )
    return question, answer, steps, distractors, size, clues


TEMPLATES = {
    "transitive_order": t_transitive_order,
    "category_chain": t_category_chain,
    "constraint_grid": t_constraint_grid,
}


def generate(n: int, seed: int, templates: list[str] | None = None,
             id_prefix: str = "lg") -> list[Problem]:
    """Generate n logic problems. Deterministic in `seed`.

    Constraint puzzles without a unique solution are regenerated rather than
    kept, so exactly n problems come back.
    """
    rng = random.Random(seed)
    names = templates or list(TEMPLATES)
    unknown = [t for t in names if t not in TEMPLATES]
    if unknown:
        raise KeyError(f"unknown logic template(s): {unknown}")

    problems: list[Problem] = []
    attempts = 0
    while len(problems) < n and attempts < n * 50 + 100:
        key = names[len(problems) % len(names)]
        attempts += 1
        made = TEMPLATES[key](rng)
        if made is None:
            continue
        question, answer, steps, distractors, difficulty, facts = made
        distractors = [d for d in distractors if d.answer != answer]
        problems.append(Problem(
            id=f"{id_prefix}_{len(problems):06d}",
            family="logic",
            template=key,
            difficulty=min(4, max(1, difficulty)),
            question=question,
            answer=answer,
            steps=steps,
            distractors=distractors,
            facts=facts,
        ))
    if len(problems) < n:
        raise RuntimeError(
            f"only generated {len(problems)} of {n} logic problems in {attempts} attempts"
        )
    return problems
