"""Multi-step arithmetic word problems, generated with their ground truth.

Each template computes the answer as it builds the sentence, so the label is
produced by the same arithmetic the question describes — it cannot drift from
the text. The difficulty of an item is its number of arithmetic steps, which
is a property of the template, not an estimate.

Distractors are the interesting part. For every template we also compute the
answer a *specific* plausible mistake would give: adding where you should
subtract, forgetting the final operation, multiplying the wrong pair. Those
become the wrong candidates the Critic learns to object to and the Verifier
learns to reject — supervision that would otherwise need a teacher model.

Vocabulary is kept deliberately close to TinyStories (names, everyday objects,
simple verbs) so the pretrained base model is not asked to learn a new
distribution of words at the same time as a new task.
"""

from __future__ import annotations

import random

from reasoning.problem import Distractor, Problem

NAMES = [
    "Maya", "Leo", "Ava", "Noah", "Mia", "Eli", "Zoe", "Finn", "Ivy", "Theo",
    "Nina", "Omar", "Lila", "Jonah", "Ruby", "Max", "Sana", "Kofi", "Elsa", "Arjun",
]

ITEMS = [
    ("pen", "pens"), ("apple", "apples"), ("marble", "marbles"), ("sticker", "stickers"),
    ("coin", "coins"), ("book", "books"), ("flower", "flowers"), ("shell", "shells"),
    ("card", "cards"), ("cookie", "cookies"), ("button", "buttons"), ("stone", "stones"),
]

CONTAINERS = [
    ("box", "boxes"), ("bag", "bags"), ("jar", "jars"), ("basket", "baskets"),
    ("packet", "packets"), ("tray", "trays"),
]


def _plural(pair: tuple[str, str], n: int) -> str:
    return pair[0] if n == 1 else pair[1]


def _usable(distractors: list[Distractor], answer: str) -> list[Distractor]:
    """Drop distractors that give the game away.

    Two kinds are useless as wrong candidates: one that happens to equal the
    correct answer (the arithmetic of the mistake coincided), and a negative
    count, which no reader would accept as an answer to "how many are left"
    and which would teach the Critic to object on the wrong grounds.
    """
    kept: list[Distractor] = []
    seen = {answer}
    for d in distractors:
        if d.answer in seen:
            continue
        if d.answer.lstrip("-").isdigit() and int(d.answer) < 0:
            continue
        seen.add(d.answer)
        kept.append(d)
    return kept


# --------------------------------------------------------------- templates
# Every template returns (question, answer, steps, distractors, difficulty).
# `rng` is passed in so the whole dataset is reproducible from one seed.


def t_group_then_remove(rng: random.Random):
    """k containers of n items, give away g. Two steps: multiply, subtract."""
    name = rng.choice(NAMES)
    item, cont = rng.choice(ITEMS), rng.choice(CONTAINERS)
    k, n = rng.randint(2, 9), rng.randint(2, 9)
    total = k * n
    g = rng.randint(1, max(1, total - 1))
    answer = total - g

    question = (f"{name} has {k} {_plural(cont, k)}. Each {cont[0]} holds {n} "
                f"{_plural(item, n)}. {name} gives away {g} {_plural(item, g)}. "
                f"How many {item[1]} does {name} have left?")
    steps = [
        f"{k} times {n} is {total}",
        f"{total} minus {g} is {answer}",
    ]
    distractors = [
        Distractor(str(total), "stopped after multiplying and never subtracted",
                   "missing_final_step"),
        Distractor(str(total + g), "added the given-away amount instead of subtracting",
                   "wrong_operation"),
        Distractor(str(k + n - g), "added the groups instead of multiplying them",
                   "wrong_operation"),
    ]
    return question, str(answer), steps, distractors, 2


def t_add_then_share(rng: random.Random):
    """Two amounts combined, then split evenly. Add, then divide."""
    name, friend = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS)
    people = rng.randint(2, 5)
    per = rng.randint(2, 9)
    total = people * per
    first = rng.randint(1, total - 1)
    second = total - first

    question = (f"{name} collects {first} {_plural(item, first)} and {friend} collects "
                f"{second} {_plural(item, second)}. They share all of them equally "
                f"between {people} children. How many {item[1]} does each child get?")
    steps = [
        f"{first} plus {second} is {total}",
        f"{total} divided by {people} is {per}",
    ]
    distractors = [
        Distractor(str(total), "added but never divided", "missing_final_step"),
        Distractor(str(abs(first - second)), "subtracted the two amounts instead of adding",
                   "wrong_operation"),
        Distractor(str(max(1, total // (people + 1))), "divided by the wrong number of children",
                   "wrong_quantity"),
    ]
    return question, str(per), steps, distractors, 2


def t_rate_over_time(rng: random.Random):
    """A rate per day over d days, minus a loss. Multiply, subtract."""
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    per_day, days = rng.randint(2, 9), rng.randint(2, 9)
    made = per_day * days
    lost = rng.randint(1, max(1, made - 1))
    answer = made - lost

    question = (f"{name} makes {per_day} {_plural(item, per_day)} every day for {days} days. "
                f"Then {lost} of them are lost. How many {item[1]} are left?")
    steps = [
        f"{per_day} times {days} is {made}",
        f"{made} minus {lost} is {answer}",
    ]
    distractors = [
        Distractor(str(made), "forgot the ones that were lost", "missing_final_step"),
        Distractor(str(per_day + days - lost), "added the rate and the days", "wrong_operation"),
        Distractor(str(made + lost), "added the lost ones back", "wrong_operation"),
    ]
    return question, str(answer), steps, distractors, 2


def t_three_step_shop(rng: random.Random):
    """Buy k at p each, pay with a note, then split the change. Three steps."""
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    k, price = rng.randint(2, 6), rng.randint(2, 9)
    cost = k * price
    note = cost + rng.randint(2, 8) * 2          # keeps the change even
    change = note - cost
    friends = 2
    each = change // friends

    question = (f"{name} buys {k} {_plural(item, k)} for {price} coins each and pays with "
                f"{note} coins. {name} shares the change equally with {friends} friends. "
                f"How many coins does each friend get?")
    steps = [
        f"{k} times {price} is {cost}",
        f"{note} minus {cost} is {change}",
        f"{change} divided by {friends} is {each}",
    ]
    distractors = [
        Distractor(str(change), "gave the whole change instead of each share",
                   "missing_final_step"),
        Distractor(str(cost), "answered the cost rather than the share", "answered_wrong_quantity"),
        Distractor(str(max(0, (note - k) // friends)), "subtracted the count, not the cost",
                   "wrong_quantity"),
    ]
    return question, str(each), steps, distractors, 3


def t_compare_two_people(rng: random.Random):
    """A has n, B has m times as many, how many more does B have. Two steps."""
    a, b = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS)
    n, m = rng.randint(2, 9), rng.randint(2, 6)
    bigger = n * m
    difference = bigger - n

    question = (f"{a} has {n} {_plural(item, n)}. {b} has {m} times as many as {a}. "
                f"How many more {item[1]} does {b} have than {a}?")
    steps = [
        f"{n} times {m} is {bigger}",
        f"{bigger} minus {n} is {difference}",
    ]
    distractors = [
        Distractor(str(bigger), "gave the total instead of the difference",
                   "answered_wrong_quantity"),
        Distractor(str(n * (m + 1)), "multiplied by one too many", "off_by_one"),
        Distractor(str(m - n if m > n else n - m), "subtracted before multiplying",
                   "wrong_order"),
    ]
    return question, str(difference), steps, distractors, 2


def t_four_step_chain(rng: random.Random):
    """Longest chain: multiply, add, subtract, divide. Difficulty 4."""
    name = rng.choice(NAMES)
    item, cont = rng.choice(ITEMS), rng.choice(CONTAINERS)
    k, n = rng.randint(2, 6), rng.randint(2, 6)
    start = k * n
    found = rng.randint(2, 9)
    total = start + found
    broken = rng.randint(1, max(1, total - 4))
    left = total - broken
    groups = 2
    while left % groups != 0:
        broken += 1
        left = total - broken
    each = left // groups

    question = (f"{name} has {k} {_plural(cont, k)} with {n} {_plural(item, n)} in each. "
                f"{name} finds {found} more {item[1]}, then {broken} of them break. "
                f"The rest are shared equally into {groups} piles. "
                f"How many {item[1]} are in each pile?")
    steps = [
        f"{k} times {n} is {start}",
        f"{start} plus {found} is {total}",
        f"{total} minus {broken} is {left}",
        f"{left} divided by {groups} is {each}",
    ]
    distractors = [
        Distractor(str(left), "stopped before sharing into piles", "missing_final_step"),
        Distractor(str(total), "forgot the broken ones", "skipped_step"),
        Distractor(str(start // groups), "ignored the ones that were found", "skipped_step"),
    ]
    return question, str(each), steps, distractors, 4


def t_single_step(rng: random.Random):
    """One operation. The easy rung of the ladder; difficulty 1."""
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    a, b = rng.randint(2, 20), rng.randint(2, 20)
    if rng.random() < 0.5:
        answer = a + b
        question = (f"{name} has {a} {_plural(item, a)} and is given {b} more. "
                    f"How many {item[1]} does {name} have now?")
        steps = [f"{a} plus {b} is {answer}"]
        distractors = [
            Distractor(str(abs(a - b)), "subtracted instead of adding", "wrong_operation"),
            Distractor(str(a * b), "multiplied instead of adding", "wrong_operation"),
        ]
    else:
        big, small = max(a, b), min(a, b)
        answer = big - small
        question = (f"{name} has {big} {_plural(item, big)} and loses {small}. "
                    f"How many {item[1]} are left?")
        steps = [f"{big} minus {small} is {answer}"]
        distractors = [
            Distractor(str(big + small), "added instead of subtracting", "wrong_operation"),
            Distractor(str(small), "answered the amount lost", "answered_wrong_quantity"),
        ]
    return question, str(answer), steps, distractors, 1


TEMPLATES = {
    "single_step": t_single_step,
    "group_then_remove": t_group_then_remove,
    "add_then_share": t_add_then_share,
    "rate_over_time": t_rate_over_time,
    "compare_two_people": t_compare_two_people,
    "three_step_shop": t_three_step_shop,
    "four_step_chain": t_four_step_chain,
}


def generate(n: int, seed: int, templates: list[str] | None = None,
             id_prefix: str = "ar") -> list[Problem]:
    """Generate n arithmetic problems. Deterministic in `seed`."""
    rng = random.Random(seed)
    names = templates or list(TEMPLATES)
    unknown = [t for t in names if t not in TEMPLATES]
    if unknown:
        raise KeyError(f"unknown arithmetic template(s): {unknown}")

    problems: list[Problem] = []
    for i in range(n):
        key = names[i % len(names)]
        question, answer, steps, distractors, difficulty = TEMPLATES[key](rng)
        problems.append(Problem(
            id=f"{id_prefix}_{i:06d}",
            family="arithmetic",
            template=key,
            difficulty=difficulty,
            question=question,
            answer=answer,
            steps=steps,
            distractors=_usable(distractors, answer),
        ))
    return problems
