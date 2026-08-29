"""The special-token registry.

These ids are reserved BEFORE any BPE merge is learned, so they occupy fixed
low indices in every MAX vocabulary.

Why this matters: the role tokens must exist in the vocabulary before
pretraining begins. Adding them later means introducing randomly initialised
embedding rows into an otherwise trained model, which trains badly and is an
entirely avoidable self-inflicted wound.

Indices 16-31 are deliberately left spare. Sixteen unused embedding rows cost
under 0.2% of the Config-A model, and mean that a later design change never
forces a tokenizer rebuild -- which would invalidate every checkpoint trained
before it.
"""

from __future__ import annotations

N_RESERVED = 32

# --- structural ---
PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
UNK = "<|unk|>"

# --- agent roles (used from Stage 2 onwards; reserved from day one) ---
SOLVER = "<|solver|>"
CRITIC = "<|critic|>"
VERIFIER = "<|verifier|>"
ALTERNATIVE = "<|alternative|>"
COORDINATOR = "<|coordinator|>"

# --- field markers for the structured reasoning schema ---
QUESTION = "<|question|>"
STEPS = "<|steps|>"
ANSWER = "<|answer|>"
CONF = "<|conf|>"
ISSUE = "<|issue|>"
VERDICT = "<|verdict|>"
EVIDENCE = "<|evidence|>"

SPECIAL_TOKENS: list[str] = [
    PAD,          # 0
    BOS,          # 1
    EOS,          # 2
    UNK,          # 3
    SOLVER,       # 4
    CRITIC,       # 5
    VERIFIER,     # 6
    ALTERNATIVE,  # 7
    COORDINATOR,  # 8
    QUESTION,     # 9
    STEPS,        # 10
    ANSWER,       # 11
    CONF,         # 12
    ISSUE,        # 13
    VERDICT,      # 14
    EVIDENCE,     # 15
]

# pad the registry out to N_RESERVED with placeholders
SPECIAL_TOKENS += [f"<|reserved_{i}|>" for i in range(len(SPECIAL_TOKENS), N_RESERVED)]

assert len(SPECIAL_TOKENS) == N_RESERVED
assert len(set(SPECIAL_TOKENS)) == N_RESERVED, "duplicate special token"

SPECIAL_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
ID_TO_SPECIAL: dict[int, str] = {i: tok for tok, i in SPECIAL_TO_ID.items()}

PAD_ID = SPECIAL_TO_ID[PAD]
BOS_ID = SPECIAL_TO_ID[BOS]
EOS_ID = SPECIAL_TO_ID[EOS]
UNK_ID = SPECIAL_TO_ID[UNK]

ROLE_TOKENS: list[str] = [SOLVER, CRITIC, VERIFIER, ALTERNATIVE, COORDINATOR]
