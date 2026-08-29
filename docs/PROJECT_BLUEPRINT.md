# Multi-Agent Explainable AI with Collaborative Neural Reasoning
## Technical analysis & development roadmap — v1.1, 25 August 2026
### STATUS: Version 1 trained. `ckpt_A_final.pt` exists.

**Project handle:** MAX · **Model family:** MAX-LM
**Review 1 model:** 1,331,968 params · **Main model:** 13,784,064 params
**Hardware target:** NVIDIA T4 / P100, 16 GB (Colab / Kaggle free tier)

---

## HARD CONSTRAINT

No pretrained weights enter this project at any stage — not as an initialisation, not as
a teacher, not as a fallback. The model is our architecture, randomly initialised, trained
on data we select, by a training loop we write. PyTorch supplies tensors, autograd, CUDA
kernels and optimisers; it does not supply the model.

Forbidden: Llama, Qwen, Mistral, GPT/GPT-2, Gemma, Phi, DeepSeek, Falcon, any HF pretrained
LM, any downloaded Transformer checkpoint, any API LLM as the core model.

## EVIDENCE LABELS

| Label | Meaning |
|---|---|
| **PROPOSED** | A design decision. Ours to change. |
| **CALCULATED** | Exact arithmetic from a stated formula. Verified by script. |
| **ESTIMATE** | A projection resting on a stated assumption. Expect 2× error. |
| **NOT YET MEASURED** | Only an experiment can supply this. Blank until we run it. |
| **MEASURED** | Produced by a run that happened, traceable to a manifest. New in v1.1 — see §D, §E, §F, §K, §L, §M. |

### MEASURED RESULTS — run `run_a_001`, Tesla T4, seed 1337

| Quantity | Value |
|---|---:|
| Parameters | 1,331,968 |
| Step-0 loss | **8.3327** (ln(4096) = 8.3178, delta +0.0150) |
| Final training loss | 2.194813 |
| Final validation loss | **2.1407** (perplexity 8.5) |
| Best validation loss | 2.1426 |
| Wall clock | **2.6 min** at 320,128 tokens/s |
| Peak VRAM | **0.83 GB** of 15.4 GB |
| Steps / tokens | 6,100 / 49,971,200; final lr 3.0e-4 = exactly 10% of peak |
| Gradient norm at end | 0.61, stable throughout, no NaNs |
| Checkpoint | 16 MB, full training state |

Validation sits *below* training loss — one epoch-equivalent of tokens, nothing memorised.

**The recovery path was tested for real.** The Colab runtime recycled mid-session and wiped
`/content`. Everything was restored from Drive and `04_verify_model.py` re-run: bit-identical
numbers, tokenizer fingerprint `d6080bac0a9a` intact. The resumability engineering in §K is no
longer a precaution — it was used.

### Level 1 evaluation — fresh process, T4

| Check | Measured | Note |
|---|---:|---|
| Validation loss reproduction | **delta +0.000000** | A process that trained nothing reproduced 2.140691 to six decimal places |
| Perplexity | 8.5053 | 3.0884 bits per token |
| Next-token accuracy@1 | **51.10%** | against a 1-in-4,096 baseline |
| distinct-1 / distinct-2 | 0.3121 / 0.7016 | over 15 samples: 5 prompts × 3 temperatures |
| Immediate repetition rate | 0.0011 | word-level only — phrase-level looping is still visible and is *not* captured by this metric |
| Agent stub format compliance | 0% | expected; the model has had no reasoning training |

Generation at 6,100 steps produces grammatical, coherent TinyStories-register English —
*"The little girl was so happy that she had made a new friend."* This reproduces the Eldan & Li
result at 1.33M parameters. It is narrative English on a restricted vocabulary, **not** general
language competence, and should be described that way.

**The V0 stub's confidences separate by decoding temperature**, with no overlap between the greedy
agents (verifier 0.34–0.49, solver 0.31–0.44) and the sampled ones (critic 0.18–0.25, alternative
0.07–0.13). That is what `exp(mean token log-probability)` should do, and it is evidence the
number measures something real rather than being a placeholder.

Everything not listed above remains PROPOSED or ESTIMATE.

---

# A. PROJECT DEFINITION

Two artefacts, in strict order. The second cannot begin until the first exists as a checkpoint.

**Artefact 1 — MAX-LM.** A decoder-only Transformer we implement in PyTorch, initialise from
random weights, and train from scratch on a corpus we clean and tokenise with a BPE vocabulary
we fit ourselves. Small by design. Its purpose is not to be good at language in general — it is
to be a language model whose every weight we can account for.

**Artefact 2 — MAX-System.** A multi-agent framework in which *one* copy of MAX-LM plays five
roles selected by role-control tokens. Each role produces an answer, a confidence, and a hidden
representation. A small learned coordinator fuses those representations and emits a final answer
plus a machine-readable decision record.

**One-sentence definition.** MAX is a controlled study of whether role-specialised multi-agent
collaboration — with fusion at the level of learned representations rather than concatenated
text — improves reasoning accuracy, confidence calibration and explanation quality over a single
agent drawn from the same model, **at a matched inference budget**.

### Research questions

| # | Question | Answered by |
|---|---|---|
| RQ1 | Do multiple specialised agents on a from-scratch LM improve accuracy, reliability and explainability over a single agent? | M1 vs M3–M5 (§J), budget held equal |
| RQ2 | Does neural representation-level collaboration beat text-level aggregation or majority voting? | M3 vs M4 vs M5 (§J) |
| RQ3 | Are the coordinator's agent-contribution weights *faithful*? | Fusion weight vs leave-one-out flip rate (§I) |

### Scope boundary — what this is NOT

- Not a new Transformer architecture. Standard pre-LN decoder block.
- Not a general-purpose assistant. A 14M model on ~275M tokens produces fluent short narrative
  English and solves problems inside a closed domain we define. Nothing more.
- Not a benchmark-topping system. Comparison is always MAX-System vs MAX-single-agent.
- Not chain-of-thought exposure. Structured reasoning summaries, not a claim to have revealed
  internal deliberation.

**The constraint that shapes everything:** the capability ceiling is set by a 1M–30M parameter
model. This is the experimental condition, not a limitation to work around. It forces a closed
reasoning domain (§E), a selection-head coordinator (§H), and method-vs-method evaluation (§J).

---

# B. RESEARCH NOVELTY

### Tier 1 — established technology we use without claiming credit

| Component | Rests on |
|---|---|
| Decoder-only Transformer, MHA causal attention, pre-LN blocks | Vaswani et al. 2017; Radford et al. 2018/2019 |
| Byte-pair encoding | Sennrich et al. 2016 |
| Next-token pretraining, AdamW, cosine+warmup, weight tying | Standard practice |
| Small models from scratch producing coherent English | Eldan & Li 2023 (TinyStories) — the result that makes this feasible at all |
| Role prompting; solver/critic/verifier decomposition | Multi-agent debate & self-critique literature (Du et al.; Madaan et al.) |
| Self-consistency by sampling and voting | Wang et al. 2022 |
| Text-level aggregation of multiple outputs | Mixture-of-Agents style aggregation |
| Depth-generalisation for synthetic rule reasoning | Tafjord et al. 2021 (ProofWriter): train depth ≤3, test depth 5 |

### Tier 2 — our four contributions

**C1 · A contamination-free experimental setting.** Every published multi-agent reasoning result
is confounded by the same problem: nobody knows what was in the base model's pretraining corpus.
Because we generate both the pretraining corpus and the reasoning benchmark, we can state with
certainty that no test item was seen during pretraining. This is a real methodological advantage
that larger-scale work cannot easily buy — and it should be the headline framing of the project.

**C2 · Representation-level vs text-level fusion, ablated at matched budget.** The literature
compares multi-agent systems against single agents, conflating "more agents" with "more compute".
We hold generated-token count approximately constant across all five methods and isolate the
effect of *how* outputs are combined.

**C3 · Disagreement-triggered escalation with measured cost.** Adaptive verification is usually
described, rarely costed. We measure escalation rate, marginal accuracy per extra forward pass,
and the coverage–risk curve under abstention.

**C4 · Causally validated agent attribution.** We do not assume fusion weights explain the
decision. We remove each agent, record whether the answer flips, and correlate flip rate against
fusion weight. A weak correlation is reported as a weak correlation.

### Claims we will NOT make

Novel architecture · state of the art · human-level reasoning · "attention proves why" ·
"fully explainable AI". The defensible framing: *a controlled, fully reproducible study of
multi-agent collaboration mechanisms, at a scale where every variable — tokenizer, corpus,
weights, benchmark — is owned and inspectable.* That is strong enough. Overclaiming beyond it
is the fastest way to lose an examiner.

---

# C. COMPLETE SYSTEM ARCHITECTURE

### Layer 0 — foundation pipeline

```
raw corpus -> clean & dedup -> train BPE -> encode -> pack to ctx -> train/val split
           -> RANDOM INIT -> pretrain -> reasoning SFT -> checkpoint.pt
```

### Layer 1 — reasoning runtime

The mechanism that matters: **every agent is the same weights called with a different
role-control token**, and every call returns two things — generated text and a pooled hidden
state. Only the coordinator is trained on top.

```
                 +--------------------+
   <|solver|>    |                    |  -> a1 h1 c1 --+
   <|critic|>    |     MAX-LM         |  -> a2 h2 c2 --+
q -><|verifier|> |  one checkpoint    |  -> a3 h3 c3 --+--> COORDINATOR --> final answer
   <|alternative|>|  weights FROZEN   |  -> a4 h4 c4 --+   (~0.2M, trainable)  + decision record
                 |  13,784,064 params |
                 +--------------------+
                    4 forward passes
```

Only ~1.4% of the model's parameter count is trainable at this stage. Every baseline in §J
consumes the same four triples and differs only in how it combines them.

### Control flow with disagreement escalation

```
q -> solver + alternative (parallel) -> critic -> verifier -> AGREEMENT CHECK
                                                                |
                              agree ------------------------> coordinator -> answer + record
                                                                |
                              disagree -> extra solver sample (T=1.0) -> 2nd verifier pass
                                       -> coordinator over 6 triples -> answer, or abstain
```

Escalation costs two extra forward passes. §J measures whether those two passes buy more accuracy
than spending them on the non-escalated path — exactly the objection an examiner should raise.

---

# D. LLM ARCHITECTURE

### Block design (PROPOSED)

| Decision | Choice | Reason |
|---|---|---|
| Normalisation | pre-LN | Post-LN needs careful warmup at depth; pre-LN trains reliably. We cannot afford a diverging run. |
| Attention | MHA + causal mask | Implemented by us including the mask |
| Activation | GELU | Fewer moving parts than SwiGLU, which adds a 3rd FFN matrix and changes the arithmetic. SwiGLU is a V2 experiment. |
| FFN width | 4 × d_model | Conventional; keeps the count easy to reason about |
| Positional | learned (A) · RoPE (B,C) | Learned absolute is fastest to implement correctly in 48h. RoPE costs 0 params and extrapolates better — which matters for the S3 depth test. |
| Bias terms | off in attention, on in FFN | Stated so the parameter count is reproducible |
| LM head | tied to token embedding | Saves vocab×d_model — an amount equal to **39% of Config A's final count** |
| Init | N(0, 0.02); residual projections × 1/√(2L) | GPT-2 recipe; residual scaling keeps activation variance stable with depth |

### The three configurations (CALCULATED)

| Parameter | A · MAX-1M | B · MAX-14M | C · MAX-29M |
|---|---:|---:|---:|
| Role | Review 1 / dev | main project | stretch |
| Vocabulary | 4,096 | 8,192 | 8,192 |
| d_model | 128 | 384 | 512 |
| Layers | 4 | 6 | 8 |
| Heads | 4 | 6 | 8 |
| Head dim | 32 | 64 | 64 |
| d_ff | 512 | 1,536 | 2,048 |
| Context length | 128 | 256 | 512 |
| Positional | learned | RoPE | RoPE |
| **Total parameters** | **1,331,968** | **13,784,064** | **29,398,016** |
| Params + AdamW state (fp32) | 21 MB | 221 MB | 470 MB |
| Step-0 loss = ln(V) | 8.318 | 9.011 | 9.011 |
| **Step-0 loss, MEASURED** | **8.3327** | — | — |

### Parameter arithmetic for Config A — check it

```
token embedding      4096 x 128                    =   524,288
positional (learned)  128 x 128                    =    16,384

per transformer block
  attention W_q,W_k,W_v,W_o   4 x (128 x 128)      =    65,536
  layernorms          2 x (2 x 128)                =       512
  FFN        128*512 + 512  +  512*128 + 128       =   131,712
  block total                                      =   197,760
  x 4 layers                                       =   791,040

final layernorm       2 x 128                      =       256
LM head               tied to token embedding      =         0
====================================================================
TOTAL                                                1,331,968   (1.33 M)
```

General formula:
`total = V*d + P + L*(4d^2 + 4d + 2*d*d_ff + d_ff + d) + 2d`, where `P = ctx*d` for learned
positions and `0` for RoPE.

### THE RANDOM-INITIALISATION PROOF

A freshly initialised LM has no information about the next token, so it assigns roughly uniform
probability across the vocabulary. Cross-entropy under a uniform distribution over V tokens is
exactly **ln(V)**.

- Config A: **ln(4096) = 8.318**
- Config B/C: **ln(8192) = 9.011**

If our step-0 loss sits at that value, the weights cannot have carried prior knowledge — a
pretrained model would start far below it. One number, one forward pass, closes the entire
question. This is the centrepiece of the Review 1 demo.

Two supporting checks: a histogram of initial weights (mean ≈ 0, std ≈ 0.02), and a live `grep`
across the repository showing zero occurrences of `from_pretrained`, `AutoModel`, or any model
download call.

---

# E. DATASET PLAN

### The three corpora (licenses verified 25 Aug 2026, not recalled)

| Dataset | Job | License | Verified details |
|---|---|---|---|
| **A · TinyStories** (`roneneldan/TinyStories`) | Pretraining | CDLA-Sharing-1.0 | **2,141,709 rows** (2.12M train / 22k validation); 7.62 GB across repo files. Use `TinyStoriesV2-GPT4-train.txt`. Chosen because Eldan & Li showed models in exactly our size range produce coherent English on it. |
| **B · MAX-Reason** (ours) | **Primary** reasoning benchmark | ours | Generated; ~60k train / 6k dev / 4×4k test. Guarantees the contamination-free claim in §B. |
| **C · GSM8K** (`openai/gsm8k`) | **Secondary** external probe, never trained on | MIT | **7,473 train / 1,319 test**; configs `main` and `socratic`. Answers carry `<< >>` calculator annotations and a final answer after `####`. |
| **D · WikiText-103** (optional, Config B/C) | Register diversity | CC BY-SA 4.0 | 1,801,350 train / 3,760 val / 4,358 test; >100M tokens. Share-alike: attribute in the report. |

ProofWriter (Allen Institute) is worth reading for its depth-generalisation protocol, but its
license is **not stated** on the paper page — verify at allenai.org/data/proofwriter before use.

### Why GSM8K cannot be the primary benchmark

A 14M model trained on ~275M tokens of children's stories will score at or near zero on
grade-school math word problems. If that is the primary metric, every method in §J scores zero,
the differences are noise, and RQ1 becomes unanswerable — the project fails not because the idea
is wrong but because the instrument cannot resolve it. GSM8K stays as an external probe with low
absolute scores expected and stated in advance.

### MAX-Reason — five task families

| Family | Form | What depth means |
|---|---|---|
| Arithmetic chain | Quantities on entities, then add/subtract/multiply chain | Number of operations |
| Rule entailment | Facts + `if…then` rules, closed-world; does a conclusion hold? | Rule applications in the proof |
| Transitive comparison | Pairwise relations over entities; ask an ordering fact | Chain length |
| Multi-hop lookup | Small relational graph (person→team→city→region) | Edges traversed |
| Set & counting | Membership and cardinality with filters | Filters composed |

All have programmatically verifiable ground truth — which is what makes the Verifier agent a real
component rather than a decorative one.

### Four defences against template memorisation

1. **Surface variation.** 8–12 phrasings per family; randomised sentence order; randomised
   distractors; entity name pools **disjoint between train and test**; different numeric ranges.
2. **Structural variation.** Depth sampled not fixed; irrelevant facts inserted so "use every
   number" fails; answer position varies. Distractors are the cheapest defence and catch the
   largest class of shortcut.
3. **Compositional variation.** Train sees rule combinations {AB, BC, CD}; the compositional test
   set contains only {AC, BD} — every rule familiar, every combination new.
4. **Generation-time holdout.** Entity pools and template IDs partitioned *before* generation,
   not filtered afterwards. Post-hoc dedup always leaks. Plus normalised-form hashing as an audit.

### Which generalisation test is scientifically appropriate

| Split | Shift | Role | Assessment |
|---|---|---|---|
| **S1 · IID** | Same generator, held-out instances | Diagnostic | Measures whether the model learned the task. **Not** a generalisation result. Runs first as a gate — if S1 is low, nothing downstream is interpretable. |
| **S2 · Compositional** | Unseen combinations of seen rules; depth and surface constant | **PRIMARY** | **The appropriate primary test.** Holds sequence length, depth and vocabulary fixed and varies only composition, so a difference is attributable to compositional generalisation and nothing else. Also where a Critic and Verifier plausibly help — the right ground for RQ1. |
| **S3 · Depth extrapolation** | Train d ≤ 2, test d = 3–4 | Secondary | Interesting and well precedented, but **confounded by sequence length** — deeper problems are longer, so failure may mean "cannot compose" or merely "never saw sequences this long". Mitigation: pad shallow training items with distractors so token-length distributions match. Report with the confound stated. |
| **S4 · Surface shift** | Unseen entity names and templates, depth unchanged | Secondary | Controls for lexical memorisation. If S1 is high and S4 collapses, the model matched strings rather than reasoned. |

Run S1 as a gate, S2 as the headline, S3 and S4 as stress tests.

### Preprocessing pipeline (PROPOSED)

```
download -> strip artefacts -> unicode NFKC -> length filter -> MinHash dedup
         -> seeded shuffle -> split 98/1/1 -> fit BPE ON TRAIN ONLY -> encode
         -> pack into ctx blocks -> uint16 memmap
```

- **Split before fitting the tokenizer.** Fitting BPE on the full corpus leaks validation
  statistics into the vocabulary. Small effect, free to avoid, and an examiner may ask.
- **Deduplicate before splitting**, so a near-duplicate cannot straddle train and validation.
- **Pack, do not pad.** Concatenate documents separated by `<|eos|>` and slice into fixed
  ctx-length blocks. Padding a 128-token context wastes a large fraction of a small budget.
- **uint16 memmap** — both 4,096 and 8,192 vocabularies fit; halves disk, enables zero-copy
  random access on Colab's slow disk.
- **Record:** source URL, revision hash, row counts before/after each filter, seed, token count.

### How much data is required (ESTIMATE)

| Config | Params | Target tokens (~20/param) | ≈ raw text | Source |
|---|---:|---:|---:|---|
| A · MAX-1M | 1,331,968 | 27 M | 0.11 GB | TinyStories subset (~150k stories) |
| B · MAX-14M | 13,784,064 | 276 M | 1.10 GB | TinyStories (+ WikiText-103) |
| C · MAX-29M | 29,398,016 | 588 M | 2.35 GB | TinyStories + WikiText-103 |

Both the 20-tokens-per-parameter ratio and ~4 chars/token are heuristics. The real token count
is only known after tokenisation.

### What the corpus actually came out as (MEASURED)

| Stage | Result | Note |
|---|---:|---|
| Documents loaded | 230,000 | TinyStories, streamed slice |
| After cleaning and dedup | 226,654 | 1.5% dropped; deduplicated *before* splitting |
| Characters | 204,352,155 | |
| BPE merges learned | 3,966 | alphabet 98 characters, vocab 4,096 |
| Round-trip exact | 1000 / 1000 | 100% — encode/decode lossless on held-out text |
| UNK rate | 0.0000% | no validation character fell outside the alphabet |
| Compression | 3.88 chars/token | close to the 4.0 heuristic used for sizing above |
| Tokenizer fingerprint | `d6080bac0a9a` | in every checkpoint; loader refuses on mismatch |
| Train / val / test tokens | 51,601,430 / 527,038 / 522,122 | 98/1/1, uint16 memmap |

The 4-chars-per-token heuristic held up: 3.88 measured, so token budgets for Config B and C can
be trusted to roughly ±5%.

For Review 1 we deliberately train Config A on ~**50M tokens**
— above the compute-optimal target — because the extra tokens cost minutes and produce a visibly
better loss curve and more coherent samples for the demo.

---

# F. TRAINING PLAN

| Stage | Objective | Trainable | Produces |
|---|---|---|---|
| **1 · Pretraining** | Next-token cross-entropy | All model weights | Base checkpoint. **This is the artefact Review 1 must contain.** |
| **2 · Reasoning SFT** | Next-token, loss masked to completion | All model weights | One checkpoint responding to all five role tokens. Role specialisation happens here — not by training five models. |
| **3 · Coordinator** | CE over candidates + calibration term | Coordinator only; MAX-LM frozen | The fusion module of §H. ~0.2M params, minutes of training. |

### Special tokens — reserve at tokenizer-fitting time

Role tokens must exist in the vocabulary **before pretraining**. Adding them at Stage 2 means
introducing randomly initialised embedding rows into an otherwise trained model — trains poorly,
avoidable. Reserve indices 0–31 at BPE fitting time:

```
0  <|pad|>      4  <|solver|>        10  <|steps|>     # structured reasoning summary
1  <|bos|>      5  <|critic|>        11  <|answer|>
2  <|eos|>      6  <|verifier|>      12  <|conf|>
3  <|unk|>      7  <|alternative|>   13  <|issue|>
                8  <|coordinator|>   14  <|verdict|>
                9  <|question|>      15  <|evidence|>
                                  16-31  reserved
```

16 spare slots cost <0.2% of Config A and mean a later design change never forces a tokenizer
rebuild — which would invalidate every checkpoint trained before it.

### Stage 1 hyperparameters (PROPOSED)

| Setting | Config A (Review 1) | Config B (main) | Note |
|---|---:|---:|---|
| Optimizer | AdamW | AdamW | β=(0.9,0.95), ε=1e-8 |
| Weight decay | 0.1 | 0.1 | Excluded from LayerNorm and embeddings |
| Peak LR **(MEASURED)** | **3e-3** | 6e-4 | Probe over 200 steps: **5e-4 → val 4.6537 · 1e-3 → val 4.2026 · 3e-3 → val 3.8820**. 3e-3 won by 0.32 nats with gradient norms *falling* as the rate rose; held stable across the full run |
| Schedule | cosine → 10% | cosine → 10% | Linear warmup over first 5% of steps |
| Grad clipping | 1.0 | 1.0 | Global norm |
| Batch (sequences) | 64 | 48 | Raise until VRAM or throughput stops improving |
| Tokens per step | 8,192 | 12,288 | batch × context |
| Grad accumulation | 1 | 2–4 | Effective batch without more memory |
| Target tokens | ≈ 50 M | ≈ 275 M | |
| Steps | ≈ 6,100 | ≈ 22,400 | tokens ÷ tokens-per-step |
| Precision | fp16 + GradScaler | fp16 + GradScaler | **T4 is Turing — bf16 unavailable.** GradScaler mandatory. |
| Validate every | 250 steps | 500 steps | Fixed held-out batches, same every time |
| Checkpoint every | 500 steps | 1,000 steps | To Drive. Non-negotiable. |
| Seed | 1337 | 1337 | Recorded in the checkpoint with the config hash |

### Stage 2 — reasoning output schema

Structured summaries, not exposed deliberation. Fixed field layout per role makes outputs
parseable and makes format compliance measurable.

```
<|solver|> <|question|> Maya has 4 boxes. Each box holds 6 pens. She gives away 7 pens.
                        How many pens does she have left?
<|steps|>   4 boxes x 6 pens = 24 pens | 24 - 7 = 17
<|answer|>  17
<|conf|>    0.86 <|eos|>

<|critic|>  <|question|> ... <|answer|> 17
<|issue|>   none | multiplication and subtraction both check out
<|conf|>    0.79 <|eos|>

<|verifier|> <|question|> ... <|answer|> 17
<|evidence|> 17 + 7 = 24 ; 24 / 6 = 4 boxes — consistent with the premise
<|verdict|>  correct
<|conf|>     0.91 <|eos|>
```

- **Loss masked to the completion only** — the model is not rewarded for predicting the question.
- **All five role datasets mixed into one SFT run.** One optimiser, one checkpoint, five
  behaviours. This is what makes the agent design affordable.
- **Confidence is a generated token sequence**, bucketed to two decimals. A self-report, poorly
  calibrated on its own; §J measures how poorly and the coordinator learns to correct it.
- Hyperparameters (PROPOSED): LR 2e-4 cosine, 3 epochs over ~60k examples, batch 32, warmup 5%.

### Stage 3 — coordinator

MAX-LM frozen, used only to produce the four (answer, hidden, confidence) triples. Because it is
frozen, **triples for the whole training set are generated once and cached** — after which
coordinator training is minutes on cached tensors rather than hours of generation. This caching
decision is what makes the §J ablation matrix affordable.

LR 1e-3, Adam, batch 256, up to 60 epochs, early stopping on dev accuracy, patience 8. (PROPOSED)

---

# G. MULTI-AGENT DESIGN

| Agent | Sees | Emits | Decoding |
|---|---|---|---|
| **Solver** | question | steps · answer · confidence | greedy — the reference attempt |
| **Alternative** | question (different role token) | steps · answer · confidence | T=0.9, top-p 0.92, different seed — deliberately decorrelated |
| **Critic** | question + both candidates | issues · confidence | T=0.7 — a deterministic critic repeats one objection |
| **Verifier** | question + candidate + critic's issues | evidence · verdict · confidence | greedy — a verifier that varies is not a verifier |
| **Coordinator** | four hidden states, confidences, candidate answers | final answer · calibrated probability · fusion weights | not generation — a trained selection head (§H) |

Decoding temperature is doing real work: it is the cheapest source of the output diversity the
whole multi-agent premise depends on.

### How to instantiate the agents — the four options

| Option | Mechanism | Extra cost | Assessment |
|---|---|---:|---|
| **A** | One model, role prompts / control tokens | 0 params | **Chosen for V4.** With role tokens trained into the SFT mix (not merely prompted at inference), specialisation is genuinely learned — the model has seen thousands of examples of what follows `<\|critic\|>`. Costs nothing beyond the SFT run we are doing anyway. |
| **B** | One model, different decoding strategies | 0 params | Diversity but no specialisation — a high-temperature Solver is not a Critic. Used *with* A (the temperature column above), not instead of it. |
| **C** | Shared frozen trunk + lightweight per-role heads/adapters | ~0.3M/role | **V5 extension, not a V4 dependency.** Genuinely different parameters per role at a fraction of the cost of separate models. Deferred because it multiplies what can break before the core comparison exists. |
| **D** | Four independently specialised full copies | 4 × 13.78M | **Rejected.** 4× training compute and storage on hardware where one main-model run already costs hours. The gain over C does not come close to justifying it. |

### Interaction protocol

1. **Round 1 — independent attempts.** Solver and Alternative run in parallel. Independence
   matters: if the Alternative sees the Solver's answer it anchors, and the two candidates stop
   being two candidates.
2. **Round 2 — criticism.** Critic sees question + both candidates, lists concrete issues.
3. **Round 3 — verification.** Verifier sees question + leading candidate + critic's issues,
   returns verdict with evidence.
4. **Round 4 — agreement check.** Answers normalised and compared. Agreement = Solver and
   Alternative match **and** Verifier returns `correct`.
5. **Round 5a — agreement path.** Coordinator fuses four triples, emits final answer.
6. **Round 5b — escalation path.** One extra Solver sample at T=1.0 + second Verifier pass;
   coordinator fuses six triples; may abstain if no candidate clears the threshold.

### INFERENCE BUDGET ACCOUNTING

Agreement path = **4 generations**. Escalation path = **6**. Self-consistency baseline fixed at
**k = 5** so it sits between them. Every baseline is reported alongside its generation count.
Without this, an examiner can dismiss any multi-agent gain as the predictable effect of sampling
more — and they would be right. **Accuracy is always reported next to cost, never alone.**

---

# H. COLLABORATIVE NEURAL REASONING

Concatenating agent text and asking the model to summarise it is text-level aggregation — the M4
baseline, not the contribution. The contribution is combining the agents' *internal
representations*, so information that never reached the generated tokens can still influence the
decision.

### Step 1 — building an agent token

For each agent i, its forward pass already computes final-layer hidden states over the tokens it
generated. Capture two pooled views:

```
h_i = [ mean over answer-span tokens ; state at <|conf|> ]           in R^(2*d_model)
```

The mean carries the content of the whole answer; the final-position state carries what the model
had accumulated at the moment it committed to a confidence. Alongside, four scalar features that
text-level methods throw away:

```
phi_i = [ self-reported confidence,
          mean token log-probability of the answer span,
          entropy of the next-token distribution at the answer position,
          fraction of other agents whose answer matches this one ]    in R^4
```

Plus a learned role embedding `r_i in R^32`. Project into a shared coordinator space, `d_c = 128`:

```
z_i = W_in . [ h_i ; r_i ; psi(phi_i) ] + b_in                        in R^128
```

### Step 2 — fusion, in two escalating variants

**V5a · Gated fusion (build first, ~16k params)**

```
e_i = w^T tanh(W_g z_i)
g   = softmax(e)              over agents
u   = sum_i g_i * z_i
```

Learns a per-instance weighting over agents. If this does not beat majority voting, cross-agent
attention almost certainly will not either — and we learned that cheaply.

**V5b · Cross-agent attention (if justified, ~0.2M params)**

```
Z    = [ z_0 ; z_1 ... z_k ]      z_0 = learned coordinator query
Z'   = Z  + MHA(LN(Z))            4 heads
Z''  = Z' + FFN(LN(Z'))
u    = Z''_0
```

Agents attend to each other, so the Verifier's representation can modulate how much the Solver's
is trusted — a genuine interaction that gating cannot express.

### Step 3 — decision head (answer selection, not generation)

```
e_j = mean of z_i over agents proposing candidate j        # candidate embedding
s_j = v^T tanh(W_s [ u ; e_j ])                            # score per candidate
p   = softmax([ s_1 ... s_m , s_abstain ])
```

**Why selection and not generation.** A generative coordinator on a 14M model would produce text
of roughly the quality of the agents it is meant to arbitrate. Selection is trainable from ~60k
examples, produces a probability that can be calibrated and scored with ECE and Brier, gives
abstention for free, and makes the output verifiable against ground truth. It is the formulation
that fits the scale.

### Training objective

```
L = CE(p, gold) + lambda_cal * Brier(max p, correct) + lambda_agr * BCE(a_hat, agreement)
```

lambda_cal = lambda_agr = 0.1, tuned on dev (PROPOSED). The calibration term stops the coordinator
becoming confidently wrong — the failure mode that matters most for an explainability system. The
auxiliary agreement head predicts whether the agents agreed: a cheap regulariser forcing the fused
representation to encode inter-agent structure rather than collapsing onto the Solver.

### Fusion ablations

- Uniform mean over z_i (no learning) → gated → cross-attention: does each step of added mechanism
  buy accuracy?
- With/without the scalar features phi — is log-probability and entropy doing any work?
- With/without role embeddings — does the coordinator need to know *who* said something?
- With/without the auxiliary agreement head.

---

# I. EXPLAINABILITY

The step that makes this research rather than presentation is **measuring whether the explanation
predicts the system's actual behaviour**. That measurement is cheap here, because we can re-run
the system with agents removed.

### Two levels, kept strictly apart

- **System-level (PRIMARY).** Which agents ran, what each concluded, confidence, where they
  disagreed, what the Verifier found, how the coordinator weighted them. Every one of these is a
  fact about execution — fully supported, no interpretation required. This carries the claim.
- **Model-level (SECONDARY, hedged).** Fusion weights, per-agent sequence log-probabilities, and
  gradient×input saliency *on the coordinator's input vector* — not on the LM's attention.
  Reported as correlational signals, never as proof of why a decision was made.

### The decision record

```json
{
  "question":        "...",
  "final_answer":    "17",
  "confidence":      0.88,
  "agents_run":      ["solver","alternative","critic","verifier"],
  "conclusions":     {"solver":"17","alternative":"17","verifier":"correct"},
  "supporting":      ["solver","alternative","verifier"],
  "conflicting":     [],
  "verification":    {"verdict":"correct","evidence":"17 + 7 = 24; 24 / 6 = 4"},
  "disagreement":    false,
  "escalated":       false,
  "fusion_weights":  {"solver":0.41,"alternative":0.29,"critic":0.08,"verifier":0.22},
  "leave_one_out":   {"solver":"flip","alternative":"same","critic":"same","verifier":"same"},
  "uncertainty":     "single unchallenged candidate; verifier evidence consistent",
  "decision_path":   ["q","solver+alt","critic","verifier","agree","fuse","answer"],
  "generations":     4,
  "seed":            1337
}
```

`leave_one_out` is what turns the record from a description into a claim that can be checked.

### The decision graph

NetworkX for layout, Plotly for the interactive view. Nodes = agent invocations + fusion +
verification + final answer. **Edges labelled with what was passed**, not merely that something
was. Node colour = agreement status; node size = fusion weight; dashed edge = escalation path.
Clicking a node reveals that agent's full output.

```
question -> solver --candidate--> critic --issues--> verifier --verdict--> FUSION
         --weights--> coordinator -> answer + record
```

### Measuring the explanations (NOT YET MEASURED)

| Property | How measured | What a bad result means |
|---|---|---|
| **Faithfulness** | Spearman correlation between each agent's fusion weight and its leave-one-out decision-flip rate | Weak correlation means the weights do not describe what drove decisions. Report plainly; fall back to leave-one-out as the attribution method — it is causal, it just costs more passes. |
| **Completeness** | % of records with every field populated and parseable | Parsing failures in the output schema; also catches Stage-2 format drift |
| **Consistency** | Same input + seed re-run; explanations must match exactly | Non-determinism somewhere — a bug, not a finding |
| **Human utility** | Pilot, 8–12 classmates. Half see answer alone, half answer+explanation. Measure (a) *simulatability* — can they predict correctness? (b) *appropriate reliance* — do they accept correct and reject wrong at higher rates? | If explanations raise acceptance of *wrong* answers, the layer is actively harmful — a genuinely valuable finding, and worth designing the study to detect. |

Report the human pilot with n stated and framed as a pilot. Ten classmates supports "suggestive
of" and nothing stronger.

### Language discipline for the report

Write *"the coordinator assigned the Verifier a weight of 0.22, and removing the Verifier changed
the answer in 14% of cases"* — not *"the model paid attention to the Verifier because it
recognised the verification was important."* The first is a measurement. The second is a story
about a measurement, and it is the sentence that will be challenged in the viva.

---

# J. EVALUATION

**The separation that answers the examiner's hardest question.** "Did the improvement come from
the language model or from the multi-agent architecture?" is answerable only if the two are
evaluated separately. Level 1 evaluates MAX-LM alone. Level 2 holds the checkpoint **frozen and
identical** across all five methods. Level 2 results are never reported without naming which
Level 1 checkpoint produced them.

### Level 1 — the language model alone (NOT YET MEASURED)

| Metric | Definition and purpose |
|---|---|
| Step-0 loss | Must equal ln(V) ± 0.05. The random-initialisation proof. |
| Training / validation loss | Every step / every 250 steps. Plotted together — a widening gap is the overfitting signal. |
| Validation perplexity | exp(val loss). Reported with bits-per-token, which is comparable across vocabulary sizes — so Config A and B can be compared honestly. |
| Next-token accuracy@1 | Fraction of positions where argmax is correct. More intuitive to a non-specialist audience than perplexity. |
| Generation quality | Fixed prompts, fixed seeds, T ∈ {0.2, 0.7, 1.0}. Plus distinct-1/distinct-2 and repetition rate to quantify degeneracy, and a short human rubric for grammaticality and coherence. |
| Throughput | Tokens/sec, wall-clock, peak VRAM. Turns §K estimates into measurements. |
| *— after Stage 2 —* | |
| Format compliance | Fraction of outputs parsing into the role schema. Below ~90% the multi-agent evaluation measures a parser, not a model. |
| Single-agent reasoning accuracy | On S1/S2/S3/S4 separately. S1 gates; S2 is the headline. |
| GSM8K exact-match | Secondary probe. Low scores expected and stated in advance as a hypothesis, not discovered as a disappointment. |

### Level 2 — the reasoning system, checkpoint held fixed

| Method | Aggregation | Generations | Isolates |
|---|---|---:|---|
| **M0** | Majority-class constant | 0 | The floor. Anything not beating this is broken. |
| **M1** | Single Solver, greedy | 1 | The baseline the project exists to beat. |
| **M2** | Self-consistency, k=5, vote | 5 | **The critical control.** Separates "gains from more compute" from "gains from role specialisation". If M3–M5 do not beat M2, the architecture is an expensive way to sample more. |
| **M3** | Four role agents, majority vote | 4 | Value of role specialisation with no learned aggregation |
| **M4** | Four role agents, text-level aggregation | 4 + 1 | Value of reading the agents' text. The extra generation is the aggregation pass — and must be counted. |
| **M5** | Four role agents, neural fusion (§H) | 4 | The contribution. Compared against M4 at a *lower* generation count, which strengthens the claim if it wins. |

### Level 2 metrics (NOT YET MEASURED)

- **Accuracy** — exact match on S1–S4 and GSM8K, per method, per seed.
- **Calibration** — ECE at 15 bins, Brier score, reliability diagram per method. AUROC of
  confidence against correctness measures ranking quality independent of calibration.
- **Selective prediction** — coverage–risk curve under abstention. A system that knows when to
  decline is more useful than one point of accuracy.
- **Disagreement detection** — precision/recall/F1 of the agreement check against whether the
  final answer was actually wrong.
- **Escalation economics** — escalation rate, accuracy gained per extra generation, compared
  against spending the same generations on M2.
- **Agent diversity** — pairwise answer-disagreement rate. A first-class metric, not a diagnostic.

### Statistical protocol

- **Three seeds** per method; mean and standard deviation, never a single run.
- **Bootstrap 95% CIs** over test items, 10,000 resamples.
- **McNemar's exact test** for paired comparisons (M1 vs M5, M2 vs M5, M4 vs M5) — the methods
  are evaluated on the same items and an unpaired test discards that information.
- **Holm correction** across the comparison family. With six methods it is easy to find a
  significant difference by accident.
- **Pre-register the hypotheses** in the repo before running Level 2, with a timestamped commit.
  A null result against a pre-registered hypothesis is a finding; the same null reported after the
  fact looks like a failure.

### Ablation matrix (NOT YET MEASURED)

| Ablation | Isolates | Why it earns its compute |
|---|---|---|
| Single agent vs multiple agents | RQ1 | Headline comparison. Only interpretable against M2. |
| Without vs with Critic | Critic's marginal value | The Critic produces no answer of its own — tests whether criticism alone changes outcomes |
| Without vs with Verifier | Verifier's marginal value | Expected largest single contributor, since MAX-Reason answers are checkable |
| Without vs with Alternative | Value of a second independent attempt | Distinguishes role diversity from mere resampling |
| Text vs neural aggregation | RQ2 | The contribution's core test |
| Uniform / gated / cross-attention fusion | Fusion mechanism | Does added mechanism pay for itself, or is a mean enough? |
| Without vs with confidence features | Value of phi | Cheap: retrain coordinator on cached triples with phi zeroed |
| Without vs with escalation | Adaptive verification | Reported as accuracy per unit cost, not accuracy alone |
| Without vs with explainability | Human utility | The §I human pilot — the only ablation with people in it |

Because agent outputs are cached once and reused, most of these cost minutes of coordinator
retraining rather than hours of generation.

---

# K. HARDWARE

| Config | Params | Optim state | Batch | Activations | Peak VRAM | Tokens | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A · MAX-1M** *(MEASURED)* | 1.33 M | 21 MB | 64 × 128 | — | **0.83 GB** | 49.97 M | **2.6 min** |
| **B · MAX-14M** | 13.78 M | 221 MB | 48 × 256 | ~1.1 GB | ~2.5 GB | 275 M | 2–6 h |
| **C · MAX-29M** | 29.40 M | 470 MB | 24 × 512 | ~2.8 GB | ~4.5 GB | 588 M | 6–20 h |

**Config A's peak-VRAM and time columns are now measurements**, taken on a Tesla T4 over 6,100
steps: **0.83 GB** and **2.6 minutes at 320,128 tokens/s**. The original 15–45 minute estimate
assumed a 1.33M-parameter model would be dominated by kernel-launch overhead; at batch 64 ×
context 128 the GPU stayed well fed, and the estimate was pessimistic by roughly 6×.

B and C remain ESTIMATES from 6·N·D FLOPs at an assumed 1.5–5 effective TFLOP/s — and given how
far Config A's estimate missed, expect them to be pessimistic too. **Re-derive them from Config
A's measured 320,128 tokens/s** before planning around them.

### THE INSIGHT THAT SHOULD SHAPE THE ENGINEERING

All three configurations fit in 16 GB with room to spare — Config A used **0.83 GB of 15.4 GB**,
about 5% of the card. **Nothing in this project is memory-bound.** What actually costs the project
is the runtime disappearing mid-run — which happened, and cost nothing, because the checkpoints
carried full state. Engineering effort belongs in checkpointing
and resumption, not memory optimisation. Gradient checkpointing, offloading and quantisation are
all irrelevant here and should not be built.

### T4-specific constraints

- **No bf16.** T4 is Turing; bf16 needs Ampere+. Use fp16 + `torch.cuda.amp.GradScaler`.
  Attempting bf16 fails at runtime or silently falls back to fp32 at a third of the speed.
- **Free Colab** caps around 12 h and can preempt earlier without warning. Kaggle offers ~30
  GPU-hours/week in ~9-hour sessions on P100 or dual T4.
- **Do not use `torch.compile`** for V1. Compile time is a significant fraction of a short run and
  the failure modes are opaque. Revisit for Config C, where it amortises.
- **Colab disk is slow.** Tokenise once, save a uint16 memmap to Drive, memory-map it.

### Mandatory resumability

Every checkpoint must contain: model state, optimiser state, scheduler state, RNG states
(Python/NumPy/Torch/CUDA), dataloader position, step counter, full config, config hash, tokenizer
hash. A checkpoint that only stores weights forces a restart from step zero — which, at hour
eleven, ends the schedule.

The tokenizer hash matters more than it looks: loading a checkpoint against a re-fitted tokenizer
produces a model that generates confident nonsense with no error message. **The loader should
refuse on mismatch.**

### CPU FALLBACK — the Review 1 insurance policy

Config A trains on CPU. Slowly — expect single-digit hours rather than minutes (ESTIMATE) — but it
trains, and produces a real loss curve and real generations. If Colab is unavailable on the night
before Review 1, start a reduced CPU run (fewer tokens, same everything else) and the
demonstration still happens. **This is the single most important reason Config A is deliberately
tiny.**

---

# L. DEVELOPMENT ROADMAP

| Version | Deliverable | Exit criterion | Effort | Risk if skipped |
|---|---|---|---:|---|
| **V0** | Repo skeleton, config system, logging, seeding, tests | `pytest` passes tokenizer round-trip and shape tests | 3 h | Every later version inherits the mess |
| **V1** | **MAX-1M: tokenizer, model, random init, trained checkpoint, generation** | **`ckpt_A_final.pt` exists, loads, generates; loss curve plotted** | 2 days | **Review 1 fails. The only hard gate before the review.** |
| **V2** | MAX-14M with RoPE, full Level 1 evaluation | Validation perplexity recorded; samples archived | 1 week | Reasoning training on a 1.3M model will likely not clear the S1 gate |
| **V3** | MAX-Reason generator; reasoning SFT; role tokens live | Format compliance > 90%; S1 clears the gate | 1.5 weeks | No benchmark means no Level 2 evaluation at all |
| **V4** | Five agents, protocol, disagreement detection; M1–M4 | M1–M4 measured on S1–S4 with CIs | 1.5 weeks | The contribution has nothing to be compared against |
| **V5** | Neural collaborative fusion — gated, then cross-attention | M5 measured; M4 vs M5 comparison exists | 1.5 weeks | RQ2 unanswered |
| **V6** | Calibration, decision record, decision graph, faithfulness test | ECE + reliability diagrams; faithfulness correlation computed | 1 week | The "explainable" in the title is unearned |
| **V7** | FastAPI backend, Streamlit UI, interactive graph, final report | End-to-end demo runs cold on a fresh machine | 1 week | Nothing to demonstrate at the final review |

Effort figures are ESTIMATES for one student working part-time alongside coursework.

### Repository structure (PROPOSED)

```
max/
├── configs/          # YAML: model_a.yaml, model_b.yaml, train_*.yaml, agents.yaml
├── data/
│   ├── raw/ interim/ processed/     # .gitignored; manifests are committed
│   └── synth/        # MAX-Reason generator + split definitions
├── tokenizer/        # BPE training, encode/decode, special-token registry
├── model/            # attention.py  block.py  transformer.py  init.py
├── training/         # trainer.py  scheduler.py  checkpoint.py  metrics.py
├── reasoning/        # SFT data builders, role schema, output parser
├── agents/           # solver critic verifier alternative coordinator, protocol.py
├── collaboration/    # fusion_gated.py  fusion_attention.py  heads.py
├── explainability/   # decision_record.py  graph.py  attribution.py
├── evaluation/       # level1/  level2/  calibration.py  stats.py  ablations.py
├── inference/        # generate.py  sampling.py
├── api/ frontend/    # FastAPI · Streamlit
├── experiments/      # one dir per run: config + manifest + logs + metrics.csv
├── checkpoints/      # .gitignored
├── notebooks/ tests/
└── requirements.txt  README.md  .gitignore
```

Two rules to enforce from day one: **no file over ~300 lines**, and **every run writes
`experiments/<run_id>/manifest.json`** containing seed, config, config hash, dataset revision,
tokenizer hash, git commit, hardware, and final metrics. The manifest is what makes a result
reproducible three months later.

---

# M. REVIEW 1 PLAN — Thursday 27 August

## THE SINGLE RULE FOR THE NEXT 48 HOURS

The most important Review 1 artefact is a checkpoint file containing weights our own training loop
produced. Slides can be assembled in three hours; a trained model cannot be assembled in the last
three hours. **Documentation is written only after `ckpt_A_final.pt` exists.** If the schedule
slips, cut slides — never cut the training run.

## 1 · The 48-hour schedule

| When | Block | Output | Gate before moving on |
|---|---|---|---|
| **Tue eve, 3 h** | V0 · skeleton + data + tokenizer | Repo structure, YAML config loader, seeded RNG, CSV logger. TinyStories subset downloaded, cleaned, deduped, split. BPE fitted to vocab 4,096 with 32 special tokens reserved. | Tokenizer round-trips 1,000 held-out strings exactly. Corpus token count printed. |
| **Tue late, 2 h** | Model + init verification | Attention, block, transformer, weight tying, GPT-2 init. Parameter counter. Weight histogram. | `1,331,968` printed, matching §D exactly. Init mean ≈ 0, std ≈ 0.02. **Step-0 loss = 8.318 ± 0.05.** |
| **Wed AM, 3 h** | Training loop | AdamW, cosine+warmup, clipping, AMP, validation pass, checkpoint save/load, resume. | **Overfit-one-batch test:** loss on a single fixed batch of 8 sequences drives below 0.1. If not, the loop is wrong and no amount of training fixes it. |
| **Wed mid, 1 h** | LR probe | Three 200-step runs at 5e-4, 1e-3, 3e-3 | Pick the lowest stable loss. Do not skip — a diverged main run costs more than this hour. |
| ✓ **Wed PM, 2 h** | **THE RUN** — *2.6 min GPU, MEASURED* | 6,100 steps, 49,971,200 tokens. `metrics.csv` opens at step 0 = 8.33214 and closes at 6100 = 2.194813. Final lr 3.0e-4 = exactly 10% of peak. | **Passed.** Stable throughout, gradient norm 0.61 at the end, no NaNs. |
| **Wed PM, 2 h** | Evaluation + artefacts | Final val loss and perplexity, loss-curve PNG, generation script, samples at T=0.2/0.7/1.0, results table. | Checkpoint reloads in a **fresh process** and generates. Proves the artefact is portable, not a live-session accident. |
| **Wed eve, 2 h** | V0 multi-agent stub | Three roles by prompt prefix on the **real** checkpoint, majority-vote coordinator, printed decision record. | Runs end to end on one question. Output will be weak — that is fine and expected. It demonstrates the architectural link, nothing more. |
| **Wed night, 3 h** | Slides + evidence pack | Deck, architecture diagrams, results table, screenshots, logs. | Every number on every slide traceable to a file in `experiments/`. |
| **Thu AM** | Rehearsal + buffer | Two full run-throughs. Screen recording of the demo as backup. | Demo completes in under 8 minutes from a cold terminal. |

~18 working hours, with the training run itself occupying under an hour. The schedule is dominated
by the things that make the run trustworthy — which is the correct allocation.

## 2 · Version 1 — exact specification

**Model**
```
vocab_size      4096      (BPE, ours)
d_model          128
n_layers           4
n_heads            4      head_dim 32
d_ff             512
context_len      128
positional     learned absolute
norm           pre-LN
activation     GELU
lm_head        tied to embedding
init           N(0, 0.02); residual x 1/sqrt(8)
-----------------------------------------------
parameters   1,331,968
step-0 loss      8.318  = ln(4096)
```

**Training**
```
corpus     TinyStoriesV2-GPT4 subset
license    CDLA-Sharing-1.0
tokens     ~50 M
split      98 / 1 / 1
optimizer  AdamW beta(0.9,0.95) wd 0.1
lr         1e-3 -> 1e-4 cosine
warmup     5% of steps
clip       1.0 global norm
batch      64 x 128 = 8,192 tok/step
steps      ~6,100
precision  fp16 + GradScaler
val every  250 steps
ckpt every 500 steps
seed       1337
```

## 3 · Review 1 acceptance criteria

**All sixteen are done.** Every box was ticked by a run that produced an artefact, never on the
strength of code existing. The evidence pack is complete and backed up to Drive. The three marked
★ carry the from-scratch claim.

- [x] Custom BPE tokenizer trained on our corpus; encode/decode round-trips exactly
- [x] Custom decoder-only Transformer implemented by us, component by component
- [x] ★ No pretrained weights anywhere — verified by live repository search
- [x] ★ Model initialised from random weights — verified by init statistics
- [x] ★ Step-0 loss equals ln(vocab_size) = 8.318
- [x] Model trains — training loss decreases meaningfully from 8.318
- [x] Validation loss measured on a held-out split
- [x] Training and validation loss plotted together
- [x] Checkpoint saved to disk
- [x] Checkpoint reloads in a fresh process and reproduces the same validation loss
- [x] Model generates text from the reloaded checkpoint
- [x] Parameter count documented and matching the config exactly
- [x] Training configuration documented in a run manifest
- [x] Dataset source, revision and license documented
- [x] At least one quantitative result — final validation loss and perplexity
- [x] The link from this checkpoint to the future multi-agent system demonstrated live

## 4 · Demonstration flow (~8 minutes)

| # | Show | What it proves |
|---|---|---|
| 1 | Config file, then the parameter counter printing `1,331,968` | The architecture is ours and fully specified. The number on the slide is the number the code produces. |
| 2 | `grep -rn "from_pretrained\|AutoModel\|hf_hub_download" .` → no matches | Nothing was downloaded. Running this **live** is far more persuasive than a claim on a slide. Rehearse so it is instant. |
| 3 | Weight histogram immediately after initialisation | Gaussian, mean ≈ 0, std ≈ 0.02. Freshly sampled weights, not loaded ones. |
| 4 | Step-0 loss printed: `8.3327`, beside `ln(4096) = 8.3178` | **The centrepiece.** The model assigns uniform probability across the vocabulary — it knows nothing. A pretrained model would start far below. One number closes the entire question. |
| 5 | Tail of the training log, then the loss curve | Loss fell from **8.33214 to 2.194813** over 6,100 steps, validation ending at **2.1407** — *below* training loss, so nothing was memorised. Our training loop did this, in 2.6 minutes. |
| 6 | Fresh process: load checkpoint, generate at T=0.7 | The checkpoint is a real, portable artefact. **Say plainly that a 1.3M model produces simple, sometimes incoherent English** — setting that expectation before the panel forms it themselves is the difference between a modest result and a disappointing one. |
| 7 | V0 multi-agent stub on one question, printing the decision record | Three roles, one checkpoint, one coordinator. The reasoning layer is wired to the real model; the path to §G is a change of degree, not of kind. |
| 8 | Roadmap slide: V2 → V7 | Where the 14M model, reasoning training, neural fusion and explainability land, with dates. |

## 5 · Evidence pack to have open

- `experiments/run_a_001/manifest.json` — seed, config, config hash, dataset revision, tokenizer hash, git commit
- `experiments/run_a_001/metrics.csv` — every logged step, starting at 8.318
- `experiments/run_a_001/loss_curve.png` — training and validation on one axis
- `experiments/run_a_001/samples.txt` — generations at three temperatures, seeds recorded
- `checkpoints/ckpt_A_final.pt` — with file size and modification time visible
- Terminal scrollback showing the overfit-one-batch test passing
- A screen recording of the full demo, in case the live run fails

## 6 · The Review 1 narrative

> "Single reasoning processes fail silently — they produce one answer, verify nothing, and expose
> no basis for trusting it. We propose a multi-agent explainable reasoning framework. Rather than
> build it on a pretrained model whose training data we cannot inspect, we are developing our own
> decoder-only language model from random initialisation and training it from scratch — which also
> means our reasoning benchmark cannot be contaminated by pretraining. **We have implemented and
> trained the first version of that model and can demonstrate it now.** The next stages specialise
> this trained model into reasoning agents, and investigate whether collaboration at the level of
> learned representations improves accuracy, calibration and explainability over text-level
> aggregation."

The load-bearing sentence is the bold one. Everything before is motivation, everything after is
plan; the middle is the only part that is already true — and it is what distinguishes this from
every other first review that day.

---

# N. RISKS

| Risk | Likely | Mitigation | Early warning signal |
|---|---|---|---|
| **Colab preemption mid-run** | High | Full-state checkpoints to Drive every 500 steps; resume tested *before* the real run. Config A is short enough to restart entirely. | Session idle warnings; falling throughput |
| **Agents produce near-identical outputs** | High | The quiet killer: if all four agents say the same thing, every aggregation method scores identically and the comparison is vacuous. Measure pairwise disagreement rate as a **first-class metric** from the first V4 run. Below ~15%, fix decoding diversity and role-token training before evaluating anything. | M3 accuracy exactly equal to M1 across all splits |
| **Model too small to reason at all** | Medium | The S1 gate exists for this. Run a cheap pilot — can Config B learn 2-hop MAX-Reason at all? — before committing weeks to the multi-agent layer. If not, reduce reasoning depth and family count. A narrower benchmark the model can do beats a broad one it cannot. | S1 accuracy near chance after reasoning SFT |
| **Multi-agent shows no gain over M2** | Medium | Pre-register the hypothesis and report the null honestly — a well-controlled negative result is a legitimate contribution, and M4 vs M5 may still differ even if both lose to self-consistency. Frame the project around the *comparison*, not a predicted winner. | Overlapping CIs in the first M1–M3 run |
| **Format non-compliance breaks parsing** | Medium | Report compliance as a Level 1 metric; permissive regex fallback parser; if below 90%, more SFT epochs or constrained decoding at field tokens | Parser exceptions during the first agent run |
| **Coordinator overfits** | Medium | 0.2M params on ~60k examples is a real risk. Held-out dev, early stopping with patience, dropout on z, train–dev gap reported alongside every accuracy figure | Dev accuracy plateauing while training accuracy climbs |
| **Tokenizer / checkpoint mismatch** | Low | Store tokenizer hash in the checkpoint; loader refuses on mismatch. Without this the failure is silent — fluent nonsense, no error | Generation quality collapsing after a tokenizer change |
| **Examiner doubts "from scratch"** | Low | The §M evidence pack — ln(V) check, live grep, init histograms, complete logs from step 0. Fully retired by preparation | Questions about where the weights came from |
| **Scope overrun** | High | Versions ordered so stopping early still leaves a coherent project. V1–V4 answer RQ1. V5 answers RQ2. V6–V7 are presentation. **Cut from the back.** | V4 not started when V5's slot arrives |

### The one risk worth internalising

Every other risk costs time. The agent-degeneracy risk costs the *project*: a multi-agent system
whose agents all say the same thing produces a full set of results, tables and graphs in which
every method scores identically, and it can take weeks to notice because nothing errors. Measuring
pairwise agent disagreement from the very first V4 run is a few lines of code and the cheapest
insurance in this document.

---

## Where this stands

Versions 0 and 1 are complete. `ckpt_A_final.pt` exists, and this document has gained its first
MEASURED labels in §D, §E, §F, §K and §M. Config A's row in the §K hardware table is now a
measurement rather than a projection.

**All sixteen Review 1 acceptance criteria in §M are met.** The checkpoint reloads in a fresh
process and reproduces its validation loss to six decimal places; the model generates coherent
English; the four-role stub runs on the frozen checkpoint and emits a full decision record.

After Review 1, Version 2 scales to the 13,784,064-parameter main model. Re-derive Config B's time
estimate from Config A's measured 320,128 tokens/s rather than from the FLOP arithmetic, which
underestimated throughput by roughly 6×.

## Sources verified 25 August 2026

- TinyStories — https://huggingface.co/datasets/roneneldan/TinyStories — CDLA-Sharing-1.0; 2,141,709 rows; 7.62 GB
- GSM8K — https://huggingface.co/datasets/openai/gsm8k — MIT; 7,473 train / 1,319 test; main and socratic configs
- WikiText — https://huggingface.co/datasets/Salesforce/wikitext — CC BY-SA 4.0; wikitext-103: 1,801,350 train rows, >100M tokens
- ProofWriter (Tafjord et al., 2021) — https://ar5iv.labs.arxiv.org/html/2012.13048 — depth-generalisation protocol; license not stated on the paper page, verify at https://allenai.org/data/proofwriter before use
