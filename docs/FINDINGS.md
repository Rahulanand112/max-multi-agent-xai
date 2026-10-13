# Bugs found while building Reviews 2–4

Three bugs in this project would have produced wrong numbers rather than
crashes. All three are the same shape — something silently removed information
that a later measurement depended on — and all three are now pinned by tests.
They are recorded here rather than quietly fixed, because the second and third
change how a Review 1 figure should be read.

---

## 1 · The parser never credited a role's first field

`reasoning/schema.build_prompt()` ends every prompt with the role's first field
marker, so the model continues from *after* it and never emits it:

```
prompt      <|solver|><|question|> Maya has 4 boxes ...<|steps|>
completion   4 times 6 is 24. 24 minus 7 is 17.<|answer|> 17<|conf|> 0.87
```

`parse()` split the completion on markers and reported `<|steps|>` as missing,
because the completion does not contain it. Every output was therefore
non-compliant by construction, and **format compliance could never exceed 0%**
regardless of what the model produced.

**Fix.** Text before the first marker is credited to the role's first expected
field. `tests/test_reasoning_data.py::test_parser_credits_the_implied_leading_field`.

---

## 2 · The agent decoded away the schema it was being scored on

`agents/base.py` called `tokenizer.decode(generated_ids)`. That defaults to
`skip_special=True`, which strips every `<|...|>` token — and the schema *is*
those tokens. The model emitted `<|answer|>` and `<|conf|>` correctly; the
decoder removed them; the parser then found nothing to parse.

Measured on a small checkpoint trained in development, with everything else
held constant:

| | format compliance |
|---|---:|
| `skip_special=True` (the bug) | **0%** |
| `skip_special=False` (the fix) | **80%** |

The model had not changed between those two rows. Only the decode flag had.

**Fix.** `skip_special=False` in `RoleAgent.run`, with a comment saying why it is
required rather than preferred.

### What this means for the Review 1 figure

Review 1 reported **0% format compliance** for the V0 agent stub, attributed to
the model having had no reasoning training. Bugs 1 and 2 were both present in
that measurement, and either alone forces the result to zero.

So the honest statement is: **that figure measured the parser and the decoder,
not the model.** The underlying claim it supported — that an untrained model
produces unusable reasoning output — is still almost certainly true, and the
agent confidences reported alongside it were computed from log-probabilities and
are unaffected. But the 0% itself cannot be read as evidence about the model,
and the number has not been restated here as though it had always meant
something else. Re-running `scripts/09_agent_stub.py` against the Review 1
checkpoint would produce a figure that does mean something; until that is run,
this correction stands in its place.

---

## 3 · The Verifier and Coordinator were asked questions in a shape they were never taught

`reasoning/render.py` builds the prompts the model is **trained** on;
`reasoning/schema.build_prompt()` builds the ones it is **asked** at inference.
They had drifted apart in two places:

| role | trained on | asked at inference |
|---|---|---|
| Verifier | `…<\|answer\|> 17<\|evidence\|>` | `…<\|answer\|> 17<\|issue\|> …<\|evidence\|>` |
| Coordinator | `<\|answer\|> 17<\|answer\|> 24<\|verdict\|>` | `<\|answer\|> 17 24<\|verdict\|>` |

Nothing raises for this. The model is simply handed a context it has never seen
and continues from it as best it can, and accuracy falls for a reason no error
message mentions.

**Fix.** The Verifier is now *trained* with the Critic's objection in its prompt —
which is the order the protocol actually runs in, so this is the better design as
well as the consistent one — and `build_prompt` gained a `candidates` argument
that emits one `<|answer|>` marker each.

`tests/test_reasoning_data.py::test_inference_prompts_match_training_prompts`
rebuilds every training prompt through `build_prompt` and asserts equality, for
every role, on both problem families.

---

## 4 · The export that could not be loaded

`scripts/17_export_weights.py` strips optimiser state so the checkpoint fits
under GitHub's 100 MB limit. The first version dropped the `format` key, which
the loader checks before anything else, so a complete and otherwise valid export
failed with `unrecognised checkpoint format: None`.

**Fix.** `format` is kept, and `tests/test_export.py` loads every export rather
than only inspecting it.

---

## The common thread

Every one of these was a silent subtraction: a marker stripped, a field not
credited, a prompt section missing, a key dropped. None produced a traceback and
none would have failed a loss curve inspection — the Stage 2 training loss fell
to 0.03 with bugs 2 and 3 both live.

The protection that worked was not code review. It was insisting that a metric
be checked against a case where its value is known in advance: our own training
targets must score 100% compliance, and every training prompt must be
reproducible by the function used at inference. Both of those are now tests.
