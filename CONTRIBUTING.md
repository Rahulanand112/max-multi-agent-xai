# Working on this project

## Before any training run, two gates must pass

```bash
python scripts/04_verify_model.py     # step-0 loss must equal ln(vocab_size)
python scripts/05_overfit_batch.py    # loss must fall below 0.1 on 8 fixed sequences
```

These are gates, not tasks. A training run started before they pass wastes GPU time on a pipeline
that may be silently wrong.

## Rules enforced in this repository

- **No file over ~300 lines.** Split it instead.
- **Every run writes a manifest** with seed, config, config hash, dataset revision, tokenizer
  fingerprint, git commit and hardware.
- **No pretrained weights, ever.** `tests/test_no_pretrained.py` fails the build if any
  pretrained-loading call appears in the source tree, or if `transformers` enters `requirements.txt`.
- **Never report an unmeasured number.** Label it `PROPOSED` or `NOT YET MEASURED`.

## Three bugs this codebase actively guards against

Each one produces a run that looks completely normal and results that are wrong:

1. **Targets off by one** — the model predicts its own input rather than the next token.
   Guarded by `test_targets_are_inputs_shifted_by_one`, and visible in the step-0 loss (7.30
   instead of 8.33, because the LM head is tied to the embedding).
2. **Gradient clipping before unscaling under fp16** — compares a scaled gradient norm against an
   unscaled threshold, so clipping never fires. Guarded by `test_clip_happens_after_unscale`.
3. **Loading a checkpoint against a different tokenizer** — every token id means something else to
   the model, producing fluent nonsense with no error. Guarded by the fingerprint check in
   `training/checkpoint.py`.

```bash
pytest tests/ -q          # 119 tests
```
