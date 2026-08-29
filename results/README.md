# Results

Every run in this project writes a `manifest.json` recording the seed, full config, config hash,
dataset revision, tokenizer fingerprint, git commit and hardware - so any number here can be traced
back to the exact conditions that produced it.

| Directory | Produced by | Contains |
|---|---|---|
| `run_a_001/` | `scripts/07_train.py`, `08_evaluate.py` | training log, loss curve, generated samples, evaluation manifest |
| `verify_model_max-1m/` | `scripts/04_verify_model.py` | initial-weight histogram, step-0 loss, parameter count |
| `tokenizer_max_bpe_a/` | `scripts/02_train_tokenizer.py` | vocabulary size, merges, round-trip and UNK rates |
| `encode_max_bpe_a/` | `scripts/03_encode_corpus.py` | train/validation/test token counts |
| `data_prep/` | `scripts/01_prepare_corpus.py` | document counts before and after each filter |
| `agent_stub_v0/` | `scripts/09_agent_stub.py` | decision records from the four-role stub |

### Two gates whose manifests are not archived here

`scripts/05_overfit_batch.py` and `scripts/06_lr_probe.py` both ran and both passed - the overfit
gate drove loss on 8 fixed sequences below 0.1, and the probe measured 5e-4 -> 4.6537,
1e-3 -> 4.2026, 3e-3 -> 3.8820, which is why 3e-3 was used for `run_a_001`. Their output
directories were lost when the Colab runtime was recycled before the artefacts were collected.
The numbers are reported because they were observed and recorded at the time; the raw manifests
are not, so they are not claimed here as archived evidence. Re-running either script regenerates
them in a few minutes.

## Headline numbers - run `run_a_001`

```
parameters            1,331,968
step-0 loss             8.33214      ln(4096) = 8.3178
final training loss     2.194813
final validation loss   2.1407       perplexity 8.5053
next-token acc@1          51.10%
reload difference       0.000000
training time            2.6 min     6,100 steps, 49,971,200 tokens
peak VRAM                0.83 GB
```

Nothing in this directory is estimated. If a quantity has not been measured, it does not appear.
