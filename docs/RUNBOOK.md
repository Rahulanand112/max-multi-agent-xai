# Runbook — training and evaluating the full system on free Colab

Everything after Review 1, as cells you can paste in order. Runtime → Change
runtime type → **T4 GPU** before you start.

Wall-clock figures below are **estimates**, extrapolated from the Review 1 run
(1.33M parameters, 320,128 tokens/s measured on a T4). They are not
measurements of these stages, and nothing in the repository reports them as
results. The real numbers land in `experiments/*/manifest.json` when you run it.

If the runtime disconnects, every stage resumes — skip to the stage you were on
and add `--resume`.

---

## 0 · Setup  (~3 min)

```python
from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/max
!git clone -q https://github.com/Rahulanand112/max-multi-agent-xai.git /content/max
%cd /content/max
!pip -q install -r requirements.txt
!mkdir -p /content/drive/MyDrive/max_v0/checkpoints
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Check the Review 1 tokenizer came with the clone — everything downstream depends
on its fingerprint being `d6080bac0a9a`:

```python
!python -c "from tokenizer.bpe import MAXTokenizer; t=MAXTokenizer.load('tokenizer/artifacts/max_bpe_a'); print(t.vocab_size, t.fingerprint())"
!python -m pytest tests/ -q
```

---

## 1 · Corpus and tokenizer  (~12 min)

The TinyStories corpus is re-downloaded and re-encoded with the **existing**
tokenizer. Nothing is refitted, so the fingerprint cannot change.

```python
%cd /content/max
!python scripts/01_prepare_corpus.py --source tinystories
!python scripts/03_encode_corpus.py
```

If you still have `data/processed/*.txt` on Drive from Review 1, copy those in
first and this step is under a minute.

---

## 2 · Generate MAX-Reason  (~2 min, CPU)

Our own benchmark: 120,000 problems, arithmetic and logic, every label computed
by the generator. No API model, no pretrained teacher, no human annotation.

```python
%cd /content/max
!python scripts/11_generate_reasoning.py --n 120000
!python scripts/12_build_sft.py --max-problems 40000
```

Watch two numbers in the output: the **duplicate rate** (should be ~1%) and
**split overlap** (must be 0 — the script exits if it is not).

---

## 3 · Gates before spending GPU time  (~2 min)

Do not skip these. They are cheap and they catch the two failures that waste an
hour each.

```python
%cd /content/max
!python scripts/04_verify_model.py --config configs/model_b.yaml
!python scripts/05_overfit_batch.py --config configs/train_b.yaml
```

Gate 1: step-0 loss must be `ln(4096) = 8.3178` within ±0.25 — the model is
randomly initialised and knows nothing. Gate 2: loss on one fixed batch must
fall below 0.1 — if it cannot memorise 8 sequences, the training loop is broken
and no amount of training fixes it.

---

## 4 · Stage 1 — pretrain MAX-12M  (est. 35–50 min)

12,211,200 parameters, random initialisation, on a mixture of TinyStories and
the reasoning narratives.

```python
%cd /content/max
D = "/content/drive/MyDrive/max_v0/checkpoints"
!python scripts/13_train_b.py --stage pretrain --checkpoint-dir {D}
```

Preempted? `!python scripts/13_train_b.py --stage pretrain --checkpoint-dir {D} --resume`

Short of time? `--max-steps 4000` halves it. The model will be weaker and the
manifest records the smaller token count, which is the honest way to report it.

---

## 5 · Stage 2 — reasoning SFT  (est. 12–18 min)

Teaches the five role behaviours. Loss is computed on completions only.

```python
%cd /content/max
D = "/content/drive/MyDrive/max_v0/checkpoints"
!python scripts/13_train_b.py --stage sft --init-from {D}/ckpt_B_final.pt --checkpoint-dir {D}
```

---

## 6 · Level 1 — is it still a language model?  (~3 min)

```python
%cd /content/max
D = "/content/drive/MyDrive/max_v0/checkpoints"
!python scripts/08_evaluate.py --checkpoint {D}/ckpt_B_sft_final.pt --config configs/sft_b.yaml
```

Perplexity and next-token accuracy. This says nothing about reasoning — that is
Level 2, and keeping them apart is deliberate.

---

## 7 · Run the five agents  (est. 15–25 min for both splits)

The only expensive step in the evaluation. Generations are cached, so every
later step is fast and needs no GPU.

```python
%cd /content/max
D = "/content/drive/MyDrive/max_v0/checkpoints"
!python scripts/14_run_agents.py --split val  --limit 1200 --checkpoint {D}/ckpt_B_sft_final.pt
!python scripts/14_run_agents.py --split test --limit 1500 --checkpoint {D}/ckpt_B_sft_final.pt
```

**Read the disagreement number it prints.** Below 15%, the agents are producing
near-identical answers and every aggregation method will score alike — the
comparison in step 9 would be weak evidence either way. The fix is more decoding
diversity (raise the Alternative's temperature) or better role separation in
Stage 2, not more evaluation.

---

## 8 · Stage 3 — train the fusion coordinator  (~2 min, CPU)

Trained on the **validation** bundles, reported on **test**. The language model
is frozen throughout.

```python
%cd /content/max
!python scripts/15_train_coordinator.py --bundles experiments/agents/bundles_val.pt
```

If it reports fewer than 50 usable rows it stops and says so: there was not
enough disagreement to learn from, and M5 will fall back to M4 **and label
itself as having done so** in the results table.

---

## 9 · Level 2 — the results table  (~1 min, CPU)

```python
%cd /content/max
!python scripts/16_evaluate_system.py
!cat experiments/level2/results_table.txt
!head -60 experiments/level2/decision_records.txt
```

Five methods on identical generations, with 95% intervals, calibration, per
family and per difficulty, ablations, and paired significance tests against both
a single agent (M1) and self-consistency (M2). M2 is the baseline that matters:
beating one generation with four is easy, and the table reports both.

---

## 10 · Collect everything and push  (~3 min)

The full checkpoint is ~140 MB — over GitHub's 100 MB hard limit, because it
carries the optimiser moments needed to resume. Export the weights first; the
export is ~49 MB, loads for inference, and keeps the fingerprint guard. Keep the
full checkpoint on Drive.

```python
%cd /content/max
D = "/content/drive/MyDrive/max_v0/checkpoints"
!python scripts/17_export_weights.py --checkpoint {D}/ckpt_B_sft_final.pt --out checkpoints/ckpt_B_sft_weights.pt
!python scripts/10_collect_results.py --experiments experiments --checkpoint checkpoints/ckpt_B_sft_weights.pt
!du -sh results checkpoints
```

`checkpoints/*` is gitignored apart from Review 1's `ckpt_A_final.pt`, so add the
export explicitly once you have checked its size:

```python
!git add -f checkpoints/ckpt_B_sft_weights.pt
```

```python
!git config user.name  "Rahul Anand"
!git config user.email "rahulanandbheemavarapu@gmail.com"
!git add -A && git commit -q -m "Reviews 2-4: reasoning benchmark, MAX-12M, five agents, fusion coordinator, Level 2 results"
```

Push from your own machine as before (`git pull` then `git push`), or with a
token here.

---

## 11 · The demo  (optional, ~2 min)

```python
!pip -q install streamlit
!npm install -q -g localtunnel
!streamlit run app/streamlit_app.py --server.port 8501 &>/content/log.txt &
!npx -y localtunnel --port 8501
```

Open the printed URL. The tunnel password is the Colab VM's public IP:

```python
!curl -s https://loca.lt/mytunnelpassword
```

Running it locally is simpler if you have the checkpoint on your machine:
`streamlit run app/streamlit_app.py`

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `tokenizer fingerprint mismatch` | checkpoint and tokenizer are from different fits | use the `tokenizer/artifacts/max_bpe_a` that came with the repo; never refit |
| `step-0 loss` far from 8.3178 | targets misaligned, or weights loaded from somewhere | stop — this is the from-scratch proof failing, not a tolerance to widen |
| overfit gate will not reach 0.1 | the training loop is broken | fix it before spending GPU time; training cannot rescue it |
| `CUDA out of memory` | batch too large for this T4 | `--batch-size 32`, or `--batch-size 24 --grad-accum 2` |
| disagreement below 15% | agents are producing the same text | raise the Alternative's temperature; report the number either way |
| fewer than 50 usable coordinator rows | same cause | M5 falls back to M4 and labels itself; that is a result, not a failure to hide |
| runtime died mid-stage | Colab preemption | re-run the same cell with `--resume` |
