# Pushing this to GitHub

The repository here contains **all the code and documentation**. It deliberately does *not* contain
your training outputs — those live on the machine that ran the training. Step 2 brings them in.

---

## 1. Create the repository on GitHub

Go to <https://github.com/new>, name it `max-multi-agent-xai` (or anything you like), leave it
**empty** — no README, no .gitignore, no license, because this repo already has all three.

---

## 2. On Colab — collect your results and push

Run this as one cell. Replace the two placeholders on the first two lines.

```python
GITHUB_USER = "your-username"
REPO        = "max-multi-agent-xai"

%cd /content/max
!python scripts/10_collect_results.py \
    --experiments experiments \
    --checkpoint /content/drive/MyDrive/max_v0/checkpoints/ckpt_A_final.pt

!git init -q 2>/dev/null; git branch -M main
!git config user.name  "Vishnu Vardhan"
!git config user.email "itsvishnugoat@gmail.com"
!git add -A
!git commit -q -m "Review 1: 1.33M-parameter language model trained from scratch

Custom BPE tokenizer (4,096 vocab, 3,966 merges) and a decoder-only Transformer
implemented from scratch, randomly initialised and trained on 49,971,200 tokens
of TinyStories.

  parameters            1,331,968
  step-0 loss             8.33214   (ln(4096) = 8.3178)
  final validation loss   2.1407    (perplexity 8.5053)
  next-token acc@1          51.10%
  reload difference       0.000000
  training time            2.6 min on a Tesla T4

No pretrained weights at any stage. 119 tests pass."
```

Then push. GitHub needs a **personal access token**, not your password — create one at
<https://github.com/settings/tokens> (classic token, tick the `repo` scope):

```python
from getpass import getpass
TOKEN = getpass("GitHub token: ")
!git remote remove origin 2>/dev/null; git remote add origin https://{GITHUB_USER}:{TOKEN}@github.com/{GITHUB_USER}/{REPO}.git
!git push -u origin main
!git remote set-url origin https://github.com/{GITHUB_USER}/{REPO}.git   # strip the token
```

That last line matters: it removes the token from the stored remote URL so it is not left sitting in
`.git/config` in a Colab session.

---

## 3. Check what you are about to commit

Before pushing, look at the size:

```python
!du -sh .git 2>/dev/null; du -sh results checkpoints tokenizer
!git status --short | head -40
```

**What should be there:** all `.py` files, `configs/`, `docs/`, `results/` (manifests, `metrics.csv`,
`samples.txt`, `loss_curve.png`, weight histogram), `tokenizer/artifacts/max_bpe_a/tokenizer.json`,
and `checkpoints/ckpt_A_final.pt` (16 MB).

**What should NOT be there:** any `.bin` file (the token streams — 100 MB and regenerable), any
`ckpt_step_*.pt`, `__pycache__`. The `.gitignore` handles all of these.

If a single file exceeds 100 MB, GitHub rejects the push. Nothing here should — the checkpoint is
16 MB — but if it ever does, either use Git LFS or drop the checkpoint and note in the README where
it can be downloaded.

---

## Alternative: push from your own computer

If you would rather not put a token into Colab, download `results/`, `checkpoints/` and
`tokenizer/artifacts/` from Drive, drop them into your local copy of this repo, then:

```bash
git init && git branch -M main
git add -A && git commit -m "Review 1: 1.33M-parameter language model trained from scratch"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

---

## After the push

Your repository front page will show the README: the results table, the loss curve, and the
from-scratch verification section with the commands a reader can run themselves. That is the page a
reviewer will look at first, so it is written to answer their questions before they ask.
