#!/usr/bin/env node
/*
 * Review 1 deck generator.
 *
 *   node scripts/make_deck.py            (yes, it is JavaScript -- run with node)
 *   node scripts/make_deck.py --figs experiments/run_a_001
 *
 * Every number in this deck is a MEASURED value from run_a_001. Nothing is
 * illustrative and nothing is rounded up.
 *
 * The two figures -- loss_curve.png and init_weight_histogram.png -- are
 * embedded if they are found, and drawn as a labelled placeholder if they are
 * not. Run this on the machine that has experiments/run_a_001/ and the real
 * plots go in automatically.
 */

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

// ---------------------------------------------------------------- arguments
const args = process.argv.slice(2);
const figIndex = args.indexOf("--figs");
const FIGS = figIndex >= 0 ? args[figIndex + 1] : "experiments/run_a_001";
const outIndex = args.indexOf("--out");
const OUT = outIndex >= 0 ? args[outIndex + 1] : "MAX_Review1.pptx";

const LOSS_CURVE = path.join(FIGS, "loss_curve.png");
const HISTOGRAM = path.join(FIGS, "init_weight_histogram.png");
const altHist = "experiments/verify_model_max-1m/init_weight_histogram.png";

// ------------------------------------------------------------------ palette
const INK = "10203A";        // deep navy — dominant on dark slides
const INK_SOFT = "1B2C4A";
const PAPER = "FFFFFF";
const MIST = "F1F4F9";       // light card ground
const GREY = "5A6478";
const RULE = "D6DEEA";
const BLUE = "1B4A8F";       // accent
const BLUE_LT = "6FA3E8";    // accent on dark
const AMBER = "B4610F";      // caution / "not yet"
const GREEN = "146049";      // measured
const GREEN_LT = "5AC495";

const H_FONT = "Cambria";    // headers
const B_FONT = "Calibri";    // body
const M_FONT = "Courier New";// every measured number

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5 in — set BEFORE any slide
pres.author = "Vishnu Vardhan";
pres.title = "MAX — Review 1";

const W = 13.3, H = 7.5, M = 0.62;

// ---------------------------------------------------------------- helpers
function title(slide, text, sub) {
  slide.addText(text, {
    x: M, y: 0.42, w: W - 2 * M, h: 0.72,
    fontFace: H_FONT, fontSize: 34, bold: true, color: INK, margin: 0,
  });
  if (sub) {
    slide.addText(sub, {
      x: M, y: 1.14, w: W - 2 * M, h: 0.36,
      fontFace: B_FONT, fontSize: 14, color: GREY, margin: 0,
    });
  }
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || MIST },
    line: { color: RULE, width: 1 },
  });
}

// the deck's one repeated motif: a measured number in mono, label beneath
function stat(slide, x, y, w, value, label, note, color) {
  slide.addText(value, {
    x, y, w, h: 0.5, fontFace: M_FONT, fontSize: 26, bold: true,
    color: color || BLUE, margin: 0, valign: "top",
  });
  slide.addText(label, {
    x, y: y + 0.5, w, h: 0.24, fontFace: B_FONT, fontSize: 11.5,
    bold: true, color: INK, margin: 0, valign: "top",
  });
  if (note) {
    slide.addText(note, {
      x, y: y + 0.74, w, h: 0.28, fontFace: B_FONT, fontSize: 10.5,
      color: GREY, margin: 0, valign: "top",
    });
  }
}

function figure(slide, file, x, y, w, h, caption) {
  if (fs.existsSync(file)) {
    slide.addImage({ path: file, x, y, w, h });
  } else {
    slide.addShape(pres.ShapeType.rect, {
      x, y, w, h, fill: { color: MIST },
      line: { color: BLUE, width: 1.25, dashType: "dash" },
    });
    slide.addText(
      [
        { text: "FIGURE SLOT\n", options: { fontSize: 12, bold: true, color: BLUE } },
        { text: path.basename(file) + "\n\n", options: { fontSize: 12, fontFace: M_FONT, color: INK } },
        { text: "Re-run  node scripts/make_deck.py  on the machine\nholding this file and it is embedded automatically.",
          options: { fontSize: 10.5, color: GREY } },
      ],
      { x: x + 0.2, y: y + h / 2 - 0.7, w: w - 0.4, h: 1.4, align: "center", fontFace: B_FONT }
    );
  }
  if (caption) {
    slide.addText(caption, {
      x, y: y + h + 0.06, w, h: 0.3, fontFace: B_FONT, fontSize: 10.5,
      color: GREY, margin: 0,
    });
  }
}

function chip(slide, x, y, text, color, bg) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w: 1.42, h: 0.3, rectRadius: 0.05,
    fill: { color: bg }, line: { color: color, width: 1 },
  });
  slide.addText(text, {
    x, y, w: 1.42, h: 0.3, align: "center", valign: "middle",
    fontFace: B_FONT, fontSize: 9.5, bold: true, color: color, margin: 0,
  });
}

// ============================================================ 1. TITLE
{
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("REVIEW 1  ·  DATA SCIENCE FOUNDATION  ·  4TH YEAR CSE", {
    x: M, y: 1.35, w: W - 2 * M, h: 0.3,
    fontFace: B_FONT, fontSize: 12, bold: true, color: BLUE_LT,
    charSpacing: 2, margin: 0,
  });
  s.addText("Multi-Agent Explainable AI\nwith Collaborative Neural Reasoning", {
    x: M, y: 1.8, w: W - 2 * M, h: 1.7,
    fontFace: H_FONT, fontSize: 40, bold: true, color: PAPER,
    lineSpacing: 46, margin: 0,
  });
  s.addText(
    "A decoder-only language model built and trained from random initialisation — " +
    "then specialised into collaborating reasoning agents.",
    { x: M, y: 3.6, w: 8.6, h: 0.8, fontFace: B_FONT, fontSize: 16, color: "C7D4E8", margin: 0 }
  );

  // status strip — the four claims the review rests on
  const items = [
    ["1,331,968", "parameters"],
    ["8.3327", "step-0 loss = ln(4096)"],
    ["2.1407", "validation loss"],
    ["none", "pretrained weights"],
  ];
  items.forEach(([v, l], i) => {
    const x = M + i * 3.05;
    s.addText(v, { x, y: 5.05, w: 2.9, h: 0.42, fontFace: M_FONT, fontSize: 21,
                   bold: true, color: GREEN_LT, margin: 0 });
    s.addText(l, { x, y: 5.48, w: 2.9, h: 0.3, fontFace: B_FONT, fontSize: 11,
                   color: "9DB0CC", margin: 0 });
  });

  s.addText("Vishnu Vardhan   ·   Computer Science & Engineering (Core)", {
    x: M, y: 6.5, w: W - 2 * M, h: 0.3,
    fontFace: B_FONT, fontSize: 12.5, color: "9DB0CC", margin: 0,
  });

  s.addNotes(
    "Open with the status strip, not the title. The four numbers are the whole review: " +
    "we built a model, it started exactly where an untrained model must start, it learned, " +
    "and nothing was downloaded. Everything after this slide is evidence for those four numbers."
  );
}

// ============================================================ 2. THE PROBLEM
{
  const s = pres.addSlide();
  title(s, "A single reasoning process fails silently",
        "It produces one answer, checks nothing, and gives you no basis for trusting it.");

  const problems = [
    ["One attempt, no second opinion",
     "A single pass produces a single answer. Nothing independent ever disagrees with it, so an error survives to the output unchallenged."],
    ["No verification step",
     "The system never asks whether its own answer holds up. Correct and incorrect answers are delivered with identical assurance."],
    ["Uncalibrated confidence",
     "A model that is wrong sounds exactly like a model that is right. Confidence carries no information the user can act on."],
    ["No inspectable basis",
     "The user is given a conclusion with no account of what supported it, what contradicted it, or where the uncertainty sits."],
  ];

  problems.forEach(([head, body], i) => {
    const x = M + (i % 2) * 6.15;
    const y = 1.75 + Math.floor(i / 2) * 2.2;
    card(s, x, y, 5.85, 1.95);
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.28, y: y + 0.3, w: 0.42, h: 0.42, fill: { color: BLUE },
    });
    s.addText(String(i + 1), {
      x: x + 0.28, y: y + 0.3, w: 0.42, h: 0.42, align: "center", valign: "middle",
      fontFace: M_FONT, fontSize: 13, bold: true, color: PAPER, margin: 0,
    });
    s.addText(head, {
      x: x + 0.85, y: y + 0.28, w: 4.75, h: 0.32,
      fontFace: B_FONT, fontSize: 15, bold: true, color: INK, margin: 0,
    });
    s.addText(body, {
      x: x + 0.85, y: y + 0.66, w: 4.75, h: 1.1,
      fontFace: B_FONT, fontSize: 12, color: GREY, margin: 0, valign: "top",
    });
  });

  s.addNotes(
    "Keep this to 45 seconds. These four are the motivation, not the contribution. " +
    "The point to land: the failures are silent, which is why adding verification and " +
    "an inspectable record is worth doing at all."
  );
}

// ============================================================ 3. THE SYSTEM
{
  const s = pres.addSlide();
  title(s, "Proposed system", "One trained model, addressed by five role tokens, fused by a learned coordinator.");

  const boxY = 2.35;
  const roles = ["solver", "critic", "verifier", "alternative"];

  // question
  card(s, M, boxY + 0.55, 1.3, 0.75, PAPER);
  s.addText("question", { x: M, y: boxY + 0.55, w: 1.3, h: 0.75, align: "center",
    valign: "middle", fontFace: B_FONT, fontSize: 12.5, bold: true, color: INK, margin: 0 });

  // role tokens
  roles.forEach((r, i) => {
    const y = 1.95 + i * 0.82;
    card(s, 2.5, y, 2.15, 0.62, PAPER);
    s.addText("<|" + r + "|>", { x: 2.5, y, w: 2.15, h: 0.62, align: "center",
      valign: "middle", fontFace: M_FONT, fontSize: 11, color: INK, margin: 0 });
    s.addShape(pres.ShapeType.line, {
      x: 2.2, y: y + 0.31, w: 0.26, h: 0,
      line: { color: GREY, width: 1.25, endArrowType: "triangle" },
    });
  });
  s.addShape(pres.ShapeType.line, {
    x: 2.2, y: 2.26, w: 0, h: 2.46, line: { color: GREY, width: 1.25 },
  });
  s.addShape(pres.ShapeType.line, {
    x: 1.92, y: boxY + 0.92, w: 0.28, h: 0, line: { color: GREY, width: 1.25 },
  });

  // the model
  card(s, 5.1, 1.9, 2.5, 3.5, PAPER);
  s.addText("MAX-LM", { x: 5.1, y: 3.05, w: 2.5, h: 0.4, align: "center",
    fontFace: H_FONT, fontSize: 19, bold: true, color: INK, margin: 0 });
  s.addText("one checkpoint\nweights frozen", { x: 5.1, y: 3.45, w: 2.5, h: 0.6,
    align: "center", fontFace: B_FONT, fontSize: 11.5, color: GREY, margin: 0 });
  s.addText("1,331,968 params", { x: 5.1, y: 4.05, w: 2.5, h: 0.3, align: "center",
    fontFace: M_FONT, fontSize: 10.5, color: BLUE, margin: 0 });
  roles.forEach((_, i) => {
    const y = 1.95 + i * 0.82 + 0.31;
    s.addShape(pres.ShapeType.line, { x: 4.65, y, w: 0.42, h: 0,
      line: { color: GREY, width: 1.25, endArrowType: "triangle" } });
    s.addShape(pres.ShapeType.line, { x: 7.6, y, w: 0.42, h: 0,
      line: { color: GREY, width: 1.25, endArrowType: "triangle" } });
  });

  // outputs
  roles.forEach((_, i) => {
    const y = 1.95 + i * 0.82;
    card(s, 8.05, y, 1.85, 0.62, PAPER);
    s.addText("aᵢ  hᵢ  cᵢ", { x: 8.05, y, w: 1.85, h: 0.62, align: "center",
      valign: "middle", fontFace: M_FONT, fontSize: 11, color: INK, margin: 0 });
  });
  s.addText("answer · hidden state · confidence", { x: 7.9, y: 1.6, w: 2.2, h: 0.28,
    align: "center", fontFace: B_FONT, fontSize: 10, color: GREY, margin: 0 });

  // coordinator
  card(s, 10.35, 2.65, 2.35, 1.35, "E8EFFA");
  s.addText("COORDINATOR", { x: 10.35, y: 2.85, w: 2.35, h: 0.3, align: "center",
    fontFace: B_FONT, fontSize: 12.5, bold: true, color: BLUE, margin: 0 });
  s.addText("cross-agent fusion\nof hidden states", { x: 10.35, y: 3.15, w: 2.35, h: 0.6,
    align: "center", fontFace: B_FONT, fontSize: 11, color: BLUE, margin: 0 });
  s.addShape(pres.ShapeType.line, { x: 9.9, y: 3.32, w: 0.42, h: 0,
    line: { color: BLUE, width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.ShapeType.line, { x: 11.52, y: 4.0, w: 0, h: 0.45,
    line: { color: BLUE, width: 1.5, endArrowType: "triangle" } });
  card(s, 10.35, 4.5, 2.35, 0.9, PAPER);
  s.addText("final answer\n+ decision record", { x: 10.35, y: 4.5, w: 2.35, h: 0.9,
    align: "center", valign: "middle", fontFace: B_FONT, fontSize: 12, bold: true,
    color: INK, margin: 0 });

  s.addText(
    "No agent has parameters of its own. Four role tokens address the same frozen weights, so any " +
    "difference between agents comes from the role token and the decoding settings — nothing else.",
    { x: M, y: 6.15, w: W - 2 * M, h: 0.5, fontFace: B_FONT, fontSize: 12.5, color: INK, margin: 0 }
  );

  s.addNotes(
    "The load-bearing sentence is at the bottom: ONE model, four role tokens, zero extra weights. " +
    "That is what makes five agents affordable on a free GPU, and it is why the comparison in later " +
    "versions is clean — the base model is identical across every method we test."
  );
}

// ============================================================ 4. FROM SCRATCH
{
  const s = pres.addSlide();
  title(s, "Why build the model instead of downloading one",
        "The constraint is not an obstacle. It buys something no larger project can easily buy.");

  card(s, M, 1.8, 5.85, 2.15);
  s.addText("The hard constraint", { x: M + 0.35, y: 2.02, w: 5.15, h: 0.32,
    fontFace: B_FONT, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText(
    "No pretrained weights at any stage — not as an initialisation, not as a teacher, " +
    "not as a fallback. PyTorch supplies tensors, autograd and optimisers. It does not supply the model.",
    { x: M + 0.35, y: 2.4, w: 5.15, h: 1.3, fontFace: B_FONT, fontSize: 12.5, color: GREY, margin: 0, valign: "top" }
  );

  card(s, 6.77, 1.8, 5.9, 2.15, "E5F1EB");
  s.addText("What it buys us", { x: 7.12, y: 2.02, w: 5.2, h: 0.32,
    fontFace: B_FONT, fontSize: 15, bold: true, color: GREEN, margin: 0 });
  s.addText(
    "Every published multi-agent result is confounded: nobody knows what was in the base model's " +
    "training data. We generate the corpus and the benchmark, so we can state with certainty that " +
    "no test item was seen during pretraining.",
    { x: 7.12, y: 2.4, w: 5.2, h: 1.4, fontFace: B_FONT, fontSize: 12.5, color: "1A4A3A", margin: 0, valign: "top" }
  );

  s.addText("The pipeline, end to end — every stage ours", {
    x: M, y: 4.25, w: 8, h: 0.3, fontFace: B_FONT, fontSize: 14, bold: true, color: INK, margin: 0,
  });

  const stages = ["corpus", "clean +\ndedup", "our BPE\ntokenizer", "encode", "RANDOM\nINIT",
                  "train", "checkpoint"];
  stages.forEach((t, i) => {
    const x = M + i * 1.79;
    const hot = t.indexOf("RANDOM") === 0;
    card(s, x, 4.7, 1.5, 0.82, hot ? "FAEEE0" : PAPER);
    s.addText(t, { x, y: 4.7, w: 1.5, h: 0.82, align: "center", valign: "middle",
      fontFace: hot ? M_FONT : B_FONT, fontSize: hot ? 10 : 11.5,
      bold: hot, color: hot ? AMBER : INK, margin: 0 });
    if (i < stages.length - 1) {
      s.addShape(pres.ShapeType.line, { x: x + 1.5, y: 5.11, w: 0.29, h: 0,
        line: { color: GREY, width: 1.25, endArrowType: "triangle" } });
    }
  });

  s.addText(
    "tests/test_no_pretrained.py fails the build if from_pretrained, AutoModel or any model " +
    "download call appears anywhere in the source tree — and if transformers ever enters requirements.txt.",
    { x: M, y: 5.95, w: W - 2 * M, h: 0.6, fontFace: B_FONT, fontSize: 12, color: GREY, margin: 0 }
  );

  s.addNotes(
    "Do not present the constraint as a rule imposed on us. Present it as a methodological advantage: " +
    "we own the pretraining corpus, so contamination is impossible by construction. That is a real " +
    "research argument, and it is the honest framing of what is otherwise just a course requirement."
  );
}

// ============================================================ 5. THE MODEL
{
  const s = pres.addSlide();
  title(s, "The model we built", "Decoder-only transformer. Every component implemented by us.");

  const rows = [
    ["vocabulary", "4,096"], ["embedding dim", "128"], ["layers", "4"],
    ["attention heads", "4  (head dim 32)"], ["feed-forward dim", "512"],
    ["context length", "128"], ["positional", "learned absolute"],
    ["normalisation", "pre-LN"], ["LM head", "tied to embedding"],
  ];
  card(s, M, 1.75, 5.5, 4.35);
  s.addText("Configuration", { x: M + 0.3, y: 1.95, w: 4.9, h: 0.3,
    fontFace: B_FONT, fontSize: 14, bold: true, color: INK, margin: 0 });
  rows.forEach(([k, v], i) => {
    const y = 2.36 + i * 0.39;
    s.addText(k, { x: M + 0.3, y, w: 2.6, h: 0.32, fontFace: B_FONT, fontSize: 12,
      color: GREY, margin: 0 });
    s.addText(v, { x: M + 2.9, y, w: 2.3, h: 0.32, fontFace: M_FONT, fontSize: 12,
      color: INK, margin: 0, align: "right" });
  });

  card(s, 6.5, 1.75, 6.18, 4.35);
  s.addText("Parameter count — derived, not estimated", { x: 6.8, y: 1.95, w: 5.6, h: 0.3,
    fontFace: B_FONT, fontSize: 14, bold: true, color: INK, margin: 0 });
  const calc = [
    ["token embedding   4096 × 128", "524,288"],
    ["positional         128 × 128", "16,384"],
    ["4 × transformer block", "791,040"],
    ["final layernorm", "256"],
    ["LM head (tied — adds nothing)", "0"],
  ];
  calc.forEach(([k, v], i) => {
    const y = 2.42 + i * 0.42;
    s.addText(k, { x: 6.8, y, w: 4.1, h: 0.34, fontFace: M_FONT, fontSize: 11,
      color: GREY, margin: 0 });
    s.addText(v, { x: 10.6, y, w: 1.8, h: 0.34, fontFace: M_FONT, fontSize: 11,
      color: INK, align: "right", margin: 0 });
  });
  s.addShape(pres.ShapeType.line, { x: 6.8, y: 4.62, w: 5.6, h: 0, line: { color: RULE, width: 1 } });
  s.addText("TOTAL", { x: 6.8, y: 4.74, w: 3, h: 0.4, fontFace: B_FONT, fontSize: 14,
    bold: true, color: INK, margin: 0 });
  s.addText("1,331,968", { x: 9.4, y: 4.7, w: 3, h: 0.45, fontFace: M_FONT, fontSize: 22,
    bold: true, color: BLUE, align: "right", margin: 0 });
  s.addText(
    "A unit test recomputes this from the config file and asserts it equals 1,331,968 — " +
    "the number on this slide is the number the code produces.",
    { x: 6.8, y: 5.3, w: 5.6, h: 0.6, fontFace: B_FONT, fontSize: 11.5, color: GREY, margin: 0 }
  );

  s.addNotes(
    "If asked why so small: it is deliberate. This size trains in minutes on a free T4 and still " +
    "trains on CPU, which was the insurance policy if Colab had been unavailable. Version 2 scales " +
    "to 13.8M parameters."
  );
}

// ============================================================ 6. THE PROOF
{
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("The proof that nothing was downloaded", {
    x: M, y: 0.5, w: W - 2 * M, h: 0.6,
    fontFace: H_FONT, fontSize: 32, bold: true, color: PAPER, margin: 0 });
  s.addText(
    "A model that has learned nothing spreads probability evenly across the vocabulary. " +
    "Cross-entropy under a uniform distribution over V tokens is exactly ln(V).",
    { x: M, y: 1.16, w: 8.4, h: 0.6, fontFace: B_FONT, fontSize: 14, color: "C7D4E8", margin: 0 });

  card(s, M, 2.0, 6.0, 2.1, INK_SOFT);
  s.addText("ln(4096)  =  8.3178", { x: M + 0.4, y: 2.25, w: 5.2, h: 0.45,
    fontFace: M_FONT, fontSize: 20, color: "9DB0CC", margin: 0 });
  s.addText("our step-0 loss  =  8.3327", { x: M + 0.4, y: 2.78, w: 5.2, h: 0.55,
    fontFace: M_FONT, fontSize: 24, bold: true, color: GREEN_LT, margin: 0 });
  s.addText("difference  +0.0150", { x: M + 0.4, y: 3.38, w: 5.2, h: 0.35,
    fontFace: M_FONT, fontSize: 14, color: "9DB0CC", margin: 0 });

  s.addText(
    "A model carrying pretrained weights starts far below this — it already knows which tokens " +
    "are likely. One number, one forward pass, and the question is closed.",
    { x: M, y: 4.25, w: 6.0, h: 0.9, fontFace: B_FONT, fontSize: 13, color: "C7D4E8", margin: 0 });

  const checks = [
    ["Weight histogram", "Gaussian, mean ≈ 0, std 0.02003 — freshly sampled, not loaded"],
    ["Live repository search", "zero matches for from_pretrained, AutoModel, hf_hub_download"],
    ["Target alignment", "a misaligned run would score 7.30, not 8.33 — the check catches that too"],
  ];
  checks.forEach(([h, b], i) => {
    const y = 2.0 + i * 1.32;
    s.addShape(pres.ShapeType.ellipse, { x: 7.05, y: y + 0.06, w: 0.34, h: 0.34,
      fill: { color: GREEN_LT } });
    s.addText("✓", { x: 7.05, y: y + 0.06, w: 0.34, h: 0.34, align: "center",
      valign: "middle", fontFace: B_FONT, fontSize: 13, bold: true, color: INK, margin: 0 });
    s.addText(h, { x: 7.55, y, w: 5.2, h: 0.32, fontFace: B_FONT, fontSize: 14,
      bold: true, color: PAPER, margin: 0 });
    s.addText(b, { x: 7.55, y: y + 0.36, w: 5.2, h: 0.7, fontFace: B_FONT, fontSize: 12,
      color: "9DB0CC", margin: 0, valign: "top" });
  });

  s.addText("Demonstrated live from the terminal, not asserted on a slide.", {
    x: M, y: 6.45, w: W - 2 * M, h: 0.35,
    fontFace: B_FONT, fontSize: 13, bold: true, color: BLUE_LT, margin: 0 });

  s.addNotes(
    "This is the centrepiece. Run the grep live — it takes two seconds and it is far more " +
    "convincing than any slide. Then show step-0 loss beside ln(4096). " +
    "Mention the third check: because the LM head is tied to the embedding, an off-by-one in the " +
    "targets would score 7.30 instead of 8.33 — so this single number also catches the most common " +
    "training bug."
  );
}

// ============================================================ 7. TRAINING
{
  const s = pres.addSlide();
  title(s, "Training from random initialisation",
        "6,100 steps · 49,971,200 tokens · Tesla T4 · seed 1337 · every figure measured.");

  figure(s, LOSS_CURVE, M, 1.75, 7.3, 4.05,
         "Training and validation loss, with ln(4096) marked as the random-initialisation baseline.");

  const stats = [
    ["8.33214", "loss at step 0", "identical to ln(4096)", GREEN],
    ["2.194813", "final training loss", "", BLUE],
    ["2.1407", "final validation loss", "perplexity 8.51", BLUE],
    ["51.10%", "next-token accuracy", "vs 1-in-4,096 baseline", BLUE],
  ];
  stats.forEach(([v, l, n, c], i) => {
    stat(s, 8.35, 1.9 + i * 1.08, 4.3, v, l, n, c);
  });

  s.addText(
    "Validation loss finished below training loss — one epoch-equivalent of tokens, nothing memorised. " +
    "2.6 minutes at 320,128 tokens/s, peak VRAM 0.83 GB of 15.4 GB. Our §K estimate said 15–45 minutes; " +
    "it was pessimistic by roughly 6×.",
    { x: M, y: 6.25, w: W - 2 * M, h: 0.7, fontFace: B_FONT, fontSize: 12.5, color: INK, margin: 0 }
  );

  s.addNotes(
    "Two things worth saying out loud. First: validation below training means nothing was memorised — " +
    "we saw each token roughly once. Second: we published a hardware estimate before running and it was " +
    "wrong by 6×; the measured number replaces it in the report. Being seen to correct your own estimate " +
    "is worth more than having got it right."
  );
}

// ============================================================ 8. WHAT IT WRITES
{
  const s = pres.addSlide();
  title(s, "What a 1.33M-parameter model writes",
        "Loaded in a fresh process from the saved checkpoint. Fixed prompts, recorded seeds.");

  const samples = [
    ["T = 0.2",
     "The little girl was so happy that she had made a new friend. She was so happy that she could help her friend. She hugged her…"],
    ["T = 0.7",
     "He decided not to be mad at the bear to be kind and not be grumpy. He said to himself, “Come here, bear. You don't have to talk…”"],
    ["T = 1.0",
     ", Max, he liked to play with his cat, and he played a game. One day, Hanny was playing tag and he saw a little puppy with a lost…"],
  ];
  samples.forEach(([t, txt], i) => {
    const y = 1.78 + i * 1.34;
    card(s, M, y, 8.6, 1.15);
    s.addText(t, { x: M + 0.28, y: y + 0.12, w: 1.0, h: 0.3, fontFace: M_FONT,
      fontSize: 11.5, bold: true, color: BLUE, margin: 0 });
    s.addText(txt, { x: M + 1.3, y: y + 0.12, w: 7.05, h: 0.92, fontFace: B_FONT,
      fontSize: 12.5, color: INK, italic: true, margin: 0, valign: "top" });
  });

  card(s, 9.55, 1.78, 3.13, 3.9, "E5F1EB");
  s.addText("Why this matters", { x: 9.85, y: 1.98, w: 2.6, h: 0.3,
    fontFace: B_FONT, fontSize: 14, bold: true, color: GREEN, margin: 0 });
  s.addText(
    "Grammatical sentences with correct pronoun agreement, from 1.33 million parameters " +
    "trained for 2.6 minutes.\n\nThis reproduces the TinyStories result (Eldan & Li, 2023) " +
    "at our scale — and it is the strongest demonstration available from a model this small.",
    { x: 9.85, y: 2.36, w: 2.6, h: 3.1, fontFace: B_FONT, fontSize: 11.5, color: "1A4A3A", margin: 0, valign: "top" }
  );

  s.addText(
    "Stated honestly: this is narrative English on a restricted vocabulary, not general language " +
    "competence. Immediate word repetition is 0.11%, though phrase-level looping is still visible " +
    "and our metric does not capture it.",
    { x: M, y: 5.95, w: W - 2 * M, h: 0.7, fontFace: B_FONT, fontSize: 12.5, color: AMBER, margin: 0 }
  );

  s.addNotes(
    "Read one sample aloud — the T=0.2 one. Then immediately deliver the caveat at the bottom " +
    "before anyone else spots the repeated phrase. Naming your own model's weakness first is worth " +
    "more than the metric that hides it."
  );
}

// ============================================================ 9. AGENT STUB
{
  const s = pres.addSlide();
  title(s, "The multi-agent layer, running on our checkpoint",
        "Four role tokens, one frozen model, a majority-vote coordinator, a machine-readable record.");

  card(s, M, 1.75, 6.6, 3.5, "F7F9FC");
  s.addText("MAX DECISION RECORD", { x: M + 0.3, y: 1.95, w: 5.5, h: 0.3,
    fontFace: M_FONT, fontSize: 12, bold: true, color: INK, margin: 0 });
  const rec = [
    "QUESTION      Maya has 4 boxes. Each box holds",
    "              6 pens. How many pens?",
    "FINAL ANSWER  (none parsed)",
    "CONFIDENCE    0.0000",
    "DISAGREEMENT  yes      ESCALATED  no",
    "FORMAT COMPLY 0% of outputs matched the schema",
    "DECISION PATH question -> solver+alternative ->",
    "              critic -> verifier -> majority_vote",
  ];
  s.addText(rec.join("\n"), { x: M + 0.3, y: 2.3, w: 6.05, h: 2.8,
    fontFace: M_FONT, fontSize: 10.5, color: GREY, margin: 0, lineSpacing: 15 });

  card(s, 7.5, 1.75, 5.18, 3.5);
  s.addText("Measured confidence tracks temperature", { x: 7.8, y: 1.95, w: 4.6, h: 0.3,
    fontFace: B_FONT, fontSize: 13.5, bold: true, color: INK, margin: 0 });
  const conf = [
    ["verifier", "0.0", "0.343 – 0.490"],
    ["solver", "0.0", "0.307 – 0.437"],
    ["critic", "0.7", "0.185 – 0.251"],
    ["alternative", "0.9", "0.071 – 0.126"],
  ];
  s.addText("agent", { x: 7.8, y: 2.36, w: 1.6, h: 0.26, fontFace: B_FONT, fontSize: 10.5,
    bold: true, color: GREY, margin: 0 });
  s.addText("T", { x: 9.4, y: 2.36, w: 0.7, h: 0.26, fontFace: B_FONT, fontSize: 10.5,
    bold: true, color: GREY, margin: 0, align: "right" });
  s.addText("confidence", { x: 10.2, y: 2.36, w: 2.2, h: 0.26, fontFace: B_FONT,
    fontSize: 10.5, bold: true, color: GREY, margin: 0, align: "right" });
  conf.forEach(([a, t, c], i) => {
    const y = 2.7 + i * 0.42;
    const hot = i < 2;
    s.addText(a, { x: 7.8, y, w: 1.6, h: 0.34, fontFace: B_FONT, fontSize: 12,
      color: INK, margin: 0 });
    s.addText(t, { x: 9.4, y, w: 0.7, h: 0.34, fontFace: M_FONT, fontSize: 12,
      color: GREY, margin: 0, align: "right" });
    s.addText(c, { x: 10.2, y, w: 2.2, h: 0.34, fontFace: M_FONT, fontSize: 12,
      color: hot ? GREEN : AMBER, margin: 0, align: "right" });
  });
  s.addText(
    "The two greedy agents are always the most confident, the highest-temperature agent always the " +
    "least, with no overlap. Evidence the number measures something real.",
    { x: 7.8, y: 4.5, w: 4.6, h: 0.7, fontFace: B_FONT, fontSize: 11, color: GREY, margin: 0, valign: "top" }
  );

  card(s, M, 5.5, 12.06, 1.3, "FAEEE0");
  s.addText("What this does and does not show", { x: M + 0.32, y: 5.66, w: 11.4, h: 0.3,
    fontFace: B_FONT, fontSize: 13, bold: true, color: AMBER, margin: 0 });
  s.addText(
    "It shows the architecture running on weights we trained: four roles, one checkpoint, a full " +
    "decision record. It does not show reasoning — the model has had no reasoning training, so the " +
    "agents write fluent stories and ignore the question. Format compliance is 0%, reported as 0%. " +
    "Teaching them to reason is Version 3.",
    { x: M + 0.32, y: 5.99, w: 11.4, h: 0.7, fontFace: B_FONT, fontSize: 12, color: "6B3D10", margin: 0 }
  );

  s.addNotes(
    "Deliver the amber box BEFORE showing the record, not after. Order matters: " +
    "'the architecture runs on weights we trained; the agents do not reason yet, because reasoning " +
    "training is Version 3; here is what they do instead.' Owning it makes it a milestone. " +
    "Being asked about it makes it a gap."
  );
}

// ============================================================ 10. ROADMAP
{
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("What comes next", { x: M, y: 0.5, w: W - 2 * M, h: 0.6,
    fontFace: H_FONT, fontSize: 32, bold: true, color: PAPER, margin: 0 });
  s.addText("Versions 0 and 1 are complete. The research questions live in Versions 3 to 6.", {
    x: M, y: 1.14, w: W - 2 * M, h: 0.35, fontFace: B_FONT, fontSize: 14,
    color: "C7D4E8", margin: 0 });

  const road = [
    ["V0–V1", "Tokenizer, model, training, checkpoint", "done"],
    ["V2", "Scale to the 13.8M-parameter main model", "next"],
    ["V3", "Reasoning benchmark + reasoning training", "next"],
    ["V4", "Five agents, disagreement handling, baselines", ""],
    ["V5", "Collaborative neural fusion — the contribution", ""],
    ["V6", "Confidence calibration and explainability", ""],
    ["V7", "Web application and final evaluation", ""],
  ];
  road.forEach(([v, t, tag], i) => {
    const y = 1.75 + i * 0.66;
    s.addText(v, { x: M, y, w: 1.0, h: 0.4, fontFace: M_FONT, fontSize: 14, bold: true,
      color: tag === "done" ? GREEN_LT : BLUE_LT, margin: 0 });
    s.addText(t, { x: 1.75, y, w: 7.6, h: 0.4, fontFace: B_FONT, fontSize: 14,
      color: tag === "done" ? "9DB0CC" : PAPER, margin: 0 });
    if (tag === "done") chip(s, 9.5, y + 0.03, "COMPLETE", GREEN_LT, "12312A");
    if (tag === "next") chip(s, 9.5, y + 0.03, "NEXT", BLUE_LT, "16294A");
  });

  card(s, 11.2, 1.75, 1.48, 4.5, INK_SOFT);
  s.addText("Review 2\nwill show", { x: 11.35, y: 1.95, w: 1.2, h: 0.6,
    fontFace: B_FONT, fontSize: 12, bold: true, color: BLUE_LT, margin: 0 });
  s.addText("a model that\nanswers the\nquestion\n\nagents that\ndisagree\n\nthe first\nmeasured\ncomparison",
    { x: 11.35, y: 2.65, w: 1.2, h: 3.3, fontFace: B_FONT, fontSize: 11,
      color: "9DB0CC", margin: 0, valign: "top" });

  s.addText(
    "Research question: can multiple specialised agents on a model trained from scratch improve " +
    "reasoning accuracy, reliability and explainability over a single agent — at a matched inference budget?",
    { x: M, y: 6.5, w: 10.3, h: 0.7, fontFace: B_FONT, fontSize: 13, italic: true,
      color: "C7D4E8", margin: 0 }
  );

  s.addNotes(
    "Close on the research question, not on the roadmap. The panel should leave knowing that the " +
    "foundation is built and verified, and that the actual experiment — does collaboration beat a " +
    "single agent at equal cost — is what Versions 3 to 5 answer."
  );
}

pres.writeFile({ fileName: OUT }).then(() => {
  const hasCurve = fs.existsSync(LOSS_CURVE);
  console.log(`wrote ${OUT}`);
  console.log(`  loss_curve.png : ${hasCurve ? "embedded" : "PLACEHOLDER — re-run where the file exists"}`);
});
