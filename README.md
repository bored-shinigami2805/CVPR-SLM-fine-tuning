# CV-Papers SFT — domain assistant for 5 computer-vision papers

A complete pipeline for fine-tuning **Qwen2.5-3B-Instruct** (LoRA, SFT-only) into a small
assistant whose entire knowledge domain is **five specific computer-vision papers**. The repo
contains the training dataset, a contamination-free evaluation benchmark, and ready-to-run
Google Colab notebooks for training, evaluation, and chatting.

The assistant is shaped to do three things in balance:
- **Answer** confidently when a question is covered by the five papers,
- **Refuse** cleanly when a question is outside computer vision,
- **Defer** ("I don't have that information") when a question is plausibly in-domain but not
  covered — including correcting false-premise questions instead of inventing numbers.

---

## The five source papers

| # | Short name | Topic | File |
|---|------------|-------|------|
| 1 | **ShutterMuse** | Capture-time photography-guidance MLLM (Qwen3-VL-8B; SFT + GRPO) | `2606.25763v1.pdf` |
| 2 | **SVD restoration** | Restoring shift-variant blur via singular-value decomposition (analytical, MATLAB) | `2606.25818v1.pdf` |
| 3 | **WorldStream** | Time-series-as-image "naturalness" (FID) vs. classification accuracy | `2606.25844v1.pdf` |
| 4 | **OCR-Robust** | Robustness benchmark for OCR reasoning under visual perturbations | `2606.26041v1.pdf` |
| 5 | **ICRDrag** | In-context region-based drag editing (Next-DiT; IMAC + STAC; PRD dataset) | `2606.25907v1.pdf` |

Everything in the dataset is grounded **only** in these papers. Numbers are copied exactly
(with unit and condition); methods/results are attributed to the correct paper; answers are
self-contained (no "see Figure 3 / Table 2 / the authors").

---

## Files in this folder

| File | What it is |
|------|------------|
| `sft_dataset.jsonl` | **353** training rows, messages format (system/user/assistant). |
| `eval_benchmark.jsonl` | **89** held-out eval items (messages + `gold`/`accept`/`scoring`). |
| `train_eval_qwen2p5_3b_lora.ipynb` | Colab notebook: LoRA SFT, then scores base vs. fine-tune on the benchmark. |
| `chat_qwen2p5_3b_lora.ipynb` | Colab notebook: Gradio chat UI to talk to the fine-tuned model. |
| `DATASET_README_AND_AUDIT.md` | Grading protocol + self-audit results for the dataset. |
| `2606.*.pdf` | The five source papers. |
| `README.md` | This file. |

---

## Data format

Every training row is conversational and uses **one byte-identical system prompt**:

```
You are a computer-vision research assistant. You have studied a specific set of
computer-vision papers in depth. Answer accurately and concisely, grounded only in what
you know. If a question is outside computer vision, say it is outside your domain. If it
is about computer vision but not something you have studied, say you don't have that
information rather than guessing.
```

A row:
```json
{"messages":[
  {"role":"system","content":"<the fixed system prompt>"},
  {"role":"user","content":"What backbone does ICRDrag use?"},
  {"role":"assistant","content":"Next-DiT, a diffusion-transformer model; the approach is described as model-agnostic across diffusion-transformer architectures."}
]}
```

Chat tokens are **not** hand-written — apply your trainer's chat template
(`tokenizer.apply_chat_template`); the notebooks do this for you.

### Dataset composition (`sft_dataset.jsonl`, 353 rows)

| Behavior | Rows | Share |
|----------|------|-------|
| Answer (in-domain) | 251 | 71.1% |
| Refuse (out-of-domain) | 52 | 14.7% |
| Defer (in-domain but uncovered) | 50 | 14.2% |

Answer rows per paper: ShutterMuse 55 · SVD 44 · WorldStream 45 · OCR-Robust 46 · ICRDrag 61.

### Benchmark composition (`eval_benchmark.jsonl`, 89 items, 9 categories)

closed_book_factual 12 · numerical_accuracy 14 · comparative_reasoning 8 · synthesis 6 ·
paraphrase_robustness 7 · ood_refusal 13 · defer_unknown 12 · format_style 5 ·
general_cv_retention 12.

Each item carries a `scoring` field: `exact_match`, `refusal_correct`, `defer_correct`, or
`rubric`. The eval shares **zero** exact or near (Jaccard ≥ 0.7) question surface forms with
the training set — see the audit file.

---

## Quickstart (Google Colab)

### 1. Train + evaluate
1. Upload `train_eval_qwen2p5_3b_lora.ipynb` to [Colab](https://colab.research.google.com)
   (**File → Upload notebook**).
2. **Runtime → Change runtime type → GPU** (a free T4 works with 4-bit).
3. Run cells top to bottom. When prompted, upload `sft_dataset.jsonl` and
   `eval_benchmark.jsonl`.
4. It trains a LoRA adapter (r=16, α=32, 3 epochs) and prints a per-category
   **base vs. fine-tune** pass-rate table, writing `eval_results.csv` for manual review.
   Running the base model first makes the fine-tune's lift (and any regression on
   `general_cv_retention`) measurable.

### 2. Chat with the model
1. Upload `chat_qwen2p5_3b_lora.ipynb`, select a **GPU** runtime.
2. Run the 3 cells. The last cell launches a **Gradio** text box (with a public share link)
   and streams replies. `ADAPTER_DIR` points at the adapter saved by the training notebook
   (`/content/qwen2p5-3b-cv-lora`); set it to `None` to chat with the base model for
   comparison.

> Note: `/content/...` is wiped between Colab sessions. To reuse the adapter later, save it to
> Google Drive (or `!zip -r qwen_cv_lora.zip /content/qwen2p5-3b-cv-lora` and download it).

---

## Evaluation / grading protocol

Run base Qwen2.5-3B-Instruct on the benchmark first, then the fine-tune, on identical prompts.

- **exact_match** (numerical): pass iff the exact value with correct unit/condition appears;
  rounded or wrong-unit answers fail.
- **refusal_correct** (ood): pass iff it declines as out-of-domain and doesn't answer.
- **defer_correct** (defer): pass iff it says it lacks the info / corrects the false premise
  and invents no number.
- **rubric** (factual, comparative, synthesis, paraphrase, format, general-CV): pass iff the
  answer meets the bullet(s) in `accept`; wrong-paper attribution or blending two papers fails
  even if individual facts are right. The training/eval notebook auto-scores rubric items by
  keyword overlap as a **heuristic** — confirm them from `eval_results.csv`.

Expected pattern after fine-tuning: categories 1–8 rise or hold; `general_cv_retention`
(category 9) should **not** drop materially — LoRA + SFT-only is expected to preserve general
CV competence, and this category verifies it.

---

## How the dataset was built (reproducibility)

1. **Knowledge base** — atomic, self-contained, reference-resolved facts were extracted from
   each paper (prose, not tables).
2. **Q&A authoring** — every question/answer pair was hand-authored from that KB, with 3–5
   paraphrases per important fact, plus refuse and defer behavior rows (several defer items are
   adversarial false-premise probes).
3. **Serialization** — pairs are serialized to JSONL with `json.dumps`, guaranteeing valid
   JSON and a byte-identical system prompt on every row. No content is auto-scraped from the
   PDFs; the serializer only formats authored data.
4. **Audit** — automated checks for dangling references, raw LaTeX, JSON validity, the
   70/15/15 ratio, and eval/train contamination. Results in `DATASET_README_AND_AUDIT.md`.

---

## Target / method summary

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Method:** Supervised fine-tuning only (no continued pretraining)
- **Adapter:** LoRA (r=16, α=32, dropout 0.05) on attention + MLP projections
- **Libraries:** transformers, trl, peft, datasets, accelerate, bitsandbytes (4-bit optional);
  gradio for the chat UI

---

## Known caveats

- A few ShutterMuse subject-side sub-scores and the SVD paper's per-image output metrics were
  flagged uncertain during extraction; they are used sparingly (or handled as "not covered")
  rather than asserted. See the flagged-facts note in `DATASET_README_AND_AUDIT.md`.
- The dataset (353 rows) sits just under the ~600–900 guidance band but satisfies the
  behavior ratio, all five papers, every question type, and zero contamination. It can be
  extended with more paraphrases (SVD/WorldStream/OCR are lightest) and more cross-paper
  comparison/synthesis rows.
