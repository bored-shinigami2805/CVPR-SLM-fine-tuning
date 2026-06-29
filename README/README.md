# CVPR-SLM-fine-tuning

Domain-specific instruction tuning of a 3B-parameter small language model into a research assistant that answers questions about **five specific CVPR computer-vision papers** — and refuses everything else with a single, deterministic out-of-domain reply.

Built on **Qwen2.5-3B-Instruct** with **LoRA** (PEFT) on a custom, behavior-balanced SFT dataset. Trained on a single Colab A100 in bf16; entire pipeline fits in two notebooks.

---

## The five papers

The model is grounded in only these papers — anything else triggers a refusal:

1. **ShutterMuse** — Capture-Time Photography Guidance with MLLMs
2. **SVD Restoration** — Shift-Variant Image Degradation and Restoration Using Singular Value Decomposition
3. **Naturalness vs. Transferability** — Naturalness Predicts but Does Not Cause Transferability in Image Encodings of Real-World Streams
4. **ICRDrag** — In-context Region-based Drag
5. **OCR-Robust** — How Robust is OCR-Reasoning?

---

## Repo layout

```
├── 01_train.ipynb     # SFT/LoRA training pipeline (Colab-ready)
├── 02_chat.ipynb      # Load adapter + interactive chat loop
├── train.jsonl        # 229 SFT samples (chat-format messages)
└── val.jsonl          # 25 held-out validation samples
```

---

## Quickstart

Both notebooks have an **Open In Colab** badge at the top. Use a GPU runtime (A100 recommended; T4 works with smaller batches).

### Train

1. Open `01_train.ipynb` in Colab.
2. Upload `train.jsonl` and `val.jsonl` when prompted.
3. Run all cells. The LoRA adapter is saved to `qwen2.5-3b-papers-lora/` and zipped for download at the end.

### Chat

1. Open `02_chat.ipynb` in Colab.
2. Upload the `qwen2.5-3b-papers-lora.zip` adapter from training.
3. Run all cells, then use the chat loop in cell 10 (`quit`/`exit` to stop, `reset` to clear history) or the batch `ask()` helper.

---

## Training setup

| Component | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | LoRA (PEFT), rank 16, α=32, dropout 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Precision | bf16 (no quantization — A100 has the VRAM) |
| Max sequence length | 1024 |
| Epochs | 8 |
| LR / schedule | 2e-4, cosine, 5% warmup |
| Batch | per-device 4 × grad-accum 4 = effective 16 |
| Loss | Response-only (system + user + assistant header masked with `-100`) |
| Gradient checkpointing | On |

### Response-only loss

Labels for everything except the assistant's reply are set to `-100`, so the model is trained only to *produce* the answer, not to reconstruct the prompt. Same idea referenced by the ShutterMuse paper. The encoding step renders the chat template to text twice — once with the full conversation, once with just the prompt (and `add_generation_prompt=True`) — then masks the prompt-length prefix of the labels.

---

## Dataset

**254 total samples** (229 train / 25 val), each a single-turn `{system, user, assistant}` conversation in Qwen's ChatML-compatible `messages` format. Identical system prompt across all samples — it names the five papers and pins the exact refusal string `"I don't have information."`

The dataset is **behavior-balanced**, covering:

- **Confident in-domain answers** — factual recall from each paper (numbers, methods, datasets, results).
- **Cross-paper reasoning** — questions that reference more than one paper.
- **Numerical exact-match** — specific metrics, sample sizes, model dimensions.
- **Adversarial / false-premise** — questions that misattribute claims or invent facts about the papers.
- **Defer-on-uncovered** — in-domain-shaped questions where the paper doesn't give the answer.
- **Out-of-domain refusals** — completely unrelated questions (weather, recipes, world cup, etc.).

The OOD and false-premise samples are what stop the model from drifting back into the base model's general chattiness after fine-tuning.

---

## Smoke tests

The training notebook ends with a sanity-check pass on questions like:

```
Q: What is ShutterMuse?
Q: What PSFs are used in the SVD restoration paper?
Q: What is the PRD dataset?
Q: What's the weather today?         → I don't have information.
Q: Who won the 2022 World Cup?       → I don't have information.
```

The chat notebook runs a similar batch (`Summarize the OCR-Robust paper.`, `What rewards does ShutterMuse use in GRPO?`, etc.) plus an OOD probe.

---

## Stack

`transformers ≥ 4.45` · `peft ≥ 0.13` · `trl ≥ 0.11` · `datasets ≥ 2.20` · `accelerate ≥ 0.34` · `bitsandbytes` · PyTorch (bf16, CUDA)
