# Recursive Hierarchical RAG

Reproduction code for *Fu, Kim, Yang — Recursive Hierarchical RAG* (MIT 6.8610). The pipeline builds a recursive *k*-means hierarchy over a BEIR corpus, then evaluates two navigators over that hierarchy:

1. **LLM Navigator** — zero-shot recursive cluster routing followed by a global BM25 rerank.
2. **Neural Policy** — a lightweight (~830K-param) MLP trained from oracle traversals, with dense-augmented final ranking at inference time.

Headline numbers from the report are in [`RESULTS.md`](RESULTS.md). The commands below reproduce them.

---

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **Data** loads from Hugging Face `datasets` into your cache.
- **LLM Navigator** needs `OPENAI_API_KEY` in a `.env` file at the repo root (default model: `gpt-4o-mini`).

The build script defaults to HF *offline* mode. On a first-time run, enable network:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 <command>
```

---

## 1. Build the hierarchies

The shipped checkpoints expect indices under `indexes/scifact_wide` and `indexes/nfcorpus_wide`. Both use the same build recipe (only the dataset and output dir change).

```bash
mkdir -p indexes/scifact_wide

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
python scripts/build_binary_leaf5_hierarchy.py \
  --dataset scifact \
  --index_dir indexes/scifact_wide \
  --embeddings_cache indexes/scifact_wide/_embeddings.npy \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --summary_mode keywords \
  --branching_factor 8 --max_leaf_size 250 --min_leaf_size 25 --max_depth 4
```

`indexes/scifact_wide/build_summary.json` should show **41 nodes / 28 leaves / depth 3 / mean leaf 185** to match the report.

For Table 3 (NFCorpus), mirror with `--dataset nfcorpus` and `--index_dir indexes/nfcorpus_wide`.

---

## 2. Reproduce the report tables

The trained checkpoints (`checkpoints/scifact_wide/phase3/phase3.pt`, `checkpoints/nfcorpus_wide/phase3/phase3.pt`) are committed. Skip to §3 only if you want to retrain.

### Table 1 — LLM Navigator on SciFact

```bash
python scripts/evaluate_llm_agent.py \
  --dataset scifact --split test \
  --index_dir indexes/scifact_wide \
  --llm_model gpt-4o-mini
```

### Table 2 — Neural Policy on SciFact

```bash
python scripts/evaluate_agentic_policy_v2.py \
  --dataset scifact --split test \
  --index_dir indexes/scifact_wide \
  --checkpoint checkpoints/scifact_wide/phase3/phase3.pt \
  --dense_augment_top_k 100 --final_alpha 0.9
```

### Table 3 — Neural Policy on NFCorpus

```bash
python scripts/evaluate_agentic_policy_v2.py \
  --dataset nfcorpus --split test \
  --index_dir indexes/nfcorpus_wide \
  --checkpoint checkpoints/nfcorpus_wide/phase3/phase3.pt \
  --dense_augment_top_k 100 --final_alpha 0.9
```

Each evaluator prints metrics to stdout and writes a timestamped `*_summary.json` under `--output_dir` (default `results/...`).

---

## 3. (Optional) Retrain the neural policy

Skip unless you want to regenerate the shipped checkpoints.

**Generate oracle supervision** from the BEIR train split:

```bash
mkdir -p data/supervision

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
python scripts/generate_agentic_scidocs_supervision.py \
  --index_dir indexes/scifact_wide \
  --beir_dataset scifact --split train \
  --out data/supervision/scifact_wide_train.jsonl
```

**Three-phase training** (matches `checkpoints/scifact_wide/phase{1,2,3}/config.json`):

```bash
# Phase 1 — action + jump (+ loop aux)
python scripts/train_agentic_policy_v2.py --phase 1 \
  --supervision_path data/supervision/scifact_wide_train.jsonl \
  --query_embeddings_path data/supervision/scifact_wide_train.query_embeddings.npz \
  --index_dir indexes/scifact_wide \
  --output_dir checkpoints/scifact_wide/phase1 \
  --epochs 40 --batch_size 64 --in_batch_negatives 0 \
  --lambda_jump_loop 0.1 --lambda_action_loop 0.05 --device cpu

# Phase 2 — warm-start + in-batch negatives
python scripts/train_agentic_policy_v2.py --phase 2 \
  --init_from checkpoints/scifact_wide/phase1/phase1.pt \
  --supervision_path data/supervision/scifact_wide_train.jsonl \
  --query_embeddings_path data/supervision/scifact_wide_train.query_embeddings.npz \
  --index_dir indexes/scifact_wide \
  --output_dir checkpoints/scifact_wide/phase2 \
  --epochs 20 --batch_size 64 --in_batch_negatives 8 \
  --lambda_jump_loop 0.1 --lambda_action_loop 0.05 --device cpu

# Phase 3 — done + retrieve heads
python scripts/train_agentic_policy_v2.py --phase 3 \
  --init_from checkpoints/scifact_wide/phase2/phase2.pt \
  --supervision_path data/supervision/scifact_wide_train.jsonl \
  --query_embeddings_path data/supervision/scifact_wide_train.query_embeddings.npz \
  --index_dir indexes/scifact_wide \
  --output_dir checkpoints/scifact_wide/phase3 \
  --epochs 20 --batch_size 64 --in_batch_negatives 8 \
  --lambda_jump_loop 0.0 --lambda_action_loop 0.0 --device cpu
```

For NFCorpus, mirror with `--beir_dataset nfcorpus`, `--index_dir indexes/nfcorpus_wide`, and matching output paths.

---

## 4. Tests

```bash
pytest tests/ -q
```

---

## 5. Citation

Please cite the *Recursive Hierarchical RAG* write-up alongside BEIR (Thakur et al., 2021), SciFact (Wadden et al., 2020), and NFCorpus (Boteva et al., 2016).
