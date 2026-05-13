# Recursive Hierarchical RAG — Results

Numbers reproduce the three tables in *Fu, Kim, Yang — Recursive Hierarchical RAG* (MIT 6.8610). All retrieval systems return up to 100 documents per query; metrics are nDCG@10 and Recall@100.

## Hierarchy

SciFact tree used for every row below: 5,183 documents, 41 nodes, 28 leaves, max depth 3, branching factor 8, max leaf size 250, mean leaf size 185.1.

```bash
python scripts/build_binary_leaf5_hierarchy.py --dataset scifact \
  --index_dir indexes/scifact_wide \
  --branching_factor 8 --max_leaf_size 250 --min_leaf_size 25 --max_depth 4
```

## Table 1 — LLM Navigator on SciFact

| Method | nDCG@10 | Recall@100 |
|---|---|---|
| Flat BM25 | 0.5880 | 0.6700 |
| LLM Navigator (Ours) | **0.6333** | **0.8522** |

Zero-shot, `gpt-4o-mini` base LLM, two-phase navigate → global BM25 rerank.

```bash
python scripts/evaluate_llm_agent.py --dataset scifact --split test \
  --index_dir indexes/scifact_wide --llm_model gpt-4o-mini
```

## Table 2 — Neural Policy on SciFact

| Method | nDCG@10 | Recall@100 |
|---|---|---|
| Dense Retrieval | 0.6451 | 0.9250 |
| MLP + Dense (Ours) | **0.6726** | **0.9367** |

Lightweight (~830K-param) policy trained on 800 SciFact queries with oracle-path supervision, evaluated with dense-augmented final ranking (K = 100, α = 0.9).

```bash
python scripts/evaluate_agentic_policy_v2.py --dataset scifact --split test \
  --index_dir indexes/scifact_wide \
  --checkpoint checkpoints/scifact_wide/phase3/phase3.pt \
  --dense_augment_top_k 100 --final_alpha 0.9
```

## Table 3 — NFCorpus Generalization

| Method | nDCG@10 | Recall@100 |
|---|---|---|
| Dense Retrieval | **0.3159** | 0.3115 |
| MLP + Dense (Ours) | 0.3115 | **0.3145** |

Same policy recipe, same α = 0.9. NFCorpus dense recall is much lower (~0.31 vs. 0.93 on SciFact), so the blend preserves the dense floor rather than lifting it. See §5 of the report.

```bash
python scripts/evaluate_agentic_policy_v2.py --dataset nfcorpus --split test \
  --index_dir indexes/nfcorpus_wide \
  --checkpoint checkpoints/nfcorpus_wide/phase3/phase3.pt \
  --dense_augment_top_k 100 --final_alpha 0.9
```