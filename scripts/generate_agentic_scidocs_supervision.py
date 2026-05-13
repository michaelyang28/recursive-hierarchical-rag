"""Generate trajectory-aligned supervision JSONL for a BEIR split.

Implements plan section 3.12. Reads the BEIR queries / qrels for the requested
split, walks an oracle trajectory through a persisted hierarchical index,
emits one JSONL row per trajectory step (plus off-path recovery samples),
and writes a sidecar ``query_embeddings.npy`` keyed by ``query_id``.

Example
-------

    python scripts/generate_agentic_scidocs_supervision.py \\
        --index_dir indexes/scidocs_hierarchy_cpu_smoke_leaf50 \\
        --beir_dataset scidocs --split test \\
        --ann_K 24 --K_max 32 --M_max 256 \\
        --n_off_path 2 --retrieve_threshold 250 --tau_done 1.0 \\
        --out data/supervision/scidocs_smoke_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from RAG.agentic_policy_v2.embedding import (  # noqa: E402
    encode_query_batch,
    make_text_encoder,
)
from RAG.agentic_policy_v2.node_ann import NodeCentroidIndex  # noqa: E402
from RAG.agentic_policy_v2.state import NodeFeatureLookup  # noqa: E402
from RAG.agentic_policy_v2.supervision import (  # noqa: E402
    SupervisionConfig,
    SupervisionGenerator,
)
from RAG.standalone_hierarchy import StandaloneHierarchyIndex  # noqa: E402

logger = logging.getLogger(__name__)


def _load_beir_split(dataset: str, split: str, cache_dir):
    from benchmarks.beir.beir_loader import BEIRBenchmark

    benchmark = BEIRBenchmark(dataset_name=dataset, split=split, cache_dir=cache_dir)
    return benchmark


def _resolve_examples(benchmark, num_examples: int | None):
    examples = benchmark.get_test_examples()
    if num_examples is not None:
        examples = examples[:num_examples]
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--beir_dataset", default="scidocs")
    parser.add_argument("--split", default="test", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--ann_K", type=int, default=24)
    parser.add_argument("--K_max", type=int, default=32)
    parser.add_argument("--M_max", type=int, default=256)
    parser.add_argument("--n_off_path", type=int, default=2)
    parser.add_argument("--retrieve_threshold", type=int, default=250)
    parser.add_argument("--tau_done", type=float, default=1.0)
    parser.add_argument("--n_random_jump_candidates", type=int, default=6)
    parser.add_argument(
        "--retrieve_action_repeats",
        type=int,
        default=1,
        help=(
            "Emit this many copies of each oracle RETRIEVE state. Values >1 "
            "increase RETRIEVE prevalence without changing rollout semantics."
        ),
    )
    parser.add_argument("--max_doc_budget", type=int, default=100)
    parser.add_argument("--max_call_budget", type=int, default=16)
    parser.add_argument(
        "--n_trajectories_per_query",
        type=int,
        default=1,
        help=(
            "Number of oracle trajectories to roll out per query. >1 generates "
            "additional trajectories with randomly shuffled target orderings, "
            "providing path diversity to combat overfitting on a single ordering."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--query_embeddings_out",
        default=None,
        help="Output .npy / .npz file for query embeddings (default: alongside --out)",
    )
    parser.add_argument(
        "--ann_dir",
        default=None,
        help="Directory containing or to write the node-centroid ANN index. "
        "Defaults to <index_dir>/ann_v2.",
    )
    parser.add_argument(
        "--rebuild_ann",
        action="store_true",
        help="Force rebuilding the node-centroid ANN index even if cached.",
    )
    parser.add_argument(
        "--query_encoder_model",
        default=None,
        help="Override the embedding model name (defaults to index config's embedding_model).",
    )
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.query_embeddings_out is None:
        query_embed_path = out_path.with_suffix("")
        query_embed_path = Path(str(query_embed_path) + ".query_embeddings.npz")
    else:
        query_embed_path = Path(args.query_embeddings_out)
    query_embed_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading hierarchy index from %s", args.index_dir)
    hierarchy_index = StandaloneHierarchyIndex.load(args.index_dir, load_embeddings=True)
    lookup = NodeFeatureLookup(hierarchy_index)

    embedding_model = args.query_encoder_model or str(
        hierarchy_index.config.get("embedding_model", "local-hash-embedding")
    )
    logger.info(
        "Embedding backbone=%s, dim=%d", embedding_model, lookup.embedding_dim
    )

    ann_dir = Path(args.ann_dir or (Path(args.index_dir) / "ann_v2"))
    if (
        not args.rebuild_ann
        and (ann_dir / "node_centroid_meta.json").exists()
        and (ann_dir / "node_centroid_matrix.npy").exists()
    ):
        logger.info("Loading existing node ANN index from %s", ann_dir)
        node_ann = NodeCentroidIndex.load(ann_dir)
    else:
        logger.info("Building node ANN index at %s", ann_dir)
        node_ann = NodeCentroidIndex.from_hierarchy(
            hierarchy_index, embedding_model=embedding_model
        )
        node_ann.save(ann_dir)

    sup_config = SupervisionConfig(
        ann_K=args.ann_K,
        K_max=args.K_max,
        M_max=args.M_max,
        n_off_path=args.n_off_path,
        retrieve_threshold=args.retrieve_threshold,
        tau_done=args.tau_done,
        n_random_jump_candidates=args.n_random_jump_candidates,
        retrieve_action_repeats=args.retrieve_action_repeats,
        max_doc_budget=args.max_doc_budget,
        max_call_budget=args.max_call_budget,
        n_trajectories_per_query=args.n_trajectories_per_query,
        seed=args.seed,
    )
    generator = SupervisionGenerator(
        hierarchy_index=hierarchy_index,
        lookup=lookup,
        node_ann=node_ann,
        config=sup_config,
    )

    logger.info("Loading BEIR/%s split=%s", args.beir_dataset, args.split)
    benchmark = _load_beir_split(args.beir_dataset, args.split, args.cache_dir)
    examples = _resolve_examples(benchmark, args.num_examples)
    logger.info("Loaded %d queries", len(examples))

    encoder_fn = make_text_encoder(embedding_model, lookup.embedding_dim)

    queries = [ex["question"] for ex in examples]
    logger.info("Encoding %d queries...", len(queries))
    query_matrix = encode_query_batch(encoder_fn, queries) if queries else np.zeros(
        (0, lookup.embedding_dim), dtype="float32"
    )
    query_id_list = [ex["id"] for ex in examples]

    np.savez(
        query_embed_path,
        query_ids=np.array(query_id_list, dtype=object),
        embeddings=query_matrix.astype("float32"),
    )
    logger.info("Wrote query embeddings to %s", query_embed_path)

    action_counter: Counter = Counter()
    written_examples = 0
    skipped_queries = 0
    n_examples_per_query: List[int] = []

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, ex in enumerate(examples):
            qid = ex["id"]
            qtext = ex["question"]
            qrels = ex.get("qrels") or {
                d: 1 for d in ex.get("ground_truth_docs", [])
            }
            relevant = [d for d, score in qrels.items() if int(score) > 0]
            if not relevant:
                skipped_queries += 1
                continue

            qvec = query_matrix[idx]
            try:
                examples_for_q = generator.generate_for_query(
                    query_id=qid,
                    query=qtext,
                    relevant_doc_ids=relevant,
                    query_vec=qvec,
                )
            except Exception as exc:
                logger.exception(
                    "Query %s failed during supervision generation: %s", qid, exc
                )
                skipped_queries += 1
                continue

            if not examples_for_q:
                skipped_queries += 1
                continue
            n_examples_per_query.append(len(examples_for_q))

            for sup in examples_for_q:
                fout.write(json.dumps(sup.to_jsonable(), ensure_ascii=False) + "\n")
                action_counter[sup.action_label] += 1
                written_examples += 1

            if (idx + 1) % args.log_every == 0:
                logger.info(
                    "Processed %d/%d queries; %d rows written",
                    idx + 1,
                    len(examples),
                    written_examples,
                )

    summary = {
        "out_path": str(out_path),
        "query_embeddings_path": str(query_embed_path),
        "ann_dir": str(ann_dir),
        "embedding_model": embedding_model,
        "embedding_dim": lookup.embedding_dim,
        "num_queries_total": len(examples),
        "num_queries_used": len(examples) - skipped_queries,
        "num_examples": written_examples,
        "skipped_queries": skipped_queries,
        "action_counts": {int(k): int(v) for k, v in action_counter.items()},
        "avg_examples_per_query": (
            float(np.mean(n_examples_per_query)) if n_examples_per_query else 0.0
        ),
        "supervision_config": sup_config.to_dict(),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary: %s", summary)
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
