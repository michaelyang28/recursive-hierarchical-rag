"""BEIR evaluation harness for the supervised agentic policy v2.

Implements plan section 5.6: loads a trained checkpoint, instantiates
``AgenticRetrieverV2``, runs it over a BEIR split, writes per-query traces /
results / summary JSON, and computes BEIR metrics (nDCG@10 etc.) plus
inference diagnostics (avg steps, frac_max_steps_hit, action distribution,
frac_loop_detected).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.beir.beir_loader import BEIRBenchmark  # noqa: E402
from benchmarks.beir.beir_metrics import BEIRMetrics, print_beir_metrics  # noqa: E402
from RAG.agentic_policy_v2.inference import (  # noqa: E402
    AgenticRetrieverV2,
    InferenceConfig,
)
from RAG.agentic_policy_v2.node_ann import NodeCentroidIndex  # noqa: E402
from RAG.agentic_policy_v2.state import NodeFeatureLookup  # noqa: E402
from RAG.standalone_hierarchy import StandaloneHierarchyIndex  # noqa: E402

logger = logging.getLogger(__name__)


def _load_or_build_node_ann(index_dir: Path, hierarchy_index, embedding_model: str) -> NodeCentroidIndex:
    ann_dir = index_dir / "ann_v2"
    try:
        return NodeCentroidIndex.load(ann_dir)
    except FileNotFoundError:
        ann = NodeCentroidIndex.from_hierarchy(hierarchy_index, embedding_model=embedding_model)
        ann.save(ann_dir)
        return ann


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scidocs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument(
        "--query_offset",
        type=int,
        default=0,
        help="Skip the first N queries (useful for held-out evaluation when training "
        "consumed the first M queries).",
    )
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--max_docs_budget", type=int, default=300)
    parser.add_argument("--max_call_budget", type=int, default=32)
    parser.add_argument("--K_max", type=int, default=32)
    parser.add_argument("--M_max", type=int, default=256)
    parser.add_argument("--ann_K", type=int, default=24)
    parser.add_argument("--retrieve_top_m", type=int, default=10)
    parser.add_argument("--tau_done", type=float, default=0.5)
    parser.add_argument("--min_evidence_to_stop", type=int, default=5)
    parser.add_argument("--final_alpha", type=float, default=0.6)
    parser.add_argument("--enable_done_head", action="store_true", default=True)
    parser.add_argument("--disable_done_head", dest="enable_done_head", action="store_false")
    parser.add_argument(
        "--retrieve_action_bias",
        type=float,
        default=0.0,
        help="Logit bias added to ACTION_RETRIEVE before argmax. Higher values make the agent retrieve more often.",
    )
    parser.add_argument(
        "--retrieve_at_leaf_bias",
        type=float,
        default=0.0,
        help="Extra logit bias added to ACTION_RETRIEVE when the current node is a leaf.",
    )
    parser.add_argument(
        "--dense_augment_top_k",
        type=int,
        default=0,
        help=(
            "If >0, after policy retrieval seed the final-ranking pool with the top-K "
            "docs by raw query/doc cosine similarity, scored as retrieve_score=0. The "
            "policy can then boost docs it retrieved (via the alpha-blend) on top of a "
            "dense baseline so the agent never does worse than flat dense retrieval."
        ),
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=0,
        help=(
            "If >0, fuse policy retrieval ranks and dense top-K ranks with Reciprocal "
            "Rank Fusion (score = sum 1/(rrf_k + rank_i)). Robust to score-scale "
            "mismatch between policy and dense; recommended k is 60."
        ),
    )
    parser.add_argument(
        "--rrf_dense_top_k",
        type=int,
        default=100,
        help="Size of the dense candidate list to fuse when --rrf_k > 0.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"agentic_v2_{args.dataset}_{args.split}_{timestamp}"
    traces_path = output_dir / f"{run_prefix}_traces.jsonl"
    results_path = output_dir / f"{run_prefix}_results.jsonl"
    summary_path = output_dir / f"{run_prefix}_summary.json"

    logger.info("Loading hierarchy index from %s", args.index_dir)
    hierarchy_index = StandaloneHierarchyIndex.load(args.index_dir, load_embeddings=True)
    lookup = NodeFeatureLookup(hierarchy_index)
    embedding_model = str(hierarchy_index.config.get("embedding_model", "local-hash-embedding"))
    node_ann = _load_or_build_node_ann(Path(args.index_dir), hierarchy_index, embedding_model)

    cfg = InferenceConfig(
        max_steps=args.max_steps,
        max_docs_budget=args.max_docs_budget,
        max_call_budget=args.max_call_budget,
        K_max=args.K_max,
        M_max=args.M_max,
        ann_K=args.ann_K,
        retrieve_top_m=args.retrieve_top_m,
        tau_done=args.tau_done,
        min_evidence_to_stop=args.min_evidence_to_stop,
        final_alpha=args.final_alpha,
        enable_done_head=args.enable_done_head,
        retrieve_action_bias=args.retrieve_action_bias,
        retrieve_at_leaf_bias=args.retrieve_at_leaf_bias,
        dense_augment_top_k=args.dense_augment_top_k,
        rrf_k=args.rrf_k,
        rrf_dense_top_k=args.rrf_dense_top_k,
    )

    retriever = AgenticRetrieverV2.from_checkpoint(
        checkpoint_path=args.checkpoint,
        hierarchy_index=hierarchy_index,
        lookup=lookup,
        node_ann=node_ann,
        config=cfg,
        device=args.device,
    )

    benchmark = BEIRBenchmark(dataset_name=args.dataset, split=args.split, cache_dir=args.cache_dir)
    examples = benchmark.get_test_examples()
    if args.query_offset > 0:
        examples = examples[args.query_offset:]
    if args.num_examples is not None:
        examples = examples[: args.num_examples]
    logger.info("Loaded %d examples (offset=%d)", len(examples), args.query_offset)

    qrels = benchmark.get_qrels()
    results_dict: Dict[str, Dict[str, float]] = {}

    action_counter: Counter = Counter()
    guard_counter: Counter = Counter()
    n_max_steps = 0
    n_loop = 0
    n_done = 0
    total_steps = 0
    evidence_sizes: List[int] = []
    latencies: List[float] = []
    n_failed = 0

    with traces_path.open("w", encoding="utf-8") as ft, results_path.open(
        "w", encoding="utf-8"
    ) as fr:
        for idx, ex in enumerate(examples):
            qid = ex["id"]
            qtext = ex["question"]
            try:
                result = retriever.retrieve(qtext, top_k=args.top_k, query_id=qid)
            except Exception:
                logger.exception("Query %s failed", qid)
                n_failed += 1
                results_dict[qid] = {}
                ft.write(json.dumps({"query_id": qid, "error": "retrieval_failed"}) + "\n")
                fr.write(json.dumps({"query_id": qid, "documents": []}) + "\n")
                continue

            results_dict[qid] = {
                doc["doc_id"]: float(doc.get("final_score", doc.get("retrieve_score", 0.0)))
                for doc in result.documents
            }
            ft.write(
                json.dumps({"query_id": qid, **result.to_dict()}, ensure_ascii=False)
                + "\n"
            )
            fr.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "question": qtext,
                        "documents": [
                            {"doc_id": d["doc_id"], "score": float(d.get("final_score", 0.0))}
                            for d in result.documents
                        ],
                        "terminated_reason": result.terminated_reason,
                        "docs_spent": result.docs_spent,
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for trace in result.traces:
                action_counter[trace.action] += 1
                if trace.guard_triggered:
                    guard_counter[trace.guard_triggered] += 1
            total_steps += len(result.traces)
            evidence_sizes.append(result.docs_spent)
            latencies.append(result.latency_ms)
            if result.terminated_reason == "max_steps":
                n_max_steps += 1
            elif result.terminated_reason == "loop_detected":
                n_loop += 1
            elif result.terminated_reason == "done":
                n_done += 1

            if (idx + 1) % args.log_every == 0:
                logger.info("Processed %d/%d queries", idx + 1, len(examples))

    k_values = [k for k in (1, 3, 5, 10, 100, 1000) if k <= args.top_k]
    metrics = BEIRMetrics.evaluate(qrels, results_dict, k_values=k_values)
    print_beir_metrics(metrics, args.dataset)

    n_q = max(1, len(examples))
    summary = {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "index_dir": args.index_dir,
            "num_queries": len(examples),
            "num_failed": n_failed,
            "timestamp": timestamp,
            "config": vars(args),
        },
        "metrics": metrics,
        "inference_diagnostics": {
            "avg_steps": total_steps / n_q,
            "frac_max_steps_hit": n_max_steps / n_q,
            "frac_done_terminated": n_done / n_q,
            "frac_loop_detected": n_loop / n_q,
            "action_distribution": dict(action_counter),
            "guard_distribution": dict(guard_counter),
            "avg_evidence_size": float(np.mean(evidence_sizes)) if evidence_sizes else 0.0,
            "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency_ms": (
                float(np.percentile(latencies, 95)) if latencies else 0.0
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
