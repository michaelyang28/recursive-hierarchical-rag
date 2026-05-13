"""Phase-vs-baseline retrieval comparison harness (plan section 5.7).

Runs the supervised v2 retriever and one or more baselines side-by-side on a
BEIR split, then writes ``comparison.json`` with:

* aggregate BEIR metrics for each retriever
* per-query nDCG@10 deltas vs. the v2 retriever
* the top-10 wins / losses (queries where one retriever beats the other most)
* action distribution and average path length per retriever

Available baselines:

* ``deterministic`` -- ``DeterministicAgenticPolicy`` from
  ``RAG.agentic_recursive_retrieval``.
* ``baseline_hierarchical`` -- existing top-down navigator (BM25 over the
  whole corpus via ``StandaloneHierarchyNavigator.search_in_cluster``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.beir.beir_loader import BEIRBenchmark  # noqa: E402
from benchmarks.beir.beir_metrics import BEIRMetrics  # noqa: E402
from RAG.agentic_policy_v2.embedding import (  # noqa: E402
    l2_normalize,
    make_text_encoder,
)
from RAG.agentic_policy_v2.inference import (  # noqa: E402
    AgenticRetrieverV2,
    InferenceConfig,
)
from RAG.agentic_policy_v2.node_ann import NodeCentroidIndex  # noqa: E402
from RAG.agentic_policy_v2.state import NodeFeatureLookup  # noqa: E402
from RAG.standalone_hierarchy import (  # noqa: E402
    StandaloneHierarchyIndex,
    StandaloneHierarchyNavigator,
)

logger = logging.getLogger(__name__)


def _load_or_build_node_ann(index_dir: Path, hierarchy_index, embedding_model: str) -> NodeCentroidIndex:
    ann_dir = index_dir / "ann_v2"
    try:
        return NodeCentroidIndex.load(ann_dir)
    except FileNotFoundError:
        ann = NodeCentroidIndex.from_hierarchy(hierarchy_index, embedding_model=embedding_model)
        ann.save(ann_dir)
        return ann


def _v2_runner(
    args: argparse.Namespace,
    hierarchy_index,
    lookup: NodeFeatureLookup,
    node_ann: NodeCentroidIndex,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
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
    )
    retriever = AgenticRetrieverV2.from_checkpoint(
        checkpoint_path=args.checkpoint,
        hierarchy_index=hierarchy_index,
        lookup=lookup,
        node_ann=node_ann,
        config=cfg,
        device=args.device,
    )
    diag: Dict[str, Any] = {
        "action_counts": Counter(),
        "path_lengths": [],
        "termination_reasons": Counter(),
        "evidence_sizes": [],
        "latencies": [],
    }

    def _run(query_id: str, query: str) -> Dict[str, Any]:
        result = retriever.retrieve(query, top_k=args.top_k, query_id=query_id)
        diag["action_counts"].update(t.action for t in result.traces)
        diag["path_lengths"].append(len(result.traces))
        diag["termination_reasons"][result.terminated_reason] += 1
        diag["evidence_sizes"].append(result.docs_spent)
        diag["latencies"].append(result.latency_ms)
        return {
            "doc_scores": {
                d["doc_id"]: float(d.get("final_score", d.get("retrieve_score", 0.0)))
                for d in result.documents
            },
            "doc_ids_ranked": [d["doc_id"] for d in result.documents],
        }

    return _run, diag


def _deterministic_runner(
    args: argparse.Namespace,
    hierarchy_index,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    from RAG.agentic_recursive_retrieval import (
        AgenticRecursiveRetriever,
        DeterministicAgenticPolicy,
    )

    navigator = StandaloneHierarchyNavigator(hierarchy_index)
    policy = DeterministicAgenticPolicy()
    retriever = AgenticRecursiveRetriever(
        navigator=navigator,
        policy=policy,
        max_steps=args.max_steps,
    )
    diag: Dict[str, Any] = {
        "action_counts": Counter(),
        "path_lengths": [],
        "evidence_sizes": [],
    }

    def _run(query_id: str, query: str) -> Dict[str, Any]:
        result = retriever.retrieve(query, top_k=args.top_k)
        for trace in result.traces:
            diag["action_counts"][trace.action.get("action_type", "UNKNOWN")] += 1
        diag["path_lengths"].append(len(result.traces))
        diag["evidence_sizes"].append(result.docs_spent)
        scores: Dict[str, float] = {}
        for rank, doc in enumerate(result.documents, start=1):
            metadata = doc.get("metadata") or {}
            doc_id = metadata.get("doc_id") or doc.get("id")
            if not doc_id:
                continue
            scores[doc_id] = max(scores.get(doc_id, 0.0), 1.0 / rank)
        return {
            "doc_scores": scores,
            "doc_ids_ranked": list(scores.keys()),
        }

    return _run, diag


def _baseline_hierarchical_runner(
    args: argparse.Namespace,
    hierarchy_index,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    navigator = StandaloneHierarchyNavigator(hierarchy_index)
    diag: Dict[str, Any] = {
        "evidence_sizes": [],
    }

    def _run(query_id: str, query: str) -> Dict[str, Any]:
        results = navigator.search_in_cluster(None, query, limit=args.top_k)
        scores: Dict[str, float] = {}
        for rank, hit in enumerate(results, start=1):
            doc_id = hit.get("metadata", {}).get("doc_id") or hit.get("id")
            if not doc_id:
                continue
            scores[doc_id] = float(hit.get("score", 1.0 / rank))
        diag["evidence_sizes"].append(len(scores))
        return {
            "doc_scores": scores,
            "doc_ids_ranked": list(scores.keys()),
        }

    return _run, diag


def _dense_global_runner(
    args: argparse.Namespace,
    hierarchy_index,
    lookup: NodeFeatureLookup,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Pure MiniLM cosine over the full corpus, no policy and no hierarchy."""

    if lookup._doc_embeddings is None:
        raise SystemExit("dense_global requires hierarchy_index loaded with embeddings")
    embedding_model = str(
        hierarchy_index.config.get("embedding_model", "local-hash-embedding")
    )
    encoder = make_text_encoder(embedding_model, lookup.embedding_dim)
    doc_mat_n = l2_normalize(lookup._doc_embeddings)
    diag: Dict[str, Any] = {"evidence_sizes": []}

    def _run(query_id: str, query: str) -> Dict[str, Any]:
        q = encoder([query])[0].astype("float32")
        q_n = q / max(1e-12, float(np.linalg.norm(q)))
        sims = doc_mat_n @ q_n
        order = np.argsort(-sims)[: args.top_k]
        scores: Dict[str, float] = {}
        for idx in order:
            doc_id = hierarchy_index.documents[int(idx)].doc_id
            scores[doc_id] = float(sims[int(idx)])
        diag["evidence_sizes"].append(len(scores))
        return {
            "doc_scores": scores,
            "doc_ids_ranked": list(scores.keys()),
        }

    return _run, diag


def _centroid_greedy_walk(
    lookup: NodeFeatureLookup,
    root_id: str,
    query_vec: np.ndarray,
) -> Tuple[str, List[str]]:
    """Greedy traversal: at each internal node, pick the child whose centroid
    has the highest cosine similarity with ``query_vec``. Returns ``(leaf_id, path)``.
    """
    node_id = root_id
    path = [node_id]
    while True:
        children = lookup.children_of(node_id)
        if not children:
            return node_id, path
        cm = lookup.node_centroid_matrix(children)
        sims = cm @ query_vec
        best = int(np.argmax(sims))
        node_id = children[best]
        path.append(node_id)


def _centroid_greedy_dense_leaf_runner(
    args: argparse.Namespace,
    hierarchy_index,
    lookup: NodeFeatureLookup,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Centroid-greedy navigation; dense MiniLM cosine within the chosen leaf."""

    if lookup._doc_embeddings is None:
        raise SystemExit(
            "centroid_greedy_dense_leaf requires hierarchy_index loaded with embeddings"
        )
    embedding_model = str(
        hierarchy_index.config.get("embedding_model", "local-hash-embedding")
    )
    encoder = make_text_encoder(embedding_model, lookup.embedding_dim)
    root_id = next(n.node_id for n in hierarchy_index.nodes if n.parent_id is None)
    diag: Dict[str, Any] = {
        "leaf_sizes": [],
        "path_lengths": [],
    }

    def _run(query_id: str, query: str) -> Dict[str, Any]:
        q = encoder([query])[0].astype("float32")
        q_n = q / max(1e-12, float(np.linalg.norm(q)))
        leaf_id, path = _centroid_greedy_walk(lookup, root_id, q_n)
        diag["path_lengths"].append(len(path))
        member = lookup.member_doc_ids(leaf_id)
        diag["leaf_sizes"].append(len(member))
        if not member:
            return {"doc_scores": {}, "doc_ids_ranked": []}
        mat = lookup.doc_embedding_matrix(member)
        mat_n = l2_normalize(mat)
        sims = mat_n @ q_n
        order = np.argsort(-sims)[: args.top_k]
        scores: Dict[str, float] = {}
        for idx in order:
            doc_id = member[int(idx)]
            scores[doc_id] = float(sims[int(idx)])
        return {
            "doc_scores": scores,
            "doc_ids_ranked": list(scores.keys()),
        }

    return _run, diag


def _centroid_greedy_bm25_leaf_runner(
    args: argparse.Namespace,
    hierarchy_index,
    lookup: NodeFeatureLookup,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Centroid-greedy navigation; BM25 scoring within the chosen leaf."""

    if lookup._doc_embeddings is None:
        raise SystemExit(
            "centroid_greedy_bm25_leaf requires hierarchy_index loaded with embeddings"
        )
    embedding_model = str(
        hierarchy_index.config.get("embedding_model", "local-hash-embedding")
    )
    encoder = make_text_encoder(embedding_model, lookup.embedding_dim)
    navigator = StandaloneHierarchyNavigator(hierarchy_index)
    root_id = next(n.node_id for n in hierarchy_index.nodes if n.parent_id is None)
    diag: Dict[str, Any] = {
        "leaf_sizes": [],
        "path_lengths": [],
    }

    def _run(query_id: str, query: str) -> Dict[str, Any]:
        q = encoder([query])[0].astype("float32")
        q_n = q / max(1e-12, float(np.linalg.norm(q)))
        leaf_id, path = _centroid_greedy_walk(lookup, root_id, q_n)
        diag["path_lengths"].append(len(path))
        diag["leaf_sizes"].append(len(lookup.member_doc_ids(leaf_id)))
        results = navigator.search_in_cluster(leaf_id, query, limit=args.top_k)
        scores: Dict[str, float] = {}
        for rank, hit in enumerate(results, start=1):
            doc_id = hit.get("metadata", {}).get("doc_id") or hit.get("id")
            if not doc_id:
                continue
            scores[doc_id] = float(hit.get("score", 1.0 / rank))
        return {
            "doc_scores": scores,
            "doc_ids_ranked": list(scores.keys()),
        }

    return _run, diag


def _per_query_ndcg(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for qid, query_qrels in qrels.items():
        retrieved = results.get(qid, {})
        sorted_docs = [d for d, _ in sorted(retrieved.items(), key=lambda kv: kv[1], reverse=True)][:k]
        out[qid] = BEIRMetrics._calculate_ndcg(sorted_docs, query_qrels, k)
    return out


def _aggregate_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in diag.items():
        if isinstance(value, Counter):
            out[key] = dict(value)
        elif isinstance(value, list):
            if value and all(isinstance(v, (int, float)) for v in value):
                out[f"{key}_mean"] = float(np.mean(value))
                out[f"{key}_p95"] = float(np.percentile(value, 95))
            elif value:
                out[key] = value
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scidocs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Phase 1/2/3 v2 checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["deterministic", "baseline_hierarchical"],
        choices=[
            "deterministic",
            "baseline_hierarchical",
            "dense_global",
            "centroid_greedy_dense_leaf",
            "centroid_greedy_bm25_leaf",
        ],
    )
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument(
        "--query_offset",
        type=int,
        default=0,
        help="Skip the first N queries before slicing num_examples.",
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top_winloss", type=int, default=10)
    parser.add_argument("--ndcg_k", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"comparison_{args.dataset}_{args.split}_{timestamp}.json"

    logger.info("Loading hierarchy index")
    hierarchy_index = StandaloneHierarchyIndex.load(args.index_dir, load_embeddings=True)
    lookup = NodeFeatureLookup(hierarchy_index)
    embedding_model = str(hierarchy_index.config.get("embedding_model", "local-hash-embedding"))
    node_ann = _load_or_build_node_ann(Path(args.index_dir), hierarchy_index, embedding_model)

    benchmark = BEIRBenchmark(dataset_name=args.dataset, split=args.split, cache_dir=args.cache_dir)
    examples = benchmark.get_test_examples()
    if args.query_offset > 0:
        examples = examples[args.query_offset:]
    if args.num_examples is not None:
        examples = examples[: args.num_examples]
    qrels = benchmark.get_qrels()
    qrels = {qid: rels for qid, rels in qrels.items() if qid in {ex["id"] for ex in examples}}

    runners: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
    runners["agentic_v2"] = _v2_runner(args, hierarchy_index, lookup, node_ann)
    if "deterministic" in args.baselines:
        runners["deterministic"] = _deterministic_runner(args, hierarchy_index)
    if "baseline_hierarchical" in args.baselines:
        runners["baseline_hierarchical"] = _baseline_hierarchical_runner(args, hierarchy_index)
    if "dense_global" in args.baselines:
        runners["dense_global"] = _dense_global_runner(args, hierarchy_index, lookup)
    if "centroid_greedy_dense_leaf" in args.baselines:
        runners["centroid_greedy_dense_leaf"] = _centroid_greedy_dense_leaf_runner(
            args, hierarchy_index, lookup
        )
    if "centroid_greedy_bm25_leaf" in args.baselines:
        runners["centroid_greedy_bm25_leaf"] = _centroid_greedy_bm25_leaf_runner(
            args, hierarchy_index, lookup
        )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in runners}
    for idx, ex in enumerate(examples):
        qid = ex["id"]
        question = ex["question"]
        for name, (run, _diag) in runners.items():
            try:
                rv = run(qid, question)
                all_results[name][qid] = rv["doc_scores"]
            except Exception as exc:
                logger.exception("%s failed on %s: %s", name, qid, exc)
                all_results[name][qid] = {}
        if (idx + 1) % args.log_every == 0:
            logger.info("Processed %d/%d", idx + 1, len(examples))

    k_values = [k for k in (1, 3, 5, 10, 100, 1000) if k <= args.top_k]
    metrics_by_runner: Dict[str, Dict[str, float]] = {}
    per_query_ndcg: Dict[str, Dict[str, float]] = {}
    for name, results in all_results.items():
        metrics_by_runner[name] = BEIRMetrics.evaluate(qrels, results, k_values=k_values)
        per_query_ndcg[name] = _per_query_ndcg(qrels, results, args.ndcg_k)

    diagnostics = {name: _aggregate_diag(diag) for name, (_, diag) in runners.items()}

    delta_histograms: Dict[str, Any] = {}
    win_loss: Dict[str, Any] = {}
    for name in runners:
        if name == "agentic_v2":
            continue
        deltas: List[Tuple[str, float]] = []
        for qid in qrels:
            v = per_query_ndcg["agentic_v2"].get(qid, 0.0)
            o = per_query_ndcg[name].get(qid, 0.0)
            deltas.append((qid, v - o))
        deltas_sorted_desc = sorted(deltas, key=lambda kv: kv[1], reverse=True)
        wins = deltas_sorted_desc[: args.top_winloss]
        losses = list(reversed(deltas_sorted_desc[-args.top_winloss:]))
        delta_values = [d for _, d in deltas]
        delta_histograms[name] = {
            "mean_delta": float(np.mean(delta_values)) if delta_values else 0.0,
            "median_delta": float(np.median(delta_values)) if delta_values else 0.0,
            "fraction_v2_wins": float(np.mean([d > 0 for d in delta_values])) if delta_values else 0.0,
            "fraction_v2_losses": float(np.mean([d < 0 for d in delta_values])) if delta_values else 0.0,
            "histogram_bins": [-1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0],
            "histogram_counts": np.histogram(
                delta_values, bins=[-1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0]
            )[0].tolist(),
        }
        win_loss[name] = {
            "top_v2_wins": [{"query_id": qid, "delta_ndcg_at_k": d} for qid, d in wins],
            "top_v2_losses": [{"query_id": qid, "delta_ndcg_at_k": d} for qid, d in losses],
        }

    summary = {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "index_dir": args.index_dir,
            "num_queries": len(examples),
            "ndcg_k": args.ndcg_k,
            "timestamp": timestamp,
            "config": vars(args),
        },
        "metrics_by_runner": metrics_by_runner,
        "delta_vs_v2": delta_histograms,
        "win_loss": win_loss,
        "diagnostics": diagnostics,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote comparison to %s", out_path)
    print(json.dumps({"output": str(out_path), **{f"ndcg@{args.ndcg_k}_{n}": metrics_by_runner[n].get(f"NDCG@{args.ndcg_k}", 0.0) for n in metrics_by_runner}}, indent=2))


if __name__ == "__main__":
    main()
