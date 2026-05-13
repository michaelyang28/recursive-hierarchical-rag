"""
Baseline evaluation of LLMAgentRetriever on BEIR SciFact (or any dataset).

Usage
-----
python scripts/evaluate_llm_agent.py \
    --index_dir   indexes/scifact/ \
    --dataset     scifact \
    --prompt_file results/llm_agent_prompts/best_prompt_YYYYMMDD_HHMMSS.txt \
    --output_dir  results/llm_agent/ \
    --max_depth   8 \
    --max_branches 2 \
    --top_k       100

Omit --prompt_file to use the built-in DEFAULT_SYSTEM_PROMPT.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from benchmarks.beir.beir_loader import BEIRBenchmark
from benchmarks.beir.beir_metrics import BEIRMetrics, print_beir_metrics
from inference import InferenceClient, LLMConfig
from RAG.standalone_hierarchy import StandaloneHierarchyNavigator
from RAG.llm_agent_retrieval import LLMAgentRetriever, DEFAULT_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM agent retriever on BEIR")
    parser.add_argument("--index_dir",    required=True)
    parser.add_argument("--dataset",      default="scifact")
    parser.add_argument("--split",        default="test")
    parser.add_argument("--prompt_file",  default=None,
                        help="Path to optimized prompt text file. "
                             "Defaults to built-in DEFAULT_SYSTEM_PROMPT.")
    parser.add_argument("--output_dir",   default="results/llm_agent/")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Limit number of queries (default: all)")
    parser.add_argument("--query_offset", type=int, default=0)
    parser.add_argument("--top_k",        type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=4)
    parser.add_argument("--max_branches", type=int, default=3)
    parser.add_argument("--max_clusters", type=int, default=8,
                        help="Max leaf clusters visited per query (caps LLM calls)")
    parser.add_argument("--min_budget",   type=int, default=5,
                        help="Legacy: no longer used in two-phase design")
    parser.add_argument("--llm_model",    default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--no_llm",       action="store_true",
                        help="Use lexical heuristic only (no LLM calls). "
                             "Fast sanity check.")
    parser.add_argument(
        "--dense_augment_top_k",
        type=int,
        default=0,
        help="If >0, augment the BM25 rerank pool with the top-K docs by "
             "MiniLM-L6-v2 dense similarity. Mirrors the agentic_policy_v2 fix.",
    )
    parser.add_argument(
        "--bm25_dense_alpha",
        type=float,
        default=0.0,
        help="If >0, blend final ranking as alpha*cosine_sim + (1-alpha)*z(BM25) "
             "instead of pure BM25. Most useful with --dense_augment_top_k.",
    )
    parser.add_argument("--log_every",    type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "llm_agent" if not args.no_llm else "heuristic_agent"
    summary_path = out_dir / f"{tag}_{args.dataset}_{args.split}_{ts}_summary.json"
    results_path = out_dir / f"{tag}_{args.dataset}_{args.split}_{ts}_results.jsonl"

    # ── Load system prompt ────────────────────────────────────────────────────
    if args.prompt_file:
        system_prompt = Path(args.prompt_file).read_text().strip()
        logger.info("Loaded prompt from %s (%d chars)", args.prompt_file, len(system_prompt))
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        logger.info("Using DEFAULT_SYSTEM_PROMPT")

    # ── Load hierarchy ────────────────────────────────────────────────────────
    logger.info("Loading hierarchy from %s", args.index_dir)
    if args.dense_augment_top_k > 0 or args.bm25_dense_alpha > 0:
        from RAG.standalone_hierarchy import StandaloneHierarchyIndex
        idx = StandaloneHierarchyIndex.load(args.index_dir, load_embeddings=True)
        idx.config["__index_dir__"] = str(args.index_dir)
        navigator = StandaloneHierarchyNavigator(idx)
    else:
        navigator = StandaloneHierarchyNavigator.load(args.index_dir)
    logger.info(
        "Hierarchy: %d nodes, depth=%d, %d docs",
        len(navigator.index.nodes),
        navigator.max_depth,
        len(navigator._all_doc_ids),
    )

    # ── Build LLM client ──────────────────────────────────────────────────────
    inference_client = None
    if not args.no_llm:
        config = LLMConfig(
            provider=os.getenv("USE_PROVIDER", "openai"),
            model=args.llm_model,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"),
            base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            max_concurrent=5,
            timeout=60,
        )
        inference_client = InferenceClient(config)
        logger.info("LLM client: %s @ %s", args.llm_model, config.base_url)

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = LLMAgentRetriever(
        navigator=navigator,
        system_prompt=system_prompt,
        llm_client=inference_client,
        max_depth=args.max_depth,
        top_k=args.top_k,
        max_branches=args.max_branches,
        max_clusters=args.max_clusters,
        dense_augment_top_k=args.dense_augment_top_k,
        bm25_dense_alpha=args.bm25_dense_alpha,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    benchmark = BEIRBenchmark(dataset_name=args.dataset, split=args.split)
    examples = benchmark.get_test_examples()
    qrels = benchmark.get_qrels()

    if args.query_offset > 0:
        examples = examples[args.query_offset:]
    if args.num_examples is not None:
        examples = examples[: args.num_examples]
    logger.info("Evaluating %d queries (offset=%d)", len(examples), args.query_offset)

    # ── Run evaluation ────────────────────────────────────────────────────────
    results_dict: Dict[str, Dict[str, float]] = {}
    n_failed = 0

    with results_path.open("w") as f:
        for idx, ex in enumerate(examples):
            qid = ex["id"]
            query = ex["question"]
            try:
                doc_ids = agent.retrieve_ids(query, top_k=args.top_k)
                # Score = 1.0 for all returned docs (ranking via order)
                results_dict[qid] = {doc_id: 1.0 / (rank + 1)
                                     for rank, doc_id in enumerate(doc_ids)}
                f.write(json.dumps({"qid": qid, "doc_ids": doc_ids}) + "\n")
            except Exception as exc:
                logger.warning("Query %s failed: %s", qid, exc)
                results_dict[qid] = {}
                n_failed += 1

            if (idx + 1) % args.log_every == 0:
                logger.info("Processed %d / %d queries", idx + 1, len(examples))

    logger.info("Done. Failed: %d / %d", n_failed, len(examples))

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = BEIRMetrics.evaluate(qrels=qrels, results=results_dict)
    print_beir_metrics(metrics, args.dataset.upper())

    summary = {
        "timestamp": ts,
        "dataset": args.dataset,
        "split": args.split,
        "index_dir": args.index_dir,
        "prompt_source": args.prompt_file or "DEFAULT_SYSTEM_PROMPT",
        "num_queries": len(examples),
        "n_failed": n_failed,
        "max_depth": args.max_depth,
        "max_branches": args.max_branches,
        "top_k": args.top_k,
        "llm_model": args.llm_model if not args.no_llm else "heuristic",
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary → %s", summary_path)


if __name__ == "__main__":
    main()
