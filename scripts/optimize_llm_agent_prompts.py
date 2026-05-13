"""
Optimize the LLM-agent system prompt via a manual TextGrad loop.

Uses chat completions directly (no AdalFlow Responses API) to generate
text gradients and propose improved prompts.

Usage
-----
python scripts/optimize_llm_agent_prompts.py \
    --standalone_index_dir indexes/scifact_wide/ \
    --dataset              scifact \
    --num_train            30 \
    --num_val              10 \
    --steps                5 \
    --output_dir           results/llm_agent_prompts/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── .env loading ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from benchmarks import BEIRBenchmark
from inference import InferenceClient, LLMConfig
from RAG.standalone_hierarchy import StandaloneHierarchyNavigator
from RAG.llm_agent_retrieval import LLMAgentRetriever, DEFAULT_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def dcg(relevances: List[float], k: int) -> float:
    import math
    score = 0.0
    for i, r in enumerate(relevances[:k], start=1):
        score += r / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved_ids: List[str], relevant: Dict[str, int], k: int = 10) -> float:
    rels = [float(relevant.get(doc_id, 0) > 0) for doc_id in retrieved_ids[:k]]
    ideal = sorted([float(v > 0) for v in relevant.values()], reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(rels, k) / ideal_dcg


def recall_at_k(retrieved_ids: List[str], relevant: Dict[str, int], k: int = 100) -> float:
    n_relevant = sum(1 for v in relevant.values() if v > 0)
    if n_relevant == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if relevant.get(doc_id, 0) > 0)
    return hits / n_relevant


def composite_score(retrieved_ids: List[str], relevant: Dict[str, int]) -> float:
    return 0.7 * ndcg_at_k(retrieved_ids, relevant, k=10) + \
           0.3 * recall_at_k(retrieved_ids, relevant, k=100)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_samples(
    agent: LLMAgentRetriever,
    samples: List[Tuple[str, str, Dict[str, int]]],  # (qid, query, relevant)
    top_k: int = 100,
    label: str = "eval",
) -> float:
    scores = []
    for i, (qid, query, relevant) in enumerate(samples):
        try:
            doc_ids = agent.retrieve_ids(query, top_k=top_k)
            s = composite_score(doc_ids, relevant)
            scores.append(s)
            logger.info("[%s %d/%d] qid=%s score=%.4f docs=%d",
                        label, i + 1, len(samples), qid, s, len(doc_ids))
        except Exception as exc:
            logger.warning("[%s %d/%d] qid=%s failed: %s", label, i + 1, len(samples), qid, exc)
            scores.append(0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    logger.info("[%s] mean=%.4f  (%d samples)", label, avg, len(scores))
    return avg


# ── TextGrad (chat completions) ───────────────────────────────────────────────

def generate_gradient(
    current_prompt: str,
    failures: List[Dict[str, Any]],
    openai_api_key: str,
    model: str,
    temperature: float = 0.7,
) -> str:
    """
    Ask the meta-LLM to identify why the agent failed and suggest improvements
    to the system prompt.  Uses chat completions directly.
    """
    import openai

    failure_text = "\n\n".join(
        f"Query: {f['query']}\n"
        f"Score: {f['score']:.3f}\n"
        f"Retrieved top-5 IDs: {f['retrieved'][:5]}\n"
        f"Relevant IDs: {list(f['relevant'].keys())[:5]}"
        for f in failures[:6]
    )

    meta_prompt = f"""You are a prompt engineer optimizing a hierarchical document navigation agent.

CURRENT SYSTEM PROMPT:
\"\"\"
{current_prompt}
\"\"\"

The agent scored poorly (nDCG@10 < 0.4) on these queries:
{failure_text}

Analyze WHY the agent failed. Common causes:
- Went too shallow (retrieve too early → few docs)
- Went too deep / too narrow (missed relevant clusters)
- Cluster selection heuristic mismatched query semantics
- Didn't explore enough branches

Then rewrite the system prompt to fix the identified issues.
Keep it concise (< 200 words). Return ONLY the new prompt text, no commentary.
"""

    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": meta_prompt}],
        max_completion_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples(
    benchmark: BEIRBenchmark,
    num_train: int,
    num_val: int,
    seed: int = 42,
) -> Tuple[List, List]:
    import random
    examples = benchmark.get_test_examples()  # [{id, question, qrels}, ...]

    samples = [
        (ex["id"], ex["question"], ex["qrels"])
        for ex in examples
        if ex.get("qrels")
    ]

    random.seed(seed)
    random.shuffle(samples)

    train = samples[:num_train]
    val = samples[num_train: num_train + num_val]
    logger.info("Dataset: %d train, %d val queries (seed=%d)", len(train), len(val), seed)
    return train, val


# ── Build retrieval client ────────────────────────────────────────────────────

def build_inference_client(args) -> InferenceClient:
    config = LLMConfig(
        provider=os.getenv("USE_PROVIDER", "openai"),
        model=os.getenv("OPENAI_MODEL", args.llm_model),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"),
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("HF_BASE_URL")
                 or "https://api.openai.com/v1",
        max_concurrent=args.max_concurrent,
        timeout=60,
    )
    return InferenceClient(config)


# ── Main optimization loop ────────────────────────────────────────────────────

def optimize(args) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load index ────────────────────────────────────────────────────────────
    logger.info("Loading standalone hierarchy from %s", args.standalone_index_dir)
    navigator = StandaloneHierarchyNavigator.load(args.standalone_index_dir)
    logger.info("Hierarchy loaded: %d nodes", len(navigator.index.nodes))

    # ── Load benchmark ────────────────────────────────────────────────────────
    benchmark = BEIRBenchmark(dataset_name=args.dataset, split="test")
    train_samples, val_samples = load_samples(benchmark, args.num_train, args.num_val, seed=args.seed)

    # ── Build LLM client ──────────────────────────────────────────────────────
    inference_client = build_inference_client(args)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    meta_model = args.meta_model
    logger.info("Meta-optimizer model : %s", meta_model)
    logger.info("Retriever model      : %s", args.llm_model)

    # ── Baseline ──────────────────────────────────────────────────────────────
    current_prompt = DEFAULT_SYSTEM_PROMPT
    agent = LLMAgentRetriever(
        navigator=navigator,
        system_prompt=current_prompt,
        llm_client=inference_client,
        max_depth=args.max_depth,
        top_k=args.top_k,
        max_branches=args.max_branches,
        max_clusters=args.max_clusters,
    )

    logger.info("=" * 60)
    logger.info("Step 0 — baseline validation")
    best_score = evaluate_samples(agent, val_samples, top_k=args.top_k, label="val/0")
    best_prompt = current_prompt

    history = [{"step": 0, "val_score": best_score, "prompt": current_prompt}]

    # ── Optimization loop ─────────────────────────────────────────────────────
    for step in range(1, args.steps + 1):
        logger.info("=" * 60)
        logger.info("Step %d — forward pass on %d train samples", step, len(train_samples))

        # Forward pass: score each train query
        failures = []
        for qid, query, relevant in train_samples:
            try:
                doc_ids = agent.retrieve_ids(query, top_k=args.top_k)
                s = composite_score(doc_ids, relevant)
                if s < args.failure_threshold:
                    failures.append({
                        "query": query,
                        "score": s,
                        "retrieved": doc_ids,
                        "relevant": relevant,
                    })
            except Exception as exc:
                logger.warning("Train forward failed for %s: %s", qid, exc)
                failures.append({
                    "query": query,
                    "score": 0.0,
                    "retrieved": [],
                    "relevant": relevant,
                })

        logger.info("Step %d — %d / %d queries scored < %.2f", step, len(failures), len(train_samples), args.failure_threshold)

        if not failures:
            logger.info("All train queries score >= 0.4, stopping early.")
            break

        # Backward pass: generate improved prompt
        logger.info("Step %d — generating gradient (text feedback)...", step)
        try:
            new_prompt = generate_gradient(
                current_prompt=best_prompt,
                failures=failures,
                openai_api_key=api_key,
                model=meta_model,
                temperature=args.temperature,
            )
            logger.info("Step %d — new prompt (first 200 chars): %s...", step, new_prompt[:200])
        except Exception as exc:
            logger.error("Gradient generation failed: %s", exc)
            history.append({"step": step, "val_score": best_score, "error": str(exc)})
            continue

        # Evaluate new prompt on val set
        agent_candidate = LLMAgentRetriever(
            navigator=navigator,
            system_prompt=new_prompt,
            llm_client=inference_client,
            max_depth=args.max_depth,
            top_k=args.top_k,
            max_branches=args.max_branches,
            max_clusters=args.max_clusters,
        )
        val_score = evaluate_samples(agent_candidate, val_samples, top_k=args.top_k, label=f"val/{step}")

        improved = val_score > best_score
        if improved:
            logger.info("Step %d — ✓ improved %.4f → %.4f, keeping new prompt", step, best_score, val_score)
            best_score = val_score
            best_prompt = new_prompt
            agent = agent_candidate
            current_prompt = new_prompt
        else:
            logger.info("Step %d — ✗ no improvement %.4f vs %.4f, keeping old prompt", step, val_score, best_score)

        history.append({
            "step": step,
            "val_score": val_score,
            "best_score": best_score,
            "improved": improved,
            "prompt": new_prompt,
        })

    # ── Save results ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_path = out_dir / f"best_prompt_{ts}.txt"
    report_path = out_dir / f"report_{ts}.json"

    prompt_path.write_text(best_prompt)
    report = {
        "timestamp": ts,
        "dataset": args.dataset,
        "best_val_score": best_score,
        "steps": args.steps,
        "history": [
            {k: v for k, v in h.items() if k != "prompt"}
            for h in history
        ],
    }
    report_path.write_text(json.dumps(report, indent=2))

    logger.info("=" * 60)
    logger.info("Best val score : %.4f", best_score)
    logger.info("Best prompt    : %s", prompt_path)
    logger.info("Report         : %s", report_path)
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimize LLM-agent navigation prompt")
    parser.add_argument("--dataset",              default="scifact")
    parser.add_argument("--standalone_index_dir", required=True)
    parser.add_argument("--num_train",  type=int,   default=30)
    parser.add_argument("--num_val",    type=int,   default=40)
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for train/val shuffle.")
    parser.add_argument("--steps",      type=int,   default=5)
    parser.add_argument("--top_k",      type=int,   default=100)
    parser.add_argument("--max_depth",  type=int,   default=4)
    parser.add_argument("--max_branches", type=int, default=3)
    parser.add_argument("--max_clusters", type=int, default=8,
                        help="Max leaf clusters visited per query")
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--failure_threshold", type=float, default=0.8,
                        help="Queries scoring below this are treated as failures for gradient generation.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llm_model",  default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        help="Model used by the retriever agent (cheap, per-query).")
    parser.add_argument("--meta_model", default=os.getenv("META_MODEL", "gpt-5.5"),
                        help="Stronger model for gradient generation (called once per step).")
    parser.add_argument("--output_dir", default="results/llm_agent_prompts/")
    args = parser.parse_args()

    optimize(args)


if __name__ == "__main__":
    main()
