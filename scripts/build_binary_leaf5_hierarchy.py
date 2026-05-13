"""Build a binary-tree (branching=2) StandaloneHierarchy on the full SciDocs
corpus with leaf size 5 and real sentence-transformers embeddings.

Two summary modes are supported, both fully local (zero API cost):

* ``--summary_mode keywords`` (default): instant; uses keyword-based summaries.
* ``--summary_mode hf_sampled``: per-node ``google/flan-t5-small`` summaries
  (CPU). About 1.5s/node => roughly 4-5 hours for ~10k nodes.

Embeddings can be cached and reused between runs:

  python scripts/build_binary_leaf5_hierarchy.py --summary_mode keywords \
         --embeddings_cache indexes/scidocs_binary_leaf5/_embeddings.npy \
         --index_dir indexes/scidocs_binary_leaf5_kw

  python scripts/build_binary_leaf5_hierarchy.py --summary_mode hf_sampled \
         --embeddings_cache indexes/scidocs_binary_leaf5/_embeddings.npy \
         --index_dir indexes/scidocs_binary_leaf5_hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from RAG.standalone_hierarchy import (  # noqa: E402
    DocumentRecord,
    HierarchyBuildConfig,
    StandaloneHierarchyBuilder,
)


def _ensure_offline_env() -> None:
    """Force HF libraries to use the local cache (fully offline build)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _load_beir_documents(
    dataset: str,
    split: str,
    cache_dir: str | None,
    max_docs: int | None,
) -> List[DocumentRecord]:
    from benchmarks.beir.beir_loader import BEIRBenchmark

    benchmark = BEIRBenchmark(dataset_name=dataset, split=split, cache_dir=cache_dir)
    corpus = benchmark.get_corpus()
    if max_docs is not None:
        corpus = corpus[:max_docs]

    documents = [
        DocumentRecord(
            doc_id=str(doc.get("_id", "")),
            title=doc.get("title") or "",
            text=doc.get("text") or "",
            metadata={"doc_id": str(doc.get("_id", "")), "source": f"beir-{dataset}"},
        )
        for doc in corpus
    ]
    return documents


def _embed_documents_st(documents: List[DocumentRecord], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence-transformers model {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [f"{doc.title}\n{doc.text}".strip() for doc in documents]
    print(f"Encoding {len(texts)} documents (batch={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    print(f"Encoded in {time.time() - t0:.1f}s -> shape {embeddings.shape}")
    return embeddings.astype("float32")


def _maybe_load_embeddings(cache_path: Path | None, expected_n: int) -> np.ndarray | None:
    if cache_path is None or not cache_path.exists():
        return None
    arr = np.load(cache_path)
    if arr.shape[0] != expected_n:
        print(
            f"Cache shape {arr.shape} mismatch expected_n={expected_n}; ignoring cache"
        )
        return None
    print(f"Loaded cached embeddings from {cache_path} shape={arr.shape}")
    return arr.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Real semantic embedding model (cached locally).",
    )
    parser.add_argument(
        "--summarization_model",
        default="google/flan-t5-small",
        help="Local HF model for hf_sampled summaries.",
    )
    parser.add_argument(
        "--summary_mode",
        default="keywords",
        choices=["none", "keywords", "hf_sampled"],
    )
    parser.add_argument("--branching_factor", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--max_leaf_size", type=int, default=5)
    parser.add_argument("--min_leaf_size", type=int, default=2)
    parser.add_argument("--docs_per_summary", type=int, default=12)
    parser.add_argument("--max_summary_new_tokens", type=int, default=120)
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--max_docs", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--dataset", default="scidocs", help="BEIR dataset name")
    parser.add_argument(
        "--split",
        default="test",
        help="Which BEIR split to load corpus from (corpus is split-independent for BEIR).",
    )
    parser.add_argument(
        "--embeddings_cache",
        default=None,
        help="Optional .npy cache for reusing embeddings across runs.",
    )
    parser.add_argument("--use_faiss", action="store_true", default=False)
    args = parser.parse_args()

    _ensure_offline_env()

    documents = _load_beir_documents(args.dataset, args.split, args.cache_dir, args.max_docs)
    if not documents:
        raise SystemExit(f"No documents loaded for BEIR/{args.dataset}.")
    print(f"Loaded {len(documents)} BEIR/{args.dataset} documents.")

    cache_path = Path(args.embeddings_cache) if args.embeddings_cache else None
    embeddings = _maybe_load_embeddings(cache_path, expected_n=len(documents))
    if embeddings is None:
        embeddings = _embed_documents_st(
            documents, model_name=args.embedding_model, batch_size=args.embedding_batch_size
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            print(f"Cached embeddings to {cache_path}")

    config = HierarchyBuildConfig(
        embedding_model=args.embedding_model,
        summarization_model=args.summarization_model,
        summarization_device=-1,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        min_leaf_size=args.min_leaf_size,
        max_leaf_size=args.max_leaf_size,
        normalize_embeddings=True,
        use_faiss=args.use_faiss,
        summary_mode=args.summary_mode,
        docs_per_summary=args.docs_per_summary,
        max_summary_new_tokens=args.max_summary_new_tokens,
    )
    builder = StandaloneHierarchyBuilder(config)
    print(f"Building hierarchy summary_mode={args.summary_mode}...")
    t0 = time.time()
    index = builder.build(documents=documents, embeddings=embeddings)
    elapsed = time.time() - t0

    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    index.save(args.index_dir)

    leaves = [n for n in index.nodes if not n.children]
    leaf_sizes = [len(n.member_doc_ids) for n in leaves]
    summary = {
        "index_dir": args.index_dir,
        "num_documents": len(index.documents),
        "num_nodes": len(index.nodes),
        "num_leaves": len(leaves),
        "max_depth": index.max_depth,
        "build_seconds": elapsed,
        "embedding_model": args.embedding_model,
        "summarization_model": args.summarization_model if args.summary_mode != "keywords" else None,
        "summary_mode": args.summary_mode,
        "branching_factor": args.branching_factor,
        "max_leaf_size": args.max_leaf_size,
        "min_leaf_size": args.min_leaf_size,
        "leaf_size_distribution": {
            "mean": float(np.mean(leaf_sizes)) if leaf_sizes else 0,
            "min": int(np.min(leaf_sizes)) if leaf_sizes else 0,
            "max": int(np.max(leaf_sizes)) if leaf_sizes else 0,
            "p50": float(np.percentile(leaf_sizes, 50)) if leaf_sizes else 0,
            "p95": float(np.percentile(leaf_sizes, 95)) if leaf_sizes else 0,
        },
    }
    summary_path = Path(args.index_dir) / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
