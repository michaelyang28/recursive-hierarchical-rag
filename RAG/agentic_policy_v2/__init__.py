"""Supervised agentic retrieval over a hierarchical corpus.

This package implements a learned policy network that navigates a
``StandaloneHierarchyIndex`` via discrete actions ``{RETRIEVE, AGGREGATE, JUMP}``
plus a separate done head. See ``plans/supervised_agentic_policy_*.plan.md``
for the full design.

Public modules:
    embedding   -- query/text encoder (sentence-transformers or hash fallback)
    node_ann    -- FAISS / numpy ANN over node centroids
    state       -- ``PolicyState`` and state-tensor builder
    network     -- shared encoder + action / done / jump / retrieve heads
    losses      -- multi-positive listwise CE + auxiliary losses
    dataset     -- supervision JSONL dataset and collator
    rlm_trace_adapter -- converts rewarded RLM traces into supervision JSONL
    inference   -- ``AgenticRetrieverV2`` end-to-end inference loop
"""

from __future__ import annotations

__all__ = [
    "embedding",
    "node_ann",
    "state",
    "network",
    "losses",
    "dataset",
    "rlm_trace_adapter",
    "inference",
]
