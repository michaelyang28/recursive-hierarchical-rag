"""Policy state representation and state tensor builder.

Implements plan section 1.3. A ``PolicyState`` is the small dataclass that
records what the agent has done so far in a trajectory; ``build_state_tensor``
materializes it into the concatenation
``[q_embed | n_embed | parent_embed | child_mean_embed | path_embed |
   evidence_embed | meta]`` consumed by the encoder.

Presence flags live inside the ``meta`` block so the encoder can distinguish a
genuine zero embedding from a "not present" zero substitution. The same module
is used at supervision-generation time, training time (offline state replay),
and inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .embedding import l2_normalize


META_FEATURE_NAMES: List[str] = [
    "depth_norm",                # depth / max_depth (clipped to [0, 1])
    "log_subtree_size",          # log1p(subtree_doc_count)
    "is_leaf",                   # 1 if no children
    "has_parent",                # 0 at root, 1 elsewhere
    "has_children",              # 1 if internal, 0 if leaf
    "has_path",                  # 1 if any non-trivial path
    "has_evidence",              # 1 if any evidence collected
    "n_visited_norm",            # min(len(path) / 16, 1.0)
    "n_evidence_norm",           # min(len(evidence) / 100, 1.0)
    "retrieve_calls_norm",       # min(retrieve_calls / 8, 1.0)
    "jump_calls_norm",           # min(jump_calls / 16, 1.0)
    "doc_budget_left_norm",      # remaining_doc_budget / max_doc_budget
    "call_budget_left_norm",     # remaining_call_budget / max_call_budget
    "step_index_norm",           # min(step / 16, 1.0)
    "fraction_evidence_in_subtree",   # |evidence ∩ subtree| / max(|evidence|, 1)
    "current_max_node_sim",      # cos(q, n_embed)
]

META_DIM: int = len(META_FEATURE_NAMES)


@dataclass
class PolicyState:
    """Trajectory state at a single decision point.

    Attributes mirror the JSONL ``state`` block in plan section 3.11 so we can
    serialize, replay, and re-tensorize identically at training and inference.
    """

    query: str
    current_node_id: str
    depth: int
    is_leaf: bool
    subtree_doc_count: int
    path_node_ids: List[str] = field(default_factory=list)
    evidence_doc_ids: List[str] = field(default_factory=list)
    step_index: int = 0
    retrieve_calls: int = 0
    jump_calls: int = 0
    aggregate_calls: int = 0
    remaining_doc_budget: int = 100
    remaining_call_budget: int = 16
    max_doc_budget: int = 100
    max_call_budget: int = 16
    max_depth: int = 8

    def as_state_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.current_node_id,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "subtree_doc_count": self.subtree_doc_count,
            "path_node_ids": list(self.path_node_ids),
            "evidence_doc_ids": list(self.evidence_doc_ids),
            "step_index": self.step_index,
            "retrieve_calls": self.retrieve_calls,
            "jump_calls": self.jump_calls,
            "aggregate_calls": self.aggregate_calls,
            "remaining_doc_budget": self.remaining_doc_budget,
            "remaining_call_budget": self.remaining_call_budget,
            "max_doc_budget": self.max_doc_budget,
            "max_call_budget": self.max_call_budget,
            "max_depth": self.max_depth,
        }


def _safe_mean(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0 or rows.shape[0] == 0:
        return np.zeros(rows.shape[1] if rows.ndim == 2 else 0, dtype="float32")
    return rows.mean(axis=0).astype("float32")


def _l2_normalize_vec(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return (vec / norm).astype("float32")


def build_state_tensor(
    state: PolicyState,
    query_embed: np.ndarray,
    node_lookup: "NodeFeatureLookup",
) -> np.ndarray:
    """Return the float32 state vector ``x in R^{6D + META_DIM}``.

    All embedding components are L2-normalized; missing parent / path / evidence
    use a zero vector and the corresponding presence flag in ``meta``.
    """

    embedding_dim = node_lookup.embedding_dim
    expected = 6 * embedding_dim + META_DIM

    q = np.asarray(query_embed, dtype="float32").reshape(-1)
    if q.shape[0] != embedding_dim:
        raise ValueError(
            f"query_embed dim {q.shape[0]} != index dim {embedding_dim}"
        )
    q = _l2_normalize_vec(q)

    n_embed = node_lookup.node_centroid(state.current_node_id)

    parent_id = node_lookup.parent_of(state.current_node_id)
    has_parent = parent_id is not None
    parent_embed = (
        node_lookup.node_centroid(parent_id) if has_parent else np.zeros(embedding_dim, dtype="float32")
    )

    child_ids = node_lookup.children_of(state.current_node_id)
    has_children = len(child_ids) > 0
    if has_children:
        child_mean_embed = _l2_normalize_vec(
            _safe_mean(node_lookup.node_centroid_matrix(child_ids))
        )
    else:
        child_mean_embed = np.zeros(embedding_dim, dtype="float32")

    has_path = len(state.path_node_ids) > 0
    if has_path:
        path_embed = _l2_normalize_vec(
            _safe_mean(node_lookup.node_centroid_matrix(state.path_node_ids))
        )
    else:
        path_embed = np.zeros(embedding_dim, dtype="float32")

    has_evidence = len(state.evidence_doc_ids) > 0
    if has_evidence:
        evidence_embed = _l2_normalize_vec(
            _safe_mean(node_lookup.doc_embedding_matrix(state.evidence_doc_ids))
        )
    else:
        evidence_embed = np.zeros(embedding_dim, dtype="float32")

    is_leaf = float(state.is_leaf)
    max_depth = max(1, int(state.max_depth))
    depth_norm = min(float(state.depth) / float(max_depth), 1.0)
    log_subtree = float(np.log1p(max(0, state.subtree_doc_count)))
    n_visited_norm = min(len(state.path_node_ids) / 16.0, 1.0)
    n_evidence_norm = min(len(state.evidence_doc_ids) / 100.0, 1.0)
    retrieve_calls_norm = min(state.retrieve_calls / 8.0, 1.0)
    jump_calls_norm = min(state.jump_calls / 16.0, 1.0)
    doc_budget_norm = (
        state.remaining_doc_budget / max(1, state.max_doc_budget)
    )
    call_budget_norm = (
        state.remaining_call_budget / max(1, state.max_call_budget)
    )
    step_norm = min(state.step_index / 16.0, 1.0)

    if has_evidence:
        subtree_doc_set = node_lookup.subtree_doc_id_set(state.current_node_id)
        intersection = sum(
            1 for d in state.evidence_doc_ids if d in subtree_doc_set
        )
        fraction_in_subtree = intersection / max(1, len(state.evidence_doc_ids))
    else:
        fraction_in_subtree = 0.0

    current_max_node_sim = float(np.dot(q, n_embed))

    meta = np.array(
        [
            depth_norm,
            log_subtree,
            is_leaf,
            float(has_parent),
            float(has_children),
            float(has_path),
            float(has_evidence),
            n_visited_norm,
            n_evidence_norm,
            retrieve_calls_norm,
            jump_calls_norm,
            doc_budget_norm,
            call_budget_norm,
            step_norm,
            fraction_in_subtree,
            current_max_node_sim,
        ],
        dtype="float32",
    )
    if meta.shape[0] != META_DIM:
        raise AssertionError("META_FEATURE_NAMES out of sync with computation")

    x = np.concatenate(
        [q, n_embed, parent_embed, child_mean_embed, path_embed, evidence_embed, meta],
        axis=0,
    ).astype("float32")
    if x.shape[0] != expected:
        raise AssertionError(f"state tensor shape {x.shape} != expected {expected}")
    return x


class NodeFeatureLookup:
    """Cached lookup over a ``StandaloneHierarchyIndex`` + ``embeddings.npy``.

    The lookup is read-only and shared across many state-tensor calls. It does
    *not* construct any tensors; it simply returns numpy views or copies of the
    underlying centroid / embedding arrays.
    """

    def __init__(self, hierarchy_index, doc_embeddings: Optional[np.ndarray] = None):
        self.hierarchy_index = hierarchy_index
        embedding_dim = self._infer_embedding_dim(hierarchy_index)
        self.embedding_dim = embedding_dim

        self._node_centroid: Dict[str, np.ndarray] = {}
        for node in hierarchy_index.nodes:
            if node.centroid is not None:
                self._node_centroid[node.node_id] = _l2_normalize_vec(
                    np.asarray(node.centroid, dtype="float32")
                )

        self._parent: Dict[str, Optional[str]] = {
            node.node_id: node.parent_id for node in hierarchy_index.nodes
        }
        self._children: Dict[str, List[str]] = {
            node.node_id: list(node.children) for node in hierarchy_index.nodes
        }
        self._member_doc_ids: Dict[str, List[str]] = {
            node.node_id: list(node.member_doc_ids) for node in hierarchy_index.nodes
        }
        self._subtree_doc_set: Dict[str, set] = {
            node.node_id: set(node.member_doc_ids) for node in hierarchy_index.nodes
        }
        self._depth: Dict[str, int] = {
            node.node_id: int(node.depth) for node in hierarchy_index.nodes
        }

        self._doc_row: Dict[str, int] = {
            doc.doc_id: idx for idx, doc in enumerate(hierarchy_index.documents)
        }
        if doc_embeddings is None:
            doc_embeddings = getattr(hierarchy_index, "embeddings", None)
        self._doc_embeddings = (
            np.ascontiguousarray(doc_embeddings.astype("float32"))
            if doc_embeddings is not None
            else None
        )

    @staticmethod
    def _infer_embedding_dim(hierarchy_index) -> int:
        for node in hierarchy_index.nodes:
            if node.centroid is not None:
                return len(node.centroid)
        emb = getattr(hierarchy_index, "embeddings", None)
        if emb is not None and emb.ndim == 2:
            return int(emb.shape[1])
        raise ValueError("Cannot infer embedding dim from hierarchy index")

    def node_centroid(self, node_id: Optional[str]) -> np.ndarray:
        if node_id is None:
            return np.zeros(self.embedding_dim, dtype="float32")
        vec = self._node_centroid.get(node_id)
        if vec is None:
            return np.zeros(self.embedding_dim, dtype="float32")
        return vec

    def node_centroid_matrix(self, node_ids: Sequence[str]) -> np.ndarray:
        if not node_ids:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        out = np.zeros((len(node_ids), self.embedding_dim), dtype="float32")
        for i, nid in enumerate(node_ids):
            vec = self._node_centroid.get(nid)
            if vec is not None:
                out[i] = vec
        return out

    def parent_of(self, node_id: str) -> Optional[str]:
        return self._parent.get(node_id)

    def children_of(self, node_id: str) -> List[str]:
        return self._children.get(node_id, [])

    def depth_of(self, node_id: str) -> int:
        return self._depth.get(node_id, 0)

    def member_doc_ids(self, node_id: str) -> List[str]:
        return self._member_doc_ids.get(node_id, [])

    def subtree_doc_id_set(self, node_id: str) -> set:
        return self._subtree_doc_set.get(node_id, set())

    def doc_embedding(self, doc_id: str) -> np.ndarray:
        if self._doc_embeddings is None:
            return np.zeros(self.embedding_dim, dtype="float32")
        row = self._doc_row.get(doc_id)
        if row is None:
            return np.zeros(self.embedding_dim, dtype="float32")
        return self._doc_embeddings[row]

    def doc_embedding_matrix(self, doc_ids: Sequence[str]) -> np.ndarray:
        if self._doc_embeddings is None or not doc_ids:
            return np.zeros((len(doc_ids), self.embedding_dim), dtype="float32")
        rows = []
        for did in doc_ids:
            row = self._doc_row.get(did)
            if row is None:
                rows.append(np.zeros(self.embedding_dim, dtype="float32"))
            else:
                rows.append(self._doc_embeddings[row])
        return np.stack(rows, axis=0).astype("float32")

    def is_leaf(self, node_id: str) -> bool:
        return len(self._children.get(node_id, [])) == 0

    def state_tensor_dim(self) -> int:
        return 6 * self.embedding_dim + META_DIM

    def make_state(
        self,
        query: str,
        current_node_id: str,
        path_node_ids: Optional[Sequence[str]] = None,
        evidence_doc_ids: Optional[Sequence[str]] = None,
        step_index: int = 0,
        retrieve_calls: int = 0,
        jump_calls: int = 0,
        aggregate_calls: int = 0,
        remaining_doc_budget: int = 100,
        remaining_call_budget: int = 16,
        max_doc_budget: int = 100,
        max_call_budget: int = 16,
    ) -> PolicyState:
        return PolicyState(
            query=query,
            current_node_id=current_node_id,
            depth=self.depth_of(current_node_id),
            is_leaf=self.is_leaf(current_node_id),
            subtree_doc_count=len(self.member_doc_ids(current_node_id)),
            path_node_ids=list(path_node_ids or []),
            evidence_doc_ids=list(evidence_doc_ids or []),
            step_index=step_index,
            retrieve_calls=retrieve_calls,
            jump_calls=jump_calls,
            aggregate_calls=aggregate_calls,
            remaining_doc_budget=remaining_doc_budget,
            remaining_call_budget=remaining_call_budget,
            max_doc_budget=max_doc_budget,
            max_call_budget=max_call_budget,
            max_depth=max(1, int(self.hierarchy_index.max_depth)),
        )
