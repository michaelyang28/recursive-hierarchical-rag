"""End-to-end inference loop (plan section 5).

The :class:`AgenticRetrieverV2` consumes a trained checkpoint plus a hierarchy
index and runs the supervised policy step-by-step over a query, accumulating
evidence. It implements the five termination guards from section 5.3, the
two-term final ranking blend from section 5.4, and emits per-step traces in
the schema from section 5.5.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .embedding import l2_normalize, make_text_encoder
from .network import (
    ACTION_AGGREGATE,
    ACTION_JUMP,
    ACTION_NAMES,
    ACTION_RETRIEVE,
    PolicyConfig,
    PolicyNetwork,
)
from .node_ann import NodeCentroidIndex
from .state import NodeFeatureLookup, PolicyState, build_state_tensor
from .training import build_model_from_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


GUARD_NONE = ""
GUARD_DONE = "done"
GUARD_MAX_STEPS = "max_steps"
GUARD_DOC_BUDGET = "doc_budget"
GUARD_LOOP = "loop_detected"
GUARD_FORCED_ROOT_AGGREGATE = "forced_root_aggregate"
GUARD_EMPTY_INTERNAL_RETRIEVE = "empty_internal_retrieve"
GUARD_EMPTY_AT_END = "empty_evidence_fallback"


@dataclass
class InferenceConfig:
    """Inference-time hyperparameters (plan section 5.2 / 5.3 / 5.4)."""

    max_steps: int = 16
    max_docs_budget: int = 300
    max_call_budget: int = 32
    K_max: int = 32
    M_max: int = 256
    ann_K: int = 24
    retrieve_top_m: int = 10
    tau_done: float = 0.5
    min_evidence_to_stop: int = 5
    final_alpha: float = 0.6
    loop_window: int = 6
    loop_repeat_threshold: int = 3
    retrieve_internal_threshold: int = 250
    enable_done_head: bool = True
    fallback_top_k: int = 100
    exclude_visited_jump_candidates: bool = True
    retrieve_action_bias: float = 0.0
    retrieve_at_leaf_bias: float = 0.0
    dense_augment_top_k: int = 0
    rrf_k: int = 0
    rrf_dense_top_k: int = 100


@dataclass
class StepTrace:
    step: int
    state: Dict[str, Any]
    action: str
    action_logits: List[float]
    done_prob: float
    jump: Optional[Dict[str, Any]]
    retrieve: Optional[Dict[str, Any]]
    guard_triggered: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state,
            "action": self.action,
            "action_logits": self.action_logits,
            "done_prob": self.done_prob,
            "jump": self.jump,
            "retrieve": self.retrieve,
            "guard_triggered": self.guard_triggered,
        }


@dataclass
class RetrievalResult:
    query_id: Optional[str]
    documents: List[Dict[str, Any]]
    docs_spent: int
    calls_spent: int
    traces: List[StepTrace]
    final_node_id: Optional[str]
    terminated_reason: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "documents": self.documents,
            "docs_spent": self.docs_spent,
            "calls_spent": self.calls_spent,
            "final_node_id": self.final_node_id,
            "terminated_reason": self.terminated_reason,
            "latency_ms": self.latency_ms,
            "traces": [t.to_dict() for t in self.traces],
        }


class AgenticRetrieverV2:
    """Run a trained policy over a hierarchy and return ranked documents."""

    def __init__(
        self,
        model: PolicyNetwork,
        hierarchy_index,
        lookup: NodeFeatureLookup,
        node_ann: NodeCentroidIndex,
        text_encoder=None,
        embedding_model: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.hierarchy_index = hierarchy_index
        self.lookup = lookup
        self.node_ann = node_ann
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cpu")

        if text_encoder is None:
            backbone = embedding_model or str(
                hierarchy_index.config.get("embedding_model", "local-hash-embedding")
            )
            text_encoder = make_text_encoder(backbone, lookup.embedding_dim)
        self.text_encoder = text_encoder
        self.root_id = self._infer_root_id(hierarchy_index)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _infer_root_id(hierarchy_index) -> str:
        for node in hierarchy_index.nodes:
            if node.parent_id is None:
                return node.node_id
        raise ValueError("Hierarchy has no root")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        hierarchy_index,
        lookup: NodeFeatureLookup,
        node_ann: NodeCentroidIndex,
        config: Optional[InferenceConfig] = None,
        device: str = "cpu",
    ) -> "AgenticRetrieverV2":
        payload = load_checkpoint(checkpoint_path, map_location=device)
        model = build_model_from_checkpoint(payload)
        return cls(
            model=model,
            hierarchy_index=hierarchy_index,
            lookup=lookup,
            node_ann=node_ann,
            config=config,
            device=torch.device(device),
        )

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.text_encoder([query])[0]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return vec.astype("float32")

    def _make_state(
        self,
        query: str,
        node_id: str,
        path: Sequence[str],
        evidence_doc_ids: Sequence[str],
        step_index: int,
        retrieve_calls: int,
        jump_calls: int,
        aggregate_calls: int,
    ) -> PolicyState:
        cfg = self.config
        return PolicyState(
            query=query,
            current_node_id=node_id,
            depth=self.lookup.depth_of(node_id),
            is_leaf=self.lookup.is_leaf(node_id),
            subtree_doc_count=len(self.lookup.member_doc_ids(node_id)),
            path_node_ids=list(path),
            evidence_doc_ids=list(evidence_doc_ids),
            step_index=step_index,
            retrieve_calls=retrieve_calls,
            jump_calls=jump_calls,
            aggregate_calls=aggregate_calls,
            remaining_doc_budget=max(0, cfg.max_docs_budget - len(evidence_doc_ids)),
            remaining_call_budget=max(
                0,
                cfg.max_call_budget - retrieve_calls - jump_calls - aggregate_calls,
            ),
            max_doc_budget=cfg.max_docs_budget,
            max_call_budget=cfg.max_call_budget,
            max_depth=max(1, int(self.hierarchy_index.max_depth)),
        )

    def _build_jump_candidates(
        self,
        current_node: str,
        query_vec: np.ndarray,
        visited: Set[str],
    ) -> Tuple[List[str], List[str]]:
        """Return ``(candidate_ids, sources)`` (no random nodes at inference)."""

        cfg = self.config
        cand_ids: List[str] = []
        sources: List[str] = []
        seen: Set[str] = set()

        def add(nid: str, source: str) -> None:
            if nid in seen or nid not in self.lookup._parent:
                return
            if cfg.exclude_visited_jump_candidates and nid in visited:
                return
            seen.add(nid)
            cand_ids.append(nid)
            sources.append(source)

        ann = self.node_ann.top_k(query_vec, cfg.ann_K)
        for nid, _ in ann:
            if len(cand_ids) >= cfg.K_max:
                break
            add(nid, "ann")

        for child in self.lookup.children_of(current_node):
            if len(cand_ids) >= cfg.K_max:
                break
            add(child, "child")

        parent = self.lookup.parent_of(current_node)
        if parent is not None and len(cand_ids) < cfg.K_max:
            add(parent, "parent")

        if parent is not None:
            for sib in self.lookup.children_of(parent):
                if sib == current_node:
                    continue
                if len(cand_ids) >= cfg.K_max:
                    break
                add(sib, "sibling")

        return cand_ids[: cfg.K_max], sources[: cfg.K_max]

    def _score_jump(
        self,
        h: torch.Tensor,
        candidate_ids: Sequence[str],
        query_vec: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        cfg = self.config
        K = cfg.K_max
        cand_emb = np.zeros((1, K, self.lookup.embedding_dim), dtype="float32")
        cand_sim = np.zeros((1, K, 1), dtype="float32")
        cand_mask = np.zeros((1, K), dtype="float32")
        for i, cid in enumerate(candidate_ids[:K]):
            vec = self.lookup.node_centroid(cid)
            cand_emb[0, i] = vec
            cand_sim[0, i, 0] = float(np.dot(vec, query_vec))
            cand_mask[0, i] = 1.0

        cand_emb_t = torch.from_numpy(cand_emb).to(self.device)
        cand_sim_t = torch.from_numpy(cand_sim).to(self.device)
        cand_mask_t = torch.from_numpy(cand_mask).to(self.device)
        scores = self.model.jump_scores(h, cand_emb_t, cand_sim_t, candidate_mask=cand_mask_t)
        return scores[0].detach().cpu(), cand_mask[0]

    def _score_retrieve(
        self,
        h: torch.Tensor,
        chunk_doc_ids: Sequence[str],
        query_vec: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
        cfg = self.config
        M = cfg.M_max
        chunk_emb = np.zeros((1, M, self.lookup.embedding_dim), dtype="float32")
        chunk_sim = np.zeros((1, M, 1), dtype="float32")
        chunk_mask = np.zeros((1, M), dtype="float32")

        if len(chunk_doc_ids) > M:
            mat = self.lookup.doc_embedding_matrix(list(chunk_doc_ids))
            mat_n = l2_normalize(mat) if mat.size else mat
            sims = mat_n @ query_vec.astype("float32") if mat_n.size else np.zeros(0)
            order = np.argsort(-sims) if sims.size else np.array([], dtype=int)
            kept = [chunk_doc_ids[int(i)] for i in order[:M]]
            chunk_doc_ids = kept

        chunk_doc_ids = list(chunk_doc_ids)
        for i, did in enumerate(chunk_doc_ids[:M]):
            vec = self.lookup.doc_embedding(did)
            n = float(np.linalg.norm(vec))
            if n > 1e-12:
                vec = vec / n
            chunk_emb[0, i] = vec
            chunk_sim[0, i, 0] = float(np.dot(vec, query_vec))
            chunk_mask[0, i] = 1.0

        chunk_emb_t = torch.from_numpy(chunk_emb).to(self.device)
        chunk_sim_t = torch.from_numpy(chunk_sim).to(self.device)
        chunk_mask_t = torch.from_numpy(chunk_mask).to(self.device)
        scores = self.model.retrieve_scores(
            h, chunk_emb_t, chunk_sim_t, chunk_mask=chunk_mask_t
        )
        return scores[0].detach().cpu(), chunk_mask[0], chunk_doc_ids

    def _final_ranking(
        self,
        evidence: List[Dict[str, Any]],
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        per_doc: Dict[str, Dict[str, Any]] = {}
        for record in evidence:
            doc_id = record["doc_id"]
            keep = per_doc.get(doc_id)
            if keep is None or record["retrieve_score"] > keep["retrieve_score"]:
                per_doc[doc_id] = dict(record)

        if self.config.rrf_k > 0:
            return self._rrf_final_ranking(per_doc, query_vec, top_k)

        if self.config.dense_augment_top_k > 0:
            for dense_rec in self._fallback_top_docs(
                query_vec, self.config.dense_augment_top_k
            ):
                doc_id = dense_rec["doc_id"]
                if doc_id not in per_doc:
                    per_doc[doc_id] = {
                        "doc_id": doc_id,
                        "retrieve_score": 0.0,
                        "source_node_id": "dense_augment",
                    }

        if not per_doc:
            return []

        scores = np.array(
            [r["retrieve_score"] for r in per_doc.values()], dtype="float32"
        )
        if scores.size > 0 and scores.std() > 1e-9:
            z = (scores - scores.mean()) / max(1e-9, scores.std())
        else:
            z = np.zeros_like(scores)

        for record, z_val in zip(per_doc.values(), z):
            sim = float(np.dot(self.lookup.doc_embedding(record["doc_id"]), query_vec))
            n = float(np.linalg.norm(self.lookup.doc_embedding(record["doc_id"])))
            if n > 1e-12:
                sim = sim / n
            record["sim"] = sim
            record["final_score"] = (
                self.config.final_alpha * sim + (1.0 - self.config.final_alpha) * float(z_val)
            )

        ranked = sorted(per_doc.values(), key=lambda r: r["final_score"], reverse=True)
        return ranked[:top_k]

    def _rrf_final_ranking(
        self,
        policy_docs: Dict[str, Dict[str, Any]],
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        k = float(self.config.rrf_k)
        policy_ranked = sorted(
            policy_docs.values(), key=lambda r: r.get("retrieve_score", 0.0), reverse=True
        )
        policy_rank: Dict[str, int] = {
            r["doc_id"]: i + 1 for i, r in enumerate(policy_ranked)
        }

        dense_recs = self._fallback_top_docs(query_vec, self.config.rrf_dense_top_k)
        dense_rank: Dict[str, int] = {
            r["doc_id"]: i + 1 for i, r in enumerate(dense_recs)
        }

        all_doc_ids = set(policy_rank) | set(dense_rank)
        if not all_doc_ids:
            return []

        merged: Dict[str, Dict[str, Any]] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in policy_rank:
                score += 1.0 / (k + policy_rank[doc_id])
            if doc_id in dense_rank:
                score += 1.0 / (k + dense_rank[doc_id])
            base = policy_docs.get(doc_id) or {}
            merged[doc_id] = {
                "doc_id": doc_id,
                "retrieve_score": float(base.get("retrieve_score", 0.0)),
                "source_node_id": base.get("source_node_id", "rrf"),
                "sim": 0.0,
                "final_score": float(score),
            }

        ranked = sorted(merged.values(), key=lambda r: r["final_score"], reverse=True)
        return ranked[:top_k]

    def _fallback_top_docs(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.lookup._doc_embeddings is None:
            return []
        mat = self.lookup._doc_embeddings
        mat_n = l2_normalize(mat)
        sims = mat_n @ query_vec
        order = np.argsort(-sims)[:top_k]
        results: List[Dict[str, Any]] = []
        doc_ids = list(self.lookup._doc_row.keys())
        for idx in order:
            doc_id = self.hierarchy_index.documents[int(idx)].doc_id
            results.append(
                {
                    "doc_id": doc_id,
                    "retrieve_score": float(sims[int(idx)]),
                    "source_node_id": None,
                    "sim": float(sims[int(idx)]),
                    "final_score": float(sims[int(idx)]),
                }
            )
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        query_id: Optional[str] = None,
    ) -> RetrievalResult:
        cfg = self.config
        start_t = time.time()
        query_vec = self._embed_query(query)

        node_id = self.root_id
        path: List[str] = [self.root_id]
        evidence_records: List[Dict[str, Any]] = []
        evidence_doc_ids: List[str] = []
        step_index = 0
        retrieve_calls = 0
        jump_calls = 0
        aggregate_calls = 0
        recent = deque(maxlen=cfg.loop_window)
        recent.append(self.root_id)
        terminated_reason = "max_steps"

        traces: List[StepTrace] = []

        while step_index < cfg.max_steps:
            state = self._make_state(
                query=query,
                node_id=node_id,
                path=path,
                evidence_doc_ids=evidence_doc_ids,
                step_index=step_index,
                retrieve_calls=retrieve_calls,
                jump_calls=jump_calls,
                aggregate_calls=aggregate_calls,
            )
            x = build_state_tensor(state, query_vec, self.lookup)
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(x_t)
            h = out["h"]
            action_logits = out["action_logits"][0].detach().cpu().tolist()
            done_logit = float(out["done_logit"][0].detach().cpu())
            done_prob = float(torch.sigmoid(torch.tensor(done_logit)).item())

            guard = GUARD_NONE
            if (
                cfg.enable_done_head
                and done_prob > cfg.tau_done
                and len(evidence_doc_ids) >= cfg.min_evidence_to_stop
            ):
                guard = GUARD_DONE
                traces.append(
                    StepTrace(
                        step=step_index,
                        state=state.as_state_dict(),
                        action="DONE",
                        action_logits=action_logits,
                        done_prob=done_prob,
                        jump=None,
                        retrieve=None,
                        guard_triggered=guard,
                    )
                )
                terminated_reason = guard
                break

            biased_logits = list(action_logits)
            if cfg.retrieve_action_bias != 0.0:
                biased_logits[ACTION_RETRIEVE] += cfg.retrieve_action_bias
            if cfg.retrieve_at_leaf_bias != 0.0 and self.lookup.is_leaf(node_id):
                biased_logits[ACTION_RETRIEVE] += cfg.retrieve_at_leaf_bias
            action = int(np.argmax(biased_logits))
            forced = False
            if action == ACTION_AGGREGATE and node_id == self.root_id:
                action = ACTION_JUMP
                forced = True
                guard = GUARD_FORCED_ROOT_AGGREGATE
            if (
                action == ACTION_RETRIEVE
                and len(self.lookup.member_doc_ids(node_id)) > cfg.retrieve_internal_threshold
                and not self.lookup.is_leaf(node_id)
            ):
                action = ACTION_JUMP
                forced = True
                guard = GUARD_EMPTY_INTERNAL_RETRIEVE

            jump_block: Optional[Dict[str, Any]] = None
            retrieve_block: Optional[Dict[str, Any]] = None

            if action == ACTION_JUMP:
                cand_ids, cand_sources = self._build_jump_candidates(
                    node_id, query_vec, set(path)
                )
                if not cand_ids:
                    terminated_reason = GUARD_EMPTY_INTERNAL_RETRIEVE
                    traces.append(
                        StepTrace(
                            step=step_index,
                            state=state.as_state_dict(),
                            action="DONE",
                            action_logits=action_logits,
                            done_prob=done_prob,
                            jump=None,
                            retrieve=None,
                            guard_triggered="no_jump_candidates",
                        )
                    )
                    break
                scores, cand_mask = self._score_jump(h, cand_ids, query_vec)
                scores_np = scores.numpy()
                masked_scores = np.where(cand_mask > 0, scores_np, -np.inf)
                best_idx = int(np.argmax(masked_scores))
                next_node = cand_ids[best_idx]
                jump_block = {
                    "candidate_node_ids": list(cand_ids),
                    "candidate_sources": list(cand_sources),
                    "scores": [float(s) for s in scores_np[: len(cand_ids)]],
                    "chosen_index": best_idx,
                    "chosen_node_id": next_node,
                }
                node_id = next_node
                path.append(node_id)
                jump_calls += 1
            elif action == ACTION_AGGREGATE:
                parent = self.lookup.parent_of(node_id)
                if parent is None:
                    terminated_reason = GUARD_FORCED_ROOT_AGGREGATE
                    break
                node_id = parent
                path.append(node_id)
                aggregate_calls += 1
            elif action == ACTION_RETRIEVE:
                member = self.lookup.member_doc_ids(node_id)
                if not member:
                    guard = GUARD_EMPTY_INTERNAL_RETRIEVE
                    traces.append(
                        StepTrace(
                            step=step_index,
                            state=state.as_state_dict(),
                            action="FORCED_JUMP",
                            action_logits=action_logits,
                            done_prob=done_prob,
                            jump=None,
                            retrieve=None,
                            guard_triggered=guard,
                        )
                    )
                    step_index += 1
                    recent.append(node_id)
                    continue
                scores, chunk_mask, chunk_ids = self._score_retrieve(h, member, query_vec)
                scores_np = scores.numpy()
                masked_scores = np.where(chunk_mask > 0, scores_np, -np.inf)
                top_m = min(cfg.retrieve_top_m, int((chunk_mask > 0).sum()))
                if top_m > 0:
                    order = np.argsort(-masked_scores)[:top_m]
                else:
                    order = np.array([], dtype=int)
                selected_ids: List[str] = []
                for idx in order:
                    did = chunk_ids[int(idx)]
                    if not did or did in {r["doc_id"] for r in evidence_records}:
                        continue
                    score = float(scores_np[int(idx)])
                    evidence_records.append(
                        {
                            "doc_id": did,
                            "retrieve_score": score,
                            "source_node_id": node_id,
                        }
                    )
                    evidence_doc_ids.append(did)
                    selected_ids.append(did)
                retrieve_block = {
                    "chunk_doc_ids": list(chunk_ids[:32]),
                    "scores": [float(s) for s in scores_np[: min(32, len(chunk_ids))]],
                    "selected_doc_ids": selected_ids,
                }
                retrieve_calls += 1

            recent.append(node_id)
            count_recent = sum(1 for n in recent if n == node_id)
            if count_recent >= cfg.loop_repeat_threshold:
                guard = GUARD_LOOP
                traces.append(
                    StepTrace(
                        step=step_index,
                        state=state.as_state_dict(),
                        action="FORCED_JUMP" if forced else ACTION_NAMES[action],
                        action_logits=action_logits,
                        done_prob=done_prob,
                        jump=jump_block,
                        retrieve=retrieve_block,
                        guard_triggered=guard,
                    )
                )
                terminated_reason = guard
                break

            if len(evidence_doc_ids) >= cfg.max_docs_budget:
                guard = GUARD_DOC_BUDGET
                traces.append(
                    StepTrace(
                        step=step_index,
                        state=state.as_state_dict(),
                        action="FORCED_JUMP" if forced else ACTION_NAMES[action],
                        action_logits=action_logits,
                        done_prob=done_prob,
                        jump=jump_block,
                        retrieve=retrieve_block,
                        guard_triggered=guard,
                    )
                )
                terminated_reason = guard
                break

            traces.append(
                StepTrace(
                    step=step_index,
                    state=state.as_state_dict(),
                    action="FORCED_JUMP" if forced else ACTION_NAMES[action],
                    action_logits=action_logits,
                    done_prob=done_prob,
                    jump=jump_block,
                    retrieve=retrieve_block,
                    guard_triggered=guard,
                )
            )
            step_index += 1
        else:
            terminated_reason = GUARD_MAX_STEPS

        ranked_docs = self._final_ranking(evidence_records, query_vec, top_k)
        if not ranked_docs:
            ranked_docs = self._fallback_top_docs(query_vec, top_k)
            if not ranked_docs:
                pass
            terminated_reason = (
                terminated_reason if terminated_reason != GUARD_DONE else GUARD_EMPTY_AT_END
            )
        latency_ms = (time.time() - start_t) * 1000.0
        return RetrievalResult(
            query_id=query_id,
            documents=ranked_docs,
            docs_spent=len(evidence_doc_ids),
            calls_spent=retrieve_calls + jump_calls + aggregate_calls,
            traces=traces,
            final_node_id=node_id,
            terminated_reason=terminated_reason,
            latency_ms=latency_ms,
        )
