"""Trajectory-aligned supervision generation.

Implements plan sections 3.1-3.13. The :class:`SupervisionGenerator` walks a
deterministic oracle trajectory for each query, emitting one example per step
plus optional off-path "recovery" samples. Every emitted row satisfies the
section 3.13 invariants (asserted at write time).

This module never touches PyTorch -- the artifacts are JSON-serializable and
fed into the dataset module separately.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .embedding import l2_normalize
from .network import ACTION_AGGREGATE, ACTION_JUMP, ACTION_RETRIEVE
from .node_ann import NodeCentroidIndex
from .state import NodeFeatureLookup, PolicyState

logger = logging.getLogger(__name__)


@dataclass
class SupervisionConfig:
    """Hyperparameters for trajectory rollout and example generation."""

    ann_K: int = 24
    K_max: int = 32
    M_max: int = 256
    n_off_path: int = 2
    retrieve_threshold: int = 250
    tau_done: float = 1.0
    n_random_jump_candidates: int = 6
    retrieve_action_repeats: int = 1
    max_steps_per_trajectory: int = 32
    max_doc_budget: int = 100
    max_call_budget: int = 16
    n_trajectories_per_query: int = 1
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ann_K": self.ann_K,
            "K_max": self.K_max,
            "M_max": self.M_max,
            "n_off_path": self.n_off_path,
            "retrieve_threshold": self.retrieve_threshold,
            "tau_done": self.tau_done,
            "n_random_jump_candidates": self.n_random_jump_candidates,
            "retrieve_action_repeats": self.retrieve_action_repeats,
            "max_steps_per_trajectory": self.max_steps_per_trajectory,
            "max_doc_budget": self.max_doc_budget,
            "max_call_budget": self.max_call_budget,
            "n_trajectories_per_query": self.n_trajectories_per_query,
            "seed": self.seed,
        }


@dataclass
class SupervisionExample:
    """One JSONL row matching the plan-3.11 schema."""

    query_id: str
    query: str
    step_index: int
    trajectory_id: str
    is_off_path: bool
    state: Dict[str, Any]
    action_label: int
    done_label: int
    jump: Optional[Dict[str, Any]]
    retrieve: Optional[Dict[str, Any]]
    debug: Dict[str, Any]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "step_index": self.step_index,
            "trajectory_id": self.trajectory_id,
            "is_off_path": self.is_off_path,
            "state": self.state,
            "action_label": self.action_label,
            "done_label": self.done_label,
            "jump": self.jump,
            "retrieve": self.retrieve,
            "debug": self.debug,
        }


class SupervisionInvariantError(AssertionError):
    """Raised when a generated example violates the plan-3.13 invariants."""


def assert_example_invariants(example: SupervisionExample, root_id: str) -> None:
    """Plan section 3.13 - validated at write time, not training time."""

    s = example.state
    if s["path_node_ids"] and s["path_node_ids"][0] != root_id:
        raise SupervisionInvariantError(
            f"path[0]={s['path_node_ids'][0]} != root={root_id}"
        )
    if s["path_node_ids"] and s["path_node_ids"][-1] != s["node_id"]:
        raise SupervisionInvariantError(
            f"path[-1]={s['path_node_ids'][-1]} != node={s['node_id']}"
        )

    if example.action_label == ACTION_AGGREGATE and s["node_id"] == root_id:
        raise SupervisionInvariantError("AGGREGATE at root is invalid")

    if example.action_label == ACTION_JUMP:
        if example.jump is None:
            raise SupervisionInvariantError("JUMP row missing jump block")
        positives = example.jump.get("positive_indices") or []
        if not positives:
            raise SupervisionInvariantError("JUMP row has no positive indices")
        K = len(example.jump.get("candidate_node_ids") or [])
        mask = example.jump.get("candidate_mask") or []
        for idx in positives:
            if not (0 <= idx < K):
                raise SupervisionInvariantError(
                    f"positive index {idx} out of range [0, {K})"
                )
            if idx >= len(mask) or not mask[idx]:
                raise SupervisionInvariantError(
                    f"positive index {idx} points to a masked candidate"
                )

    if example.action_label == ACTION_RETRIEVE:
        target_nodes = example.debug.get("target_node_ids") or []
        if s["node_id"] not in target_nodes:
            raise SupervisionInvariantError(
                f"RETRIEVE at non-target node {s['node_id']}; targets={target_nodes}"
            )

    if example.done_label == 1:
        evidence = set(s.get("evidence_doc_ids") or [])
        relevant = set(example.debug.get("relevant_doc_ids") or [])
        if relevant:
            recall = len(evidence & relevant) / len(relevant)
            tau = float(example.debug.get("tau_done", 1.0))
            if recall + 1e-9 < tau:
                raise SupervisionInvariantError(
                    f"done=1 with recall={recall:.3f} < tau_done={tau}"
                )


def ancestors_of(node_id: str, lookup: NodeFeatureLookup) -> List[str]:
    """Root-to-node chain (inclusive of both)."""

    chain: List[str] = []
    current: Optional[str] = node_id
    while current is not None:
        chain.append(current)
        current = lookup.parent_of(current)
    chain.reverse()
    return chain


def lca_of(a: str, b: str, lookup: NodeFeatureLookup) -> str:
    a_chain = ancestors_of(a, lookup)
    b_set = set(ancestors_of(b, lookup))
    last_common: str = a_chain[0]
    for nid in a_chain:
        if nid in b_set:
            last_common = nid
    return last_common


def descent_path(u: str, t: str, lookup: NodeFeatureLookup) -> List[str]:
    """List of nodes from ``u`` to ``t`` inclusive, assuming ``u`` ancestors ``t``."""

    chain = ancestors_of(t, lookup)
    if u not in chain:
        raise ValueError(f"node {u} is not an ancestor of {t}")
    start = chain.index(u)
    return chain[start:]


def _doc_id_to_leaf_map(lookup: NodeFeatureLookup) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for node in lookup.hierarchy_index.nodes:
        if not node.children:
            for doc_id in node.member_doc_ids:
                mapping[doc_id] = node.node_id
    return mapping


def compute_target_nodes(
    relevant_doc_ids: Iterable[str],
    lookup: NodeFeatureLookup,
    doc_to_leaf: Dict[str, str],
    retrieve_threshold: int,
) -> List[str]:
    """Plan section 3.2.

    For each relevant doc, take its leaf. Optionally collapse a set of
    co-leaves under a small internal ancestor whose subtree size fits the
    retrieve threshold; this lets RETRIEVE happen at internal nodes when many
    relevant docs are clustered together.
    """

    leaves: List[str] = []
    for doc_id in relevant_doc_ids:
        leaf = doc_to_leaf.get(doc_id)
        if leaf is not None:
            leaves.append(leaf)
    if not leaves:
        return []

    targets: List[str] = list(dict.fromkeys(leaves))

    if retrieve_threshold > 0 and len(targets) > 1:
        relevant_set = set(relevant_doc_ids)
        deduped: List[str] = []
        used_subtree: Set[str] = set()
        for leaf in targets:
            current = leaf
            best = leaf
            while True:
                parent = lookup.parent_of(current)
                if parent is None:
                    break
                parent_member = lookup.subtree_doc_id_set(parent)
                if len(parent_member) > retrieve_threshold:
                    break
                if not (parent_member & relevant_set):
                    break
                best = parent
                current = parent
            if best not in used_subtree:
                deduped.append(best)
                used_subtree.add(best)
        targets = deduped

    return list(dict.fromkeys(targets))


def _order_targets_by_query_sim(
    targets: Sequence[str],
    query_vec: np.ndarray,
    node_ann: NodeCentroidIndex,
) -> List[str]:
    if len(targets) <= 1:
        return list(targets)
    sims = []
    for nid in targets:
        v = node_ann.vector_for(nid)
        sims.append((nid, float(np.dot(query_vec, v))))
    sims.sort(key=lambda kv: kv[1], reverse=True)
    return [nid for nid, _ in sims]


@dataclass
class _RolloutState:
    node: str
    path: List[str]
    evidence: List[str]
    step_index: int = 0
    retrieve_calls: int = 0
    jump_calls: int = 0
    aggregate_calls: int = 0


class SupervisionGenerator:
    """Roll out one or more deterministic oracle trajectories per query."""

    def __init__(
        self,
        hierarchy_index,
        lookup: NodeFeatureLookup,
        node_ann: NodeCentroidIndex,
        config: Optional[SupervisionConfig] = None,
    ):
        self.hierarchy_index = hierarchy_index
        self.lookup = lookup
        self.node_ann = node_ann
        self.config = config or SupervisionConfig()
        self.root_id = self._infer_root_id(hierarchy_index)
        self.doc_to_leaf = _doc_id_to_leaf_map(lookup)
        self._all_node_ids = [n.node_id for n in hierarchy_index.nodes]
        self._rng = random.Random(self.config.seed)

    @staticmethod
    def _infer_root_id(hierarchy_index) -> str:
        for node in hierarchy_index.nodes:
            if node.parent_id is None:
                return node.node_id
        raise ValueError("Hierarchy has no root")

    def generate_for_query(
        self,
        query_id: str,
        query: str,
        relevant_doc_ids: Sequence[str],
        query_vec: np.ndarray,
    ) -> List[SupervisionExample]:
        relevant_doc_ids = [d for d in relevant_doc_ids if d in self.lookup._doc_row]
        if not relevant_doc_ids:
            return []

        targets = compute_target_nodes(
            relevant_doc_ids,
            self.lookup,
            self.doc_to_leaf,
            self.config.retrieve_threshold,
        )
        if not targets:
            return []

        n_traj = max(1, int(self.config.n_trajectories_per_query))
        all_examples: List[SupervisionExample] = []
        for traj_idx in range(n_traj):
            if traj_idx == 0:
                ordered_targets = _order_targets_by_query_sim(
                    targets, query_vec, self.node_ann
                )
            else:
                shuffled = list(targets)
                self._rng.shuffle(shuffled)
                ordered_targets = shuffled
            all_examples.extend(
                self._rollout_one_trajectory(
                    query_id=query_id,
                    query=query,
                    query_vec=query_vec,
                    relevant_doc_ids=relevant_doc_ids,
                    ordered_targets=ordered_targets,
                    traj_index=traj_idx,
                )
            )
        return all_examples

    def _rollout_one_trajectory(
        self,
        query_id: str,
        query: str,
        query_vec: np.ndarray,
        relevant_doc_ids: Sequence[str],
        ordered_targets: List[str],
        traj_index: int,
    ) -> List[SupervisionExample]:
        traj_id = f"{query_id}:{traj_index}"
        examples: List[SupervisionExample] = []

        rollout = _RolloutState(node=self.root_id, path=[self.root_id], evidence=[])
        relevant_set = set(relevant_doc_ids)
        retrieved_targets: Set[str] = set()

        for target_idx, target in enumerate(ordered_targets):
            if target not in ancestors_of(target, self.lookup):
                continue
            try:
                path_to_t = descent_path(rollout.node, target, self.lookup)
            except ValueError:
                lca = lca_of(rollout.node, target, self.lookup)
                while rollout.node != lca:
                    self._emit_aggregate_step(
                        examples,
                        traj_id,
                        rollout,
                        ordered_targets,
                        target,
                        retrieved_targets,
                        relevant_set,
                        query,
                        query_id,
                        query_vec,
                        is_terminal=False,
                    )
                path_to_t = descent_path(rollout.node, target, self.lookup)

            for next_node in path_to_t[1:]:
                self._emit_jump_step(
                    examples,
                    traj_id,
                    rollout,
                    next_node,
                    ordered_targets,
                    target,
                    retrieved_targets,
                    relevant_set,
                    query,
                    query_id,
                    query_vec,
                )

            self._emit_retrieve_step(
                examples,
                traj_id,
                rollout,
                target,
                ordered_targets,
                retrieved_targets,
                relevant_set,
                query,
                query_id,
                query_vec,
            )

            retrieved_targets.add(target)
            remaining = [t for t in ordered_targets if t not in retrieved_targets]
            if remaining:
                next_target = remaining[0]
                lca = lca_of(rollout.node, next_target, self.lookup)
                while rollout.node != lca:
                    self._emit_aggregate_step(
                        examples,
                        traj_id,
                        rollout,
                        ordered_targets,
                        next_target,
                        retrieved_targets,
                        relevant_set,
                        query,
                        query_id,
                        query_vec,
                        is_terminal=False,
                    )

        self._emit_terminal_done_step(
            examples,
            traj_id,
            rollout,
            ordered_targets,
            retrieved_targets,
            relevant_set,
            query,
            query_id,
            query_vec,
        )

        for ex in examples:
            assert_example_invariants(ex, self.root_id)
        return examples

    def _make_state_dict(self, rollout: _RolloutState) -> Dict[str, Any]:
        node_id = rollout.node
        return {
            "node_id": node_id,
            "depth": self.lookup.depth_of(node_id),
            "is_leaf": self.lookup.is_leaf(node_id),
            "subtree_doc_count": len(self.lookup.member_doc_ids(node_id)),
            "path_node_ids": list(rollout.path),
            "evidence_doc_ids": list(rollout.evidence),
            "step_index": rollout.step_index,
            "retrieve_calls": rollout.retrieve_calls,
            "jump_calls": rollout.jump_calls,
            "aggregate_calls": rollout.aggregate_calls,
            "remaining_doc_budget": max(
                0, self.config.max_doc_budget - len(rollout.evidence)
            ),
            "remaining_call_budget": max(
                0,
                self.config.max_call_budget
                - rollout.retrieve_calls
                - rollout.jump_calls
                - rollout.aggregate_calls,
            ),
            "max_doc_budget": self.config.max_doc_budget,
            "max_call_budget": self.config.max_call_budget,
            "max_depth": max(1, int(self.hierarchy_index.max_depth)),
        }

    def _build_jump_candidates(
        self,
        current_node: str,
        positive_node_ids: Iterable[str],
        all_remaining_path_nodes: Set[str],
        query_vec: np.ndarray,
        target_path: Set[str],
    ) -> Dict[str, Any]:
        positives = list(dict.fromkeys(positive_node_ids))
        candidate_ids: List[str] = []
        sources: List[str] = []
        seen: Set[str] = set()

        def add(node_id: str, source: str) -> None:
            if node_id in seen:
                return
            if node_id not in self.lookup._parent:
                return
            seen.add(node_id)
            candidate_ids.append(node_id)
            sources.append(source)

        for pos in positives:
            add(pos, "positive")

        try:
            ann = self.node_ann.top_k(query_vec, self.config.ann_K)
            for nid, _ in ann:
                if len(candidate_ids) >= self.config.K_max:
                    break
                add(nid, "ann")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ANN lookup failed: %s", exc)

        for child in self.lookup.children_of(current_node):
            if len(candidate_ids) >= self.config.K_max:
                break
            add(child, "child")

        parent = self.lookup.parent_of(current_node)
        if parent is not None and len(candidate_ids) < self.config.K_max:
            add(parent, "parent")

        if parent is not None:
            for sib in self.lookup.children_of(parent):
                if sib == current_node:
                    continue
                if len(candidate_ids) >= self.config.K_max:
                    break
                add(sib, "sibling")

        n_random_target = min(
            self.config.n_random_jump_candidates,
            max(0, self.config.K_max - len(candidate_ids)),
        )
        random_pool = [n for n in self._all_node_ids if n not in seen]
        if random_pool and n_random_target > 0:
            picks = self._rng.sample(random_pool, min(n_random_target, len(random_pool)))
            for nid in picks:
                if len(candidate_ids) >= self.config.K_max:
                    break
                add(nid, "random")

        candidate_ids = candidate_ids[: self.config.K_max]
        sources = sources[: self.config.K_max]

        positive_set: Set[str] = set(positives)
        positive_indices: List[int] = []
        for i, nid in enumerate(candidate_ids):
            if nid in positive_set:
                positive_indices.append(i)
            elif (
                nid in target_path
                and nid != current_node
                and current_node in ancestors_of(nid, self.lookup)
            ):
                positive_indices.append(i)

        positive_indices = sorted(set(positive_indices))

        candidate_mask = [1] * len(candidate_ids)
        while len(candidate_ids) < self.config.K_max:
            candidate_ids.append("")
            sources.append("padding")
            candidate_mask.append(0)

        return {
            "candidate_node_ids": candidate_ids,
            "candidate_mask": candidate_mask,
            "candidate_sources": sources,
            "positive_indices": positive_indices,
        }

    def _build_retrieve_block(
        self,
        node_id: str,
        relevant_set: Set[str],
        query_vec: np.ndarray,
    ) -> Dict[str, Any]:
        member = self.lookup.member_doc_ids(node_id)
        positives = [d for d in member if d in relevant_set]

        if len(member) <= self.config.M_max:
            chunk_doc_ids = list(member)
        else:
            non_pos = [d for d in member if d not in relevant_set]
            mat = self.lookup.doc_embedding_matrix(non_pos)
            mat = l2_normalize(mat)
            sims = mat @ query_vec.astype("float32") if mat.size else np.zeros(0)
            order = np.argsort(-sims) if sims.size else np.array([], dtype=int)
            keep_n = max(0, self.config.M_max - len(positives))
            kept_non_pos = [non_pos[int(i)] for i in order[:keep_n]]
            chunk_doc_ids = list(positives) + kept_non_pos
            chunk_doc_ids = chunk_doc_ids[: self.config.M_max]

        positive_set = set(positives)
        positive_indices = sorted(
            i for i, d in enumerate(chunk_doc_ids) if d in positive_set
        )
        chunk_mask = [1] * len(chunk_doc_ids)
        while len(chunk_doc_ids) < self.config.M_max:
            chunk_doc_ids.append("")
            chunk_mask.append(0)
        return {
            "chunk_doc_ids": chunk_doc_ids,
            "chunk_mask": chunk_mask,
            "positive_indices": positive_indices,
        }

    def _debug_block(
        self,
        ordered_targets: Sequence[str],
        current_target: Optional[str],
        relevant_set: Set[str],
    ) -> Dict[str, Any]:
        oracle_path: List[str] = []
        if current_target is not None:
            oracle_path = ancestors_of(current_target, self.lookup)
        return {
            "target_node_ids": list(ordered_targets),
            "current_target_node_id": current_target,
            "oracle_path_node_ids": oracle_path,
            "relevant_doc_ids": sorted(relevant_set),
            "tau_done": float(self.config.tau_done),
        }

    def _compute_done(self, evidence: Sequence[str], relevant_set: Set[str]) -> int:
        if not relevant_set:
            return 0
        recall = len(set(evidence) & relevant_set) / len(relevant_set)
        return 1 if recall + 1e-9 >= self.config.tau_done else 0

    def _emit_jump_step(
        self,
        examples: List[SupervisionExample],
        traj_id: str,
        rollout: _RolloutState,
        next_node: str,
        ordered_targets: Sequence[str],
        current_target: str,
        retrieved_targets: Set[str],
        relevant_set: Set[str],
        query: str,
        query_id: str,
        query_vec: np.ndarray,
    ) -> None:
        target_path = set(ancestors_of(current_target, self.lookup))
        for t in ordered_targets:
            if t not in retrieved_targets:
                target_path.update(ancestors_of(t, self.lookup))

        jump_block = self._build_jump_candidates(
            rollout.node,
            [next_node],
            all_remaining_path_nodes=target_path,
            query_vec=query_vec,
            target_path=target_path,
        )
        if next_node not in jump_block["candidate_node_ids"]:
            jump_block["candidate_node_ids"][-1] = next_node
            jump_block["candidate_mask"][-1] = 1
            jump_block["candidate_sources"][-1] = "positive"
            new_pos = jump_block["candidate_node_ids"].index(next_node)
            if new_pos not in jump_block["positive_indices"]:
                jump_block["positive_indices"].append(new_pos)
                jump_block["positive_indices"].sort()

        example = SupervisionExample(
            query_id=query_id,
            query=query,
            step_index=rollout.step_index,
            trajectory_id=traj_id,
            is_off_path=False,
            state=self._make_state_dict(rollout),
            action_label=ACTION_JUMP,
            done_label=self._compute_done(rollout.evidence, relevant_set),
            jump=jump_block,
            retrieve=None,
            debug=self._debug_block(ordered_targets, current_target, relevant_set),
        )
        examples.append(example)

        for off in self._build_off_path_examples(
            rollout=rollout,
            gold_next=next_node,
            current_target=current_target,
            ordered_targets=ordered_targets,
            retrieved_targets=retrieved_targets,
            relevant_set=relevant_set,
            query=query,
            query_id=query_id,
            traj_id=traj_id,
            query_vec=query_vec,
        ):
            examples.append(off)

        rollout.node = next_node
        rollout.path.append(next_node)
        rollout.step_index += 1
        rollout.jump_calls += 1

    def _emit_retrieve_step(
        self,
        examples: List[SupervisionExample],
        traj_id: str,
        rollout: _RolloutState,
        target: str,
        ordered_targets: Sequence[str],
        retrieved_targets: Set[str],
        relevant_set: Set[str],
        query: str,
        query_id: str,
        query_vec: np.ndarray,
    ) -> None:
        n_repeats = max(1, int(self.config.retrieve_action_repeats))
        for repeat_idx in range(n_repeats):
            debug = self._debug_block(ordered_targets, target, relevant_set)
            debug["retrieve_repeat_index"] = repeat_idx
            debug["retrieve_action_repeats"] = n_repeats
            example = SupervisionExample(
                query_id=query_id,
                query=query,
                step_index=rollout.step_index,
                trajectory_id=traj_id,
                is_off_path=False,
                state=self._make_state_dict(rollout),
                action_label=ACTION_RETRIEVE,
                done_label=self._compute_done(rollout.evidence, relevant_set),
                jump=None,
                retrieve=self._build_retrieve_block(target, relevant_set, query_vec),
                debug=debug,
            )
            examples.append(example)

        member_set = set(self.lookup.member_doc_ids(target))
        new_evidence = [d for d in relevant_set if d in member_set and d not in rollout.evidence]
        rollout.evidence.extend(new_evidence)
        rollout.step_index += 1
        rollout.retrieve_calls += 1

    def _emit_aggregate_step(
        self,
        examples: List[SupervisionExample],
        traj_id: str,
        rollout: _RolloutState,
        ordered_targets: Sequence[str],
        current_target: str,
        retrieved_targets: Set[str],
        relevant_set: Set[str],
        query: str,
        query_id: str,
        query_vec: np.ndarray,
        is_terminal: bool,
    ) -> None:
        if rollout.node == self.root_id:
            return

        example = SupervisionExample(
            query_id=query_id,
            query=query,
            step_index=rollout.step_index,
            trajectory_id=traj_id,
            is_off_path=False,
            state=self._make_state_dict(rollout),
            action_label=ACTION_AGGREGATE,
            done_label=1 if is_terminal else self._compute_done(rollout.evidence, relevant_set),
            jump=None,
            retrieve=None,
            debug=self._debug_block(ordered_targets, current_target, relevant_set),
        )
        examples.append(example)

        parent = self.lookup.parent_of(rollout.node)
        if parent is not None:
            rollout.node = parent
            rollout.path.append(parent)
        rollout.step_index += 1
        rollout.aggregate_calls += 1

    def _emit_terminal_done_step(
        self,
        examples: List[SupervisionExample],
        traj_id: str,
        rollout: _RolloutState,
        ordered_targets: Sequence[str],
        retrieved_targets: Set[str],
        relevant_set: Set[str],
        query: str,
        query_id: str,
        query_vec: np.ndarray,
    ) -> None:
        if rollout.node == self.root_id:
            example = SupervisionExample(
                query_id=query_id,
                query=query,
                step_index=rollout.step_index,
                trajectory_id=traj_id,
                is_off_path=False,
                state=self._make_state_dict(rollout),
                action_label=ACTION_JUMP,
                done_label=1 if self._compute_done(rollout.evidence, relevant_set) else 0,
                jump=self._build_jump_candidates(
                    rollout.node,
                    [],
                    all_remaining_path_nodes=set(),
                    query_vec=query_vec,
                    target_path=set(),
                ),
                retrieve=None,
                debug=self._debug_block(ordered_targets, None, relevant_set),
            )
            jump_block = example.jump
            if jump_block and not jump_block["positive_indices"]:
                ann_idx = next(
                    (i for i, src in enumerate(jump_block["candidate_sources"]) if src == "ann"),
                    None,
                )
                if ann_idx is not None:
                    jump_block["positive_indices"] = [ann_idx]
            examples.append(example)
            return

        example = SupervisionExample(
            query_id=query_id,
            query=query,
            step_index=rollout.step_index,
            trajectory_id=traj_id,
            is_off_path=False,
            state=self._make_state_dict(rollout),
            action_label=ACTION_AGGREGATE,
            done_label=1 if self._compute_done(rollout.evidence, relevant_set) else 0,
            jump=None,
            retrieve=None,
            debug=self._debug_block(ordered_targets, None, relevant_set),
        )
        examples.append(example)

    def _build_off_path_examples(
        self,
        rollout: _RolloutState,
        gold_next: str,
        current_target: str,
        ordered_targets: Sequence[str],
        retrieved_targets: Set[str],
        relevant_set: Set[str],
        query: str,
        query_id: str,
        traj_id: str,
        query_vec: np.ndarray,
    ) -> List[SupervisionExample]:
        if self.config.n_off_path <= 0:
            return []

        target_chain = ancestors_of(current_target, self.lookup)
        oracle_path_set: Set[str] = set(target_chain)

        siblings = [
            sib
            for sib in self.lookup.children_of(self.lookup.parent_of(gold_next) or "")
            if sib != gold_next and sib != rollout.node and sib not in oracle_path_set
        ]
        ann_results = self.node_ann.top_k(query_vec, self.config.ann_K)
        ann_distractors = [
            nid
            for nid, _ in ann_results
            if nid not in oracle_path_set and nid != rollout.node
        ]

        candidates_pool = list(dict.fromkeys(siblings + ann_distractors))
        if not candidates_pool:
            return []

        n_off = min(self.config.n_off_path, len(candidates_pool))
        picks = candidates_pool[:n_off]

        examples: List[SupervisionExample] = []
        for off_idx, m in enumerate(picks):
            off_traj = f"{traj_id}.off{rollout.step_index}_{off_idx}"
            off_path = list(rollout.path) + [m]
            off_state = _RolloutState(
                node=m,
                path=off_path,
                evidence=list(rollout.evidence),
                step_index=rollout.step_index + 1,
                retrieve_calls=rollout.retrieve_calls,
                jump_calls=rollout.jump_calls + 1,
                aggregate_calls=rollout.aggregate_calls,
            )

            recovery_anchor = lca_of(m, current_target, self.lookup)
            anchor_chain = ancestors_of(current_target, self.lookup)
            try:
                anchor_idx = anchor_chain.index(recovery_anchor)
            except ValueError:
                continue
            recovery_chain = ancestors_of(m, self.lookup)
            try:
                m_anchor_idx = recovery_chain.index(recovery_anchor)
            except ValueError:
                continue

            recover_via_aggregate = False
            recovery_target: Optional[str] = None
            steps_up = (len(recovery_chain) - 1) - m_anchor_idx
            if steps_up == 1:
                parent_m = self.lookup.parent_of(m)
                if parent_m is not None and parent_m in oracle_path_set:
                    recover_via_aggregate = True
                    recovery_target = parent_m

            if not recover_via_aggregate:
                if anchor_idx + 1 < len(anchor_chain):
                    recovery_target = anchor_chain[anchor_idx + 1]
                else:
                    recovery_target = current_target

            if recover_via_aggregate:
                example = SupervisionExample(
                    query_id=query_id,
                    query=query,
                    step_index=off_state.step_index,
                    trajectory_id=off_traj,
                    is_off_path=True,
                    state=self._make_state_dict(off_state),
                    action_label=ACTION_AGGREGATE,
                    done_label=self._compute_done(off_state.evidence, relevant_set),
                    jump=None,
                    retrieve=None,
                    debug=self._debug_block(ordered_targets, current_target, relevant_set),
                )
            else:
                if recovery_target is None:
                    continue
                jump_block = self._build_jump_candidates(
                    off_state.node,
                    [recovery_target],
                    all_remaining_path_nodes=oracle_path_set,
                    query_vec=query_vec,
                    target_path=oracle_path_set,
                )
                if recovery_target not in jump_block["candidate_node_ids"]:
                    jump_block["candidate_node_ids"][-1] = recovery_target
                    jump_block["candidate_mask"][-1] = 1
                    jump_block["candidate_sources"][-1] = "positive"
                    pos = jump_block["candidate_node_ids"].index(recovery_target)
                    if pos not in jump_block["positive_indices"]:
                        jump_block["positive_indices"].append(pos)
                        jump_block["positive_indices"].sort()
                example = SupervisionExample(
                    query_id=query_id,
                    query=query,
                    step_index=off_state.step_index,
                    trajectory_id=off_traj,
                    is_off_path=True,
                    state=self._make_state_dict(off_state),
                    action_label=ACTION_JUMP,
                    done_label=self._compute_done(off_state.evidence, relevant_set),
                    jump=jump_block,
                    retrieve=None,
                    debug=self._debug_block(ordered_targets, current_target, relevant_set),
                )

            examples.append(example)
        return examples


def to_policy_state(state_block: Dict[str, Any], query: str) -> PolicyState:
    """Reconstruct a :class:`PolicyState` from a serialized state dict."""

    return PolicyState(
        query=query,
        current_node_id=state_block["node_id"],
        depth=int(state_block.get("depth", 0)),
        is_leaf=bool(state_block.get("is_leaf", False)),
        subtree_doc_count=int(state_block.get("subtree_doc_count", 0)),
        path_node_ids=list(state_block.get("path_node_ids", [])),
        evidence_doc_ids=list(state_block.get("evidence_doc_ids", [])),
        step_index=int(state_block.get("step_index", 0)),
        retrieve_calls=int(state_block.get("retrieve_calls", 0)),
        jump_calls=int(state_block.get("jump_calls", 0)),
        aggregate_calls=int(state_block.get("aggregate_calls", 0)),
        remaining_doc_budget=int(state_block.get("remaining_doc_budget", 100)),
        remaining_call_budget=int(state_block.get("remaining_call_budget", 16)),
        max_doc_budget=int(state_block.get("max_doc_budget", 100)),
        max_call_budget=int(state_block.get("max_call_budget", 16)),
        max_depth=int(state_block.get("max_depth", 8)),
    )
