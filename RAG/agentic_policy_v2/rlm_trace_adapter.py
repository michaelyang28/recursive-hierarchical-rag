"""Convert rewarded RLM traces into agentic_policy_v2 supervision rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from RAG.rlm_adaptation import RLMTrajectory, append_jsonl

from .network import ACTION_AGGREGATE, ACTION_JUMP, ACTION_RETRIEVE
from .state import NodeFeatureLookup


@dataclass
class RLMTraceAdapterConfig:
    """Controls conversion of sparse RLM traces into policy supervision."""

    K_max: int = 32
    M_max: int = 256
    max_doc_budget: int = 100
    max_call_budget: int = 16
    min_reward: float = 0.5
    include_terminal_done: bool = True
    include_low_reward_recovery: bool = False


class RLMTraceToPolicySupervisionAdapter:
    """Map recursive RLM decisions into the policy-v2 action space.

    Recursive RLM traces are tool/code execution traces rather than native
    policy states. This adapter reconstructs enough hierarchy state to produce
    valid training rows for ``SupervisionDataset``.
    """

    def __init__(
        self,
        lookup: NodeFeatureLookup,
        config: Optional[RLMTraceAdapterConfig] = None,
    ):
        self.lookup = lookup
        self.config = config or RLMTraceAdapterConfig()
        self.root_id = self._infer_root_id()
        self.max_depth = max(self.lookup.depth_of(nid) for nid in self.lookup._parent) if self.lookup._parent else 1

    def _infer_root_id(self) -> str:
        for node_id, parent_id in self.lookup._parent.items():
            if parent_id is None:
                return node_id
        raise ValueError("Hierarchy has no root node")

    def _ancestors(self, node_id: str) -> List[str]:
        chain: List[str] = []
        current: Optional[str] = node_id
        while current is not None:
            chain.append(current)
            current = self.lookup.parent_of(current)
        chain.reverse()
        return chain or [self.root_id]

    def _state(
        self,
        node_id: str,
        path: Sequence[str],
        evidence: Sequence[str],
        step_index: int,
        retrieve_calls: int,
        jump_calls: int,
        aggregate_calls: int,
    ) -> Dict[str, Any]:
        return {
            "node_id": node_id,
            "depth": self.lookup.depth_of(node_id),
            "is_leaf": self.lookup.is_leaf(node_id),
            "subtree_doc_count": len(self.lookup.member_doc_ids(node_id)),
            "path_node_ids": list(path),
            "evidence_doc_ids": list(dict.fromkeys(evidence)),
            "step_index": step_index,
            "retrieve_calls": retrieve_calls,
            "jump_calls": jump_calls,
            "aggregate_calls": aggregate_calls,
            "remaining_doc_budget": max(0, self.config.max_doc_budget - len(evidence)),
            "remaining_call_budget": max(
                0,
                self.config.max_call_budget - retrieve_calls - jump_calls - aggregate_calls,
            ),
            "max_doc_budget": self.config.max_doc_budget,
            "max_call_budget": self.config.max_call_budget,
            "max_depth": max(1, self.max_depth),
        }

    def _debug(
        self,
        trajectory: RLMTrajectory,
        action_source: str,
        relevant_doc_ids: Set[str],
        target_node_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        hits = [doc_id for doc_id in trajectory.retrieved_doc_ids if doc_id in relevant_doc_ids]
        return {
            "source": "rlm_trace",
            "action_source": action_source,
            "trajectory_reward": trajectory.reward.score,
            "trajectory_id": trajectory.trajectory_id,
            "prompt_version": trajectory.prompt_version,
            "relevant_doc_ids": sorted(relevant_doc_ids),
            "hit_doc_ids": hits,
            "target_node_ids": list(target_node_ids or []),
            "oracle_path_node_ids": self._ancestors(target_node_ids[0]) if target_node_ids else [],
            "is_high_reward": trajectory.reward.score >= self.config.min_reward,
        }

    def _jump_block(self, current_node: str, positive_node: str) -> Dict[str, Any]:
        candidate_ids: List[str] = []
        sources: List[str] = []
        seen: Set[str] = set()

        def add(node_id: Optional[str], source: str) -> None:
            if not node_id or node_id in seen or node_id not in self.lookup._parent:
                return
            seen.add(node_id)
            candidate_ids.append(node_id)
            sources.append(source)

        add(positive_node, "rlm_positive")
        for child in self.lookup.children_of(current_node):
            add(child, "child")
        add(self.lookup.parent_of(current_node), "parent")
        parent = self.lookup.parent_of(current_node)
        if parent:
            for sibling in self.lookup.children_of(parent):
                if sibling != current_node:
                    add(sibling, "sibling")

        candidate_ids = candidate_ids[: self.config.K_max]
        sources = sources[: self.config.K_max]
        positive_indices = [i for i, nid in enumerate(candidate_ids) if nid == positive_node]
        mask = [1] * len(candidate_ids)
        while len(candidate_ids) < self.config.K_max:
            candidate_ids.append("")
            sources.append("padding")
            mask.append(0)
        return {
            "candidate_node_ids": candidate_ids,
            "candidate_mask": mask,
            "candidate_sources": sources,
            "positive_indices": positive_indices,
        }

    def _doc_from_result(self, result: Dict[str, Any]) -> Optional[str]:
        metadata = result.get("metadata")
        if isinstance(metadata, dict) and metadata.get("doc_id"):
            return str(metadata["doc_id"])
        if result.get("beir_id"):
            return str(result["beir_id"])
        if result.get("doc_id"):
            return str(result["doc_id"])
        if result.get("id"):
            return str(result["id"])
        return None

    def _observed_doc_ids(self, event: Dict[str, Any]) -> List[str]:
        docs = []
        for result in event.get("top_results", []) or []:
            if isinstance(result, dict):
                doc_id = self._doc_from_result(result)
                if doc_id:
                    docs.append(doc_id)
        return list(dict.fromkeys(docs))

    def _retrieve_block(
        self,
        node_id: str,
        relevant_doc_ids: Set[str],
        observed_doc_ids: Sequence[str],
    ) -> Dict[str, Any]:
        member = list(self.lookup.member_doc_ids(node_id))
        positives = [d for d in member if d in relevant_doc_ids]
        for doc_id in observed_doc_ids:
            if doc_id in self.lookup._doc_row and doc_id not in member:
                member.append(doc_id)
                if doc_id in relevant_doc_ids or not positives:
                    positives.append(doc_id)

        if not member:
            member = [d for d in observed_doc_ids if d in self.lookup._doc_row]
        chunk_doc_ids = list(dict.fromkeys(positives + [d for d in member if d not in positives]))
        chunk_doc_ids = chunk_doc_ids[: self.config.M_max]
        positive_set = set(positives)
        positive_indices = [i for i, doc_id in enumerate(chunk_doc_ids) if doc_id in positive_set]
        if not positive_indices and chunk_doc_ids:
            positive_indices = [0]
        mask = [1] * len(chunk_doc_ids)
        while len(chunk_doc_ids) < self.config.M_max:
            chunk_doc_ids.append("")
            mask.append(0)
        return {
            "chunk_doc_ids": chunk_doc_ids,
            "chunk_mask": mask,
            "positive_indices": positive_indices,
        }

    def _row(
        self,
        trajectory: RLMTrajectory,
        step_index: int,
        node_id: str,
        path: Sequence[str],
        evidence: Sequence[str],
        action_label: int,
        done_label: int,
        jump: Optional[Dict[str, Any]],
        retrieve: Optional[Dict[str, Any]],
        debug: Dict[str, Any],
        retrieve_calls: int,
        jump_calls: int,
        aggregate_calls: int,
    ) -> Dict[str, Any]:
        return {
            "query_id": trajectory.query_id,
            "query": trajectory.query,
            "step_index": step_index,
            "trajectory_id": trajectory.trajectory_id,
            "is_off_path": False,
            "state": self._state(
                node_id,
                path,
                evidence,
                step_index,
                retrieve_calls,
                jump_calls,
                aggregate_calls,
            ),
            "action_label": action_label,
            "done_label": done_label,
            "jump": jump,
            "retrieve": retrieve,
            "debug": debug,
        }

    def convert(self, trajectory: RLMTrajectory) -> List[Dict[str, Any]]:
        """Convert one replay trajectory into zero or more policy rows."""

        if trajectory.reward.score < self.config.min_reward and not self.config.include_low_reward_recovery:
            return []

        relevant_doc_ids = {
            str(doc_id)
            for doc_id, rel in (trajectory.reward.qrels or trajectory.labels.get("qrels", {})).items()
            if int(rel) > 0
        }
        rows: List[Dict[str, Any]] = []
        current_node = self.root_id
        path = [self.root_id]
        evidence: List[str] = []
        retrieve_calls = 0
        jump_calls = 0
        aggregate_calls = 0

        for event in trajectory.trace:
            name = event.get("event")
            if name == "enter" and event.get("cluster_id") in self.lookup._parent:
                current_node = str(event["cluster_id"])
                path = self._ancestors(current_node)
                continue

            if name == "recurse":
                target = event.get("cluster_id")
                if target not in self.lookup._parent:
                    continue
                target = str(target)
                rows.append(
                    self._row(
                        trajectory=trajectory,
                        step_index=len(rows),
                        node_id=current_node,
                        path=path,
                        evidence=evidence,
                        action_label=ACTION_JUMP,
                        done_label=0,
                        jump=self._jump_block(current_node, target),
                        retrieve=None,
                        debug=self._debug(trajectory, "recurse", relevant_doc_ids, [target]),
                        retrieve_calls=retrieve_calls,
                        jump_calls=jump_calls,
                        aggregate_calls=aggregate_calls,
                    )
                )
                jump_calls += 1
                current_node = target
                path = self._ancestors(current_node)
                continue

            if name == "search_in_cluster":
                node_id = event.get("cluster_id")
                if node_id not in self.lookup._parent:
                    continue
                node_id = str(node_id)
                observed = self._observed_doc_ids(event)
                retrieve_block = self._retrieve_block(node_id, relevant_doc_ids, observed)
                if not retrieve_block["positive_indices"]:
                    continue
                rows.append(
                    self._row(
                        trajectory=trajectory,
                        step_index=len(rows),
                        node_id=node_id,
                        path=self._ancestors(node_id),
                        evidence=evidence,
                        action_label=ACTION_RETRIEVE,
                        done_label=0,
                        jump=None,
                        retrieve=retrieve_block,
                        debug=self._debug(trajectory, "search_in_cluster", relevant_doc_ids, [node_id]),
                        retrieve_calls=retrieve_calls,
                        jump_calls=jump_calls,
                        aggregate_calls=aggregate_calls,
                    )
                )
                retrieve_calls += 1
                evidence.extend([d for d in observed if d in self.lookup._doc_row])
                current_node = node_id
                path = self._ancestors(current_node)
                continue

            if name == "return":
                evidence.extend(str(x) for x in event.get("final_doc_ids", []) if x in self.lookup._doc_row)

        if self.config.include_terminal_done and rows and trajectory.reward.score >= self.config.min_reward:
            rows.append(
                self._row(
                    trajectory=trajectory,
                    step_index=len(rows),
                    node_id=current_node,
                    path=path,
                    evidence=evidence,
                    action_label=ACTION_RETRIEVE,
                    done_label=1,
                    jump=None,
                    retrieve=self._retrieve_block(current_node, relevant_doc_ids, evidence),
                    debug=self._debug(trajectory, "terminal_done", relevant_doc_ids, [current_node]),
                    retrieve_calls=retrieve_calls,
                    jump_calls=jump_calls,
                    aggregate_calls=aggregate_calls,
                )
            )
        return rows

    def convert_many(self, trajectories: Iterable[RLMTrajectory]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for trajectory in trajectories:
            rows.extend(self.convert(trajectory))
        return rows

    def write_jsonl(self, path: str | Path, trajectories: Iterable[RLMTrajectory]) -> int:
        rows = self.convert_many(trajectories)
        return append_jsonl(path, rows)


def load_replay_jsonl(path: str | Path) -> List[RLMTrajectory]:
    trajectories: List[RLMTrajectory] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(RLMTrajectory.from_dict(json.loads(line)))
    return trajectories
