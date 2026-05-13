"""Shared data contracts for joint RLM prompt and policy adaptation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _now() -> float:
    return time.time()


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@dataclass
class RewardRecord:
    """Reward and label provenance for one RLM trajectory."""

    score: float
    source: str = "qrels"
    metrics: Dict[str, float] = field(default_factory=dict)
    qrels: Dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RewardRecord":
        return cls(
            score=float(payload.get("score", 0.0)),
            source=str(payload.get("source", "qrels")),
            metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
            qrels={str(k): int(v) for k, v in (payload.get("qrels") or {}).items()},
            created_at=float(payload.get("created_at", _now())),
        )


@dataclass
class DistilledTrajectorySummary:
    """Compact behavior summary distilled from successful RLM traces."""

    trajectory_id: str
    reward: float
    successful_queries: List[str] = field(default_factory=list)
    successful_clusters: List[str] = field(default_factory=list)
    successful_doc_ids: List[str] = field(default_factory=list)
    generated_code_snippets: List[str] = field(default_factory=list)
    sandbox_errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DistilledTrajectorySummary":
        return cls(
            trajectory_id=str(payload.get("trajectory_id", "")),
            reward=float(payload.get("reward", 0.0)),
            successful_queries=[str(x) for x in payload.get("successful_queries", [])],
            successful_clusters=[str(x) for x in payload.get("successful_clusters", [])],
            successful_doc_ids=[str(x) for x in payload.get("successful_doc_ids", [])],
            generated_code_snippets=[str(x) for x in payload.get("generated_code_snippets", [])],
            sandbox_errors=[str(x) for x in payload.get("sandbox_errors", [])],
            notes=[str(x) for x in payload.get("notes", [])],
        )


@dataclass
class PromptVersion:
    """Versioned prompt snapshot used by online RLM inference."""

    version: str
    prompts: Dict[str, str]
    created_at: float = field(default_factory=_now)
    parent_version: Optional[str] = None
    source_run: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    promoted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PromptVersion":
        return cls(
            version=str(payload["version"]),
            prompts={str(k): str(v) for k, v in (payload.get("prompts") or {}).items()},
            created_at=float(payload.get("created_at", _now())),
            parent_version=payload.get("parent_version"),
            source_run=payload.get("source_run"),
            metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
            promoted=bool(payload.get("promoted", False)),
        )


@dataclass
class PolicyCheckpointVersion:
    """Registry record for a candidate or promoted agentic_policy_v2 checkpoint."""

    version: str
    checkpoint_path: str
    created_at: float = field(default_factory=_now)
    parent_version: Optional[str] = None
    source_supervision_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    promoted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyCheckpointVersion":
        return cls(
            version=str(payload["version"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            created_at=float(payload.get("created_at", _now())),
            parent_version=payload.get("parent_version"),
            source_supervision_path=payload.get("source_supervision_path"),
            metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
            promoted=bool(payload.get("promoted", False)),
        )


@dataclass
class RLMTrajectory:
    """Replay-buffer record for one rewarded RLM rollout."""

    trajectory_id: str
    query_id: str
    query: str
    retrieved_doc_ids: List[str]
    trace: List[Dict[str, Any]]
    reward: RewardRecord
    prompt_version: Optional[str] = None
    prompts: Dict[str, str] = field(default_factory=dict)
    docs_spent: int = 0
    inspects_spent: int = 0
    answer: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    trace_path: Optional[str] = None
    created_at: float = field(default_factory=_now)

    def compact_trace(self, max_events: int = 12) -> List[Dict[str, Any]]:
        return compact_trace_events(self.trace, max_events=max_events)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reward"] = self.reward.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RLMTrajectory":
        return cls(
            trajectory_id=str(payload["trajectory_id"]),
            query_id=str(payload["query_id"]),
            query=str(payload["query"]),
            retrieved_doc_ids=[str(x) for x in payload.get("retrieved_doc_ids", [])],
            trace=list(payload.get("trace", [])),
            reward=RewardRecord.from_dict(payload.get("reward", {})),
            prompt_version=payload.get("prompt_version"),
            prompts={str(k): str(v) for k, v in (payload.get("prompts") or {}).items()},
            docs_spent=int(payload.get("docs_spent", 0)),
            inspects_spent=int(payload.get("inspects_spent", 0)),
            answer=payload.get("answer"),
            labels=dict(payload.get("labels") or {}),
            trace_path=payload.get("trace_path"),
            created_at=float(payload.get("created_at", _now())),
        )


def new_version(prefix: str) -> str:
    """Return a short sortable-ish version id."""

    return f"{prefix}-{int(_now())}-{uuid.uuid4().hex[:8]}"


def compact_trace_events(trace: Sequence[Dict[str, Any]], max_events: int = 12) -> List[Dict[str, Any]]:
    """Create a bounded trace payload suitable for optimizer metadata."""

    compact: List[Dict[str, Any]] = []
    for event in list(trace)[:max_events]:
        name = event.get("event")
        if name == "generated_code":
            compact.append({
                "event": name,
                "depth": event.get("depth"),
                "step": event.get("step"),
                "cluster_id": event.get("cluster_id"),
                "code": (event.get("code") or "")[:800],
            })
        elif name in {"search_in_cluster", "search_documents", "ground_query"}:
            compact.append({
                "event": name,
                "depth": event.get("depth"),
                "cluster_id": event.get("cluster_id"),
                "query": event.get("query"),
                "num_results": event.get("num_results"),
                "top_results": event.get("top_results", [])[:3],
            })
        elif name == "rank_clusters_llm":
            compact.append({
                "event": name,
                "depth": event.get("depth"),
                "query": event.get("query"),
                "ranked": event.get("ranked", [])[:3],
            })
        elif name == "sandbox_result":
            compact.append({
                "event": name,
                "depth": event.get("depth"),
                "step": event.get("step"),
                "success": event.get("success"),
                "error": event.get("error"),
                "done": event.get("done"),
            })
        else:
            compact.append({
                k: v for k, v in event.items()
                if k in {
                    "event",
                    "depth",
                    "query",
                    "cluster_id",
                    "sub_query",
                    "num_final_doc_ids",
                    "docs_spent",
                    "inspects_spent",
                }
            })
    return compact


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append JSON-serializable rows to a JSONL file and return row count."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


class RLMReplayBuffer:
    """JSONL replay buffer for rewarded RLM trajectories."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, trajectory: RLMTrajectory) -> None:
        append_jsonl(self.path, [trajectory.to_dict()])

    def extend(self, trajectories: Iterable[RLMTrajectory]) -> int:
        return append_jsonl(self.path, [t.to_dict() for t in trajectories])

    def load(self) -> List[RLMTrajectory]:
        if not self.path.exists():
            return []
        trajectories: List[RLMTrajectory] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trajectories.append(RLMTrajectory.from_dict(json.loads(line)))
        return trajectories

    def select_successful(
        self,
        min_reward: float = 0.5,
        limit: Optional[int] = None,
    ) -> List[RLMTrajectory]:
        rows = [t for t in self.load() if t.reward.score >= min_reward]
        rows.sort(key=lambda t: (t.reward.score, t.created_at), reverse=True)
        return rows[:limit] if limit is not None else rows


class VersionedRegistry:
    """Small JSON registry with atomic-ish current-version pointers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _empty(self) -> Dict[str, Any]:
        return {"current_version": None, "versions": []}

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        payload = _read_json(self.path)
        payload.setdefault("current_version", None)
        payload.setdefault("versions", [])
        return payload

    def save(self, payload: Dict[str, Any]) -> None:
        _write_json(self.path, payload)

    def add(self, record: Dict[str, Any], promote: bool = False) -> None:
        payload = self.load()
        versions = [v for v in payload["versions"] if v.get("version") != record.get("version")]
        if promote:
            for version in versions:
                version["promoted"] = False
            record["promoted"] = True
            payload["current_version"] = record.get("version")
        versions.append(record)
        payload["versions"] = versions
        self.save(payload)

    def current(self) -> Optional[Dict[str, Any]]:
        payload = self.load()
        current = payload.get("current_version")
        for record in payload.get("versions", []):
            if record.get("version") == current:
                return record
        return None


class PromptRegistry(VersionedRegistry):
    """Versioned registry for prompt snapshots."""

    def add_prompt_version(self, version: PromptVersion, promote: bool = False) -> None:
        self.add(version.to_dict(), promote=promote)

    def current_prompt_version(self) -> Optional[PromptVersion]:
        record = self.current()
        return PromptVersion.from_dict(record) if record else None


class PolicyCheckpointRegistry(VersionedRegistry):
    """Versioned registry for policy checkpoints."""

    def add_checkpoint_version(self, version: PolicyCheckpointVersion, promote: bool = False) -> None:
        self.add(version.to_dict(), promote=promote)

    def current_checkpoint_version(self) -> Optional[PolicyCheckpointVersion]:
        record = self.current()
        return PolicyCheckpointVersion.from_dict(record) if record else None


def should_promote(
    baseline_metrics: Optional[Dict[str, float]],
    candidate_metrics: Dict[str, float],
    metric: str,
    min_delta: float = 0.0,
) -> bool:
    """Return whether candidate metrics improve over the current baseline."""

    candidate = float(candidate_metrics.get(metric, float("-inf")))
    if baseline_metrics is None:
        return candidate > float("-inf")
    baseline = float(baseline_metrics.get(metric, float("-inf")))
    return candidate >= baseline + min_delta


def distill_trajectory(trajectory: RLMTrajectory, max_snippets: int = 3) -> DistilledTrajectorySummary:
    """Extract concise successful/failed behavior from an RLM trace."""

    queries: List[str] = []
    clusters: List[str] = []
    docs: List[str] = []
    snippets: List[str] = []
    errors: List[str] = []

    for event in trajectory.trace:
        name = event.get("event")
        if name == "generated_code" and len(snippets) < max_snippets:
            code = (event.get("code") or "").strip()
            if code:
                snippets.append(code[:1200])
        if name in {"search_in_cluster", "rank_clusters_llm", "ground_query", "search_documents"}:
            query = event.get("query")
            if query:
                queries.append(str(query))
        cluster_id = event.get("cluster_id")
        if cluster_id:
            clusters.append(str(cluster_id))
        if name == "return":
            docs.extend(str(x) for x in event.get("final_doc_ids", [])[:20] if x)
        if name == "sandbox_result" and event.get("error"):
            errors.append(str(event.get("error"))[:500])

    return DistilledTrajectorySummary(
        trajectory_id=trajectory.trajectory_id,
        reward=trajectory.reward.score,
        successful_queries=list(dict.fromkeys(queries)),
        successful_clusters=list(dict.fromkeys(clusters)),
        successful_doc_ids=list(dict.fromkeys(docs or trajectory.retrieved_doc_ids[:20])),
        generated_code_snippets=snippets,
        sandbox_errors=errors,
        notes=[
            f"docs_spent={trajectory.docs_spent}",
            f"inspects_spent={trajectory.inspects_spent}",
            f"prompt_version={trajectory.prompt_version}",
        ],
    )
