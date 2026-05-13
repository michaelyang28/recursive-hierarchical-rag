"""Supervision JSONL dataset and collator (plan section 1.8).

The dataset reads supervision rows produced by ``SupervisionGenerator`` and
materializes everything needed for one forward pass: state tensor, action /
done labels, and (when applicable) jump candidate / retrieve chunk blocks
with multi-positive masks.

Embeddings are looked up at ``__getitem__`` time from the (frozen) hierarchy
index + per-query embedding cache, never stored in the JSONL.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .embedding import l2_normalize
from .network import ACTION_NAMES, ACTION_RETRIEVE, ACTION_AGGREGATE, ACTION_JUMP
from .state import META_DIM, NodeFeatureLookup, build_state_tensor
from .supervision import to_policy_state


def load_query_embeddings(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    qids = data["query_ids"].tolist()
    embeds = data["embeddings"].astype("float32")
    return {str(qid): embeds[i] for i, qid in enumerate(qids)}


def _load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_jsonl_many(paths: Sequence[str | Path]) -> List[Dict]:
    rows: List[Dict] = []
    for p in paths:
        rows.extend(_load_jsonl(p))
    return rows


@dataclass
class CollatedBatch:
    """Output of ``policy_collate`` matching plan section 1.8."""

    x: torch.Tensor
    action_label: torch.Tensor
    done_label: torch.Tensor
    jump_present: torch.Tensor
    jump_cand_emb: torch.Tensor
    jump_cand_sim: torch.Tensor
    jump_cand_mask: torch.Tensor
    jump_pos_mask: torch.Tensor
    jump_loop_mask: torch.Tensor
    action_loop_mask: torch.Tensor
    retrieve_present: torch.Tensor
    retrieve_emb: torch.Tensor
    retrieve_sim: torch.Tensor
    retrieve_mask: torch.Tensor
    retrieve_pos_mask: torch.Tensor

    def to(self, device: torch.device) -> "CollatedBatch":
        return CollatedBatch(
            x=self.x.to(device),
            action_label=self.action_label.to(device),
            done_label=self.done_label.to(device),
            jump_present=self.jump_present.to(device),
            jump_cand_emb=self.jump_cand_emb.to(device),
            jump_cand_sim=self.jump_cand_sim.to(device),
            jump_cand_mask=self.jump_cand_mask.to(device),
            jump_pos_mask=self.jump_pos_mask.to(device),
            jump_loop_mask=self.jump_loop_mask.to(device),
            action_loop_mask=self.action_loop_mask.to(device),
            retrieve_present=self.retrieve_present.to(device),
            retrieve_emb=self.retrieve_emb.to(device),
            retrieve_sim=self.retrieve_sim.to(device),
            retrieve_mask=self.retrieve_mask.to(device),
            retrieve_pos_mask=self.retrieve_pos_mask.to(device),
        )


class SupervisionDataset(Dataset):
    """Reads a supervision JSONL + paired query embeddings and yields tensors.

    Each ``__getitem__`` call produces a flat dict so multiple samples can be
    stacked by :func:`policy_collate`.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        query_embeddings_path: str | Path,
        lookup: NodeFeatureLookup,
        K_max: int,
        M_max: int,
        extra_jsonl_paths: Optional[Sequence[str | Path]] = None,
    ):
        if extra_jsonl_paths:
            self.rows = _load_jsonl_many([jsonl_path, *extra_jsonl_paths])
        else:
            self.rows = _load_jsonl(jsonl_path)
        self.query_embeds = load_query_embeddings(query_embeddings_path)
        self.lookup = lookup
        self.embedding_dim = lookup.embedding_dim
        self.K_max = K_max
        self.M_max = M_max

    def __len__(self) -> int:
        return len(self.rows)

    def action_distribution(self) -> Dict[str, int]:
        counter = Counter(r["action_label"] for r in self.rows)
        return {ACTION_NAMES[k]: int(v) for k, v in counter.items()}

    def action_class_weights(self, smoothing: float = 1.0) -> torch.Tensor:
        counts = [0, 0, 0]
        for row in self.rows:
            counts[int(row["action_label"])] += 1
        total = sum(counts)
        weights = []
        for c in counts:
            inv = total / (3.0 * (c + smoothing))
            weights.append(inv)
        return torch.tensor(weights, dtype=torch.float32)

    def done_pos_weight(self, default: float = 10.0) -> float:
        n_pos = sum(1 for r in self.rows if int(r.get("done_label", 0)) == 1)
        n_neg = sum(1 for r in self.rows if int(r.get("done_label", 0)) == 0)
        if n_pos == 0:
            return default
        return float(n_neg) / float(max(1, n_pos))

    def _query_vec(self, query_id: str) -> np.ndarray:
        vec = self.query_embeds.get(query_id)
        if vec is None:
            return np.zeros(self.embedding_dim, dtype="float32")
        return np.asarray(vec, dtype="float32")

    def _state_tensor(self, row: Dict) -> np.ndarray:
        ps = to_policy_state(row["state"], row.get("query", ""))
        q_vec = self._query_vec(row["query_id"])
        return build_state_tensor(ps, q_vec, self.lookup)

    def _jump_block(self, row: Dict, q_vec: np.ndarray) -> Dict[str, torch.Tensor]:
        jump = row.get("jump")
        K = self.K_max
        cand_emb = np.zeros((K, self.embedding_dim), dtype="float32")
        cand_sim = np.zeros((K, 1), dtype="float32")
        cand_mask = np.zeros(K, dtype="float32")
        pos_mask = np.zeros(K, dtype="float32")
        loop_mask = np.zeros(K, dtype="float32")
        state = row.get("state", {})
        current_node = state.get("node_id")
        visited = set(state.get("path_node_ids") or [])

        if jump is not None:
            cand_ids = jump.get("candidate_node_ids") or []
            mask = jump.get("candidate_mask") or []
            positives = set(jump.get("positive_indices") or [])
            for i in range(min(K, len(cand_ids))):
                if i >= len(mask) or not mask[i]:
                    continue
                cand_id = cand_ids[i]
                if not cand_id:
                    continue
                vec = self.lookup.node_centroid(cand_id)
                cand_emb[i] = vec
                cand_sim[i, 0] = float(np.dot(vec, q_vec))
                cand_mask[i] = 1.0
                if i in positives:
                    pos_mask[i] = 1.0
                elif cand_id == current_node or cand_id in visited:
                    loop_mask[i] = 1.0

        return {
            "jump_cand_emb": torch.from_numpy(cand_emb),
            "jump_cand_sim": torch.from_numpy(cand_sim),
            "jump_cand_mask": torch.from_numpy(cand_mask),
            "jump_pos_mask": torch.from_numpy(pos_mask),
            "jump_loop_mask": torch.from_numpy(loop_mask),
        }

    def _action_loop_mask(self, row: Dict) -> torch.Tensor:
        """Actions that would immediately undo progress in this teacher state."""

        mask = torch.zeros(3, dtype=torch.float32)
        state = row.get("state", {})
        current = state.get("node_id")
        path = list(state.get("path_node_ids") or [])
        if len(path) >= 2 and row.get("action_label") != ACTION_AGGREGATE:
            previous = path[-2]
            if previous == self.lookup.parent_of(current):
                mask[ACTION_AGGREGATE] = 1.0
        return mask

    def _retrieve_block(self, row: Dict, q_vec: np.ndarray) -> Dict[str, torch.Tensor]:
        retrieve = row.get("retrieve")
        M = self.M_max
        chunk_emb = np.zeros((M, self.embedding_dim), dtype="float32")
        chunk_sim = np.zeros((M, 1), dtype="float32")
        chunk_mask = np.zeros(M, dtype="float32")
        pos_mask = np.zeros(M, dtype="float32")

        if retrieve is not None:
            chunk_ids = retrieve.get("chunk_doc_ids") or []
            mask = retrieve.get("chunk_mask") or []
            positives = set(retrieve.get("positive_indices") or [])
            doc_ids_present = [cid for cid in chunk_ids if cid]
            doc_emb = self.lookup.doc_embedding_matrix(doc_ids_present)
            doc_emb = l2_normalize(doc_emb) if doc_emb.size else doc_emb
            real_pos = 0
            for i in range(min(M, len(chunk_ids))):
                if i >= len(mask) or not mask[i]:
                    continue
                cid = chunk_ids[i]
                if not cid:
                    continue
                vec = doc_emb[real_pos]
                real_pos += 1
                chunk_emb[i] = vec
                chunk_sim[i, 0] = float(np.dot(vec, q_vec))
                chunk_mask[i] = 1.0
                if i in positives:
                    pos_mask[i] = 1.0

        return {
            "retrieve_emb": torch.from_numpy(chunk_emb),
            "retrieve_sim": torch.from_numpy(chunk_sim),
            "retrieve_mask": torch.from_numpy(chunk_mask),
            "retrieve_pos_mask": torch.from_numpy(pos_mask),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        x = self._state_tensor(row)
        action_label = int(row["action_label"])
        done_label = int(row.get("done_label", 0))

        q_vec = self._query_vec(row["query_id"])
        q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

        positive_node_ids: List[str] = []
        if row.get("jump"):
            cand_ids = row["jump"].get("candidate_node_ids", [])
            for i in row["jump"].get("positive_indices", []):
                if 0 <= i < len(cand_ids) and cand_ids[i]:
                    positive_node_ids.append(cand_ids[i])
        oracle_path = list(row.get("debug", {}).get("oracle_path_node_ids", []))

        item: Dict[str, Any] = {
            "x": torch.from_numpy(x),
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "done_label": torch.tensor(done_label, dtype=torch.float32),
            "jump_present": torch.tensor(
                1.0 if action_label == ACTION_JUMP and row.get("jump") else 0.0,
                dtype=torch.float32,
            ),
            "retrieve_present": torch.tensor(
                1.0 if action_label == ACTION_RETRIEVE and row.get("retrieve") else 0.0,
                dtype=torch.float32,
            ),
            "_query_id": row["query_id"],
            "_query_vec": q_vec.astype("float32"),
            "_positive_node_ids": positive_node_ids,
            "_oracle_path": oracle_path,
        }
        item.update(self._jump_block(row, q_vec))
        item.update(self._retrieve_block(row, q_vec))
        item["action_loop_mask"] = self._action_loop_mask(row)
        return item


def policy_collate(batch: Sequence[Dict[str, Any]]) -> CollatedBatch:
    return CollatedBatch(
        x=torch.stack([b["x"] for b in batch], dim=0),
        action_label=torch.stack([b["action_label"] for b in batch], dim=0),
        done_label=torch.stack([b["done_label"] for b in batch], dim=0),
        jump_present=torch.stack([b["jump_present"] for b in batch], dim=0),
        jump_cand_emb=torch.stack([b["jump_cand_emb"] for b in batch], dim=0),
        jump_cand_sim=torch.stack([b["jump_cand_sim"] for b in batch], dim=0),
        jump_cand_mask=torch.stack([b["jump_cand_mask"] for b in batch], dim=0),
        jump_pos_mask=torch.stack([b["jump_pos_mask"] for b in batch], dim=0),
        jump_loop_mask=torch.stack([b["jump_loop_mask"] for b in batch], dim=0),
        action_loop_mask=torch.stack([b["action_loop_mask"] for b in batch], dim=0),
        retrieve_present=torch.stack([b["retrieve_present"] for b in batch], dim=0),
        retrieve_emb=torch.stack([b["retrieve_emb"] for b in batch], dim=0),
        retrieve_sim=torch.stack([b["retrieve_sim"] for b in batch], dim=0),
        retrieve_mask=torch.stack([b["retrieve_mask"] for b in batch], dim=0),
        retrieve_pos_mask=torch.stack([b["retrieve_pos_mask"] for b in batch], dim=0),
    )


class InBatchNegativeCollator:
    """Collator that splices cross-query positives into padding slots as negatives.

    For each row in the batch we look at *other* rows' positive node ids and
    pick up to ``n_in_batch_negs`` of them that are not in this row's
    ``_oracle_path`` (overlap guard, plan section 4.6). The chosen negatives
    are written into padding slots of ``jump_cand_emb`` / ``jump_cand_sim`` and
    ``jump_cand_mask`` is set to 1; ``jump_pos_mask`` stays 0 so they
    contribute only to the denominator of the listwise softmax.
    """

    def __init__(
        self,
        lookup: NodeFeatureLookup,
        n_in_batch_negs: int = 8,
    ):
        self.lookup = lookup
        self.n_in_batch_negs = n_in_batch_negs

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> CollatedBatch:
        collated = policy_collate(batch)
        if self.n_in_batch_negs <= 0:
            return collated

        cand_mask = collated.jump_cand_mask.clone()
        cand_emb = collated.jump_cand_emb.clone()
        cand_sim = collated.jump_cand_sim.clone()
        loop_mask = collated.jump_loop_mask.clone()

        B = cand_mask.shape[0]
        K = cand_mask.shape[1]
        for b in range(B):
            if collated.jump_present[b].item() <= 0.5:
                continue
            row_oracle = set(batch[b].get("_oracle_path", []))
            row_qid = batch[b].get("_query_id", "")
            row_qvec = np.asarray(
                batch[b].get("_query_vec", np.zeros(self.lookup.embedding_dim)),
                dtype="float32",
            )
            existing_ids = set()
            for other in batch:
                if other.get("_query_id") == row_qid:
                    existing_ids.update(other.get("_positive_node_ids", []))
            existing_ids.update(row_oracle)

            pool: List[str] = []
            for o, other in enumerate(batch):
                if o == b:
                    continue
                if other.get("_query_id") == row_qid:
                    continue
                for nid in other.get("_positive_node_ids", []):
                    if nid and nid not in existing_ids and nid not in pool:
                        pool.append(nid)

            free_slots = (cand_mask[b] <= 0.5).nonzero(as_tuple=False).flatten().tolist()
            n_to_add = min(self.n_in_batch_negs, len(pool), len(free_slots))
            for i in range(n_to_add):
                slot = free_slots[i]
                nid = pool[i]
                vec = self.lookup.node_centroid(nid)
                cand_emb[b, slot] = torch.from_numpy(vec)
                cand_sim[b, slot, 0] = float(np.dot(vec, row_qvec))
                cand_mask[b, slot] = 1.0
                loop_mask[b, slot] = 0.0

        return CollatedBatch(
            x=collated.x,
            action_label=collated.action_label,
            done_label=collated.done_label,
            jump_present=collated.jump_present,
            jump_cand_emb=cand_emb,
            jump_cand_sim=cand_sim,
            jump_cand_mask=cand_mask,
            jump_pos_mask=collated.jump_pos_mask,
            jump_loop_mask=loop_mask,
            action_loop_mask=collated.action_loop_mask,
            retrieve_present=collated.retrieve_present,
            retrieve_emb=collated.retrieve_emb,
            retrieve_sim=collated.retrieve_sim,
            retrieve_mask=collated.retrieve_mask,
            retrieve_pos_mask=collated.retrieve_pos_mask,
        )


def stratified_sampler_weights(dataset: SupervisionDataset) -> torch.Tensor:
    """Per-row weight = 1 / count(action_class). Used by ``WeightedRandomSampler``."""

    counts = [0, 0, 0]
    for row in dataset.rows:
        counts[int(row["action_label"])] += 1
    weights = torch.zeros(len(dataset.rows), dtype=torch.float32)
    for i, row in enumerate(dataset.rows):
        c = counts[int(row["action_label"])]
        weights[i] = 1.0 / max(1, c)
    return weights


def split_train_val(
    dataset_rows: Sequence[Dict],
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Return train / val indices, splitting on ``query_id`` so all rollouts of
    one query stay together (prevents leakage when multiple trajectories per
    query are generated)."""

    rng = np.random.default_rng(seed)
    qid_to_idx: Dict[str, List[int]] = {}
    for i, row in enumerate(dataset_rows):
        qid = str(row.get("query_id"))
        qid_to_idx.setdefault(qid, []).append(i)

    qids = list(qid_to_idx.keys())
    rng.shuffle(qids)
    n_val = int(round(val_fraction * len(qids)))
    val_qids = set(qids[:n_val])

    train_idx: List[int] = []
    val_idx: List[int] = []
    for qid, idxs in qid_to_idx.items():
        target = val_idx if qid in val_qids else train_idx
        target.extend(idxs)
    return train_idx, val_idx
