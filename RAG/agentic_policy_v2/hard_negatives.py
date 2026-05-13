"""Hard-negative mining loop (plan section 4.6).

Three sources, by priority:

1. **Model errors**: for every JUMP step where the gold positive is not in
   the model's masked top-3 jump scores, augment the candidate set with the
   model's top-3 incorrect predictions.
2. **Centroid near-neighbors of the positive**: top-5 nearest other nodes in
   centroid space that are *not* on the oracle path.
3. **In-batch negatives**: handled at training time by
   :class:`InBatchNegativeAugmenter` (the collator-time augmentation in
   ``training`` overrides ``policy_collate`` if enabled).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SupervisionDataset, policy_collate
from .network import ACTION_JUMP, PolicyNetwork
from .node_ann import NodeCentroidIndex

logger = logging.getLogger(__name__)


@dataclass
class MiningStats:
    n_jump_rows: int = 0
    n_errors: int = 0
    n_centroid_neighbors_added: int = 0
    n_augmented_rows: int = 0
    n_dropped_for_K_max: int = 0


class HardNegativeMiner:
    """Mine hard negatives by running the model over the supervision set."""

    def __init__(
        self,
        model: PolicyNetwork,
        dataset: SupervisionDataset,
        node_ann: NodeCentroidIndex,
        K_max: int,
        device: torch.device = torch.device("cpu"),
        top_n_errors: int = 3,
        n_centroid_neighbors: int = 5,
    ):
        self.model = model
        self.dataset = dataset
        self.node_ann = node_ann
        self.K_max = K_max
        self.device = device
        self.top_n_errors = top_n_errors
        self.n_centroid_neighbors = n_centroid_neighbors
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def mine(
        self,
        out_path: str | Path,
        max_rows: Optional[int] = None,
    ) -> MiningStats:
        """Stream the dataset and write a JSONL of augmented rows.

        Each emitted row has identical shape to the input row but with extra
        ``hard_neg`` candidates spliced into ``jump.candidate_node_ids`` and
        a ``"hard_neg"`` source tag in ``candidate_sources``.
        """

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats = MiningStats()

        loader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=policy_collate,
            num_workers=0,
        )

        row_idx = 0
        with out_path.open("w", encoding="utf-8") as fout:
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                out = self.model(batch.x)
                h = out["h"]
                jump_scores = self.model.jump_scores(
                    h,
                    batch.jump_cand_emb,
                    batch.jump_cand_sim,
                    candidate_mask=batch.jump_cand_mask,
                )
                K = jump_scores.shape[-1]
                masked_scores = torch.where(
                    batch.jump_cand_mask.bool(),
                    jump_scores,
                    torch.full_like(jump_scores, float("-inf")),
                )
                top_k = min(self.top_n_errors, K)
                _, top_idx = torch.topk(masked_scores, k=top_k, dim=-1)
                in_top = batch.jump_pos_mask.gather(-1, top_idx).any(dim=-1)
                jump_present_b = batch.jump_present.bool()

                for b in range(batch.x.shape[0]):
                    row = self.dataset.rows[row_idx]
                    row_idx += 1
                    if not jump_present_b[b].item():
                        continue
                    stats.n_jump_rows += 1
                    if in_top[b].item():
                        continue
                    augmented = self._augment_row(
                        row=row,
                        wrong_top_idx=top_idx[b].cpu().tolist(),
                        wrong_scores=masked_scores[b].cpu().tolist(),
                        stats=stats,
                    )
                    if augmented is None:
                        continue
                    fout.write(json.dumps(augmented, ensure_ascii=False) + "\n")
                    stats.n_augmented_rows += 1
                    stats.n_errors += 1
                    if max_rows is not None and stats.n_augmented_rows >= max_rows:
                        return stats
        return stats

    def _augment_row(
        self,
        row: Dict[str, Any],
        wrong_top_idx: List[int],
        wrong_scores: List[float],
        stats: MiningStats,
    ) -> Optional[Dict[str, Any]]:
        jump = row.get("jump")
        if jump is None:
            return None
        new_row = copy.deepcopy(row)
        new_jump = new_row["jump"]
        cand_ids: List[str] = list(new_jump["candidate_node_ids"])
        cand_mask: List[int] = list(new_jump["candidate_mask"])
        sources: List[str] = list(new_jump.get("candidate_sources") or [""] * len(cand_ids))
        positive_indices = list(new_jump.get("positive_indices") or [])
        positive_set: Set[str] = {cand_ids[i] for i in positive_indices if 0 <= i < len(cand_ids)}

        wrong_node_ids: List[str] = []
        for idx in wrong_top_idx:
            if 0 <= idx < len(cand_ids):
                nid = cand_ids[idx]
                if nid and nid not in positive_set:
                    wrong_node_ids.append(nid)

        oracle_path = set(row.get("debug", {}).get("oracle_path_node_ids", []))
        positives_for_centroid = [
            cand_ids[i] for i in positive_indices if 0 <= i < len(cand_ids)
        ]
        centroid_neighbors: List[str] = []
        for pos in positives_for_centroid:
            neighbors = self.node_ann.top_k(
                self.node_ann.vector_for(pos), self.n_centroid_neighbors + 1
            )
            for nid, _ in neighbors:
                if (
                    nid not in oracle_path
                    and nid not in positive_set
                    and nid not in centroid_neighbors
                ):
                    centroid_neighbors.append(nid)
                if len(centroid_neighbors) >= self.n_centroid_neighbors:
                    break
        stats.n_centroid_neighbors_added += len(centroid_neighbors)

        added = 0
        for nid in wrong_node_ids + centroid_neighbors:
            if nid in cand_ids:
                continue
            if added >= self.K_max:
                break
            replaced = False
            for i, m in enumerate(cand_mask):
                if not m:
                    cand_ids[i] = nid
                    cand_mask[i] = 1
                    if i < len(sources):
                        sources[i] = "hard_neg"
                    else:
                        sources.append("hard_neg")
                    replaced = True
                    added += 1
                    break
            if not replaced:
                stats.n_dropped_for_K_max += 1

        new_jump["candidate_node_ids"] = cand_ids
        new_jump["candidate_mask"] = cand_mask
        new_jump["candidate_sources"] = sources
        new_jump["positive_indices"] = positive_indices
        new_row["is_hard_neg_augmented"] = True
        return new_row
