"""ANN over node centroids for jump-candidate generation.

Wraps a persisted matrix of L2-normalized centroids (one row per
``HierarchyNode``) with cosine top-K lookup. Uses ``faiss`` if available and
falls back to a numpy dot-product when not.

Persisted artifacts (alongside an existing index dir, or in a sibling
``ann_v2/`` subdir):

* ``node_centroid_matrix.npy`` - ``(N_nodes, D)`` float32, L2-normalized.
* ``node_centroid_meta.json``  - ``{node_ids: [...], embedding_dim: int,
  embedding_model: str, normalize: bool}``.
* ``node_centroid_index.faiss`` (optional) - serialized faiss index.

The matrix file is the source of truth; the FAISS file is a cache and is
rebuilt automatically if missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .embedding import l2_normalize

logger = logging.getLogger(__name__)


META_FILE = "node_centroid_meta.json"
MATRIX_FILE = "node_centroid_matrix.npy"
FAISS_FILE = "node_centroid_index.faiss"


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as exc:  # pragma: no cover - optional dep
        logger.info("faiss unavailable (%s); using numpy fallback for node ANN", exc)
        return None


class NodeCentroidIndex:
    """ANN over hierarchy node centroids.

    Build once from a ``StandaloneHierarchyIndex``, persist alongside the
    hierarchy, then query at training and inference time.
    """

    def __init__(
        self,
        node_ids: Sequence[str],
        matrix: np.ndarray,
        embedding_dim: int,
        embedding_model: str,
        use_faiss: bool = True,
    ):
        if matrix.ndim != 2 or matrix.shape[0] != len(node_ids):
            raise ValueError(
                f"matrix shape {matrix.shape} inconsistent with {len(node_ids)} node ids"
            )
        if matrix.shape[1] != embedding_dim:
            raise ValueError(
                f"matrix dim {matrix.shape[1]} != embedding_dim {embedding_dim}"
            )
        self.node_ids: List[str] = list(node_ids)
        self.matrix: np.ndarray = np.ascontiguousarray(matrix.astype("float32"))
        self.embedding_dim: int = int(embedding_dim)
        self.embedding_model: str = str(embedding_model)
        self._id_to_row = {nid: idx for idx, nid in enumerate(self.node_ids)}
        self._faiss = _try_import_faiss() if use_faiss else None
        self._faiss_index = None
        if self._faiss is not None:
            self._faiss_index = self._build_faiss(self.matrix)

    @classmethod
    def from_hierarchy(
        cls,
        hierarchy_index,
        embedding_model: Optional[str] = None,
        use_faiss: bool = True,
    ) -> "NodeCentroidIndex":
        node_ids: List[str] = []
        rows: List[np.ndarray] = []
        for node in hierarchy_index.nodes:
            if node.centroid is None:
                continue
            node_ids.append(node.node_id)
            rows.append(np.asarray(node.centroid, dtype="float32"))
        if not rows:
            raise ValueError("Hierarchy index has no node centroids")
        matrix = l2_normalize(np.stack(rows, axis=0))
        embedding_dim = matrix.shape[1]
        embedding_model = embedding_model or str(
            hierarchy_index.config.get("embedding_model", "unknown")
        )
        return cls(
            node_ids=node_ids,
            matrix=matrix,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            use_faiss=use_faiss,
        )

    def _build_faiss(self, matrix: np.ndarray):
        if self._faiss is None:
            return None
        index = self._faiss.IndexFlatIP(self.embedding_dim)
        index.add(np.ascontiguousarray(matrix))
        return index

    def save(self, dir_path: str | Path) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / MATRIX_FILE, self.matrix)
        meta = {
            "node_ids": self.node_ids,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "num_nodes": len(self.node_ids),
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self._faiss is not None and self._faiss_index is not None:
            try:
                self._faiss.write_index(self._faiss_index, str(path / FAISS_FILE))
            except Exception as exc:  # pragma: no cover - best-effort
                logger.warning("Failed to persist faiss index: %s", exc)

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        use_faiss: bool = True,
    ) -> "NodeCentroidIndex":
        path = Path(dir_path)
        meta_path = path / META_FILE
        matrix_path = path / MATRIX_FILE
        if not meta_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(f"Missing node centroid index files under {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        matrix = np.load(matrix_path).astype("float32")
        return cls(
            node_ids=meta["node_ids"],
            matrix=matrix,
            embedding_dim=int(meta["embedding_dim"]),
            embedding_model=str(meta.get("embedding_model", "unknown")),
            use_faiss=use_faiss,
        )

    def has(self, node_id: str) -> bool:
        return node_id in self._id_to_row

    def vector_for(self, node_id: str) -> np.ndarray:
        row = self._id_to_row.get(node_id)
        if row is None:
            return np.zeros(self.embedding_dim, dtype="float32")
        return self.matrix[row]

    def vectors_for(self, node_ids: Iterable[str]) -> np.ndarray:
        ids = list(node_ids)
        if not ids:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        out = np.zeros((len(ids), self.embedding_dim), dtype="float32")
        for i, nid in enumerate(ids):
            row = self._id_to_row.get(nid)
            if row is not None:
                out[i] = self.matrix[row]
        return out

    def top_k(
        self,
        query_vec: np.ndarray,
        k: int,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, float]]:
        if k <= 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(-1)
        if q.shape[0] != self.embedding_dim:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self.embedding_dim}"
            )
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        excluded_rows = set()
        if exclude_ids is not None:
            for nid in exclude_ids:
                row = self._id_to_row.get(nid)
                if row is not None:
                    excluded_rows.add(row)

        oversample = max(k + len(excluded_rows), k * 2)
        oversample = min(oversample, len(self.node_ids))

        if self._faiss is not None and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(
                q.reshape(1, -1).astype("float32"), oversample
            )
            scores = scores[0]
            indices = indices[0]
        else:
            sims = self.matrix @ q
            indices = np.argsort(-sims)[:oversample]
            scores = sims[indices]

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            if int(idx) in excluded_rows:
                continue
            results.append((self.node_ids[int(idx)], float(score)))
            if len(results) >= k:
                break
        return results
