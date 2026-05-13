"""Frozen query/text embedding helpers shared across the policy package.

The embedding *backbone* is frozen everywhere in the v2 policy: we only ever
encode strings into vectors that match the dimensionality of the persisted
``StandaloneHierarchyIndex`` centroids. We support two backbones:

* sentence-transformers (default for production indexes) - lazily loaded.
* a deterministic hash embedder used by the CPU smoke index where
  ``config.embedding_model == 'local-hash-embedding'``.

The choice is driven by the index ``config.json``: callers pass
``index.config['embedding_model']`` to :func:`make_text_encoder`.
"""

from __future__ import annotations

import hashlib
import logging
import string
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


HASH_MODEL_TAG = "local-hash-embedding"


def _tokenize(text: str) -> List[str]:
    return (text or "").lower().translate(str.maketrans("", "", string.punctuation)).split()


def _hash_embed(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype="float32")
    for token in _tokenize(text):
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vector[bucket] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def hash_embed_batch(texts: Sequence[str], dim: int) -> np.ndarray:
    return np.stack([_hash_embed(t, dim) for t in texts], axis=0).astype("float32")


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model %s", model_name)
    return SentenceTransformer(model_name)


def make_text_encoder(
    model_name: str,
    embedding_dim: int,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a function ``encode(texts) -> (N, D) np.float32`` matching the index dim.

    The returned encoder is *frozen* - it never participates in autograd.
    """

    if model_name == HASH_MODEL_TAG:
        def _hash_encoder(texts: Sequence[str]) -> np.ndarray:
            return hash_embed_batch(list(texts), embedding_dim)

        return _hash_encoder

    try:
        model = _load_sentence_transformer(model_name)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning(
            "Failed to load sentence-transformers '%s' (%s); falling back to hash embedder",
            model_name,
            exc,
        )

        def _fallback(texts: Sequence[str]) -> np.ndarray:
            return hash_embed_batch(list(texts), embedding_dim)

        return _fallback

    def _st_encoder(texts: Sequence[str]) -> np.ndarray:
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=device,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    return _st_encoder


def encode_one(encoder: Callable[[Sequence[str]], np.ndarray], text: str) -> np.ndarray:
    return encoder([text])[0]


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def encode_query_batch(
    encoder: Callable[[Sequence[str]], np.ndarray],
    queries: Iterable[str],
) -> np.ndarray:
    items = list(queries)
    if not items:
        return np.zeros((0, 0), dtype="float32")
    return l2_normalize(encoder(items))
