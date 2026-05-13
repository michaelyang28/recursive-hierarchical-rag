"""
LLM-Agent Hierarchical Retriever — Two-Phase Design.

Phase 1 (Navigate):  LLM walks the cluster tree and selects leaf clusters
                     purely based on semantic reasoning.  No budget splitting.
Phase 2 (Retrieve):  All doc IDs from visited leaf clusters are pooled and
                     re-ranked by global BM25 (full-corpus IDF).

This separation lets the LLM focus on what it does well (semantic cluster
selection) while BM25 handles what it does well (lexical scoring with correct
global statistics).

The system prompt controls Phase 1 and is the parameter being optimized.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Default system prompt (the parameter being optimized) ────────────────────
DEFAULT_SYSTEM_PROMPT = """You are an expert scientific literature navigator.
You receive a user query and the cluster hierarchy of a document collection.
Your goal is to identify clusters that are likely to contain relevant documents.

RULES:
1. Select 1-3 child clusters whose labels, summaries and keywords are most
   semantically related to the query — including synonyms and related concepts
   that BM25 keyword matching would miss.
2. Prefer clusters with larger subtree_doc_count when relevance is unclear.
3. Output RETRIEVE if the current cluster is a leaf (no children), or if
   none of the children look relevant — this will search all docs here.
4. Output ONLY a JSON object, no other text.

OUTPUT FORMAT:
  {"action": "JUMP",     "targets": ["<cluster_id>", ...]}   — navigate deeper
  {"action": "RETRIEVE"}                                      — collect here
"""
# ─────────────────────────────────────────────────────────────────────────────


class LLMAgentRetriever:
    """
    Two-phase hierarchical retriever:

    Phase 1 — Navigation (LLM):
        Walk the cluster tree and collect a set of leaf cluster IDs.
        The LLM makes pure semantic cluster selections with no budget constraint.

    Phase 2 — Retrieval (global BM25):
        Gather all doc IDs from the selected leaf clusters.
        Re-rank the entire candidate pool with global BM25 (full-corpus IDF).
        Return the top-k.

    Parameters
    ----------
    navigator : StandaloneHierarchyNavigator
    system_prompt : str, optional
        Instruction prompt — the optimizable parameter.
    llm_client : InferenceClient, optional
        Falls back to lexical heuristic when None.
    max_depth : int
        Max navigation depth before forcing RETRIEVE.
    top_k : int
        Documents returned after global BM25 rerank.
    max_branches : int
        Max child clusters to explore per node.
    max_clusters : int
        Cap on total leaf clusters visited per query.
    """

    def __init__(
        self,
        navigator,
        system_prompt: Optional[str] = None,
        llm_client=None,
        max_depth: int = 4,
        top_k: int = 100,
        max_branches: int = 3,
        max_clusters: int = 8,
        # Legacy parameter kept for API compatibility (no longer used)
        min_budget: int = 5,
        dense_augment_top_k: int = 0,
        embedding_model: Optional[str] = None,
        bm25_dense_alpha: float = 0.0,
    ):
        self.navigator = navigator
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm_client = llm_client
        self.max_depth = max_depth
        self.top_k = top_k
        self.max_branches = max_branches
        self.max_clusters = max_clusters
        self.dense_augment_top_k = dense_augment_top_k
        self.bm25_dense_alpha = bm25_dense_alpha
        self._dense_embeddings: Optional[Any] = None
        self._dense_doc_ids: Optional[List[str]] = None
        self._dense_doc_id_to_row: Dict[str, int] = {}
        self._query_encoder = None
        if dense_augment_top_k > 0 or bm25_dense_alpha > 0:
            self._init_dense_resources(embedding_model)

    def _init_dense_resources(self, embedding_model: Optional[str]) -> None:
        import numpy as np
        from RAG.agentic_policy_v2.embedding import make_text_encoder, l2_normalize

        emb = getattr(self.navigator.index, "embeddings", None)
        if emb is None:
            from RAG.standalone_hierarchy import StandaloneHierarchyIndex
            idx_cfg = self.navigator.index.config
            index_dir = idx_cfg.get("__index_dir__")
            if index_dir is None:
                raise RuntimeError(
                    "Navigator was loaded without embeddings. Use "
                    "StandaloneHierarchyIndex.load(..., load_embeddings=True)."
                )
            full = StandaloneHierarchyIndex.load(index_dir, load_embeddings=True)
            self.navigator.index.embeddings = full.embeddings
            emb = full.embeddings
        self._dense_embeddings = l2_normalize(np.asarray(emb, dtype="float32"))
        self._dense_doc_ids = [d.doc_id for d in self.navigator.index.documents]
        self._dense_doc_id_to_row = {d: i for i, d in enumerate(self._dense_doc_ids)}
        dim = self._dense_embeddings.shape[1]
        model_name = embedding_model or str(
            self.navigator.index.config.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self._query_encoder = make_text_encoder(model_name, dim)

    def _dense_top_k(self, query: str, top_k: int) -> List[str]:
        if not (self.dense_augment_top_k > 0 and top_k > 0):
            return []
        import numpy as np
        from RAG.agentic_policy_v2.embedding import l2_normalize

        qvec = np.asarray(self._query_encoder([query])[0], dtype="float32")
        qvec = l2_normalize(qvec[None, :])[0]
        sims = self._dense_embeddings @ qvec
        order = np.argsort(-sims)[:top_k]
        return [self._dense_doc_ids[int(i)] for i in order]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the two-phase agent and return document dicts."""
        k = top_k or self.top_k
        doc_ids = self.retrieve_ids(query, top_k=k)
        return self.navigator.fetch_documents(doc_ids)

    def retrieve_ids(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Phase 1: navigate to leaf clusters via LLM.
        Phase 2: pool all their docs and rerank by global BM25.
        """
        k = top_k or self.top_k

        # Phase 1 — collect visited leaf cluster IDs
        visited_leaves: Set[str] = set()
        self._navigate(
            query=query,
            cluster_id=None,
            depth=0,
            visited_leaves=visited_leaves,
        )

        if not visited_leaves:
            # Complete fallback: search from root
            visited_leaves.add("root")

        logger.debug("Visited %d leaf clusters: %s", len(visited_leaves), list(visited_leaves)[:6])

        # Phase 2 — gather docs from all visited leaves + global BM25 rerank
        candidate_ids: List[str] = []
        for cid in visited_leaves:
            candidate_ids.extend(self.navigator._doc_ids_for_cluster(cid))

        # Optionally augment with top-K dense candidates so the BM25 rerank
        # competes the LLM-selected pool against a strong dense-retrieval
        # baseline.
        if self.dense_augment_top_k > 0:
            candidate_ids.extend(self._dense_top_k(query, self.dense_augment_top_k))

        # Deduplicate preserving first occurrence
        seen: Set[str] = set()
        unique_candidates = [d for d in candidate_ids if not (d in seen or seen.add(d))]

        if not unique_candidates:
            return []

        # Global BM25 rerank over the full candidate pool
        try:
            if self.bm25_dense_alpha > 0:
                return self._bm25_dense_blend_rank(query, unique_candidates, k)
            scored = self.navigator._rank_doc_ids(unique_candidates, query, k)
            return [doc.doc_id for doc, _score in scored]
        except Exception as exc:
            logger.warning("Global BM25 rerank failed (%s); returning unranked", exc)
            return unique_candidates[:k]

    def _bm25_dense_blend_rank(
        self, query: str, candidate_ids: List[str], top_k: int
    ) -> List[str]:
        """Score candidates by alpha*cosine_sim + (1-alpha)*z(BM25)."""
        import numpy as np
        from RAG.agentic_policy_v2.embedding import l2_normalize

        nav = self.navigator
        valid = [d for d in candidate_ids if d in nav._doc_index]
        if not valid:
            return []

        if nav._bm25 is not None:
            all_bm25 = nav._bm25.get_scores(nav._tokenize(query))
            bm25 = np.array(
                [float(all_bm25[nav._doc_index[d]]) for d in valid], dtype="float32"
            )
        else:
            q_terms = set(nav._tokenize(query))
            bm25 = np.zeros(len(valid), dtype="float32")
            for i, d in enumerate(valid):
                doc = nav.index.doc_by_id[d]
                t = set(nav._tokenize(f"{doc.title} {doc.text}"))
                bm25[i] = len(q_terms & t) / max(1, len(q_terms))

        if bm25.std() > 1e-9:
            bm25_z = (bm25 - bm25.mean()) / bm25.std()
        else:
            bm25_z = np.zeros_like(bm25)

        qvec = np.asarray(self._query_encoder([query])[0], dtype="float32")
        qvec = l2_normalize(qvec[None, :])[0]
        sims = np.zeros(len(valid), dtype="float32")
        for i, d in enumerate(valid):
            row = self._dense_doc_id_to_row.get(d)
            if row is not None:
                sims[i] = float(np.dot(self._dense_embeddings[row], qvec))

        a = float(self.bm25_dense_alpha)
        final = a * sims + (1.0 - a) * bm25_z

        order = np.argsort(-final)[:top_k]
        return [valid[int(i)] for i in order]

    # ── Phase 1: recursive navigation ────────────────────────────────────────

    def _navigate(
        self,
        query: str,
        cluster_id: Optional[str],
        depth: int,
        visited_leaves: Set[str],
    ) -> None:
        """Recursively navigate the cluster tree, collecting leaf cluster IDs."""
        if len(visited_leaves) >= self.max_clusters:
            return

        children = self.navigator.get_children(cluster_id)
        node_id = cluster_id or "root"

        # Leaf or depth limit → collect this cluster
        if not children or depth >= self.max_depth:
            visited_leaves.add(node_id)
            return

        # Ask LLM (or heuristic) which children to explore
        decision = self._decide(query, cluster_id, children, depth)

        if decision["action"] == "RETRIEVE":
            visited_leaves.add(node_id)
            return

        targets = decision.get("targets", [])
        valid_child_ids = {c["id"] for c in children}
        targets = [t for t in targets if t in valid_child_ids]

        if not targets:
            # Fallback: pick top-2 by lexical score
            cards = self.navigator.get_cluster_cards(
                [c["id"] for c in children[:8]], token_budget=800, query=query
            )
            cards.sort(key=lambda c: c.get("lexical_score", 0), reverse=True)
            targets = [c["id"] for c in cards[:2]]

        targets = targets[: self.max_branches]
        for target_id in targets:
            if len(visited_leaves) >= self.max_clusters:
                break
            self._navigate(query, target_id, depth + 1, visited_leaves)

    # ── LLM decision ─────────────────────────────────────────────────────────

    def _decide(
        self,
        query: str,
        cluster_id: Optional[str],
        children: List[Dict],
        depth: int,
    ) -> Dict[str, Any]:
        if not self.llm_client:
            return self._heuristic_decision(query, children)

        child_ids = [c["id"] for c in children[:12]]
        cards = self.navigator.get_cluster_cards(
            child_ids, token_budget=1500, query=query
        )
        user_msg = self._build_user_message(query, cluster_id, cards, depth)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self.llm_client.generate(
                        user_msg, system_prompt=self.system_prompt, max_tokens=128
                    )
                )
            finally:
                loop.close()
            return self._parse_decision(response, cards)
        except Exception as exc:
            logger.warning("LLM decision failed (%s); using heuristic", exc)
            return self._heuristic_decision(query, children)

    def _heuristic_decision(
        self, query: str, children: List[Dict]
    ) -> Dict[str, Any]:
        """Lexical-score fallback when LLM is unavailable."""
        child_ids = [c["id"] for c in children[:8]]
        cards = self.navigator.get_cluster_cards(
            child_ids, token_budget=800, query=query
        )
        cards.sort(key=lambda c: c.get("lexical_score", 0), reverse=True)
        targets = [c["id"] for c in cards[:2]] if cards else []
        return {"action": "JUMP", "targets": targets}

    # ── Prompt / response helpers ─────────────────────────────────────────────

    def _build_user_message(
        self,
        query: str,
        cluster_id: Optional[str],
        cards: List[Dict[str, Any]],
        depth: int,
    ) -> str:
        lines = [
            f"Query: {query}",
            f"Current cluster: {cluster_id or 'ROOT'}",
            f"Depth: {depth} / {self.max_depth}",
            f"Clusters collected so far: exploring...",
            "",
            "Child clusters to choose from:",
        ]
        for card in cards:
            summary = (card.get("summary") or "")[:200]
            kw = ", ".join((card.get("keywords") or [])[:6])
            lines += [
                f"  ID: {card['id']}",
                f"  Label: {card.get('label', '?')}",
                f"  Summary: {summary}",
                f"  Keywords: {kw}",
                f"  Docs: {card.get('subtree_doc_count', 0)}",
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def _parse_decision(
        response: str, cards: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Robust JSON parse; falls back to top-2 children."""
        try:
            text = (response or "").strip()
            text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                action = str(obj.get("action", "JUMP")).upper()
                if action == "RETRIEVE":
                    return {"action": "RETRIEVE"}
                targets = obj.get("targets", [])
                if isinstance(targets, list) and targets:
                    return {"action": "JUMP", "targets": [str(t) for t in targets]}
        except Exception:
            pass
        targets = [c["id"] for c in cards[:2]]
        return {"action": "JUMP", "targets": targets}
