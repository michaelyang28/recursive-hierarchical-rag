"""
Standalone hierarchical document index and navigator.

Builds a portable hierarchy artifact (embeddings, k-means tree, BM25, cluster
cards) from document records and exposes query-time navigation and retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi

    RANK_BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    RANK_BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """A canonical document/chunk in the standalone hierarchy."""

    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_rag_document(self, score: float = 1.0, cluster_id: Optional[str] = None) -> Dict[str, Any]:
        metadata = dict(self.metadata or {})
        metadata.setdefault("doc_id", self.doc_id)
        return {
            "id": self.doc_id,
            "text": self.text or "",
            "title": self.title or "",
            "metadata": metadata,
            "cluster_id": cluster_id,
            "score": float(score),
        }


@dataclass
class HierarchyNode:
    """A node in the standalone hierarchy."""

    node_id: str
    parent_id: Optional[str]
    depth: int
    children: List[str] = field(default_factory=list)
    member_doc_ids: List[str] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    label: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)

    @property
    def subtree_doc_count(self) -> int:
        return len(self.member_doc_ids)


@dataclass
class HierarchyBuildConfig:
    """Configuration for standalone recursive k-means hierarchy construction."""

    embedding_model: str = "all-MiniLM-L6-v2"
    summarization_model: str = "google/flan-t5-base"
    summarization_device: Optional[int] = None
    branching_factor: int = 8
    max_depth: int = 4
    min_leaf_size: int = 25
    max_leaf_size: int = 250
    batch_size: int = 64
    normalize_embeddings: bool = True
    use_faiss: bool = True
    random_state: int = 42
    summary_mode: str = "keywords"
    docs_per_summary: int = 12
    max_input_chars_per_doc: int = 1200
    max_summary_new_tokens: int = 160


class HierarchyNavigatorProtocol(Protocol):
    """Protocol consumed by RecursiveLanguageModelRAG."""

    max_depth: int

    def get_children(self, cluster_id: Optional[str], depth: int = 1) -> List[Dict[str, str]]: ...

    def get_cluster_cards(
        self, cluster_ids: List[str], token_budget: int = 2000, query: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    def peek_cluster_documents(
        self, cluster_id: Optional[str], limit: int = 10, query: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    def search_in_cluster(
        self, cluster_id: Optional[str], query: str, limit: int = 20
    ) -> List[Dict[str, Any]]: ...

    def inspect_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]: ...

    def get_cluster_info(self, cluster_id: str) -> Dict[str, Any]: ...

    def get_full_hierarchy_text(self, mapping: Optional[Dict[str, str]] = None) -> str: ...

    def fetch_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]: ...


class StandaloneHierarchyIndex:
    """Portable hierarchy artifact with JSON/NumPy persistence."""

    CONFIG_FILE = "config.json"
    DOCS_FILE = "documents.jsonl"
    NODES_FILE = "nodes.jsonl"
    EMBEDDINGS_FILE = "embeddings.npy"

    def __init__(
        self,
        documents: List[DocumentRecord],
        nodes: List[HierarchyNode],
        config: Dict[str, Any],
        embeddings: Optional[np.ndarray] = None,
    ):
        self.documents = documents
        self.nodes = nodes
        self.config = config
        self.embeddings = embeddings
        self.doc_by_id = {doc.doc_id: doc for doc in documents}
        self.node_by_id = {node.node_id: node for node in nodes}
        self.root_ids = [node.node_id for node in nodes if node.parent_id is None]
        self.max_depth = max((node.depth for node in nodes), default=0)

    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        config = dict(self.config)
        config["num_documents"] = len(self.documents)
        config["num_nodes"] = len(self.nodes)
        config["max_depth"] = self.max_depth

        (path / self.CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
        with (path / self.DOCS_FILE).open("w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

        with (path / self.NODES_FILE).open("w", encoding="utf-8") as f:
            for node in self.nodes:
                f.write(json.dumps(asdict(node), ensure_ascii=False) + "\n")

        if self.embeddings is not None:
            np.save(path / self.EMBEDDINGS_FILE, self.embeddings.astype("float32"))

    @classmethod
    def load(cls, index_dir: str | Path, load_embeddings: bool = False) -> "StandaloneHierarchyIndex":
        path = Path(index_dir)
        config_path = path / cls.CONFIG_FILE
        docs_path = path / cls.DOCS_FILE
        nodes_path = path / cls.NODES_FILE
        if not config_path.exists() or not docs_path.exists() or not nodes_path.exists():
            raise FileNotFoundError(f"Missing standalone hierarchy files under {path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        documents = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(DocumentRecord(**json.loads(line)))

        nodes = []
        with nodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    nodes.append(HierarchyNode(**json.loads(line)))

        embeddings = None
        embeddings_path = path / cls.EMBEDDINGS_FILE
        if load_embeddings and embeddings_path.exists():
            embeddings = np.load(embeddings_path)

        return cls(documents=documents, nodes=nodes, config=config, embeddings=embeddings)


class StandaloneHierarchyBuilder:
    """Builds a recursive k-means hierarchy."""

    def __init__(self, config: Optional[HierarchyBuildConfig] = None):
        self.config = config or HierarchyBuildConfig()
        self._rng = np.random.default_rng(self.config.random_state)
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_device = "cpu"

    def build(
        self,
        documents: List[DocumentRecord],
        embeddings: Optional[np.ndarray] = None,
    ) -> StandaloneHierarchyIndex:
        if not documents:
            raise ValueError("Cannot build hierarchy with no documents")

        start = time.time()
        if embeddings is None:
            embeddings = self._embed_documents(documents)
        embeddings = np.asarray(embeddings, dtype="float32")
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if self.config.normalize_embeddings:
            embeddings = self._normalize(embeddings)

        nodes_by_id: Dict[str, HierarchyNode] = {}
        all_indices = list(range(len(documents)))
        self._build_node(
            node_id="root",
            parent_id=None,
            depth=0,
            member_indices=all_indices,
            embeddings=embeddings,
            documents=documents,
            nodes_by_id=nodes_by_id,
        )

        nodes = self._ordered_nodes(nodes_by_id)
        config = asdict(self.config)
        config.update(
            {
                "builder": "standalone_recursive_kmeans",
                "created_at_unix": time.time(),
                "build_seconds": time.time() - start,
            }
        )
        return StandaloneHierarchyIndex(
            documents=documents,
            nodes=nodes,
            config=config,
            embeddings=embeddings,
        )

    def _embed_documents(self, documents: List[DocumentRecord]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.config.embedding_model)
        texts = [f"{doc.title}\n{doc.text}".strip() for doc in documents]
        return model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def _build_node(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        member_indices: List[int],
        embeddings: np.ndarray,
        documents: List[DocumentRecord],
        nodes_by_id: Dict[str, HierarchyNode],
    ) -> None:
        member_doc_ids = [documents[i].doc_id for i in member_indices]
        centroid = embeddings[member_indices].mean(axis=0) if member_indices else np.zeros(embeddings.shape[1])
        keywords = self._keywords_for_docs(documents, member_indices)
        label = " / ".join(keywords[:3]) if keywords else f"Cluster {node_id}"
        summary = self._summary_for_docs(
            documents=documents,
            member_indices=member_indices,
            keywords=keywords,
            embeddings=embeddings,
            centroid=centroid,
        )

        node = HierarchyNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            member_doc_ids=member_doc_ids,
            centroid=centroid.astype("float32").tolist(),
            label=label,
            summary=summary,
            keywords=keywords,
        )
        nodes_by_id[node_id] = node

        if not self._should_split(depth, len(member_indices)):
            return

        k = self._num_children(len(member_indices))
        if k <= 1:
            return

        labels = self._cluster_subset(embeddings[member_indices], k)
        child_groups: Dict[int, List[int]] = defaultdict(list)
        for local_idx, label_id in enumerate(labels):
            child_groups[int(label_id)].append(member_indices[local_idx])

        if len(child_groups) <= 1 or any(len(group) < self.config.min_leaf_size for group in child_groups.values()):
            return

        for child_ordinal, (_, child_indices) in enumerate(sorted(child_groups.items(), key=lambda kv: kv[0])):
            child_id = f"{node_id}.{child_ordinal}" if node_id != "root" else f"c{child_ordinal}"
            node.children.append(child_id)
            self._build_node(
                node_id=child_id,
                parent_id=node_id,
                depth=depth + 1,
                member_indices=child_indices,
                embeddings=embeddings,
                documents=documents,
                nodes_by_id=nodes_by_id,
            )

    def _should_split(self, depth: int, n_members: int) -> bool:
        return depth < self.config.max_depth and n_members > self.config.max_leaf_size

    def _num_children(self, n_members: int) -> int:
        target_children = math.ceil(n_members / max(1, self.config.max_leaf_size))
        return max(2, min(self.config.branching_factor, target_children, n_members))

    def _cluster_subset(self, vectors: np.ndarray, k: int) -> np.ndarray:
        if self.config.use_faiss:
            try:
                import faiss

                d = vectors.shape[1]
                kmeans = faiss.Kmeans(
                    d,
                    k,
                    niter=20,
                    nredo=1,
                    verbose=False,
                    seed=self.config.random_state,
                    spherical=self.config.normalize_embeddings,
                )
                kmeans.train(vectors.astype("float32"))
                _, labels = kmeans.index.search(vectors.astype("float32"), 1)
                return labels.reshape(-1)
            except Exception as exc:
                logger.warning("Faiss k-means failed; falling back to MiniBatchKMeans: %s", exc)

        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.config.random_state,
            batch_size=max(1024, k * 32),
            n_init=10,
        )
        return model.fit_predict(vectors)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    @staticmethod
    def _ordered_nodes(nodes_by_id: Dict[str, HierarchyNode]) -> List[HierarchyNode]:
        return sorted(nodes_by_id.values(), key=lambda node: (node.depth, node.node_id))

    @staticmethod
    def _keywords_for_docs(documents: List[DocumentRecord], member_indices: List[int], limit: int = 8) -> List[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
            "has", "have", "into", "using", "used", "use", "its", "their", "our", "can",
            "will", "not", "but", "all", "also", "such", "these", "those", "between",
        }
        counter: Counter[str] = Counter()
        for idx in member_indices[:1000]:
            doc = documents[idx]
            text = f"{doc.title} {doc.text}".lower()
            for token in re.findall(r"[a-z][a-z0-9-]{2,}", text):
                if token not in stopwords:
                    counter[token] += 1
        return [term for term, _ in counter.most_common(limit)]

    def _summary_for_docs(
        self,
        documents: List[DocumentRecord],
        member_indices: List[int],
        keywords: List[str],
        embeddings: Optional[np.ndarray] = None,
        centroid: Optional[np.ndarray] = None,
    ) -> str:
        if self.config.summary_mode == "none":
            return ""
        if self.config.summary_mode == "hf_sampled":
            try:
                return self._hf_summary_for_docs(documents, member_indices, keywords, embeddings, centroid)
            except Exception as exc:
                logger.warning("HF summary generation failed; falling back to keyword summary: %s", exc)

        return self._keyword_summary_for_docs(documents, member_indices, keywords)

    @staticmethod
    def _keyword_summary_for_docs(
        documents: List[DocumentRecord],
        member_indices: List[int],
        keywords: List[str],
    ) -> str:
        sample_titles = [documents[i].title for i in member_indices[:5] if documents[i].title]
        keyword_text = ", ".join(keywords[:8]) if keywords else "mixed topics"
        title_text = "; ".join(sample_titles[:3])
        if title_text:
            return f"Documents about {keyword_text}. Representative titles: {title_text}."
        return f"Documents about {keyword_text}."

    def _hf_summary_for_docs(
        self,
        documents: List[DocumentRecord],
        member_indices: List[int],
        keywords: List[str],
        embeddings: Optional[np.ndarray],
        centroid: Optional[np.ndarray],
    ) -> str:
        representative_indices = self._representative_indices(member_indices, embeddings, centroid)
        paper_blocks = []
        for ordinal, idx in enumerate(representative_indices, start=1):
            doc = documents[idx]
            text = (doc.text or "")[: self.config.max_input_chars_per_doc]
            title = doc.title or "Untitled"
            paper_blocks.append(f"Paper {ordinal}: {title}\n{text}")

        keyword_text = ", ".join(keywords[:8]) if keywords else "unknown"
        prompt = (
            "Summarize the common research topic shared by these papers in 2-3 concise sentences. "
            "Then add a short 'Keywords:' line with 5-8 terms.\n\n"
            f"Initial keyword hints: {keyword_text}\n\n"
            + "\n\n".join(paper_blocks)
        )

        summary = self._generate_hf_text(prompt)
        if not summary:
            return self._keyword_summary_for_docs(documents, member_indices, keywords)
        summary = summary.strip()
        return summary or self._keyword_summary_for_docs(documents, member_indices, keywords)

    def _representative_indices(
        self,
        member_indices: List[int],
        embeddings: Optional[np.ndarray],
        centroid: Optional[np.ndarray],
    ) -> List[int]:
        sample_size = max(1, min(self.config.docs_per_summary, len(member_indices)))
        if embeddings is None or centroid is None or len(member_indices) <= sample_size:
            return member_indices[:sample_size]

        centroid_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        member_vectors = embeddings[member_indices]
        scores = member_vectors @ centroid_norm
        order = np.argsort(scores)[::-1][:sample_size]
        return [member_indices[int(i)] for i in order]

    def _get_hf_model(self):
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return self._hf_tokenizer, self._hf_model, self._hf_device

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = self.config.summarization_model
        preferred = self.config.summarization_device
        if preferred is not None and preferred >= 0 and torch.cuda.is_available():
            device = f"cuda:{preferred}"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        logger.info("Loading HF seq2seq model %s on %s", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        model.eval()

        self._hf_tokenizer = tokenizer
        self._hf_model = model
        self._hf_device = device
        return self._hf_tokenizer, self._hf_model, self._hf_device

    def _generate_hf_text(self, prompt: str) -> str:
        import torch

        tokenizer, model, device = self._get_hf_model()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_summary_new_tokens,
                do_sample=False,
                min_new_tokens=20,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)


class StandaloneHierarchyNavigator:
    """Navigator over a StandaloneHierarchyIndex compatible with recursive RLM."""

    def __init__(self, index: StandaloneHierarchyIndex):
        self.index = index
        self.max_depth = index.max_depth
        self._cluster_cache = {
            node.node_id: {
                "id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "label": node.label,
                "summary": node.summary,
                "keywords": node.keywords,
                "subtree_doc_count": node.subtree_doc_count,
            }
            for node in index.nodes
        }
        self._children_index = {node.node_id: list(node.children) for node in index.nodes}
        self._doc_cluster = self._build_doc_cluster_map()
        self._all_doc_ids = [doc.doc_id for doc in index.documents]
        self._build_search_index()

    @classmethod
    def load(cls, index_dir: str | Path) -> "StandaloneHierarchyNavigator":
        return cls(StandaloneHierarchyIndex.load(index_dir))

    def get_children(self, cluster_id: Optional[str], depth: int = 1) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        seen = set()

        def collect(parent_id: str, current_distance: int) -> None:
            if current_distance > depth:
                return
            for child_id in self._children_index.get(parent_id, []):
                if child_id in seen:
                    continue
                child = self.index.node_by_id[child_id]
                if child.subtree_doc_count > 0:
                    results.append({"id": child_id, "label": child.label})
                    seen.add(child_id)
                collect(child_id, current_distance + 1)

        if cluster_id is None:
            for root_id in self.index.root_ids:
                root = self.index.node_by_id[root_id]
                if root_id not in seen and root.subtree_doc_count > 0:
                    results.append({"id": root_id, "label": root.label})
                    seen.add(root_id)
                collect(root_id, 2)
        else:
            collect(cluster_id, 1)
        return results

    def get_cluster_cards(
        self, cluster_ids: List[str], token_budget: int = 2000, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_terms = set(self._tokenize(query or ""))
        cards: List[Dict[str, Any]] = []
        chars_used = 0
        max_chars = token_budget * 5
        for cluster_id in cluster_ids:
            node = self.index.node_by_id.get(cluster_id)
            if not node:
                continue
            summary = node.summary or ""
            text_terms = set(self._tokenize(" ".join([node.label, summary, " ".join(node.keywords)])))
            lexical_score = len(query_terms & text_terms) / max(1, len(query_terms)) if query_terms else 0.0
            if chars_used + len(summary) > max_chars:
                break
            cards.append(
                {
                    "id": node.node_id,
                    "label": node.label,
                    "summary": summary[:1000],
                    "keywords": node.keywords,
                    "depth": node.depth,
                    "lexical_score": lexical_score,
                    "subtree_doc_count": node.subtree_doc_count,
                }
            )
            chars_used += len(summary)
        return cards

    def peek_cluster_documents(
        self, cluster_id: Optional[str], limit: int = 10, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        docs = self._rank_doc_ids(self._doc_ids_for_cluster(cluster_id), query or "", limit)
        return [
            {
                "id": doc.doc_id,
                "title": doc.title or "Untitled",
                "snippet": self._snippet(doc.text),
            }
            for doc, _score in docs
        ]

    def search_in_cluster(
        self, cluster_id: Optional[str], query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        results = []
        for doc, score in self._rank_doc_ids(self._doc_ids_for_cluster(cluster_id), query, limit):
            results.append(
                {
                    "id": doc.doc_id,
                    "title": doc.title or "Untitled",
                    "snippet": self._snippet(doc.text),
                    "score": float(score),
                    "metadata": dict(doc.metadata or {}, doc_id=doc.doc_id),
                }
            )
        return results

    def inspect_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        return self.fetch_documents(doc_ids)

    def fetch_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        documents = []
        for doc_id in doc_ids:
            doc = self.index.doc_by_id.get(str(doc_id))
            if doc:
                documents.append(doc.to_rag_document(cluster_id=self._doc_cluster.get(doc.doc_id)))
        return documents

    def get_cluster_info(self, cluster_id: str) -> Dict[str, Any]:
        node = self.index.node_by_id.get(cluster_id)
        if not node:
            raise ValueError(f"Cluster {cluster_id} not found")
        return {
            "id": node.node_id,
            "label": node.label,
            "summary": node.summary,
            "keywords": node.keywords,
            "depth": node.depth,
        }

    def get_full_hierarchy_text(self, mapping: Optional[Dict[str, str]] = None) -> str:
        if not self.index.nodes:
            return "No clusters available."
        lines: List[str] = []
        next_id = 1

        def render(node_id: str, indent_level: int) -> None:
            nonlocal next_id
            node = self.index.node_by_id[node_id]
            display_id = node_id
            if mapping is not None:
                shorthand = f"CID-{next_id}"
                mapping[shorthand] = node_id
                display_id = shorthand
                next_id += 1
            indent = "  " * indent_level
            lines.append(f"{indent}- Cluster {node.label} (ID: {display_id}, Docs: {node.subtree_doc_count})")
            summary = node.summary or "No summary available."
            lines.append(f"{indent}  Summary: {summary[:200]}")
            for child_id in node.children:
                render(child_id, indent_level + 1)

        for root_id in self.index.root_ids:
            render(root_id, 0)
        return "\n".join(lines)

    def ground_query_infx(self, query: str, k: int = 40, align: bool = True) -> List[Dict[str, Any]]:
        logger.warning("StandaloneHierarchyNavigator does not provide INF-X retrieval; using lexical search.")
        return [
            {
                "doc_id": result["id"],
                "beir_id": result["metadata"].get("doc_id"),
                "title": result["title"],
                "snippet": result["snippet"],
                "infx_score": result["score"],
                "cluster_id": self._doc_cluster.get(result["id"], ""),
                "metadata": result["metadata"],
            }
            for result in self.search_in_cluster(None, query, k)
        ]

    def _build_doc_cluster_map(self) -> Dict[str, str]:
        leaves = [node for node in self.index.nodes if not node.children]
        doc_cluster: Dict[str, str] = {}
        for leaf in leaves:
            for doc_id in leaf.member_doc_ids:
                doc_cluster[doc_id] = leaf.node_id
        return doc_cluster

    def _build_search_index(self) -> None:
        corpus = [self._tokenize(f"{doc.title} {doc.text}") for doc in self.index.documents]
        self._doc_index = {doc.doc_id: idx for idx, doc in enumerate(self.index.documents)}
        self._tokenized_corpus = corpus
        self._bm25 = BM25Okapi(corpus) if RANK_BM25_AVAILABLE and corpus else None

    def _doc_ids_for_cluster(self, cluster_id: Optional[str]) -> List[str]:
        """Return all doc IDs belonging to a cluster and all its descendants."""
        if cluster_id is None:
            return self._all_doc_ids
        node = self.index.node_by_id.get(cluster_id)
        if not node:
            return []
        # Collect direct members first
        direct = list(node.member_doc_ids)
        if direct:
            return direct
        # Internal node: walk all descendants and collect their leaf docs
        all_ids: List[str] = []
        queue = list(self._children_index.get(cluster_id, []))
        visited: set = set()
        while queue:
            child_id = queue.pop(0)
            if child_id in visited:
                continue
            visited.add(child_id)
            child_node = self.index.node_by_id.get(child_id)
            if child_node:
                all_ids.extend(child_node.member_doc_ids)
                queue.extend(self._children_index.get(child_id, []))
        return all_ids

    def _rank_doc_ids(
        self, candidate_doc_ids: Iterable[str], query: str, limit: int
    ) -> List[Tuple[DocumentRecord, float]]:
        candidate_doc_ids = list(candidate_doc_ids)
        if not candidate_doc_ids:
            return []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(self._tokenize(query))
            scored = [
                (self.index.doc_by_id[doc_id], float(scores[self._doc_index[doc_id]]))
                for doc_id in candidate_doc_ids
                if doc_id in self._doc_index
            ]
        else:
            query_terms = set(self._tokenize(query))
            scored = []
            for doc_id in candidate_doc_ids:
                doc = self.index.doc_by_id.get(doc_id)
                if not doc:
                    continue
                doc_terms = set(self._tokenize(f"{doc.title} {doc.text}"))
                score = len(query_terms & doc_terms) / max(1, len(query_terms))
                scored.append((doc, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        if any(score > 0 for _doc, score in scored):
            scored = [(doc, score) for doc, score in scored if score > 0]
        return scored[:limit]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return (text or "").lower().translate(str.maketrans("", "", string.punctuation)).split()

    @staticmethod
    def _snippet(text: str, limit: int = 300) -> str:
        text = text or ""
        return text[:limit] + "..." if len(text) > limit else text
