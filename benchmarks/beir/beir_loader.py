"""BEIR benchmark loader from HuggingFace datasets."""

from typing import List, Dict, Any, Optional
from datasets import load_dataset
import pandas as pd

class BEIRBenchmark:
    """Loader for BEIR benchmark datasets (NFCorpus, SciFact, etc.)."""

    def __init__(self, dataset_name: str = "nfcorpus", split: str = "test", cache_dir: Optional[str] = None):
        """
        Initialize the BEIR benchmark loader.

        Args:
            dataset_name: Name of the BEIR dataset (e.g., "nfcorpus", "scifact")
            split: Dataset split to load (usually "test" or "dev" for evaluation)
            cache_dir: Optional cache directory for dataset
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.cache_dir = cache_dir
        
        # BEIR datasets on HuggingFace are prefixed with "BeIR/" (uppercase IR)
        self.hf_path = f"BeIR/{self.dataset_name}"
        
        self.corpus = None
        self.queries = None
        self.qrels = None
        
        self._load_dataset()

    def _load_dataset(self):
        """Load corpus, queries, and qrels from HuggingFace."""
        print(f"Loading BEIR/{self.dataset_name} ({self.split} split)...")
        
        # Load corpus and queries from the main Beir/{dataset_name} dataset
        self.corpus_ds = load_dataset(self.hf_path, "corpus", split="corpus", cache_dir=self.cache_dir)
        self.queries_ds = load_dataset(self.hf_path, "queries", split="queries", cache_dir=self.cache_dir)
        
        # Load qrels from a separate Beir/{dataset_name}-qrels dataset
        qrels_path = f"{self.hf_path}-qrels"
        self.qrels_ds = load_dataset(qrels_path, split=self.split, cache_dir=self.cache_dir)
        
        print(f"Loaded {len(self.corpus_ds)} documents and {len(self.queries_ds)} queries")

    def get_corpus(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the corpus.
        
        Returns:
            List of dictionaries with '_id', 'title', 'text'
        """
        return [doc for doc in self.corpus_ds]

    def get_test_examples(self) -> List[Dict[str, Any]]:
        """
        Get queries that have at least one relevance judgment in the split.

        Returns:
            List of dictionaries with 'id', 'question', 'ground_truth_docs', 'qrels'
            where qrels is a dict mapping doc_id to relevance score (preserves graded relevance)
        """
        # Map query_id -> dict of {doc_id: relevance_score}
        # This preserves graded relevance (e.g., NFCorpus uses 1 and 2)
        qrels_map = {}
        for row in self.qrels_ds:
            q_id = str(row['query-id'])
            doc_id = str(row['corpus-id'])
            score = row['score']
            if score > 0:
                if q_id not in qrels_map:
                    qrels_map[q_id] = {}
                qrels_map[q_id][doc_id] = score

        # Map query_id -> query_text
        query_map = {str(q['_id']): q['text'] for q in self.queries_ds}

        examples = []
        for q_id, doc_scores in qrels_map.items():
            if q_id in query_map:
                examples.append({
                    "id": str(q_id),
                    "question": query_map[q_id],
                    "ground_truth_docs": [str(d) for d in doc_scores.keys()],  # For backward compatibility
                    "qrels": {str(d): s for d, s in doc_scores.items()}  # Preserve graded relevance
                })

        return examples

    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """
        Get qrels in standard BEIR/TREC format: {query_id: {doc_id: relevance_score}}

        Returns:
            Nested dictionary with graded relevance judgments
        """
        qrels = {}
        for row in self.qrels_ds:
            q_id = str(row['query-id'])
            doc_id = str(row['corpus-id'])
            score = row['score']
            if score > 0:
                if q_id not in qrels:
                    qrels[q_id] = {}
                qrels[q_id][doc_id] = score
        return qrels

    def __len__(self) -> int:
        """Return the number of queries in the evaluation split."""
        return len(self.qrels_ds)
