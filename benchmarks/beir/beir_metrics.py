"""
BEIR standard evaluation metrics.

Implements the official BEIR evaluation metrics:
- nDCG@k (primary metric for BEIR)
- MAP@k (Mean Average Precision)
- Recall@k
- MRR@k (Mean Reciprocal Rank)
- Precision@k

Supports both pytrec_eval (fast, official TREC implementation) and manual calculation.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pytrec_eval for official TREC metrics
try:
    import pytrec_eval
    HAS_PYTREC_EVAL = True
except ImportError:
    HAS_PYTREC_EVAL = False
    logger.warning(
        "pytrec_eval not found. Install with: pip install pytrec_eval\n"
        "Falling back to manual metric calculation (slower, may differ slightly from official BEIR)."
    )


class BEIRMetrics:
    """
    BEIR standard evaluation metrics calculator.

    Computes IR metrics with graded relevance support (e.g., NFCorpus uses grades 1 and 2).
    """

    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int] = [1, 3, 5, 10, 100, 1000]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results using BEIR standard metrics.

        Args:
            qrels: Ground truth relevance judgments
                   Format: {query_id: {doc_id: relevance_score}}
            results: Retrieval results with scores
                     Format: {query_id: {doc_id: score}}
            k_values: List of k values to compute metrics for

        Returns:
            Dictionary of metric_name -> score
            Example: {"NDCG@10": 0.322, "MAP@10": 0.245, ...}
        """
        if HAS_PYTREC_EVAL:
            return BEIRMetrics._evaluate_with_pytrec(qrels, results, k_values)
        else:
            return BEIRMetrics._evaluate_manual(qrels, results, k_values)

    @staticmethod
    def _evaluate_with_pytrec(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """Evaluate using official pytrec_eval library (matches BEIR toolkit)."""

        # Build measure strings for pytrec_eval
        measures = set()
        for k in k_values:
            measures.add(f'ndcg_cut.{k}')
            measures.add(f'map_cut.{k}')
            measures.add(f'recall.{k}')
            measures.add(f'P.{k}')  # Precision
        measures.add('recip_rank')  # MRR

        # Create evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)

        # Evaluate
        scores = evaluator.evaluate(results)

        # Aggregate across queries.
        # NB: pytrec_eval emits metric keys with '.' replaced by '_'
        # (e.g. ``ndcg_cut.10`` -> ``ndcg_cut_10``), so we look up the
        # underscore form in each query's score dict.
        aggregated = {}
        for measure in measures:
            lookup_key = measure.replace('.', '_')
            if measure == 'recip_rank':
                values = [query_scores.get(lookup_key, 0.0) for query_scores in scores.values()]
                aggregated['MRR'] = np.mean(values) if values else 0.0
            else:
                values = [query_scores.get(lookup_key, 0.0) for query_scores in scores.values()]
                mean_val = np.mean(values) if values else 0.0

                if 'ndcg_cut' in measure:
                    k = measure.split('.')[-1]
                    aggregated[f'NDCG@{k}'] = mean_val
                elif 'map_cut' in measure:
                    k = measure.split('.')[-1]
                    aggregated[f'MAP@{k}'] = mean_val
                elif 'recall' in measure:
                    k = measure.split('.')[-1]
                    aggregated[f'Recall@{k}'] = mean_val
                elif measure.startswith('P.'):
                    k = measure.split('.')[-1]
                    aggregated[f'P@{k}'] = mean_val

        return aggregated

    @staticmethod
    def _evaluate_manual(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """Manual implementation of BEIR metrics (fallback when pytrec_eval unavailable)."""

        metrics = {}

        for k in k_values:
            ndcg_scores = []
            map_scores = []
            recall_scores = []
            precision_scores = []
            mrr_scores = []

            for query_id in qrels.keys():
                if query_id not in results:
                    continue

                # Get ground truth relevance for this query
                query_qrels = qrels[query_id]

                # Get retrieved docs sorted by score (descending)
                query_results = results[query_id]
                sorted_docs = sorted(query_results.items(), key=lambda x: x[1], reverse=True)

                # Get top-k docs
                top_k_docs = sorted_docs[:k]
                top_k_doc_ids = [doc_id for doc_id, _ in top_k_docs]

                # Calculate metrics for this query
                ndcg_scores.append(BEIRMetrics._calculate_ndcg(top_k_doc_ids, query_qrels, k))
                map_scores.append(BEIRMetrics._calculate_average_precision(top_k_doc_ids, query_qrels))
                recall_scores.append(BEIRMetrics._calculate_recall(top_k_doc_ids, query_qrels))
                precision_scores.append(BEIRMetrics._calculate_precision(top_k_doc_ids, query_qrels))

                # MRR only needs to be calculated once per query (not per k)
                if k == max(k_values):
                    all_docs = [doc_id for doc_id, _ in sorted_docs]
                    mrr_scores.append(BEIRMetrics._calculate_reciprocal_rank(all_docs, query_qrels))

            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            metrics[f'MAP@{k}'] = np.mean(map_scores) if map_scores else 0.0
            metrics[f'Recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            metrics[f'P@{k}'] = np.mean(precision_scores) if precision_scores else 0.0

            if k == max(k_values):
                metrics['MRR'] = np.mean(mrr_scores) if mrr_scores else 0.0

        return metrics

    @staticmethod
    def _calculate_ndcg(retrieved_docs: List[str], qrels: Dict[str, int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        nDCG = DCG / IDCG
        DCG = sum((2^rel_i - 1) / log2(i + 1)) for i in positions
        """
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], start=1):
            rel = qrels.get(doc_id, 0)
            dcg += (2**rel - 1) / np.log2(i + 1)

        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_rels = sorted(qrels.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, start=1):
            idcg += (2**rel - 1) / np.log2(i + 1)

        # Normalize
        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _calculate_average_precision(retrieved_docs: List[str], qrels: Dict[str, int]) -> float:
        """
        Calculate Average Precision.

        AP = (sum of P@k for each relevant doc) / number of relevant docs
        """
        num_relevant = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in qrels and qrels[doc_id] > 0:
                num_relevant += 1
                precision_at_i = num_relevant / i
                precision_sum += precision_at_i

        total_relevant = sum(1 for rel in qrels.values() if rel > 0)
        if total_relevant == 0:
            return 0.0

        return precision_sum / total_relevant

    @staticmethod
    def _calculate_recall(retrieved_docs: List[str], qrels: Dict[str, int]) -> float:
        """
        Calculate Recall@k.

        Recall = (number of relevant docs retrieved) / (total number of relevant docs)
        """
        num_retrieved_relevant = sum(1 for doc_id in retrieved_docs if doc_id in qrels and qrels[doc_id] > 0)
        total_relevant = sum(1 for rel in qrels.values() if rel > 0)

        if total_relevant == 0:
            return 0.0

        return num_retrieved_relevant / total_relevant

    @staticmethod
    def _calculate_precision(retrieved_docs: List[str], qrels: Dict[str, int]) -> float:
        """
        Calculate Precision@k.

        Precision = (number of relevant docs retrieved) / k
        """
        if len(retrieved_docs) == 0:
            return 0.0

        num_relevant = sum(1 for doc_id in retrieved_docs if doc_id in qrels and qrels[doc_id] > 0)
        return num_relevant / len(retrieved_docs)

    @staticmethod
    def _calculate_reciprocal_rank(retrieved_docs: List[str], qrels: Dict[str, int]) -> float:
        """
        Calculate Reciprocal Rank (for MRR).

        RR = 1 / (rank of first relevant document)
        """
        for i, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in qrels and qrels[doc_id] > 0:
                return 1.0 / i
        return 0.0


def print_beir_metrics(metrics: Dict[str, float], dataset_name: str = ""):
    """
    Pretty print BEIR metrics in standard format.

    Args:
        metrics: Dictionary of metric_name -> score
        dataset_name: Optional dataset name for header
    """
    print("\n" + "="*70)
    if dataset_name:
        print(f"BEIR EVALUATION RESULTS - {dataset_name.upper()}")
    else:
        print("BEIR EVALUATION RESULTS")
    print("="*70)

    # Print in standard BEIR order: nDCG@10 (primary), then others
    metric_order = [
        'NDCG@10',   # PRIMARY METRIC
        'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@100',
        'MAP@10',
        'MAP@1', 'MAP@3', 'MAP@5', 'MAP@100',
        'Recall@100',  # SECONDARY METRIC
        'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@1000',
        'P@10',
        'P@1', 'P@3', 'P@5', 'P@100',
        'MRR'
    ]

    # Group by metric type
    print("\n📊 PRIMARY METRICS (for leaderboard comparison):")
    for metric_name in ['NDCG@10', 'Recall@100']:
        if metric_name in metrics:
            print(f"  {metric_name:15s}: {metrics[metric_name]:.4f}")

    print("\n📈 nDCG (Normalized Discounted Cumulative Gain):")
    for metric_name in metric_order:
        if metric_name.startswith('NDCG') and metric_name in metrics:
            print(f"  {metric_name:15s}: {metrics[metric_name]:.4f}")

    print("\n📍 MAP (Mean Average Precision):")
    for metric_name in metric_order:
        if metric_name.startswith('MAP') and metric_name in metrics:
            print(f"  {metric_name:15s}: {metrics[metric_name]:.4f}")

    print("\n🎯 Recall:")
    for metric_name in metric_order:
        if metric_name.startswith('Recall') and metric_name in metrics:
            print(f"  {metric_name:15s}: {metrics[metric_name]:.4f}")

    print("\n✓ Precision:")
    for metric_name in metric_order:
        if metric_name.startswith('P@') and metric_name in metrics:
            print(f"  {metric_name:15s}: {metrics[metric_name]:.4f}")

    if 'MRR' in metrics:
        print("\n⚡ MRR (Mean Reciprocal Rank):")
        print(f"  MRR            : {metrics['MRR']:.4f}")

    print("="*70)
    print("\nNote: nDCG@10 is the primary metric for BEIR benchmark comparison.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    qrels = {
        'q1': {'doc1': 2, 'doc2': 1, 'doc5': 1},  # Graded relevance
        'q2': {'doc3': 1, 'doc4': 2}
    }

    results = {
        'q1': {'doc1': 0.9, 'doc2': 0.8, 'doc3': 0.7, 'doc5': 0.6},
        'q2': {'doc4': 0.95, 'doc3': 0.85, 'doc1': 0.5}
    }

    metrics = BEIRMetrics.evaluate(qrels, results, k_values=[1, 3, 5, 10])
    print_beir_metrics(metrics, "Example Dataset")
