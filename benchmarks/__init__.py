"""Benchmarks package for RAG evaluation."""

from .beir.beir_loader import BEIRBenchmark
from .beir.beir_metrics import BEIRMetrics, print_beir_metrics

__all__ = [
    "BEIRBenchmark",
    "BEIRMetrics",
    "print_beir_metrics",
]
