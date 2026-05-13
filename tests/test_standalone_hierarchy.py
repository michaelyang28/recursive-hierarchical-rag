import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.standalone_hierarchy import (
    DocumentRecord,
    HierarchyBuildConfig,
    StandaloneHierarchyBuilder,
    StandaloneHierarchyIndex,
    StandaloneHierarchyNavigator,
)


def _toy_documents():
    return [
        DocumentRecord("d1", "Neural retrieval models rank scientific papers.", "Neural retrieval"),
        DocumentRecord("d2", "Dense embeddings improve semantic search.", "Dense search"),
        DocumentRecord("d3", "Tomatoes and basil are used in pasta sauce.", "Pasta"),
        DocumentRecord("d4", "Bread recipes often use yeast and flour.", "Bread"),
    ]


def _toy_embeddings():
    return np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.1, 1.0],
            [0.0, 0.0, 0.9],
        ],
        dtype="float32",
    )


def test_standalone_hierarchy_round_trip(tmp_path):
    config = HierarchyBuildConfig(
        branching_factor=2,
        max_depth=2,
        min_leaf_size=1,
        max_leaf_size=2,
        use_faiss=False,
    )
    index = StandaloneHierarchyBuilder(config).build(_toy_documents(), embeddings=_toy_embeddings())
    index.save(tmp_path)

    loaded = StandaloneHierarchyIndex.load(tmp_path)
    assert len(loaded.documents) == 4
    assert "root" in loaded.node_by_id
    assert loaded.max_depth >= 1

    navigator = StandaloneHierarchyNavigator(loaded)
    children = navigator.get_children(None)
    assert children

    results = navigator.search_in_cluster(None, "neural retrieval", limit=2)
    assert results[0]["metadata"]["doc_id"] == "d1"
