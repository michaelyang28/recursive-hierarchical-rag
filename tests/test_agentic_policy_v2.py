"""Unit tests for the supervised agentic policy v2.

Plan section 1.9 / 6 invariants:

(a) state tensor shape and presence flags
(b) multi-positive listwise jump CE matches hand computation on a 4-candidate toy
(c) padding mask zeros gradient on padded slots
(d) oracle rollout on a synthetic 3-level toy tree visits all targets and
    emits §3.13-valid action/done sequence
(e) parameter-sharing invariants (single JumpScorer instance, candidate
    permutation invariance, no node_id in state_dict keys)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from RAG.agentic_policy_v2 import losses as v2_losses  # noqa: E402
from RAG.agentic_policy_v2.dataset import SupervisionDataset  # noqa: E402
from RAG.agentic_policy_v2.network import (  # noqa: E402
    ACTION_AGGREGATE,
    ACTION_JUMP,
    ACTION_RETRIEVE,
    PolicyConfig,
    PolicyNetwork,
)
from RAG.agentic_policy_v2.node_ann import NodeCentroidIndex  # noqa: E402
from RAG.agentic_policy_v2.state import (  # noqa: E402
    META_DIM,
    NodeFeatureLookup,
    build_state_tensor,
)
from RAG.agentic_policy_v2.supervision import (  # noqa: E402
    SupervisionConfig,
    SupervisionGenerator,
    ancestors_of,
    assert_example_invariants,
)
from RAG.standalone_hierarchy import (  # noqa: E402
    DocumentRecord,
    HierarchyNode,
    StandaloneHierarchyIndex,
)


def _make_toy_tree(embedding_dim: int = 8, seed: int = 0) -> StandaloneHierarchyIndex:
    """3-level binary tree with 8 leaves, each leaf carrying 2 docs.

    root
    ├── c0
    │   ├── c0.0  (leaf, docs d0, d1)
    │   ├── c0.1  (leaf, docs d2, d3)
    │   ├── c0.2  (leaf, docs d4, d5)
    │   └── c0.3  (leaf, docs d6, d7)
    └── c1
        ├── c1.0  (leaf, docs d8, d9)
        ├── c1.1  (leaf, docs d10, d11)
        ├── c1.2  (leaf, docs d12, d13)
        └── c1.3  (leaf, docs d14, d15)
    """

    rng = np.random.default_rng(seed)
    documents = [
        DocumentRecord(doc_id=f"d{i}", text=f"document {i} text", title=f"Doc {i}")
        for i in range(16)
    ]
    embeddings = rng.standard_normal((16, embedding_dim)).astype("float32")
    embeddings = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )

    nodes = [HierarchyNode(node_id="root", parent_id=None, depth=0)]
    for i in range(2):
        nodes.append(HierarchyNode(node_id=f"c{i}", parent_id="root", depth=1))
    for i in range(2):
        for j in range(4):
            doc_ids = [f"d{i*8 + j*2}", f"d{i*8 + j*2 + 1}"]
            nodes.append(
                HierarchyNode(
                    node_id=f"c{i}.{j}",
                    parent_id=f"c{i}",
                    depth=2,
                    member_doc_ids=doc_ids,
                )
            )

    nodes_by_id = {n.node_id: n for n in nodes}
    nodes_by_id["root"].children = ["c0", "c1"]
    nodes_by_id["c0"].children = ["c0.0", "c0.1", "c0.2", "c0.3"]
    nodes_by_id["c1"].children = ["c1.0", "c1.1", "c1.2", "c1.3"]

    leaf_indices = {f"c{i}.{j}": list(range(i * 8 + j * 2, i * 8 + j * 2 + 2)) for i in range(2) for j in range(4)}
    for leaf_id, idxs in leaf_indices.items():
        nodes_by_id[leaf_id].centroid = np.asarray(
            embeddings[idxs].mean(axis=0), dtype="float32"
        ).tolist()
    for c in ("c0", "c1"):
        leaf_ids = nodes_by_id[c].children
        leaf_centroids = np.stack(
            [np.asarray(nodes_by_id[lid].centroid, dtype="float32") for lid in leaf_ids]
        )
        nodes_by_id[c].centroid = leaf_centroids.mean(axis=0).astype("float32").tolist()
    nodes_by_id["root"].member_doc_ids = [d.doc_id for d in documents]
    nodes_by_id["root"].centroid = embeddings.mean(axis=0).astype("float32").tolist()

    for c in ("c0", "c1"):
        member: list = []
        for lid in nodes_by_id[c].children:
            member.extend(nodes_by_id[lid].member_doc_ids)
        nodes_by_id[c].member_doc_ids = member

    config = {
        "embedding_model": "local-hash-embedding",
        "max_depth": 2,
        "num_documents": 16,
        "num_nodes": len(nodes),
    }
    return StandaloneHierarchyIndex(
        documents=documents,
        nodes=nodes,
        config=config,
        embeddings=embeddings,
    )


def test_state_tensor_shape_and_presence_flags():
    idx = _make_toy_tree(embedding_dim=8)
    lookup = NodeFeatureLookup(idx)

    state_root = lookup.make_state(query="q", current_node_id="root")
    state_leaf = lookup.make_state(
        query="q",
        current_node_id="c0.0",
        path_node_ids=["root", "c0", "c0.0"],
        evidence_doc_ids=["d0"],
    )
    q = np.zeros(lookup.embedding_dim, dtype="float32")
    q[0] = 1.0

    x_root = build_state_tensor(state_root, q, lookup)
    x_leaf = build_state_tensor(state_leaf, q, lookup)
    assert x_root.shape == (6 * lookup.embedding_dim + META_DIM,)
    assert x_leaf.shape == x_root.shape

    meta_root = x_root[-META_DIM:]
    meta_leaf = x_leaf[-META_DIM:]
    feature_names = [
        "depth_norm",
        "log_subtree_size",
        "is_leaf",
        "has_parent",
        "has_children",
        "has_path",
        "has_evidence",
    ]
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    assert meta_root[name_to_idx["has_parent"]] == 0.0
    assert meta_root[name_to_idx["has_children"]] == 1.0
    assert meta_root[name_to_idx["is_leaf"]] == 0.0
    assert meta_root[name_to_idx["has_path"]] == 0.0
    assert meta_root[name_to_idx["has_evidence"]] == 0.0

    assert meta_leaf[name_to_idx["has_parent"]] == 1.0
    assert meta_leaf[name_to_idx["has_children"]] == 0.0
    assert meta_leaf[name_to_idx["is_leaf"]] == 1.0
    assert meta_leaf[name_to_idx["has_path"]] == 1.0
    assert meta_leaf[name_to_idx["has_evidence"]] == 1.0


def test_multi_positive_listwise_ce_hand_computation():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    pos_mask = torch.tensor([[1, 1, 0, 0]])
    cand_mask = torch.tensor([[1, 1, 1, 1]])
    keep = torch.tensor([1.0])

    loss = v2_losses.multi_positive_listwise_ce(logits, pos_mask, cand_mask, keep)

    pos_lse = math.log(math.exp(2.0) + math.exp(1.0))
    full_lse = math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.5) + math.exp(-1.0))
    expected = full_lse - pos_lse
    assert pytest.approx(expected, rel=1e-6) == loss.item()


def test_padding_mask_zeros_padded_gradient():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, requires_grad=True)
    pos_mask = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    cand_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    keep = torch.tensor([1.0, 1.0])

    loss = v2_losses.multi_positive_listwise_ce(logits, pos_mask, cand_mask, keep)
    loss.backward()
    grad = logits.grad
    assert torch.all(grad[0, 3:].abs() < 1e-6)
    assert torch.all(grad[1, 2:].abs() < 1e-6)
    assert torch.any(grad[0, :3].abs() > 1e-6)
    assert torch.any(grad[1, :2].abs() > 1e-6)


def test_loop_probability_mass_penalty_increases_with_bad_mass():
    loop_mask = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.float32)
    cand_mask = torch.ones_like(loop_mask)

    low_bad = torch.tensor([[3.0, -3.0, 2.0], [2.0, -2.0, 1.0]])
    high_bad = torch.tensor([[-3.0, 3.0, -2.0], [-2.0, 2.0, -1.0]])

    low_loss = v2_losses.loop_probability_mass_penalty(low_bad, loop_mask, cand_mask)
    high_loss = v2_losses.loop_probability_mass_penalty(high_bad, loop_mask, cand_mask)

    assert high_loss > low_loss
    assert low_loss >= 0.0


def test_dataset_marks_visited_jump_candidates_as_loop_risks(tmp_path):
    idx = _make_toy_tree(embedding_dim=8)
    lookup = NodeFeatureLookup(idx)
    q_vec = lookup.doc_embedding_matrix(["d0"]).mean(axis=0)
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)

    row = {
        "query_id": "q-loop",
        "query": "toy query",
        "step_index": 2,
        "trajectory_id": "q-loop:0",
        "is_off_path": False,
        "state": {
            "node_id": "c0",
            "depth": 1,
            "is_leaf": False,
            "subtree_doc_count": 8,
            "path_node_ids": ["root", "c0.1", "c0"],
            "evidence_doc_ids": [],
        },
        "action_label": ACTION_JUMP,
        "done_label": 0,
        "jump": {
            "candidate_node_ids": ["c0.0", "c0.1", "c1", ""],
            "candidate_mask": [1, 1, 1, 0],
            "positive_indices": [0],
            "candidate_sources": ["positive", "child", "ann", ""],
        },
        "retrieve": None,
        "debug": {"oracle_path_node_ids": ["root", "c0", "c0.0"]},
    }
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    qemb_path = tmp_path / "qemb.npz"
    np.savez(qemb_path, query_ids=np.array(["q-loop"]), embeddings=np.stack([q_vec]))

    ds = SupervisionDataset(jsonl_path, qemb_path, lookup, K_max=4, M_max=4)
    item = ds[0]

    assert item["jump_pos_mask"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert item["jump_loop_mask"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_oracle_rollout_visits_targets_and_invariants_hold():
    idx = _make_toy_tree(embedding_dim=8)
    lookup = NodeFeatureLookup(idx)
    ann = NodeCentroidIndex.from_hierarchy(idx, embedding_model="local-hash-embedding", use_faiss=False)
    cfg = SupervisionConfig(ann_K=4, K_max=8, M_max=4, n_off_path=1, retrieve_threshold=0)
    gen = SupervisionGenerator(idx, lookup, ann, cfg)

    q_vec = lookup.doc_embedding_matrix(["d0", "d11"]).mean(axis=0)
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)
    examples = gen.generate_for_query(
        query_id="q1",
        query="toy query",
        relevant_doc_ids=["d0", "d11"],
        query_vec=q_vec.astype("float32"),
    )
    assert examples, "oracle rollout produced no examples"
    on_path = [e for e in examples if not e.is_off_path]
    nodes_visited = {e.state["node_id"] for e in on_path}
    assert "c0.0" in nodes_visited
    assert "c1.1" in nodes_visited

    actions = [e.action_label for e in on_path]
    assert ACTION_RETRIEVE in actions
    assert ACTION_JUMP in actions
    assert ACTION_AGGREGATE in actions

    done_count = sum(1 for e in examples if e.done_label == 1)
    assert done_count >= 1
    assert any(
        e.done_label == 1
        and set(e.state["evidence_doc_ids"]) >= {"d0", "d11"}
        for e in on_path
    )

    for ex in examples:
        assert_example_invariants(ex, root_id="root")


def test_retrieve_repeats_increase_retrieve_rows_without_moving_rollout():
    idx = _make_toy_tree(embedding_dim=8)
    lookup = NodeFeatureLookup(idx)
    ann = NodeCentroidIndex.from_hierarchy(idx, embedding_model="local-hash-embedding", use_faiss=False)
    cfg = SupervisionConfig(
        ann_K=4,
        K_max=8,
        M_max=4,
        n_off_path=0,
        retrieve_threshold=0,
        retrieve_action_repeats=4,
    )
    gen = SupervisionGenerator(idx, lookup, ann, cfg)

    q_vec = lookup.doc_embedding_matrix(["d0"]).mean(axis=0)
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)
    examples = gen.generate_for_query(
        query_id="q-repeat",
        query="toy query",
        relevant_doc_ids=["d0"],
        query_vec=q_vec.astype("float32"),
    )

    retrieve_rows = [e for e in examples if e.action_label == ACTION_RETRIEVE]
    assert len(retrieve_rows) == 4
    assert {e.state["node_id"] for e in retrieve_rows} == {"c0.0"}
    assert {e.step_index for e in retrieve_rows} == {2}
    assert [e.debug["retrieve_repeat_index"] for e in retrieve_rows] == [0, 1, 2, 3]


def test_jump_path_positives_exclude_ancestors_of_current_node():
    idx = _make_toy_tree(embedding_dim=8)
    lookup = NodeFeatureLookup(idx)
    ann = NodeCentroidIndex.from_hierarchy(idx, embedding_model="local-hash-embedding", use_faiss=False)
    cfg = SupervisionConfig(
        ann_K=8,
        K_max=12,
        M_max=4,
        n_off_path=0,
        retrieve_threshold=0,
    )
    gen = SupervisionGenerator(idx, lookup, ann, cfg)

    q_vec = lookup.doc_embedding_matrix(["d0"]).mean(axis=0)
    q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)
    jump_block = gen._build_jump_candidates(
        current_node="c0",
        positive_node_ids=["c0.0"],
        all_remaining_path_nodes=set(),
        query_vec=q_vec.astype("float32"),
        target_path={"root", "c0", "c0.0"},
    )
    positive_ids = {
        jump_block["candidate_node_ids"][i] for i in jump_block["positive_indices"]
    }
    assert "c0.0" in positive_ids
    assert "root" not in positive_ids
    assert "c0" not in positive_ids


def test_parameter_sharing_invariants():
    cfg = PolicyConfig(embedding_dim=8, meta_dim=META_DIM, hidden_dim=16, jump_hidden_dim=16)
    model = PolicyNetwork(cfg)
    keys = list(model.state_dict().keys())
    for k in keys:
        assert "node_id" not in k

    seen_jump_param_ids = {id(p) for p in model.jump_scorer.parameters()}
    assert len(seen_jump_param_ids) > 0
    for p in model.parameters():
        if id(p) in seen_jump_param_ids:
            seen_jump_param_ids.discard(id(p))
    assert seen_jump_param_ids == set()


def test_jump_scorer_permutation_invariance():
    torch.manual_seed(7)
    cfg = PolicyConfig(embedding_dim=8, meta_dim=META_DIM, hidden_dim=16, jump_hidden_dim=16)
    model = PolicyNetwork(cfg)
    model.eval()

    h = torch.randn(1, cfg.hidden_dim)
    K = 5
    cand_emb = torch.randn(1, K, cfg.embedding_dim)
    sim = torch.randn(1, K, 1)
    cand_mask = torch.ones(1, K)

    with torch.no_grad():
        scores_orig = model.jump_scores(h, cand_emb, sim, candidate_mask=cand_mask)
        perm = torch.tensor([4, 2, 0, 3, 1])
        cand_perm = cand_emb[:, perm, :]
        sim_perm = sim[:, perm, :]
        mask_perm = cand_mask[:, perm]
        scores_perm = model.jump_scores(h, cand_perm, sim_perm, candidate_mask=mask_perm)

    inv_perm = torch.argsort(perm)
    assert torch.allclose(scores_orig, scores_perm[:, inv_perm], atol=1e-6)


def test_jump_scorer_same_features_same_score():
    cfg = PolicyConfig(embedding_dim=8, meta_dim=META_DIM, hidden_dim=16, jump_hidden_dim=16)
    model = PolicyNetwork(cfg)
    model.eval()
    h = torch.randn(2, cfg.hidden_dim)
    cand_emb = torch.randn(2, 4, cfg.embedding_dim)
    sim = torch.randn(2, 4, 1)
    cand_emb[1] = cand_emb[0]
    sim[1] = sim[0]
    h[1] = h[0]
    with torch.no_grad():
        scores = model.jump_scores(h, cand_emb, sim)
    assert torch.allclose(scores[0], scores[1], atol=1e-6)


def test_ancestors_descent_path_and_lca():
    from RAG.agentic_policy_v2.supervision import descent_path, lca_of

    idx = _make_toy_tree(embedding_dim=8)
    lookup = NodeFeatureLookup(idx)
    assert ancestors_of("c0.1", lookup) == ["root", "c0", "c0.1"]
    assert descent_path("root", "c1.2", lookup) == ["root", "c1", "c1.2"]
    assert lca_of("c0.1", "c1.2", lookup) == "root"
    assert lca_of("c0.1", "c0.3", lookup) == "c0"
