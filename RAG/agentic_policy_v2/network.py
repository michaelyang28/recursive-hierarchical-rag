"""Policy network modules.

Implements plan section 1.5. Every module here corresponds to a single
parameter group ``theta_*`` that is applied identically at every step, every
node, and every candidate. The network is intentionally tiny (~830K params)
so it trains on CPU.

Action labels (constant ints used everywhere):

    ACTION_RETRIEVE = 0
    ACTION_AGGREGATE = 1
    ACTION_JUMP = 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


ACTION_RETRIEVE: int = 0
ACTION_AGGREGATE: int = 1
ACTION_JUMP: int = 2
ACTION_NAMES = ["RETRIEVE", "AGGREGATE", "JUMP"]
NUM_ACTIONS: int = 3


@dataclass
class PolicyConfig:
    """Hyperparameters that are baked into the saved checkpoint."""

    embedding_dim: int = 384
    meta_dim: int = 16
    hidden_dim: int = 256
    jump_hidden_dim: int = 128
    retrieve_hidden_dim: int = 128
    dropout: float = 0.1
    use_done_head: bool = False
    use_retrieve_head: bool = False

    @property
    def state_dim(self) -> int:
        return 6 * self.embedding_dim + self.meta_dim

    def to_dict(self) -> Dict[str, object]:
        return {
            "embedding_dim": self.embedding_dim,
            "meta_dim": self.meta_dim,
            "hidden_dim": self.hidden_dim,
            "jump_hidden_dim": self.jump_hidden_dim,
            "retrieve_hidden_dim": self.retrieve_hidden_dim,
            "dropout": self.dropout,
            "use_done_head": self.use_done_head,
            "use_retrieve_head": self.use_retrieve_head,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PolicyConfig":
        return cls(
            embedding_dim=int(payload.get("embedding_dim", 384)),
            meta_dim=int(payload.get("meta_dim", 16)),
            hidden_dim=int(payload.get("hidden_dim", 256)),
            jump_hidden_dim=int(payload.get("jump_hidden_dim", 128)),
            retrieve_hidden_dim=int(payload.get("retrieve_hidden_dim", 128)),
            dropout=float(payload.get("dropout", 0.1)),
            use_done_head=bool(payload.get("use_done_head", False)),
            use_retrieve_head=bool(payload.get("use_retrieve_head", False)),
        )


class Encoder(nn.Module):
    """Shared 2-layer MLP with LayerNorm + GELU."""

    def __init__(self, state_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        h = self.dropout(h)
        h = self.act(self.norm2(self.fc2(h)))
        return h


class ActionHead(nn.Module):
    """3-way logits over {RETRIEVE, AGGREGATE, JUMP}."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, NUM_ACTIONS)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class DoneHead(nn.Module):
    """Sigmoid done classifier."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)


class _CandidateScorer(nn.Module):
    """Shared MLP applied identically to each (h, candidate, sim) triple.

    Accepts ``h`` of shape ``(B, H)`` and ``candidate_emb`` of shape
    ``(B, K, D)`` plus ``sim`` of shape ``(B, K, 1)``. Returns ``(B, K)``
    scalar scores. Implementation tiles ``h`` across ``K`` and applies a
    single ``nn.Linear`` over the last dim, so PyTorch broadcasts the same
    parameters across every candidate slot.
    """

    def __init__(self, hidden_dim: int, embedding_dim: int, scorer_hidden: int):
        super().__init__()
        self.input_dim = hidden_dim + embedding_dim + 1
        self.fc1 = nn.Linear(self.input_dim, scorer_hidden)
        self.norm = nn.LayerNorm(scorer_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(scorer_hidden, 1)

    def forward(
        self,
        h: torch.Tensor,
        candidate_emb: torch.Tensor,
        sim: torch.Tensor,
    ) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"h expected 2-D, got {tuple(h.shape)}")
        if candidate_emb.dim() != 3:
            raise ValueError(
                f"candidate_emb expected 3-D, got {tuple(candidate_emb.shape)}"
            )
        if sim.dim() == 2:
            sim = sim.unsqueeze(-1)
        if sim.shape[:2] != candidate_emb.shape[:2] or sim.shape[-1] != 1:
            raise ValueError(
                f"sim shape {tuple(sim.shape)} mismatch with candidate_emb "
                f"{tuple(candidate_emb.shape)}"
            )
        K = candidate_emb.shape[1]
        h_tiled = h.unsqueeze(1).expand(-1, K, -1)
        f = torch.cat([h_tiled, candidate_emb, sim], dim=-1)
        z = self.fc1(f)
        z = self.norm(z)
        z = self.act(z)
        scores = self.fc2(z).squeeze(-1)
        return scores


class JumpScorer(_CandidateScorer):
    """Listwise scorer over jump candidates (nodes)."""


class RetrieveScorer(_CandidateScorer):
    """Listwise scorer over chunk candidates (docs)."""


class PolicyNetwork(nn.Module):
    """The shared agentic policy.

    The done and retrieve heads are constructed always, but their parameters
    are only updated in Phase 3 (controlled by ``PolicyConfig.use_done_head``
    and ``use_retrieve_head``). Keeping them in the module unconditionally
    simplifies checkpoint backwards compatibility.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.action_head = ActionHead(hidden_dim=config.hidden_dim)
        self.done_head = DoneHead(hidden_dim=config.hidden_dim)
        self.jump_scorer = JumpScorer(
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            scorer_hidden=config.jump_hidden_dim,
        )
        self.retrieve_scorer = RetrieveScorer(
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            scorer_hidden=config.retrieve_hidden_dim,
        )

    def encode(self, state_x: torch.Tensor) -> torch.Tensor:
        return self.encoder(state_x)

    def action_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.action_head(h)

    def done_logit(self, h: torch.Tensor) -> torch.Tensor:
        return self.done_head(h)

    def jump_scores(
        self,
        h: torch.Tensor,
        candidate_emb: torch.Tensor,
        candidate_sim: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.jump_scorer(h, candidate_emb, candidate_sim)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), float("-inf"))
        return scores

    def retrieve_scores(
        self,
        h: torch.Tensor,
        chunk_emb: torch.Tensor,
        chunk_sim: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.retrieve_scorer(h, chunk_emb, chunk_sim)
        if chunk_mask is not None:
            scores = scores.masked_fill(~chunk_mask.bool(), float("-inf"))
        return scores

    def forward(
        self,
        state_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode(state_x)
        return {
            "h": h,
            "action_logits": self.action_logits(h),
            "done_logit": self.done_logit(h),
        }


def make_param_groups(
    model: PolicyNetwork,
    phase: int,
    base_lr: float = 1e-3,
    warm_lr: float = 5e-4,
    encoder_lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    """Return ``torch.optim`` param groups per plan section 4.3.

    Phase 1 / 2: single LR for encoder + action + jump, done/retrieve frozen.
    Phase 3   : warm encoder at ``encoder_lr``, warm action+jump at ``warm_lr``,
                fresh done+retrieve at ``base_lr``.
    """

    if phase in (1, 2):
        for p in model.done_head.parameters():
            p.requires_grad_(False)
        for p in model.retrieve_scorer.parameters():
            p.requires_grad_(False)
        params = [
            *model.encoder.parameters(),
            *model.action_head.parameters(),
            *model.jump_scorer.parameters(),
        ]
        return [
            {
                "params": params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "policy_phase1",
            }
        ]

    for p in model.parameters():
        p.requires_grad_(True)
    return [
        {
            "params": list(model.encoder.parameters()),
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder_warm",
        },
        {
            "params": [
                *model.action_head.parameters(),
                *model.jump_scorer.parameters(),
            ],
            "lr": warm_lr,
            "weight_decay": weight_decay,
            "name": "warm_heads",
        },
        {
            "params": [
                *model.done_head.parameters(),
                *model.retrieve_scorer.parameters(),
            ],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "fresh_heads",
        },
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
