"""Training utilities (warmup + cosine schedule, checkpoint save/load).

The actual training loop lives in :mod:`scripts.train_agentic_policy_v2`; this
module hosts pure-Python helpers that don't depend on argparse so they can be
unit-tested.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .network import PolicyConfig, PolicyNetwork

logger = logging.getLogger(__name__)


@dataclass
class TrainSchedule:
    total_steps: int
    warmup_fraction: float = 0.05
    final_lr_fraction: float = 0.1


def lr_multiplier(step: int, sched: TrainSchedule) -> float:
    """Linear warmup over ``warmup_fraction`` of ``total_steps`` then cosine decay
    to ``final_lr_fraction`` of peak."""

    total = max(1, sched.total_steps)
    warmup = max(1, int(sched.warmup_fraction * total))
    if step < warmup:
        return max(1e-8, step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return sched.final_lr_fraction + (1.0 - sched.final_lr_fraction) * cosine


def state_dict_sha256(model: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: PolicyNetwork,
    config: PolicyConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "policy_config": config.to_dict(),
    }
    if extra:
        payload.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    sha = state_dict_sha256(model)
    sidecar = {
        "checkpoint_path": str(out),
        "policy_config": config.to_dict(),
        "state_dict_sha256": sha,
        "saved_at_unix": time.time(),
    }
    if extra:
        sidecar.update({k: v for k, v in extra.items() if k != "model_state_dict"})
    sidecar_path = out.with_suffix(out.suffix + ".sha256.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def load_checkpoint(
    path: str | Path,
    map_location: Optional[str] = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if "policy_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint {path} missing required fields")
    return payload


def build_model_from_checkpoint(payload: Dict[str, Any]) -> PolicyNetwork:
    config = PolicyConfig.from_dict(payload["policy_config"])
    model = PolicyNetwork(config)
    model.load_state_dict(payload["model_state_dict"])
    return model
