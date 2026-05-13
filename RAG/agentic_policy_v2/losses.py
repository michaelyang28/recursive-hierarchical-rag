"""Loss formulations from plan section 4.1.

All loss functions accept dense logits and integer / float labels per the
batch contract in section 1.8. The two listwise losses use multi-positive
masked softmax CE; the action loss is class-weighted CE; the done loss is
pos-weighted BCE-with-logits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _safe_logsumexp(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Numerically stable masked logsumexp along the last dim."""

    if mask is None:
        return torch.logsumexp(logits, dim=-1)
    very_neg = torch.tensor(
        torch.finfo(logits.dtype).min / 2.0, dtype=logits.dtype, device=logits.device
    )
    masked = torch.where(mask.bool(), logits, very_neg)
    return torch.logsumexp(masked, dim=-1)


def multi_positive_listwise_ce(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: Optional[torch.Tensor],
    keep_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Multi-positive masked listwise softmax CE per plan section 3.6 / 3.8.

    L_b = -log( sum_{i in pos_b} exp(s_b,i) / sum_j exp(s_b,j) )

    The numerator and denominator both respect ``cand_mask`` (padding) so
    padded slots contribute neither positive mass nor negative mass.

    Parameters
    ----------
    logits : (B, K) float
    pos_mask : (B, K) bool/0-1
    cand_mask : (B, K) bool/0-1 - 1 for real, 0 for padding
    keep_mask : (B,) bool/0-1 - 1 if this row contributes to the loss

    Returns
    -------
    Scalar loss, averaged over rows where ``keep_mask`` is True (or the whole
    batch if ``keep_mask`` is None). Returns ``0`` if no rows to keep.
    """

    pos_mask = pos_mask.bool()
    if cand_mask is None:
        cand_mask = torch.ones_like(pos_mask, dtype=torch.bool)
    cand_mask = cand_mask.bool()

    pos_mask = pos_mask & cand_mask

    very_neg = torch.tensor(
        torch.finfo(logits.dtype).min / 2.0, dtype=logits.dtype, device=logits.device
    )
    pos_logits = torch.where(pos_mask, logits, very_neg)
    cand_logits = torch.where(cand_mask, logits, very_neg)

    has_pos = pos_mask.any(dim=-1)
    log_num = torch.logsumexp(pos_logits, dim=-1)
    log_den = torch.logsumexp(cand_logits, dim=-1)
    per_row = log_den - log_num

    if keep_mask is None:
        keep = has_pos
    else:
        keep = keep_mask.bool() & has_pos

    n = keep.sum()
    if n.item() == 0:
        return logits.new_zeros(())
    return (per_row * keep.float()).sum() / (n.float() + eps)


def action_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Class-weighted CE over ``{RETRIEVE, AGGREGATE, JUMP}``."""

    if class_weight is not None:
        class_weight = class_weight.to(dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits, labels.long(), weight=class_weight)


def done_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Pos-weighted BCE-with-logits."""

    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    else:
        pw = None
    return F.binary_cross_entropy_with_logits(
        logits, labels.float(), pos_weight=pw
    )


def jump_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: torch.Tensor,
    jump_present: torch.Tensor,
) -> torch.Tensor:
    return multi_positive_listwise_ce(logits, pos_mask, cand_mask, jump_present)


def retrieve_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    chunk_mask: torch.Tensor,
    retrieve_present: torch.Tensor,
) -> torch.Tensor:
    return multi_positive_listwise_ce(logits, pos_mask, chunk_mask, retrieve_present)


def loop_probability_mass_penalty(
    logits: torch.Tensor,
    loop_mask: torch.Tensor,
    cand_mask: Optional[torch.Tensor] = None,
    keep_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unlikelihood penalty for probability mass assigned to loop-prone choices.

    This is intentionally auxiliary: gold positives are removed from
    ``loop_mask`` by the dataset, so this term only discourages choices such as
    self-jumps, revisiting nodes already in the path, or immediate aggregate
    reversals when the teacher label says another action is correct.
    """

    loop_mask = loop_mask.bool()
    if cand_mask is None:
        cand_mask = torch.ones_like(loop_mask, dtype=torch.bool)
    cand_mask = cand_mask.bool()
    loop_mask = loop_mask & cand_mask

    if keep_mask is None:
        keep = loop_mask.any(dim=-1)
    else:
        keep = keep_mask.bool() & loop_mask.any(dim=-1)
    n = keep.sum()
    if n.item() == 0:
        return logits.new_zeros(())

    very_neg = torch.tensor(
        torch.finfo(logits.dtype).min / 2.0, dtype=logits.dtype, device=logits.device
    )
    masked_logits = torch.where(cand_mask, logits, very_neg)
    probs = torch.softmax(masked_logits, dim=-1)
    bad_mass = (probs * loop_mask.float()).sum(dim=-1).clamp(max=1.0 - eps)
    per_row = -torch.log1p(-bad_mass)
    return (per_row * keep.float()).sum() / (n.float() + eps)


def jump_top_k_accuracy(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: torch.Tensor,
    jump_present: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """Fraction of jump-rows where some positive is in the masked top-k."""

    pos_mask = pos_mask.bool() & cand_mask.bool()
    jump_present = jump_present.bool()
    keep = jump_present & pos_mask.any(dim=-1)
    if keep.sum().item() == 0:
        return logits.new_zeros(())
    very_neg = torch.tensor(
        torch.finfo(logits.dtype).min / 2.0, dtype=logits.dtype, device=logits.device
    )
    masked = torch.where(cand_mask.bool(), logits, very_neg)
    k_eff = min(k, masked.shape[-1])
    _, top_idx = torch.topk(masked, k=k_eff, dim=-1)
    gather = pos_mask.gather(-1, top_idx)
    hits = gather.any(dim=-1)
    return (hits & keep).float().sum() / (keep.float().sum() + 1e-9)


def jump_mrr(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: torch.Tensor,
    jump_present: torch.Tensor,
) -> torch.Tensor:
    pos_mask = pos_mask.bool() & cand_mask.bool()
    jump_present = jump_present.bool()
    keep = jump_present & pos_mask.any(dim=-1)
    if keep.sum().item() == 0:
        return logits.new_zeros(())
    very_neg = torch.tensor(
        torch.finfo(logits.dtype).min / 2.0, dtype=logits.dtype, device=logits.device
    )
    masked = torch.where(cand_mask.bool(), logits, very_neg)
    order = torch.argsort(masked, dim=-1, descending=True)
    pos_in_order = pos_mask.gather(-1, order)
    rank = torch.argmax(pos_in_order.long(), dim=-1).float() + 1.0
    rr = 1.0 / rank
    return (rr * keep.float()).sum() / (keep.float().sum() + 1e-9)
