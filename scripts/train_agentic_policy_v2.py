"""Phase-aware training entrypoint for the supervised agentic policy.

Implements plan section 4.5 (Phase 1) and the Phase 2 / Phase 3 protocols
from sections 4.6 and 4.7. Phase selection is via ``--phase``; each phase
warm-starts from the previous one's checkpoint when ``--init_from`` is given.

The loop performs:
* Stratified sampling (``WeightedRandomSampler``) per section 4.4.
* AdamW + linear-warmup -> cosine decay schedule (section 4.3).
* Class-weighted action CE + multi-positive listwise jump CE (Phase 1+).
* Pos-weighted BCE done + multi-positive listwise retrieve CE (Phase 3+).
* Step-level validation (action_acc, jump_top_1, jump_mrr, retrieve_ndcg, done_f1).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from RAG.agentic_policy_v2.dataset import (  # noqa: E402
    InBatchNegativeCollator,
    SupervisionDataset,
    policy_collate,
    split_train_val,
    stratified_sampler_weights,
)
from RAG.agentic_policy_v2.losses import (  # noqa: E402
    action_cross_entropy,
    done_bce,
    jump_loss,
    jump_mrr,
    jump_top_k_accuracy,
    loop_probability_mass_penalty,
    multi_positive_listwise_ce,
)
from RAG.agentic_policy_v2.network import (  # noqa: E402
    ACTION_JUMP,
    ACTION_RETRIEVE,
    NUM_ACTIONS,
    PolicyConfig,
    PolicyNetwork,
    count_parameters,
    make_param_groups,
)
from RAG.agentic_policy_v2.state import META_DIM, NodeFeatureLookup  # noqa: E402
from RAG.agentic_policy_v2.training import (  # noqa: E402
    TrainSchedule,
    build_model_from_checkpoint,
    load_checkpoint,
    lr_multiplier,
    save_checkpoint,
)
from RAG.standalone_hierarchy import StandaloneHierarchyIndex  # noqa: E402

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_phase(args: argparse.Namespace) -> int:
    phase = int(args.phase)
    if phase not in (1, 2, 3):
        raise ValueError(f"--phase must be 1, 2 or 3 (got {phase})")
    return phase


def _build_policy_config(
    args: argparse.Namespace, embedding_dim: int, phase: int
) -> PolicyConfig:
    return PolicyConfig(
        embedding_dim=embedding_dim,
        meta_dim=META_DIM,
        hidden_dim=args.hidden_dim,
        jump_hidden_dim=args.jump_hidden_dim,
        retrieve_hidden_dim=args.retrieve_hidden_dim,
        dropout=args.dropout,
        use_done_head=phase >= 3,
        use_retrieve_head=phase >= 3,
    )


def _build_dataset(
    args: argparse.Namespace,
    lookup: NodeFeatureLookup,
) -> SupervisionDataset:
    extra = getattr(args, "extra_supervision_path", None)
    extra_paths = [extra] if extra else None
    return SupervisionDataset(
        jsonl_path=args.supervision_path,
        query_embeddings_path=args.query_embeddings_path,
        lookup=lookup,
        K_max=args.K_max,
        M_max=args.M_max,
        extra_jsonl_paths=extra_paths,
    )


def _make_dataloaders(
    args: argparse.Namespace,
    dataset: SupervisionDataset,
    lookup: NodeFeatureLookup,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    train_idx, val_idx = split_train_val(
        dataset.rows, val_fraction=args.val_fraction, seed=args.seed
    )

    weights_full = stratified_sampler_weights(dataset)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx) if val_idx else Subset(dataset, train_idx[: max(1, len(train_idx) // 10)])

    train_weights = weights_full[torch.tensor(train_idx, dtype=torch.long)]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_idx),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.in_batch_negatives > 0:
        train_collate = InBatchNegativeCollator(lookup, n_in_batch_negs=args.in_batch_negatives)
    else:
        train_collate = policy_collate

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=policy_collate,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, train_idx, val_idx


def _train_step(
    model: PolicyNetwork,
    batch,
    phase: int,
    class_weight: torch.Tensor,
    pos_weight_done: float,
    lambdas: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    out = model(batch.x)
    h = out["h"]
    action_logits = out["action_logits"]
    done_logit = out["done_logit"]

    losses: Dict[str, torch.Tensor] = {}
    losses["L_action"] = action_cross_entropy(
        action_logits, batch.action_label, class_weight=class_weight
    )

    jump_scores = model.jump_scores(
        h,
        batch.jump_cand_emb,
        batch.jump_cand_sim,
        candidate_mask=batch.jump_cand_mask,
    )
    losses["L_jump"] = jump_loss(
        jump_scores,
        batch.jump_pos_mask,
        batch.jump_cand_mask,
        batch.jump_present,
    )
    losses["L_jump_loop"] = loop_probability_mass_penalty(
        jump_scores,
        batch.jump_loop_mask,
        cand_mask=batch.jump_cand_mask,
        keep_mask=batch.jump_present,
    )
    losses["L_action_loop"] = loop_probability_mass_penalty(
        action_logits,
        batch.action_loop_mask,
    )

    if phase >= 3:
        losses["L_done"] = done_bce(done_logit, batch.done_label, pos_weight=pos_weight_done)
        retrieve_scores = model.retrieve_scores(
            h,
            batch.retrieve_emb,
            batch.retrieve_sim,
            chunk_mask=batch.retrieve_mask,
        )
        losses["L_retrieve"] = multi_positive_listwise_ce(
            retrieve_scores,
            batch.retrieve_pos_mask,
            batch.retrieve_mask,
            batch.retrieve_present,
        )
    else:
        losses["L_done"] = action_logits.new_zeros(())
        losses["L_retrieve"] = action_logits.new_zeros(())

    total = (
        losses["L_action"]
        + lambdas["jump"] * losses["L_jump"]
        + lambdas["done"] * losses["L_done"]
        + lambdas["retrieve"] * losses["L_retrieve"]
        + lambdas["jump_loop"] * losses["L_jump_loop"]
        + lambdas["action_loop"] * losses["L_action_loop"]
    )
    losses["total"] = total
    losses["jump_scores"] = jump_scores.detach()
    return losses


@torch.no_grad()
def _validate(
    model: PolicyNetwork,
    loader: DataLoader,
    phase: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    n_total = 0
    n_action_correct = 0
    n_jump = 0
    jump_top1 = 0.0
    jump_top3 = 0.0
    jump_mrr_sum = 0.0
    n_done = 0
    n_done_correct_pos = 0
    n_done_correct_neg = 0
    n_done_pos = 0
    n_done_neg = 0
    n_retrieve = 0
    retrieve_top5 = 0.0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x)
        h = out["h"]
        action_logits = out["action_logits"]
        done_logit = out["done_logit"]
        action_pred = action_logits.argmax(dim=-1)
        n_action_correct += int((action_pred == batch.action_label).sum().item())
        n_total += int(batch.action_label.numel())

        if batch.jump_present.any():
            jump_scores = model.jump_scores(
                h,
                batch.jump_cand_emb,
                batch.jump_cand_sim,
                candidate_mask=batch.jump_cand_mask,
            )
            jt1 = jump_top_k_accuracy(
                jump_scores, batch.jump_pos_mask, batch.jump_cand_mask, batch.jump_present, k=1
            ).item()
            jt3 = jump_top_k_accuracy(
                jump_scores, batch.jump_pos_mask, batch.jump_cand_mask, batch.jump_present, k=3
            ).item()
            mrr = jump_mrr(
                jump_scores, batch.jump_pos_mask, batch.jump_cand_mask, batch.jump_present
            ).item()
            n_jump_batch = int(batch.jump_present.sum().item())
            jump_top1 += jt1 * n_jump_batch
            jump_top3 += jt3 * n_jump_batch
            jump_mrr_sum += mrr * n_jump_batch
            n_jump += n_jump_batch

        if phase >= 3 and batch.done_label.numel() > 0:
            done_pred = (torch.sigmoid(done_logit) > 0.5).long()
            n_done += int(batch.done_label.numel())
            mask_pos = batch.done_label > 0.5
            mask_neg = batch.done_label <= 0.5
            n_done_pos += int(mask_pos.sum().item())
            n_done_neg += int(mask_neg.sum().item())
            n_done_correct_pos += int(((done_pred == 1) & mask_pos).sum().item())
            n_done_correct_neg += int(((done_pred == 0) & mask_neg).sum().item())

            if batch.retrieve_present.any():
                retrieve_scores = model.retrieve_scores(
                    h,
                    batch.retrieve_emb,
                    batch.retrieve_sim,
                    chunk_mask=batch.retrieve_mask,
                )
                rt5 = jump_top_k_accuracy(
                    retrieve_scores,
                    batch.retrieve_pos_mask,
                    batch.retrieve_mask,
                    batch.retrieve_present,
                    k=5,
                ).item()
                n_ret_batch = int(batch.retrieve_present.sum().item())
                retrieve_top5 += rt5 * n_ret_batch
                n_retrieve += n_ret_batch

    metrics = {
        "action_acc": n_action_correct / max(1, n_total),
        "jump_top_1": jump_top1 / max(1, n_jump),
        "jump_top_3": jump_top3 / max(1, n_jump),
        "jump_mrr": jump_mrr_sum / max(1, n_jump),
    }
    if phase >= 3:
        precision = n_done_correct_pos / max(1, n_done_correct_pos + (n_done_neg - n_done_correct_neg))
        recall = n_done_correct_pos / max(1, n_done_pos)
        f1 = 2 * precision * recall / max(1e-9, (precision + recall))
        metrics.update(
            {
                "done_acc": (n_done_correct_pos + n_done_correct_neg) / max(1, n_done),
                "done_precision": precision,
                "done_recall": recall,
                "done_f1": f1,
                "retrieve_top_5": retrieve_top5 / max(1, n_retrieve),
            }
        )
    model.train()
    return metrics


def _setup_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--supervision_path", required=True)
    parser.add_argument(
        "--extra_supervision_path",
        default=None,
        help="Optional second JSONL (e.g. hard-negatives sidecar) concatenated with --supervision_path.",
    )
    parser.add_argument("--query_embeddings_path", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_from", default=None, help="Optional warm-start checkpoint")
    parser.add_argument(
        "--in_batch_negatives",
        type=int,
        default=0,
        help="Number of cross-query positives to splice into each row's masked padding slots as negatives.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--K_max", type=int, default=32)
    parser.add_argument("--M_max", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--jump_hidden_dim", type=int, default=128)
    parser.add_argument("--retrieve_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr_base", type=float, default=1e-3)
    parser.add_argument("--lr_warm", type=float, default=5e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_jump", type=float, default=1.0)
    parser.add_argument("--lambda_done", type=float, default=0.3)
    parser.add_argument("--lambda_retrieve", type=float, default=1.0)
    parser.add_argument(
        "--lambda_jump_loop",
        type=float,
        default=0.0,
        help="Auxiliary unlikelihood penalty for self/visited jump candidates.",
    )
    parser.add_argument(
        "--lambda_action_loop",
        type=float,
        default=0.0,
        help="Auxiliary unlikelihood penalty for immediate aggregate reversals.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stop_metric", default="jump_top_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)
    _seed_everything(args.seed)

    phase = _resolve_phase(args)

    logger.info("Loading hierarchy index from %s", args.index_dir)
    hierarchy_index = StandaloneHierarchyIndex.load(args.index_dir, load_embeddings=True)
    lookup = NodeFeatureLookup(hierarchy_index)

    dataset = _build_dataset(args, lookup)
    logger.info(
        "Loaded supervision: %d rows, action distribution=%s, embedding_dim=%d",
        len(dataset),
        dataset.action_distribution(),
        lookup.embedding_dim,
    )

    train_loader, val_loader, train_idx, val_idx = _make_dataloaders(args, dataset, lookup)
    logger.info(
        "Train rows=%d, val rows=%d, batch_size=%d, batches/epoch=%d",
        len(train_idx),
        len(val_idx) if val_idx else 0,
        args.batch_size,
        len(train_loader),
    )

    config = _build_policy_config(args, lookup.embedding_dim, phase)
    model = PolicyNetwork(config)
    if args.init_from:
        logger.info("Warm-starting from %s", args.init_from)
        payload = load_checkpoint(args.init_from)
        prev_state = payload.get("model_state_dict", {})
        missing, unexpected = model.load_state_dict(prev_state, strict=False)
        logger.info(
            "Warm-start state_dict loaded: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

    device = torch.device(args.device)
    model.to(device)
    logger.info("Trainable parameters: %d", count_parameters(model))

    param_groups = make_param_groups(
        model,
        phase=phase,
        base_lr=args.lr_base,
        warm_lr=args.lr_warm,
        encoder_lr=args.lr_encoder,
        weight_decay=args.weight_decay,
    )
    optimizer = AdamW(param_groups, betas=(0.9, 0.95))
    total_steps = max(1, args.epochs * len(train_loader))
    sched_cfg = TrainSchedule(total_steps=total_steps)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[lambda step: lr_multiplier(step, sched_cfg) for _ in optimizer.param_groups],
    )

    class_weight = dataset.action_class_weights().to(device)
    pos_weight_done = dataset.done_pos_weight()
    lambdas = {
        "jump": args.lambda_jump,
        "done": args.lambda_done if phase >= 3 else 0.0,
        "retrieve": args.lambda_retrieve if phase >= 3 else 0.0,
        "jump_loop": args.lambda_jump_loop,
        "action_loop": args.lambda_action_loop,
    }
    logger.info(
        "class_weight=%s, pos_weight_done=%.2f, lambdas=%s",
        class_weight.tolist(),
        pos_weight_done,
        lambdas,
    )

    best_metric = -float("inf")
    best_path = output_dir / f"phase{phase}.pt"
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    config_dump = {
        "args": vars(args),
        "policy_config": config.to_dict(),
        "lambdas": lambdas,
        "pos_weight_done": pos_weight_done,
        "class_weight": class_weight.tolist(),
        "embedding_dim": lookup.embedding_dim,
        "num_train_rows": len(train_idx),
        "num_val_rows": len(val_idx),
        "action_distribution": dataset.action_distribution(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_dump, indent=2, default=str), encoding="utf-8"
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running: Dict[str, float] = {
            "L_action": 0.0,
            "L_jump": 0.0,
            "L_jump_loop": 0.0,
            "L_action_loop": 0.0,
            "L_done": 0.0,
            "L_retrieve": 0.0,
            "total": 0.0,
        }
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = _train_step(model, batch, phase, class_weight, pos_weight_done, lambdas)
            total = losses["total"]
            if not torch.isfinite(total):
                logger.warning(
                    "Non-finite loss at epoch %d step %d; skipping batch", epoch, global_step
                )
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            for k in running:
                running[k] += float(losses[k].detach().cpu())
            n_batches += 1
            global_step += 1
            if global_step % args.log_every == 0:
                logger.info(
                    "epoch=%d step=%d L_total=%.4f L_action=%.4f L_jump=%.4f L_jloop=%.4f L_aloop=%.4f L_done=%.4f L_ret=%.4f lr=%s",
                    epoch,
                    global_step,
                    float(losses["total"].detach().cpu()),
                    float(losses["L_action"].detach().cpu()),
                    float(losses["L_jump"].detach().cpu()),
                    float(losses["L_jump_loop"].detach().cpu()),
                    float(losses["L_action_loop"].detach().cpu()),
                    float(losses["L_done"].detach().cpu()),
                    float(losses["L_retrieve"].detach().cpu()),
                    [f"{g['lr']:.2e}" for g in optimizer.param_groups],
                )

        avg_losses = {k: v / max(1, n_batches) for k, v in running.items()}
        val_metrics = _validate(model, val_loader, phase, device)
        epoch_time = time.time() - epoch_start
        log_row = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_time_sec": epoch_time,
            "train_avg_losses": avg_losses,
            "val": val_metrics,
        }
        history.append(log_row)
        logger.info(
            "EPOCH %d: train_total=%.4f val=%s elapsed=%.1fs",
            epoch,
            avg_losses["total"],
            val_metrics,
            epoch_time,
        )

        metric_value = val_metrics.get(args.early_stop_metric, val_metrics.get("jump_top_1", 0.0))
        if metric_value > best_metric + 1e-6:
            best_metric = metric_value
            epochs_no_improve = 0
            sidecar = save_checkpoint(
                best_path,
                model,
                config,
                extra={
                    "phase": phase,
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
            )
            logger.info("Saved best checkpoint: %s (sha256=%s)", best_path, sidecar["state_dict_sha256"])
        else:
            epochs_no_improve += 1
            logger.info(
                "No improvement (best %s=%.4f, epochs since=%d)",
                args.early_stop_metric,
                best_metric,
                epochs_no_improve,
            )
            if epochs_no_improve >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote history to %s", history_path)


if __name__ == "__main__":
    main()
