"""Background trainer and checkpoint promoter for agentic_policy_v2."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from RAG.rlm_adaptation import (  # noqa: E402
    PolicyCheckpointRegistry,
    PolicyCheckpointVersion,
    new_version,
    should_promote,
)


logger = logging.getLogger(__name__)


def _count_jsonl_rows(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_history(output_dir: Path) -> Dict[str, Any]:
    history_path = output_dir / "history.json"
    if not history_path.exists():
        return {}
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    if not rows:
        return {}
    return rows[-1].get("val", {})


def _best_checkpoint_path(output_dir: Path, phase: int) -> Path:
    return output_dir / f"phase{phase}.pt"


def _current_checkpoint(registry_path: Optional[str]) -> Optional[PolicyCheckpointVersion]:
    if not registry_path:
        return None
    return PolicyCheckpointRegistry(registry_path).current_checkpoint_version()


def _build_train_command(args: argparse.Namespace, output_dir: Path, init_from: Optional[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_agentic_policy_v2.py"),
        "--phase",
        str(args.phase),
        "--supervision_path",
        args.supervision_path,
        "--query_embeddings_path",
        args.query_embeddings_path,
        "--index_dir",
        args.index_dir,
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--K_max",
        str(args.K_max),
        "--M_max",
        str(args.M_max),
        "--device",
        args.device,
        "--early_stop_metric",
        args.early_stop_metric,
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
    ]
    if args.extra_supervision_path:
        cmd.extend(["--extra_supervision_path", args.extra_supervision_path])
    if init_from:
        cmd.extend(["--init_from", init_from])
    if args.in_batch_negatives:
        cmd.extend(["--in_batch_negatives", str(args.in_batch_negatives)])
    if args.lambda_jump_loop:
        cmd.extend(["--lambda_jump_loop", str(args.lambda_jump_loop)])
    if args.lambda_action_loop:
        cmd.extend(["--lambda_action_loop", str(args.lambda_action_loop)])
    return cmd


def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    extra_rows = _count_jsonl_rows(args.extra_supervision_path) if args.extra_supervision_path else 0
    if extra_rows < args.min_new_rows:
        return {
            "status": "skipped",
            "reason": "not_enough_new_rows",
            "extra_rows": extra_rows,
            "min_new_rows": args.min_new_rows,
        }

    registry = PolicyCheckpointRegistry(args.policy_registry) if args.policy_registry else None
    current = _current_checkpoint(args.policy_registry)
    init_from = args.init_from or (current.checkpoint_path if current else None)

    run_id = args.run_id or new_version(f"policy-phase{args.phase}")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_train_command(args, output_dir, init_from)
    logger.info("Starting policy training: %s", " ".join(cmd))

    if args.dry_run:
        return {
            "status": "dry_run",
            "command": cmd,
            "output_dir": str(output_dir),
            "init_from": init_from,
            "extra_rows": extra_rows,
        }

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "output_dir": str(output_dir),
            "init_from": init_from,
        }

    metrics = _load_history(output_dir)
    checkpoint_path = _best_checkpoint_path(output_dir, args.phase)
    if not checkpoint_path.exists():
        return {
            "status": "failed",
            "reason": "checkpoint_missing",
            "output_dir": str(output_dir),
            "metrics": metrics,
        }

    baseline_metrics = current.metrics if current else None
    promote = should_promote(
        baseline_metrics,
        metrics,
        metric=args.promote_metric,
        min_delta=args.promote_min_delta,
    )
    record = PolicyCheckpointVersion(
        version=run_id,
        checkpoint_path=str(checkpoint_path),
        parent_version=current.version if current else None,
        source_supervision_path=args.extra_supervision_path or args.supervision_path,
        metrics=metrics,
        promoted=promote,
    )
    if registry:
        registry.add_checkpoint_version(record, promote=promote)

    return {
        "status": "completed",
        "promoted": promote,
        "version": run_id,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "extra_rows": extra_rows,
        "init_from": init_from,
    }


def run_loop(args: argparse.Namespace) -> None:
    while True:
        report = run_once(args)
        print(json.dumps(report, indent=2, default=str))
        if args.once:
            return
        time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument("--supervision_path", required=True)
    parser.add_argument("--extra_supervision_path", default=None)
    parser.add_argument("--query_embeddings_path", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/agentic_policy_v2/background")
    parser.add_argument("--policy_registry", default="outputs/agentic_policy_v2/policy_registry.json")
    parser.add_argument("--init_from", default=None)
    parser.add_argument("--min_new_rows", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--K_max", type=int, default=32)
    parser.add_argument("--M_max", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--early_stop_metric", default="jump_top_1")
    parser.add_argument("--promote_metric", default="jump_top_1")
    parser.add_argument("--promote_min_delta", type=float, default=0.0)
    parser.add_argument("--in_batch_negatives", type=int, default=0)
    parser.add_argument("--lambda_jump_loop", type=float, default=0.0)
    parser.add_argument("--lambda_action_loop", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--poll_seconds", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    run_loop(args)


if __name__ == "__main__":
    main()
