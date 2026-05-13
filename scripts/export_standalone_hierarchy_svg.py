#!/usr/bin/env python3
"""
Export a standalone hierarchy index to an SVG tree diagram.

Each node box includes:
- node id / depth / size
- label
- member doc IDs (optionally truncated)
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
STANDALONE_HIERARCHY_PATH = REPO_ROOT / "RAG" / "standalone_hierarchy.py"


def _load_standalone_hierarchy_classes():
    module_name = "standalone_hierarchy_module"
    spec = importlib.util.spec_from_file_location(module_name, STANDALONE_HIERARCHY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load standalone hierarchy module from {STANDALONE_HIERARCHY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.HierarchyNode, module.StandaloneHierarchyIndex


HierarchyNode, StandaloneHierarchyIndex = _load_standalone_hierarchy_classes()


def _wrap_lines(text: str, max_chars: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        extra = len(word) if not current else len(word) + 1
        if current and current_len + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return lines


def _levels(index: StandaloneHierarchyIndex) -> Dict[int, List[HierarchyNode]]:
    by_depth: Dict[int, List[HierarchyNode]] = {}
    q = deque(index.root_ids)
    seen = set()
    while q:
        node_id = q.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = index.node_by_id[node_id]
        by_depth.setdefault(node.depth, []).append(node)
        for child in node.children:
            q.append(child)

    for depth in by_depth:
        by_depth[depth].sort(key=lambda n: n.node_id)
    return by_depth


def _node_text_lines(node: HierarchyNode, max_doc_ids_per_node: int, include_all_doc_ids: bool) -> List[str]:
    lines = [
        f"{node.node_id} | depth={node.depth} | docs={len(node.member_doc_ids)}",
    ]
    if node.label:
        lines.extend(_wrap_lines(f"label: {node.label}", max_chars=44))

    ids = node.member_doc_ids
    if include_all_doc_ids:
        doc_line = "doc_ids: " + ", ".join(ids)
    else:
        shown = ids[: max(0, max_doc_ids_per_node)]
        suffix = f" ... (+{len(ids) - len(shown)} more)" if len(ids) > len(shown) else ""
        doc_line = "doc_ids: " + ", ".join(shown) + suffix
    lines.extend(_wrap_lines(doc_line, max_chars=64))
    return lines


def _truncate_lines(lines: List[str], max_lines: int) -> List[str]:
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    truncated = lines[:max_lines]
    last = truncated[-1]
    truncated[-1] = (last[:-3] + "...") if len(last) >= 3 else "..."
    return truncated


def render_svg(
    index: StandaloneHierarchyIndex,
    output_path: Path,
    max_doc_ids_per_node: int = 20,
    include_all_doc_ids: bool = False,
    node_width: int = 520,
    line_height: int = 16,
    column_gap: int = 90,
    row_gap: int = 24,
    margin: int = 30,
    show_summary: bool = True,
    max_summary_lines: int = 4,
) -> None:
    levels = _levels(index)
    if not levels:
        raise ValueError("No nodes found in index.")

    node_lines: Dict[str, List[str]] = {}
    node_height: Dict[str, int] = {}
    node_pos: Dict[str, Tuple[int, int]] = {}

    for depth, nodes in levels.items():
        for node in nodes:
            lines = _node_text_lines(node, max_doc_ids_per_node=max_doc_ids_per_node, include_all_doc_ids=include_all_doc_ids)
            if show_summary and node.summary:
                summary_lines = _wrap_lines(f"summary: {node.summary}", max_chars=64)
                lines.extend(_truncate_lines(summary_lines, max_summary_lines))
            node_lines[node.node_id] = lines
            node_height[node.node_id] = 20 + max(1, len(lines)) * line_height

    depth_keys = sorted(levels.keys())

    max_row_width = 0
    for depth in depth_keys:
        x = margin
        for node in levels[depth]:
            node_pos[node.node_id] = (x, margin + depth * (220 + row_gap))
            x += node_width + column_gap
        max_row_width = max(max_row_width, x)

    # Recompute y with per-row dynamic heights to avoid overlaps.
    current_y = margin
    for depth in depth_keys:
        row_nodes = levels[depth]
        row_height = max(node_height[n.node_id] for n in row_nodes)
        x = margin
        for node in row_nodes:
            h = node_height[node.node_id]
            # Vertically center each node in the row's max height.
            node_pos[node.node_id] = (x, current_y + (row_height - h) // 2)
            x += node_width + column_gap
        max_row_width = max(max_row_width, x)
        current_y += row_height + row_gap

    svg_width = max_row_width + margin
    svg_height = current_y + margin

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">'
    )
    parts.append('<style>')
    parts.append(".edge { stroke: #9ca3af; stroke-width: 1.25; fill: none; }")
    parts.append(".node { fill: #f8fafc; stroke: #334155; stroke-width: 1.2; rx: 8; ry: 8; }")
    parts.append(".text { font: 12px monospace; fill: #0f172a; }")
    parts.append('</style>')

    # Edges first (under nodes)
    for node in index.nodes:
        x1, y1 = node_pos[node.node_id]
        h1 = node_height[node.node_id]
        start_x = x1 + node_width / 2
        start_y = y1 + h1
        for child_id in node.children:
            if child_id not in node_pos:
                continue
            x2, y2 = node_pos[child_id]
            end_x = x2 + node_width / 2
            end_y = y2
            mid_y = (start_y + end_y) / 2
            parts.append(
                f'<path class="edge" d="M {start_x:.1f},{start_y:.1f} C {start_x:.1f},{mid_y:.1f} {end_x:.1f},{mid_y:.1f} {end_x:.1f},{end_y:.1f}" />'
            )

    # Node boxes + text
    for depth in depth_keys:
        for node in levels[depth]:
            node_id = node.node_id
            x, y = node_pos[node_id]
            h = node_height[node_id]
            parts.append(f'<rect class="node" x="{x}" y="{y}" width="{node_width}" height="{h}" />')
            lines = node_lines[node_id]
            for i, line in enumerate(lines):
                txt = html.escape(line)
                ty = y + 18 + i * line_height
                parts.append(f'<text class="text" x="{x + 10}" y="{ty}">{txt}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone hierarchy to SVG with member doc IDs.")
    parser.add_argument("--index_dir", type=str, required=True, help="Path to standalone hierarchy index dir")
    parser.add_argument("--output_svg", type=str, required=True, help="Path to output SVG file")
    parser.add_argument(
        "--max_doc_ids_per_node",
        type=int,
        default=20,
        help="Max doc IDs shown per node unless --include_all_doc_ids is set",
    )
    parser.add_argument(
        "--include_all_doc_ids",
        action="store_true",
        help="Render all member doc IDs in each node (can produce very large SVGs)",
    )
    parser.add_argument(
        "--hide_summary",
        action="store_true",
        help="Do not include node summaries in SVG labels.",
    )
    parser.add_argument(
        "--max_summary_lines",
        type=int,
        default=4,
        help="Maximum wrapped summary lines per node (ignored with --hide_summary).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = StandaloneHierarchyIndex.load(args.index_dir, load_embeddings=False)
    output_path = Path(args.output_svg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_svg(
        index=index,
        output_path=output_path,
        max_doc_ids_per_node=args.max_doc_ids_per_node,
        include_all_doc_ids=args.include_all_doc_ids,
        show_summary=not args.hide_summary,
        max_summary_lines=max(0, args.max_summary_lines),
    )
    size_kb = math.ceil(output_path.stat().st_size / 1024)
    print(f"SVG written to: {output_path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
