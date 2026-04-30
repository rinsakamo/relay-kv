#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_mean_abs_diff(payload: dict[str, Any]) -> float:
    return float(payload["attention_compare"]["mean_abs_diff"])


def extract_meta(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": payload.get("model"),
        "seq_len_target": payload.get("seq_len_target"),
        "prompt_type": payload.get("prompt_type"),
        "hot_window": payload.get("hot_window"),
        "block_size": payload.get("block_size"),
        "scoring_variant": payload.get("scoring_variant"),
        "kv_working_budget_tokens": payload.get("kv_working_budget_tokens"),
        "recent_window_tokens": payload.get("recent_window_tokens"),
        "anchor_blocks": payload.get("anchor_blocks"),
        "budget_block_size": payload.get("budget_block_size"),
        "anchor_budget_tokens": payload.get("anchor_budget_tokens"),
        "retrieval_budget_tokens": payload.get("retrieval_budget_tokens"),
        "retrieval_block_budget": payload.get("retrieval_block_budget"),
        "retrieval_top_k_requested": payload.get("retrieval_top_k_requested"),
        "retrieval_top_k_effective": payload.get("retrieval_top_k_effective"),
        "budget_overflow": payload.get("budget_overflow"),
        "budget_policy_reason": payload.get("budget_policy_reason"),
    }


def extract_budget_row(plan_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": plan_name,
        "kv_working_budget_tokens": payload.get("kv_working_budget_tokens"),
        "recent_window_tokens": payload.get("recent_window_tokens"),
        "budget_block_size": payload.get("budget_block_size"),
        "anchor_blocks": payload.get("anchor_blocks"),
        "anchor_budget_tokens": payload.get("anchor_budget_tokens"),
        "retrieval_budget_tokens": payload.get("retrieval_budget_tokens"),
        "retrieval_block_budget": payload.get("retrieval_block_budget"),
        "retrieval_top_k_requested": payload.get("retrieval_top_k_requested"),
        "retrieval_top_k_effective": payload.get("retrieval_top_k_effective"),
        "budget_overflow": payload.get("budget_overflow"),
        "budget_policy_reason": payload.get("budget_policy_reason"),
        "top_k": payload.get("top_k"),
        "num_selected_blocks": payload.get("num_selected_blocks"),
        "working_ratio": payload.get("working_ratio"),
        "mean_abs_diff": payload.get(
            "mean_abs_diff",
            payload.get("attention_compare", {}).get("mean_abs_diff"),
        ),
    }


def validate_compatible(plan_name: str, runs: dict[int, dict[str, Any]]) -> None:
    metas = {layer: extract_meta(payload) for layer, payload in runs.items()}
    first_layer = sorted(metas)[0]
    ref = metas[first_layer]

    for layer, meta in metas.items():
        if meta != ref:
            raise ValueError(
                f"Incompatible settings within plan '{plan_name}' between "
                f"layer {first_layer} and layer {layer}:\n"
                f"ref={ref}\ncur={meta}"
            )


def summarize_plan(plan_name: str, runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    validate_compatible(plan_name, runs)

    l0 = extract_mean_abs_diff(runs[0])
    l14 = extract_mean_abs_diff(runs[14])
    l27 = extract_mean_abs_diff(runs[27])

    topk_sum = sum(int(runs[layer]["top_k"]) for layer in (0, 14, 27))

    row = extract_budget_row(plan_name, runs[0])
    row.update({
        "plan": plan_name,
        "layer0_mean_abs_diff": l0,
        "layer14_mean_abs_diff": l14,
        "layer27_mean_abs_diff": l27,
        "avg_mean_abs_diff": mean([l0, l14, l27]),
        "max_mean_abs_diff": max([l0, l14, l27]),
        "total_top_k": topk_sum,
        "meta": extract_meta(runs[0]),
    })
    return row


def format_float(x: float) -> str:
    if x is None:
        return "None"
    return f"{x:.9f}"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return format_float(value)
    if value is None:
        return "None"
    return str(value)


def make_budget_markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "plan",
        "kv_working_budget_tokens",
        "recent_window_tokens",
        "budget_block_size",
        "anchor_blocks",
        "anchor_budget_tokens",
        "retrieval_budget_tokens",
        "retrieval_block_budget",
        "retrieval_top_k_requested",
        "retrieval_top_k_effective",
        "budget_overflow",
        "budget_policy_reason",
        "top_k",
        "num_selected_blocks",
        "working_ratio",
        "mean_abs_diff",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def make_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| plan | working_tokens | recent | budget_block_size | anchor_blocks | anchor_tokens | retrieval_tokens | retrieval_block_budget | retrieval_top_k_requested | retrieval_top_k_effective | budget_overflow | budget_reason | top_k | num_selected_blocks | working_ratio | mean_abs_diff | layer 0 mean_abs_diff | layer 14 mean_abs_diff | layer 27 mean_abs_diff | avg | max | total_top_k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        meta = row["meta"]
        lines.append(
            f"| {row['plan']} "
            f"| {meta.get('kv_working_budget_tokens')} "
            f"| {meta.get('recent_window_tokens')} "
            f"| {meta.get('budget_block_size')} "
            f"| {meta.get('anchor_blocks')} "
            f"| {meta.get('anchor_budget_tokens')} "
            f"| {meta.get('retrieval_budget_tokens')} "
            f"| {meta.get('retrieval_block_budget')} "
            f"| {meta.get('retrieval_top_k_requested')} "
            f"| {meta.get('retrieval_top_k_effective')} "
            f"| {meta.get('budget_overflow')} "
            f"| {meta.get('budget_policy_reason')} "
            f"| {row.get('top_k')} "
            f"| {row.get('num_selected_blocks')} "
            f"| {format_cell(row.get('working_ratio'))} "
            f"| {format_cell(row.get('mean_abs_diff'))} "
            f"| {format_float(row['layer0_mean_abs_diff'])} "
            f"| {format_float(row['layer14_mean_abs_diff'])} "
            f"| {format_float(row['layer27_mean_abs_diff'])} "
            f"| {format_float(row['avg_mean_abs_diff'])} "
            f"| {format_float(row['max_mean_abs_diff'])} "
            f"| {row['total_top_k']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a comparison table for layer-wise budget plans."
    )

    parser.add_argument("--uniform-layer0", type=Path)
    parser.add_argument("--uniform-layer14", type=Path)
    parser.add_argument("--uniform-layer27", type=Path)

    parser.add_argument("--balanced-layer0", type=Path)
    parser.add_argument("--balanced-layer14", type=Path)
    parser.add_argument("--balanced-layer27", type=Path)

    parser.add_argument("--hard-layer0", type=Path)
    parser.add_argument("--hard-layer14", type=Path)
    parser.add_argument("--hard-layer27", type=Path)

    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/processed/layer_budget_comparison.md"),
        help="Output markdown table path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/processed/layer_budget_comparison.json"),
        help="Output summary json path.",
    )
    parser.add_argument(
        "--single-json",
        type=Path,
        help="Single RelayKV pipeline JSON to render as a budget metadata table.",
    )

    args = parser.parse_args()

    plans: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []

    if args.single_json:
        single_rows.append(extract_budget_row(args.single_json.stem, load_json(args.single_json)))

    if args.uniform_layer0 and args.uniform_layer14 and args.uniform_layer27:
        uniform = {
            0: load_json(args.uniform_layer0),
            14: load_json(args.uniform_layer14),
            27: load_json(args.uniform_layer27),
        }
        plans.append(summarize_plan("uniform", uniform))

    if args.balanced_layer0 and args.balanced_layer14 and args.balanced_layer27:
        balanced = {
            0: load_json(args.balanced_layer0),
            14: load_json(args.balanced_layer14),
            27: load_json(args.balanced_layer27),
        }
        plans.append(summarize_plan("balanced", balanced))

    if args.hard_layer0 and args.hard_layer14 and args.hard_layer27:
        hard = {
            0: load_json(args.hard_layer0),
            14: load_json(args.hard_layer14),
            27: load_json(args.hard_layer27),
        }
        plans.append(summarize_plan("hard-layer-heavy", hard))

    if not plans and not single_rows:
        raise ValueError("No complete plan was provided.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    md = make_budget_markdown_table(single_rows) if single_rows else make_markdown_table(plans)
    args.output_md.write_text(md + "\n", encoding="utf-8")

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(single_rows if single_rows else plans, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {args.output_md}")
    print(f"saved json: {args.output_json}")


if __name__ == "__main__":
    main()
