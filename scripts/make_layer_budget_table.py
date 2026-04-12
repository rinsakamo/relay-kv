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

    return {
        "plan": plan_name,
        "layer0_mean_abs_diff": l0,
        "layer14_mean_abs_diff": l14,
        "layer27_mean_abs_diff": l27,
        "avg_mean_abs_diff": mean([l0, l14, l27]),
        "max_mean_abs_diff": max([l0, l14, l27]),
        "total_top_k": topk_sum,
        "meta": extract_meta(runs[0]),
    }


def format_float(x: float) -> str:
    return f"{x:.9f}"


def make_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| plan | layer 0 mean_abs_diff | layer 14 mean_abs_diff | layer 27 mean_abs_diff | avg | max | total_top_k |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['plan']} "
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

    args = parser.parse_args()

    plans: list[dict[str, Any]] = []

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

    if not plans:
        raise ValueError("No complete plan was provided.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    md = make_markdown_table(plans)
    args.output_md.write_text(md + "\n", encoding="utf-8")

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {args.output_md}")
    print(f"saved json: {args.output_json}")


if __name__ == "__main__":
    main()