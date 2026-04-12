#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROCESSED_RESULTS_DIR = Path("results/processed")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_prompt_row(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_type = payload["prompt_type"]
    full_gen = payload["full_generation"]
    relaykv = payload["relaykv"]

    uniform = relaykv["uniform"]["layers"]
    very_heavy = relaykv["very-heavy"]["layers"]

    uniform_l0 = float(uniform["0"]["attention_compare"]["mean_abs_diff"])
    uniform_l14 = float(uniform["14"]["attention_compare"]["mean_abs_diff"])
    uniform_l27 = float(uniform["27"]["attention_compare"]["mean_abs_diff"])

    very_l0 = float(very_heavy["0"]["attention_compare"]["mean_abs_diff"])
    very_l14 = float(very_heavy["14"]["attention_compare"]["mean_abs_diff"])
    very_l27 = float(very_heavy["27"]["attention_compare"]["mean_abs_diff"])

    uniform_avg = (uniform_l0 + uniform_l14 + uniform_l27) / 3.0
    uniform_max = max(uniform_l0, uniform_l14, uniform_l27)

    very_avg = (very_l0 + very_l14 + very_l27) / 3.0
    very_max = max(very_l0, very_l14, very_l27)

    layer27_abs_improvement = uniform_l27 - very_l27
    layer27_rel_improvement = (
        (layer27_abs_improvement / uniform_l27) if uniform_l27 > 0 else 0.0
    )

    avg_abs_improvement = uniform_avg - very_avg
    avg_rel_improvement = (
        (avg_abs_improvement / uniform_avg) if uniform_avg > 0 else 0.0
    )

    return {
        "prompt_type": prompt_type,
        "full_tokens_per_sec": float(full_gen["tokens_per_sec"]),
        "full_elapsed_sec": float(full_gen["elapsed_sec"]),
        "generated_tokens": int(full_gen["generated_tokens"]),
        "uniform_avg": uniform_avg,
        "uniform_max": uniform_max,
        "uniform_layer27": uniform_l27,
        "very_heavy_avg": very_avg,
        "very_heavy_max": very_max,
        "very_heavy_layer27": very_l27,
        "avg_abs_improvement": avg_abs_improvement,
        "avg_rel_improvement": avg_rel_improvement,
        "layer27_abs_improvement": layer27_abs_improvement,
        "layer27_rel_improvement": layer27_rel_improvement,
    }


def format_float(x: float) -> str:
    return f"{x:.9f}"


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def make_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RelayKV Quick Benchmark Cross-Style Summary",
        "",
        "| prompt | full tok/s | uniform avg | uniform max | very-heavy avg | very-heavy max | layer27 uniform | layer27 very-heavy | avg improvement | layer27 improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['prompt_type']} "
            f"| {row['full_tokens_per_sec']:.6f} "
            f"| {format_float(row['uniform_avg'])} "
            f"| {format_float(row['uniform_max'])} "
            f"| {format_float(row['very_heavy_avg'])} "
            f"| {format_float(row['very_heavy_max'])} "
            f"| {format_float(row['uniform_layer27'])} "
            f"| {format_float(row['very_heavy_layer27'])} "
            f"| {format_pct(row['avg_rel_improvement'])} "
            f"| {format_pct(row['layer27_rel_improvement'])} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cross-style markdown table from RelayKV quick benchmark JSON files."
    )
    parser.add_argument("--repetitive", type=Path, required=True)
    parser.add_argument("--prose", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROCESSED_RESULTS_DIR / "relaykv_quick_benchmark_crossstyle.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROCESSED_RESULTS_DIR / "relaykv_quick_benchmark_crossstyle.json",
    )
    args = parser.parse_args()

    rows = [
        extract_prompt_row(load_json(args.repetitive)),
        extract_prompt_row(load_json(args.prose)),
        extract_prompt_row(load_json(args.structured)),
    ]

   # fixed order
    prompt_order = {"repetitive": 0, "prose": 1, "structured": 2}
    rows.sort(key=lambda r: prompt_order.get(r["prompt_type"], 999))

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    md = make_markdown_table(rows)
    args.output_md.write_text(md + "\n", encoding="utf-8")

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {args.output_md}")
    print(f"saved json: {args.output_json}")


if __name__ == "__main__":
    main()