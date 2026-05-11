#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CASE_COLUMNS = [
    "case_name",
    "case_class",
    "global_working_kv_budget_mib",
    "target_concurrent_requests",
    "request_working_kv_budget_mib",
    "derived_target_keep_blocks",
    "kept_blocks_count",
    "dropped_blocks_count",
    "working_ratio",
    "mean_abs_diff",
    "max_abs_diff",
    "fallback_reason",
    "error",
]


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.9f}"


def derive_output_path(input_json: Path, suffix: str) -> Path:
    return input_json.with_name(f"{input_json.stem}{suffix}")


def classify_case(case: dict[str, Any]) -> str:
    error = case.get("error")
    fallback_reason = case.get("fallback_reason")
    demotion_target_resolution = case.get("demotion_target_resolution") or {}
    demotion_fallback_reason = demotion_target_resolution.get("fallback_reason")
    drop_block_ids = case.get("drop_block_ids") or []

    if error is not None and (
        fallback_reason == "vram_budget_not_ok"
        or demotion_fallback_reason == "vram_budget_not_ok"
    ):
        return "budget_not_ok"
    if error is not None:
        return "case_error"
    if fallback_reason == "fullkv_within_budget":
        return "fullkv_within_budget"
    if fallback_reason is not None and fallback_reason != "fullkv_within_budget":
        return "demotion_failed"
    if error is None and fallback_reason is None and len(drop_block_ids) > 0:
        return "actual_demotion"
    return "unknown"


def make_case_row(case: dict[str, Any]) -> dict[str, Any]:
    keep_block_ids = case.get("keep_block_ids") or []
    drop_block_ids = case.get("drop_block_ids") or []
    return {
        "case_name": case.get("case_name"),
        "case_class": classify_case(case),
        "global_working_kv_budget_mib": case.get("global_working_kv_budget_mib"),
        "target_concurrent_requests": case.get("target_concurrent_requests"),
        "request_working_kv_budget_mib": case.get("request_working_kv_budget_mib"),
        "derived_target_keep_blocks": case.get("derived_target_keep_blocks"),
        "kept_blocks_count": len(keep_block_ids),
        "dropped_blocks_count": len(drop_block_ids),
        "working_ratio": case.get("working_ratio"),
        "mean_abs_diff": case.get("mean_abs_diff"),
        "max_abs_diff": case.get("max_abs_diff"),
        "fallback_reason": case.get("fallback_reason"),
        "error": case.get("error"),
    }


def build_aggregate_counts(case_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total_cases": len(case_rows),
        "budget_not_ok_cases": 0,
        "fullkv_within_budget_cases": 0,
        "actual_demotion_cases": 0,
        "demotion_failed_cases": 0,
        "case_error_cases": 0,
        "unknown_cases": 0,
    }

    for row in case_rows:
        case_class = row["case_class"]
        if case_class == "budget_not_ok":
            counts["budget_not_ok_cases"] += 1
        elif case_class == "fullkv_within_budget":
            counts["fullkv_within_budget_cases"] += 1
        elif case_class == "actual_demotion":
            counts["actual_demotion_cases"] += 1
        elif case_class == "demotion_failed":
            counts["demotion_failed_cases"] += 1
        elif case_class == "case_error":
            counts["case_error_cases"] += 1
        else:
            counts["unknown_cases"] += 1

    return counts


def make_markdown(
    source_input_path: Path,
    summary: dict[str, Any],
    aggregate_counts: dict[str, int],
    case_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# VRAM Budget Demotion Sweep Summary",
        "",
        f"Source: `{source_input_path}`",
        "",
        "## Metadata",
        "",
        f"- model: `{summary.get('model')}`",
        f"- device: `{summary.get('device')}`",
        f"- seq_len_actual: `{summary.get('seq_len_actual')}`",
        f"- block_size: `{summary.get('block_size')}`",
        f"- layer_idx: `{summary.get('layer_idx')}`",
        f"- prompt_type: `{summary.get('prompt_type')}`",
        f"- total_blocks: `{summary.get('total_blocks')}`",
        "",
        "## Aggregate Counts",
        "",
        "| metric | value |",
        "|---|---:|",
    ]

    for key, value in aggregate_counts.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| case_name | case_class | global_working_kv_budget_mib | target_concurrent_requests | request_working_kv_budget_mib | derived_target_keep_blocks | kept_blocks_count | dropped_blocks_count | working_ratio | mean_abs_diff | max_abs_diff | fallback_reason | error |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for row in case_rows:
        lines.append(
            f"| {row['case_name']} "
            f"| {row['case_class']} "
            f"| {row['global_working_kv_budget_mib']} "
            f"| {row['target_concurrent_requests']} "
            f"| {format_float(row['request_working_kv_budget_mib'])} "
            f"| {row['derived_target_keep_blocks']} "
            f"| {row['kept_blocks_count']} "
            f"| {row['dropped_blocks_count']} "
            f"| {format_float(row['working_ratio'])} "
            f"| {format_float(row['mean_abs_diff'])} "
            f"| {format_float(row['max_abs_diff'])} "
            f"| {row['fallback_reason'] or ''} "
            f"| {row['error'] or ''} |"
        )

    return "\n".join(lines)


def write_csv(output_csv: Path, case_rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        writer.writerows(case_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a RelayKV VRAM-budget demotion sweep JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input vram_budget_demotion_sweep JSON path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown path; defaults to <input_stem>_summary.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path; defaults to <input_stem>_summary.csv",
    )
    args = parser.parse_args()

    output_md = args.output_md or derive_output_path(args.input_json, "_summary.md")
    output_csv = args.output_csv or derive_output_path(args.input_json, "_summary.csv")

    with args.input_json.open(encoding="utf-8") as f:
        summary = json.load(f)

    case_rows = [make_case_row(case) for case in summary.get("cases", [])]
    aggregate_counts = build_aggregate_counts(case_rows)
    markdown = make_markdown(
        source_input_path=args.input_json,
        summary=summary,
        aggregate_counts=aggregate_counts,
        case_rows=case_rows,
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w", encoding="utf-8") as f:
        f.write(markdown)

    write_csv(output_csv, case_rows)

    print(f"saved md: {output_md}")
    print(f"saved csv: {output_csv}")
    print(json.dumps(aggregate_counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
