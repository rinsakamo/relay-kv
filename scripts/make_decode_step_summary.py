#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make step-level summary from RelayKV decode prototype output.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    data = load_json(args.input)
    rows = []

    for step in data["step_logs"]:
        row = {
            "step_idx": step["step_idx"],
            "generated_token_id": step["generated_token_id"],
        }
        for layer in ["0", "14", "27"]:
            item = step["layers"][layer]
            row[f"layer{layer}_top_k"] = item["top_k"]
            row[f"layer{layer}_selected_block_ids"] = item["selected_block_ids"]
            row[f"layer{layer}_candidate_k_len"] = item["candidate_k_len"]
            row[f"layer{layer}_working_k_len"] = item["working_k_len"]
            row[f"layer{layer}_coverage_ratio"] = item["coverage_ratio"]
            row[f"layer{layer}_working_ratio"] = item["working_ratio"]
            row[f"layer{layer}_is_contiguous"] = item["candidate_kv"]["is_contiguous"]
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step_idx", "generated_token_id",
            "layer0_top_k", "layer0_selected_block_ids", "layer0_candidate_k_len", "layer0_working_k_len", "layer0_coverage_ratio", "layer0_working_ratio", "layer0_is_contiguous",
            "layer14_top_k", "layer14_selected_block_ids", "layer14_candidate_k_len", "layer14_working_k_len", "layer14_coverage_ratio", "layer14_working_ratio", "layer14_is_contiguous",
            "layer27_top_k", "layer27_selected_block_ids", "layer27_candidate_k_len", "layer27_working_k_len", "layer27_coverage_ratio", "layer27_working_ratio", "layer27_is_contiguous",
        ])
        for r in rows:
            writer.writerow([
                r["step_idx"], r["generated_token_id"],
                r["layer0_top_k"], str(r["layer0_selected_block_ids"]), r["layer0_candidate_k_len"], r["layer0_working_k_len"], f'{r["layer0_coverage_ratio"]:.6f}', f'{r["layer0_working_ratio"]:.6f}', r["layer0_is_contiguous"],
                r["layer14_top_k"], str(r["layer14_selected_block_ids"]), r["layer14_candidate_k_len"], r["layer14_working_k_len"], f'{r["layer14_coverage_ratio"]:.6f}', f'{r["layer14_working_ratio"]:.6f}', r["layer14_is_contiguous"],
                r["layer27_top_k"], str(r["layer27_selected_block_ids"]), r["layer27_candidate_k_len"], r["layer27_working_k_len"], f'{r["layer27_coverage_ratio"]:.6f}', f'{r["layer27_working_ratio"]:.6f}', r["layer27_is_contiguous"],
            ])

    md_lines = []
    md_lines.append(f"# RelayKV Decode Prototype — Step-Level Summary ({data['prompt_type']})")
    md_lines.append("")
    md_lines.append("## Run-level")
    md_lines.append("")
    md_lines.append(f"- budget_plan: `{data['budget_plan']}`")
    md_lines.append(f"- layer_budget_map: `{data['layer_budget_map']}`")
    md_lines.append(f"- generated_tokens: `{data['generated_tokens']}`")
    md_lines.append(f"- elapsed_sec: `{data['elapsed_sec']:.6f}`")
    md_lines.append(f"- tokens_per_sec: `{data['tokens_per_sec']:.6f}`")
    md_lines.append("")
    md_lines.append("## Layer 27 focus")
    md_lines.append("")
    md_lines.append("| step | token_id | selected_block_ids | candidate_k_len | working_k_len | coverage_ratio | working_ratio | contiguous |")
    md_lines.append("|---:|---:|---|---:|---:|---:|---:|---|")
    for r in rows:
        md_lines.append(
            f"| {r['step_idx']} | {r['generated_token_id']} | `{r['layer27_selected_block_ids']}` | "
            f"{r['layer27_candidate_k_len']} | {r['layer27_working_k_len']} | "
            f"{r['layer27_coverage_ratio']:.6f} | {r['layer27_working_ratio']:.6f} | {r['layer27_is_contiguous']} |"
        )

    md_lines.append("")
    md_lines.append("## Compact all-layer view")
    md_lines.append("")
    md_lines.append("| step | layer0 ids / cand / work | layer14 ids / cand / work | layer27 ids / cand / work |")
    md_lines.append("|---:|---|---|---|")
    for r in rows:
        md_lines.append(
            f"| {r['step_idx']} | "
            f"`{r['layer0_selected_block_ids']}` / {r['layer0_candidate_k_len']} / {r['layer0_working_k_len']} | "
            f"`{r['layer14_selected_block_ids']}` / {r['layer14_candidate_k_len']} / {r['layer14_working_k_len']} | "
            f"`{r['layer27_selected_block_ids']}` / {r['layer27_candidate_k_len']} / {r['layer27_working_k_len']} |"
        )

    args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"saved markdown: {args.output_md}")
    print(f"saved csv: {args.output_csv}")


if __name__ == "__main__":
    main()