#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = (
    ROOT
    / "results/processed/budget_metadata_3b_repetitive_layer27_topk_summary.md"
)
OUTPUT_JSON = (
    ROOT
    / "results/processed/budget_metadata_3b_repetitive_layer27_topk_summary.json"
)

INPUTS = {
    3: ROOT
    / "results/raw/budget_metadata_cases/budget_tokens_seq1024_repetitive_layer27_qwen2p5_3b/tokens_4096.json",
    6: ROOT
    / "results/raw/budget_metadata_cases/budget_tokens_seq1024_repetitive_layer27_qwen2p5_3b_topk6/tokens_4096.json",
    8: ROOT
    / "results/raw/budget_metadata_cases/budget_tokens_seq1024_repetitive_layer27_qwen2p5_3b_topk8/tokens_4096.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return "None"
    return str(value)


def extract_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": payload.get("model_name", payload.get("model")),
        "prompt_type": payload.get("prompt_type"),
        "layer_idx": payload.get("layer_idx"),
        "seq_len": payload.get("seq_len_actual", payload.get("seq_len_target")),
        "top_k": payload.get("top_k"),
        "num_selected_blocks": payload.get("num_selected_blocks"),
        "selected_block_ids": payload.get("selected_block_ids"),
        "working_ratio": payload.get("working_ratio"),
        "coverage_ratio": payload.get("coverage_ratio"),
        "candidate_k_len": payload.get("candidate_k_len"),
        "working_k_len": payload.get("working_k_len"),
        "cold_k_len": payload.get("cold_k_len"),
        "mean_abs_diff": payload.get("mean_abs_diff"),
        "max_abs_diff": payload.get("attention_compare", {}).get("max_abs_diff"),
    }


def make_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "model_name",
        "prompt_type",
        "layer_idx",
        "seq_len",
        "top_k",
        "num_selected_blocks",
        "selected_block_ids",
        "working_ratio",
        "coverage_ratio",
        "candidate_k_len",
        "working_k_len",
        "cold_k_len",
        "mean_abs_diff",
        "max_abs_diff",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = [extract_row(load_json(path)) for _, path in sorted(INPUTS.items())]
    md = make_markdown(rows)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md + "\n", encoding="utf-8")
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {OUTPUT_MD}")
    print(f"saved json: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
