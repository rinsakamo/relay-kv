#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ("repetitive", "prose", "structured")
DEFAULT_LAYERS = (0, 14, 27)
DEFAULT_MODEL_SUFFIX = "qwen2p5_3b"
DEFAULT_OUTPUT_MD = ROOT / "results/processed/budget_metadata_3b_sweep_seq1024_summary.md"
DEFAULT_OUTPUT_JSON = ROOT / "results/processed/budget_metadata_3b_sweep_seq1024_summary.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload: {path}")
    return payload


def format_float(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def input_path(
    processed_dir: Path,
    case_set: str,
    seq_len: int,
    prompt_type: str,
    layer_idx: int,
    model_suffix: str,
) -> Path:
    return (
        processed_dir
        / f"budget_metadata_cases_{case_set}_seq{seq_len}_{prompt_type}"
        f"_layer{layer_idx}_{model_suffix}.json"
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_type in args.prompt_type:
        for layer_idx in args.layer_idx:
            path = input_path(
                args.processed_dir,
                args.case_set,
                args.seq_len,
                prompt_type,
                layer_idx,
                args.model_suffix,
            )
            for row in load_json(path):
                rows.append(
                    {
                        "model_name": row.get("model_name"),
                        "prompt_type": prompt_type,
                        "layer_idx": layer_idx,
                        "plan": row.get("plan"),
                        "kv_bytes_per_token": row.get("kv_bytes_per_token"),
                        "kv_working_budget_tokens": row.get(
                            "kv_working_budget_tokens"
                        ),
                        "retrieval_top_k_effective": row.get(
                            "retrieval_top_k_effective"
                        ),
                        "budget_overflow": row.get("budget_overflow"),
                        "budget_policy_reason": row.get("budget_policy_reason"),
                        "top_k": row.get("top_k"),
                        "num_selected_blocks": row.get("num_selected_blocks"),
                        "working_ratio": row.get("working_ratio"),
                        "mean_abs_diff": row.get("mean_abs_diff"),
                    }
                )
    return rows


def make_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "model_name",
        "prompt_type",
        "layer_idx",
        "plan",
        "kv_bytes_per_token",
        "kv_working_budget_tokens",
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
        "|---|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize RelayKV budget metadata case tables across prompts/layers."
    )
    parser.add_argument("--case-set", default="budget_tokens")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--model-suffix", default=DEFAULT_MODEL_SUFFIX)
    parser.add_argument(
        "--prompt-type",
        action="append",
        choices=DEFAULT_PROMPTS,
        default=[],
        help="Prompt type to include. Repeat to select multiple. Defaults to all.",
    )
    parser.add_argument(
        "--layer-idx",
        action="append",
        type=int,
        default=[],
        help="Layer index to include. Repeat to select multiple. Defaults to 0/14/27.",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "results/processed"
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    if not args.prompt_type:
        args.prompt_type = list(DEFAULT_PROMPTS)
    if not args.layer_idx:
        args.layer_idx = list(DEFAULT_LAYERS)

    rows = build_rows(args)
    md = make_markdown(rows)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md + "\n", encoding="utf-8")
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {args.output_md}")
    print(f"saved json: {args.output_json}")


if __name__ == "__main__":
    main()
