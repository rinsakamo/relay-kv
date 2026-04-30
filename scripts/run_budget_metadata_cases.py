#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from make_layer_budget_table import (
    extract_budget_row,
    load_json,
    make_budget_markdown_table,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results/raw/budget_metadata_cases"
PROCESSED_MD = ROOT / "results/processed/budget_metadata_cases_table.md"
PROCESSED_JSON = ROOT / "results/processed/budget_metadata_cases_table.json"

DEFAULT_PIPELINE_ARGS = {
    "seq_len": 32,
    "hot_window": 8,
    "block_size": 8,
    "top_k": 1,
    "layer_idx": 0,
}

CASES: list[dict[str, Any]] = [
    {
        "name": "tiny_16",
        "kv_working_budget_tokens": 16,
        "recent_window": 8,
        "budget_block_size": 8,
        "anchor_blocks": 1,
        "retrieval_top_k": 2,
    },
    {
        "name": "small_32",
        "kv_working_budget_tokens": 32,
        "recent_window": 8,
        "budget_block_size": 8,
        "anchor_blocks": 1,
        "retrieval_top_k": 2,
    },
    {
        "name": "medium_64",
        "kv_working_budget_tokens": 64,
        "recent_window": 16,
        "budget_block_size": 8,
        "anchor_blocks": 1,
        "retrieval_top_k": 4,
    },
    {
        "name": "mib_512",
        "available_kv_budget_mib": 512,
        "recent_window": 768,
        "budget_block_size": 128,
        "anchor_blocks": 4,
        "retrieval_top_k": 8,
    },
]


def build_command(case: dict[str, Any], output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_relaykv_pipeline.py"),
        "--seq-len",
        str(DEFAULT_PIPELINE_ARGS["seq_len"]),
        "--hot-window",
        str(DEFAULT_PIPELINE_ARGS["hot_window"]),
        "--block-size",
        str(DEFAULT_PIPELINE_ARGS["block_size"]),
        "--top-k",
        str(DEFAULT_PIPELINE_ARGS["top_k"]),
        "--layer-idx",
        str(DEFAULT_PIPELINE_ARGS["layer_idx"]),
        "--recent-window-tokens",
        str(case["recent_window"]),
        "--budget-block-size",
        str(case["budget_block_size"]),
        "--anchor-blocks",
        str(case["anchor_blocks"]),
        "--retrieval-top-k",
        str(case["retrieval_top_k"]),
        "--output",
        str(output_path),
    ]
    if case.get("kv_working_budget_tokens"):
        cmd.extend(["--kv-working-budget-tokens", str(case["kv_working_budget_tokens"])])
    if case.get("available_kv_budget_mib"):
        cmd.extend(["--available-kv-budget-mib", str(case["available_kv_budget_mib"])])
    return cmd


def run_case(case: dict[str, Any], raw_dir: Path, env: dict[str, str]) -> Path:
    output_path = raw_dir / f"{case['name']}.json"
    cmd = build_command(case, output_path)
    print(f"running {case['name']}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"case {case['name']} failed. If the model is not already cached, "
            "rerun only after preparing the cache; this script does not download models."
        ) from exc
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run metadata-only RelayKV budget cases via run_relaykv_pipeline.py."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-md", type=Path, default=PROCESSED_MD)
    parser.add_argument("--output-json", type=Path, default=PROCESSED_JSON)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/relaykv_pycache")

    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        output_path = run_case(case, raw_dir, env)
        rows.append(extract_budget_row(case["name"], load_json(output_path)))

    md = make_budget_markdown_table(rows)
    args.output_md.write_text(md + "\n", encoding="utf-8")
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {args.output_md}")
    print(f"saved json: {args.output_json}")


if __name__ == "__main__":
    main()
