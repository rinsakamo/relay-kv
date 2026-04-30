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
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "seq_len": 32,
    "hot_window": 8,
    "block_size": 8,
    "top_k": 1,
    "layer_idx": 0,
    "prompt_type": "default",
}

CASE_SETS: dict[str, list[dict[str, Any]]] = {
    "smoke": [
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
    ],
    "budget_tokens": [
        {
            "name": "tokens_1024",
            "kv_working_budget_tokens": 1024,
            "recent_window": 768,
            "budget_block_size": 128,
            "anchor_blocks": 4,
            "retrieval_top_k": 8,
        },
        {
            "name": "tokens_2048",
            "kv_working_budget_tokens": 2048,
            "recent_window": 768,
            "budget_block_size": 128,
            "anchor_blocks": 4,
            "retrieval_top_k": 8,
        },
        {
            "name": "tokens_4096",
            "kv_working_budget_tokens": 4096,
            "recent_window": 768,
            "budget_block_size": 128,
            "anchor_blocks": 4,
            "retrieval_top_k": 8,
        },
    ],
    "residual_mib": [
        {
            "name": "mib_128",
            "available_kv_budget_mib": 128,
            "recent_window": 768,
            "budget_block_size": 128,
            "anchor_blocks": 4,
            "retrieval_top_k": 8,
        },
        {
            "name": "mib_256",
            "available_kv_budget_mib": 256,
            "recent_window": 768,
            "budget_block_size": 128,
            "anchor_blocks": 4,
            "retrieval_top_k": 8,
        },
        {
            "name": "mib_512",
            "available_kv_budget_mib": 512,
            "recent_window": 768,
            "budget_block_size": 128,
            "anchor_blocks": 4,
            "retrieval_top_k": 8,
        },
    ],
}


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def model_suffix(model_name: str) -> str:
    if model_name == DEFAULT_PIPELINE_ARGS["model_name"]:
        return ""
    name = model_name.lower().split("/")[-1]
    name = name.replace("qwen2.5", "qwen2p5").replace("-instruct", "")
    name = name.replace("-", "_")
    return f"_{slug(name)}"


def default_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    prompt = slug(args.prompt_type)
    run_name = (
        f"{args.case_set}_seq{args.seq_len}_{prompt}_layer{args.layer_idx}"
        f"{model_suffix(args.model_name)}"
    )
    return (
        RAW_DIR / run_name,
        ROOT / f"results/processed/budget_metadata_cases_{run_name}.md",
        ROOT / f"results/processed/budget_metadata_cases_{run_name}.json",
    )


def build_command(
    case: dict[str, Any], output_path: Path, args: argparse.Namespace
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_relaykv_pipeline.py"),
        "--model",
        args.model_name,
        "--seq-len",
        str(args.seq_len),
        "--hot-window",
        str(args.hot_window),
        "--block-size",
        str(args.block_size),
        "--top-k",
        str(args.top_k),
        "--layer-idx",
        str(args.layer_idx),
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
    if args.prompt_type != "default":
        cmd.extend(["--prompt-type", args.prompt_type])
    if case.get("kv_working_budget_tokens"):
        cmd.extend(["--kv-working-budget-tokens", str(case["kv_working_budget_tokens"])])
    if case.get("available_kv_budget_mib"):
        cmd.extend(["--available-kv-budget-mib", str(case["available_kv_budget_mib"])])
    return cmd


def run_case(
    case: dict[str, Any], raw_dir: Path, env: dict[str, str], args: argparse.Namespace
) -> Path:
    output_path = raw_dir / f"{case['name']}.json"
    cmd = build_command(case, output_path, args)
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
        description="Run metadata-only RelayKV budget cases via run_relaykv_pipeline.py.",
        epilog=(
            "Example: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache "
            ".venv/bin/python scripts/run_budget_metadata_cases.py "
            "--model-name Qwen/Qwen2.5-1.5B-Instruct "
            "--case-set budget_tokens --seq-len 1024 --prompt-type structured "
            "--layer-idx 0 --top-k 1"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_PIPELINE_ARGS["model_name"],
        help="Model name passed through to run_relaykv_pipeline.py --model.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_PIPELINE_ARGS["seq_len"])
    parser.add_argument(
        "--prompt-type",
        type=str,
        default=DEFAULT_PIPELINE_ARGS["prompt_type"],
        help="Pipeline prompt type. Use 'default' to keep run_relaykv_pipeline.py default.",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=DEFAULT_PIPELINE_ARGS["layer_idx"]
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_PIPELINE_ARGS["top_k"])
    parser.add_argument(
        "--hot-window", type=int, default=DEFAULT_PIPELINE_ARGS["hot_window"]
    )
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_PIPELINE_ARGS["block_size"]
    )
    parser.add_argument(
        "--case-set",
        choices=sorted(CASE_SETS),
        default="smoke",
        help="Budget metadata case set to run.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/relaykv_pycache")

    default_raw_dir, default_output_md, default_output_json = default_output_paths(args)
    raw_dir = args.raw_dir or default_raw_dir
    output_md = args.output_md or default_output_md
    output_json = args.output_json or default_output_json
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASE_SETS[args.case_set]:
        output_path = run_case(case, raw_dir, env, args)
        rows.append(extract_budget_row(case["name"], load_json(output_path)))

    md = make_budget_markdown_table(rows)
    output_md.write_text(md + "\n", encoding="utf-8")
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(md)
    print(f"\nsaved markdown: {output_md}")
    print(f"saved json: {output_json}")


if __name__ == "__main__":
    main()
