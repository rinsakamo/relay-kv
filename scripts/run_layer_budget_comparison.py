#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from relaykv import (
    TierManager,
    split_dynamic_cache_layers,
    build_metadata_for_blocks,
    score_blocks_with_query,
    top_k_blocks,
    retrieve_blocks,
    build_candidate_kv,
    build_working_kv,
    compare_attention_outputs,
)


RAW_RESULTS_DIR = Path("results/raw/prototype_checks")
PROCESSED_RESULTS_DIR = Path("results/processed")


def ensure_results_dirs() -> None:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return tokenizer, model, device


def make_prompt_for_target_tokens(target_tokens: int, prompt_type: str) -> str:
    repetitive_unit = (
        "RelayKV checks recent context retrieval behavior. "
        "RelayKV checks recent context retrieval behavior. "
        "RelayKV checks recent context retrieval behavior. "
    )

    prose_unit = (
        "RelayKV is a prototype for splitting KV cache into hot and cold regions, "
        "retrieving a smaller working set, and comparing approximate attention outputs "
        "against full attention. The current experiments examine how coverage ratio, "
        "block granularity, and layer difficulty affect approximation quality across "
        "different sequence lengths and prompt styles. "
    )

    structured_unit = (
        "Experiment summary:\n"
        "- system: RelayKV\n"
        "- goal: compare approximate attention against full attention\n"
        "- factors: coverage ratio, block size, hot window, layer index\n"
        "- observation: harder layers may require larger retrieval budgets\n"
        "- note: scoring changes and block granularity should be evaluated separately\n"
    )

    if prompt_type == "repetitive":
        base = repetitive_unit
    elif prompt_type == "prose":
        base = prose_unit
    elif prompt_type == "structured":
        base = structured_unit
    else:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    words = base.split()
    chunks: list[str] = []

    while len(" ".join(chunks).split()) < target_tokens:
        chunks.extend(words)

    return " ".join(chunks[:target_tokens])


def extract_plan_spec(plan_name: str) -> dict[int, int]:
    if plan_name == "uniform":
        return {0: 3, 14: 3, 27: 3}
    if plan_name == "balanced":
        return {0: 3, 14: 2, 27: 4}
    if plan_name == "hard-layer-heavy":
        return {0: 2, 14: 1, 27: 6}
    if plan_name == "mild-heavy":
        return {0: 2, 14: 2, 27: 5}
    if plan_name == "very-heavy":
        return {0: 1, 14: 1, 27: 7}
    raise ValueError(f"Unsupported plan: {plan_name}")


def run_single_layer(
    *,
    model_name: str,
    device: str,
    layers,
    seq_len_target: int,
    seq_len_actual: int,
    prompt_type: str,
    hot_window: int,
    block_size: int,
    layer_idx: int,
    top_k: int,
    scoring_variant: str,
) -> dict[str, Any]:
    tier_manager = TierManager(hot_window=hot_window)
    split = tier_manager.split_range(seq_len_actual)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)
    all_blocks = cold_cache.blockify(block_size=block_size)
    metadata = build_metadata_for_blocks(all_blocks)

    query = layers[layer_idx].keys[:, :, -1:, :]  # [1, heads, 1, head_dim]
    layer_metadata = [m for m in metadata if m.layer_idx == layer_idx]

    scores = score_blocks_with_query(
        layer_metadata,
        query[:, :, 0, :],
        variant=scoring_variant,
        norm_weight=1e-3,
        all_blocks=all_blocks,
    )
    top_scores = top_k_blocks(scores, k=min(top_k, len(scores)))
    retrieved = retrieve_blocks(all_blocks, top_scores)

    selected_summaries = [s.summary() for s in top_scores]
    selected_block_ranks = list(range(len(selected_summaries)))
    selected_block_ids = [s["block_id"] for s in selected_summaries]
    selected_block_scores = [s["score"] for s in selected_summaries]
    selected_block_spans = [[s["start"], s["end"]] for s in selected_summaries]

    candidate_kv = build_candidate_kv(retrieved)

    hot_k = hot_kv.keys[layer_idx]
    hot_v = hot_kv.values[layer_idx]

    working_kv = build_working_kv(
        candidate_kv=candidate_kv,
        hot_k=hot_k,
        hot_v=hot_v,
        hot_range=split.hot_range,
    )

    full_k = layers[layer_idx].keys
    full_v = layers[layer_idx].values

    attention_result = compare_attention_outputs(
        query=query,
        full_k=full_k,
        full_v=full_v,
        approx_k=working_kv.k,
        approx_v=working_kv.v,
    )

    cold_k_len = split.cold_range[1] - split.cold_range[0]
    candidate_k_len = candidate_kv.k.shape[2]
    working_k_len = working_kv.k.shape[2]
    full_k_len = full_k.shape[2]

    coverage_ratio = candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    working_ratio = working_k_len / full_k_len if full_k_len > 0 else 0.0

    return {
        "model": model_name,
        "device": device,
        "seq_len_target": seq_len_target,
        "seq_len_actual": seq_len_actual,
        "layer_idx": layer_idx,
        "hot_window": hot_window,
        "block_size": block_size,
        "top_k": top_k,
        "cold_range": list(split.cold_range),
        "hot_range": list(split.hot_range),
        "num_layers": len(layers),
        "num_all_blocks": len(all_blocks),
        "num_layer_blocks": len(layer_metadata),
        "num_selected_blocks": len(top_scores),
        "cold_k_len": cold_k_len,
        "candidate_k_len": candidate_k_len,
        "full_k_len": full_k_len,
        "working_k_len": working_k_len,
        "coverage_ratio": coverage_ratio,
        "working_ratio": working_ratio,
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
        "attention_compare": attention_result.summary(),
        "top_scores": selected_summaries,
        "selected_block_ranks": selected_block_ranks,
        "selected_block_ids": selected_block_ids,
        "selected_block_scores": selected_block_scores,
        "selected_block_spans": selected_block_spans,
        "scoring_variant": scoring_variant,
        "prompt_type": prompt_type,
    }


def save_layer_summary(plan_name: str, prompt_type: str, layer_idx: int, summary: dict[str, Any]) -> Path:
    path = RAW_RESULTS_DIR / f"relaykv_layer{layer_idx}_{plan_name}_{prompt_type}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def summarize_plan(plan_name: str, runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    l0 = float(runs[0]["attention_compare"]["mean_abs_diff"])
    l14 = float(runs[14]["attention_compare"]["mean_abs_diff"])
    l27 = float(runs[27]["attention_compare"]["mean_abs_diff"])
    total_top_k = sum(int(runs[layer]["top_k"]) for layer in (0, 14, 27))

    return {
        "plan": plan_name,
        "layer0_mean_abs_diff": l0,
        "layer14_mean_abs_diff": l14,
        "layer27_mean_abs_diff": l27,
        "avg_mean_abs_diff": mean([l0, l14, l27]),
        "max_mean_abs_diff": max([l0, l14, l27]),
        "total_top_k": total_top_k,
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
    parser = argparse.ArgumentParser(description="Run one-shot layer budget comparison and emit a table.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hot-window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--prompt-type", type=str, default="prose", choices=["repetitive", "prose", "structured"])
    parser.add_argument("--scoring-variant", type=str, default="mean_plus_norm")
    parser.add_argument(
        "--plans",
        nargs="+",
        default=[
            "uniform",
            "balanced",
            "hard-layer-heavy",
            "mild-heavy",
            "very-heavy",
        ],
        choices=[
            "uniform",
            "balanced",
            "hard-layer-heavy",
            "mild-heavy",
            "very-heavy",
        ],
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown output path. Defaults to results/processed/layer_budget_comparison_<prompt>.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional json output path. Defaults to results/processed/layer_budget_comparison_<prompt>.json",
    )
    args = parser.parse_args()

    ensure_results_dirs()

    tokenizer, model, device = load_model(args.model)

    prompt = make_prompt_for_target_tokens(args.seq_len, args.prompt_type)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.seq_len,
    )

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    layers = outputs.past_key_values.layers
    seq_len_actual = layers[0].keys.shape[2]

    plan_rows: list[dict[str, Any]] = []
    saved_files: dict[str, dict[int, str]] = {}

    for plan_name in args.plans:
        spec = extract_plan_spec(plan_name)
        plan_runs: dict[int, dict[str, Any]] = {}
        saved_files[plan_name] = {}

        for layer_idx in (0, 14, 27):
            top_k = spec[layer_idx]
            summary = run_single_layer(
                model_name=args.model,
                device=device,
                layers=layers,
                seq_len_target=args.seq_len,
                seq_len_actual=seq_len_actual,
                prompt_type=args.prompt_type,
                hot_window=args.hot_window,
                block_size=args.block_size,
                layer_idx=layer_idx,
                top_k=top_k,
                scoring_variant=args.scoring_variant,
            )
            plan_runs[layer_idx] = summary
            path = save_layer_summary(plan_name, args.prompt_type, layer_idx, summary)
            saved_files[plan_name][layer_idx] = str(path)

        plan_rows.append(summarize_plan(plan_name, plan_runs))

    output_md = args.output_md or (PROCESSED_RESULTS_DIR / f"layer_budget_comparison_{args.prompt_type}.md")
    output_json = args.output_json or (PROCESSED_RESULTS_DIR / f"layer_budget_comparison_{args.prompt_type}.json")

    md = make_markdown_table(plan_rows)
    output_md.write_text(md + "\n", encoding="utf-8")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_type": args.prompt_type,
                "seq_len": args.seq_len,
                "hot_window": args.hot_window,
                "block_size": args.block_size,
                "scoring_variant": args.scoring_variant,
                "plans": plan_rows,
                "saved_layer_files": saved_files,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(md)
    print(f"\nsaved markdown: {output_md}")
    print(f"saved json: {output_json}")


if __name__ == "__main__":
    main()