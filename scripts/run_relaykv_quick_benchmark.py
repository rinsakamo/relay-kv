#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any

from statistics import mean

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


RESULTS_DIR = Path("results/processed")


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def get_plan_spec(plan_name: str) -> dict[int, int]:
    if plan_name == "uniform":
        return {0: 3, 14: 3, 27: 3}
    if plan_name == "very-heavy":
        return {0: 1, 14: 1, 27: 7}
    raise ValueError(f"Unsupported plan: {plan_name}")


def run_layer_eval(
    *,
    layers,
    layer_idx: int,
    top_k: int,
    hot_window: int,
    block_size: int,
    scoring_variant: str,
) -> dict[str, Any]:
    seq_len_actual = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=hot_window)
    split = tier_manager.split_range(seq_len_actual)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)
    all_blocks = cold_cache.blockify(block_size=block_size)
    metadata = build_metadata_for_blocks(all_blocks)

    query = layers[layer_idx].keys[:, :, -1:, :]
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

    return {
        "layer_idx": layer_idx,
        "top_k": top_k,
        "selected_block_ids": [s.block_id for s in top_scores],
        "selected_block_spans": [[s.start, s.end] for s in top_scores],
        "candidate_k_len": int(candidate_kv.k.shape[2]),
        "working_k_len": int(working_kv.k.shape[2]),
        "coverage_ratio": float(candidate_kv.k.shape[2] / (split.cold_range[1] - split.cold_range[0])),
        "working_ratio": float(working_kv.k.shape[2] / full_k.shape[2]),
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
        "attention_compare": attention_result.summary(),
    }


def run_generation(
    *,
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    prompt_len = int(inputs["input_ids"].shape[1])

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start

    total_len = int(outputs.shape[1])
    new_tokens = total_len - prompt_len

    generated_only = outputs[0][prompt_len:]
    decoded = tokenizer.decode(generated_only, skip_special_tokens=True)

    tps = (new_tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": new_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tps,
        "generated_text": decoded,
    }

def summarize_relaykv_plan(plan_payload: dict[str, Any]) -> dict[str, float]:
    layer_items = plan_payload["layers"]

    l0 = float(layer_items["0"]["attention_compare"]["mean_abs_diff"])
    l14 = float(layer_items["14"]["attention_compare"]["mean_abs_diff"])
    l27 = float(layer_items["27"]["attention_compare"]["mean_abs_diff"])

    return {
        "layer0_mean_abs_diff": l0,
        "layer14_mean_abs_diff": l14,
        "layer27_mean_abs_diff": l27,
        "avg_mean_abs_diff": mean([l0, l14, l27]),
        "max_mean_abs_diff": max([l0, l14, l27]),
    }


def format_float(x: float) -> str:
    return f"{x:.9f}"


def make_markdown_summary(summary: dict[str, Any]) -> str:
    full_gen = summary["full_generation"]
    relaykv = summary["relaykv"]

    uniform_metrics = summarize_relaykv_plan(relaykv["uniform"])
    very_heavy_metrics = summarize_relaykv_plan(relaykv["very-heavy"])

    lines: list[str] = []

    lines.append("# RelayKV Quick Benchmark Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- model: `{summary['model']}`")
    lines.append(f"- seq_len_target: `{summary['seq_len_target']}`")
    lines.append(f"- seq_len_actual: `{summary['seq_len_actual']}`")
    lines.append(f"- prompt_type: `{summary['prompt_type']}`")
    lines.append(f"- hot_window: `{summary['hot_window']}`")
    lines.append(f"- block_size: `{summary['block_size']}`")
    lines.append(f"- scoring_variant: `{summary['scoring_variant']}`")
    lines.append(f"- max_new_tokens: `{summary['max_new_tokens']}`")
    lines.append("")
    lines.append("## Full generation")
    lines.append("")
    lines.append(f"- prompt_tokens: `{full_gen['prompt_tokens']}`")
    lines.append(f"- generated_tokens: `{full_gen['generated_tokens']}`")
    lines.append(f"- elapsed_sec: `{full_gen['elapsed_sec']:.6f}`")
    lines.append(f"- tokens_per_sec: `{full_gen['tokens_per_sec']:.6f}`")
    lines.append("")
    lines.append("## RelayKV comparison")
    lines.append("")
    lines.append("| plan | layer 0 mean_abs_diff | layer 14 mean_abs_diff | layer 27 mean_abs_diff | avg | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| uniform "
        f"| {format_float(uniform_metrics['layer0_mean_abs_diff'])} "
        f"| {format_float(uniform_metrics['layer14_mean_abs_diff'])} "
        f"| {format_float(uniform_metrics['layer27_mean_abs_diff'])} "
        f"| {format_float(uniform_metrics['avg_mean_abs_diff'])} "
        f"| {format_float(uniform_metrics['max_mean_abs_diff'])} |"
    )
    lines.append(
        f"| very-heavy "
        f"| {format_float(very_heavy_metrics['layer0_mean_abs_diff'])} "
        f"| {format_float(very_heavy_metrics['layer14_mean_abs_diff'])} "
        f"| {format_float(very_heavy_metrics['layer27_mean_abs_diff'])} "
        f"| {format_float(very_heavy_metrics['avg_mean_abs_diff'])} "
        f"| {format_float(very_heavy_metrics['max_mean_abs_diff'])} |"
    )
    lines.append("")
    lines.append("## Layer details")
    lines.append("")

    for plan_name in ["uniform", "very-heavy"]:
        lines.append(f"### {plan_name}")
        lines.append("")
        lines.append("| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for layer_idx in ["0", "14", "27"]:
            item = relaykv[plan_name]["layers"][layer_idx]
            lines.append(
                f"| {layer_idx} "
                f"| {item['top_k']} "
                f"| {item['candidate_k_len']} "
                f"| {item['working_k_len']} "
                f"| {item['coverage_ratio']:.6f} "
                f"| {item['working_ratio']:.6f} "
                f"| `{item['selected_block_ids']}` |"
            )
        lines.append("")

    lines.append("## Generated continuation")
    lines.append("")
    lines.append("```text")
    lines.append(full_gen["generated_text"])
    lines.append("```")
    lines.append("")

    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick benchmark for full vs RelayKV settings.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--prompt-type", type=str, default="prose", choices=["repetitive", "prose", "structured"])
    parser.add_argument("--hot-window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--scoring-variant", type=str, default="mean_plus_norm")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output",
        type=str,
        default="relaykv_quick_benchmark.json",
        help="Output filename under results/processed/",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional markdown summary filename under results/processed/",
    )
    args = parser.parse_args()

    ensure_results_dir()

    tokenizer, model, device = load_model(args.model)
    prompt = make_prompt_for_target_tokens(args.seq_len, args.prompt_type)

    # 1. Full generation benchmark
    full_generation = run_generation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )

    # 2. One forward pass for attention-comparison based RelayKV summaries
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
    seq_len_actual = int(layers[0].keys.shape[2])

    relaykv_runs: dict[str, Any] = {}
    for plan_name in ["uniform", "very-heavy"]:
        spec = get_plan_spec(plan_name)
        layer_results = {}
        for layer_idx in (0, 14, 27):
            layer_results[str(layer_idx)] = run_layer_eval(
                layers=layers,
                layer_idx=layer_idx,
                top_k=spec[layer_idx],
                hot_window=args.hot_window,
                block_size=args.block_size,
                scoring_variant=args.scoring_variant,
            )
        relaykv_runs[plan_name] = {
            "plan": plan_name,
            "top_k_spec": spec,
            "layers": layer_results,
        }

    summary = {
        "model": args.model,
        "seq_len_target": args.seq_len,
        "seq_len_actual": seq_len_actual,
        "prompt_type": args.prompt_type,
        "hot_window": args.hot_window,
        "block_size": args.block_size,
        "scoring_variant": args.scoring_variant,
        "max_new_tokens": args.max_new_tokens,
        "full_generation": full_generation,
        "relaykv": relaykv_runs,
    }

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

        markdown_text = make_markdown_summary(summary)

        if args.output_md is None:
            md_name = Path(args.output).stem + ".md"
        else:
            md_name = args.output_md

        md_path = RESULTS_DIR / md_name
        with md_path.open("w", encoding="utf-8") as f:
            f.write(markdown_text + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved json: {output_path}")
    print(f"saved markdown: {md_path}")




if __name__ == "__main__":
    main()