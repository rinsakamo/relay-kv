#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
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


RESULTS_DIR = Path("results/raw/prototype_checks")


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


def get_layer_budget_map(plan_name: str) -> dict[int, int]:
    if plan_name == "uniform":
        return {0: 3, 14: 3, 27: 3}
    if plan_name == "very-heavy":
        return {0: 1, 14: 1, 27: 7}
    raise ValueError(f"Unsupported plan: {plan_name}")


def summarize_step_layer(
    *,
    layers,
    layer_idx: int,
    top_k: int,
    hot_window: int,
    block_size: int,
    scoring_variant: str,
    include_attention_compare: bool = False,
) -> dict[str, Any]:
    seq_len_actual = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=hot_window)
    split = tier_manager.split_range(seq_len_actual)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)
    all_blocks = cold_cache.blockify(block_size=block_size)
    metadata = build_metadata_for_blocks(all_blocks)

    layer_metadata = [m for m in metadata if m.layer_idx == layer_idx]
    query = layers[layer_idx].keys[:, :, -1:, :]  # [1, heads, 1, head_dim]

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

    cold_k_len = split.cold_range[1] - split.cold_range[0]
    full_k = layers[layer_idx].keys
    full_v = layers[layer_idx].values
    full_k_len = full_k.shape[2]
    candidate_k_len = int(candidate_kv.k.shape[2])
    working_k_len = int(working_kv.k.shape[2])

    coverage_ratio = candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    working_ratio = working_k_len / full_k_len if full_k_len > 0 else 0.0

    result = {
        "layer_idx": layer_idx,
        "top_k": top_k,
        "selected_block_ids": [s.block_id for s in top_scores],
        "selected_block_scores": [float(s.score) for s in top_scores],
        "selected_block_spans": [[s.start, s.end] for s in top_scores],
        "candidate_k_len": candidate_k_len,
        "working_k_len": working_k_len,
        "coverage_ratio": coverage_ratio,
        "working_ratio": working_ratio,
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
    }

    if include_attention_compare:
        attention_result = compare_attention_outputs(
            query=query,
            full_k=full_k,
            full_v=full_v,
            approx_k=working_kv.k,
            approx_v=working_kv.v,
        )
        result["attention_compare"] = attention_result.summary()

    return result


def run_prefill(
    *,
    model,
    tokenizer,
    device: str,
    prompt: str,
    seq_len: int,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    return inputs, outputs


def run_decode_loop_with_shadow_logging(
    *,
    model,
    tokenizer,
    device: str,
    prefill_inputs,
    prefill_outputs,
    max_new_tokens: int,
    hot_window: int,
    block_size: int,
    scoring_variant: str,
    layer_budget_map: dict[int, int],
    include_attention_compare: bool = False,
) -> dict[str, Any]:
    input_ids = prefill_inputs["input_ids"]
    attention_mask = prefill_inputs["attention_mask"]
    past_key_values = prefill_outputs.past_key_values

    generated_token_ids: list[int] = []
    step_logs: list[dict[str, Any]] = []

    start = time.perf_counter()

    current_input_ids = input_ids[:, -1:]
    current_attention_mask = attention_mask

    for step_idx in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        generated_token_ids.append(next_token_id)

        past_key_values = outputs.past_key_values
        layers = past_key_values.layers

        per_layer_logs = {}
        for layer_idx, top_k in layer_budget_map.items():
            per_layer_logs[str(layer_idx)] = summarize_step_layer(
                layers=layers,
                layer_idx=layer_idx,
                top_k=top_k,
                hot_window=hot_window,
                block_size=block_size,
                scoring_variant=scoring_variant,
                include_attention_compare=include_attention_compare,
            )

        step_logs.append(
            {
                "step_idx": step_idx,
                "generated_token_id": next_token_id,
                "layers": per_layer_logs,
            }
        )

        next_token = torch.tensor([[next_token_id]], device=input_ids.device)
        current_input_ids = next_token
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((current_attention_mask.shape[0], 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
            ],
            dim=1,
        )

    elapsed = time.perf_counter() - start
    tokens_per_sec = (len(generated_token_ids) / elapsed) if elapsed > 0 else 0.0
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return {
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "generated_tokens": len(generated_token_ids),
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "step_logs": step_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-mode decode-time RelayKV prototype.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--prompt-type", type=str, default="prose", choices=["repetitive", "prose", "structured"])
    parser.add_argument("--hot-window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--scoring-variant", type=str, default="mean_plus_norm")
    parser.add_argument("--budget-plan", type=str, default="very-heavy", choices=["uniform", "very-heavy"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--include-step-attention-compare",
        action="store_true",
        help="If set, compute full-vs-working attention comparison at each decode step for representative layers.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="relaykv_decode_prototype_summary.json",
    )
    args = parser.parse_args()

    ensure_results_dir()

    tokenizer, model, device = load_model(args.model)
    prompt = make_prompt_for_target_tokens(args.seq_len, args.prompt_type)
    layer_budget_map = get_layer_budget_map(args.budget_plan)

    prefill_inputs, prefill_outputs = run_prefill(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        seq_len=args.seq_len,
    )

    seq_len_actual = int(prefill_outputs.past_key_values.layers[0].keys.shape[2])

    decode_result = run_decode_loop_with_shadow_logging(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefill_inputs=prefill_inputs,
        prefill_outputs=prefill_outputs,
        max_new_tokens=args.max_new_tokens,
        hot_window=args.hot_window,
        block_size=args.block_size,
        scoring_variant=args.scoring_variant,
        layer_budget_map=layer_budget_map,
        include_attention_compare=args.include_step_attention_compare,
    )

    summary = {
        "mode": "shadow_decode_prototype",
        "model": args.model,
        "prompt_type": args.prompt_type,
        "seq_len_target": args.seq_len,
        "seq_len_actual": seq_len_actual,
        "hot_window": args.hot_window,
        "block_size": args.block_size,
        "scoring_variant": args.scoring_variant,
        "budget_plan": args.budget_plan,
        "layer_budget_map": layer_budget_map,
        "include_step_attention_compare": args.include_step_attention_compare,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": decode_result["generated_tokens"],
        "generated_text": decode_result["generated_text"],
        "elapsed_sec": decode_result["elapsed_sec"],
        "tokens_per_sec": decode_result["tokens_per_sec"],
        "step_logs": decode_result["step_logs"],
    }

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()