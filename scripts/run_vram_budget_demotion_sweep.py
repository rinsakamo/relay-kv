#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from relaykv import (
    ColdSegment,
    build_candidate_kv,
    build_demotion_decision,
    build_vram_budget_decision,
    compare_attention_outputs,
    retrieve_blocks_by_ids,
)
from scripts.run_relaykv_pipeline import (
    load_model,
    make_prompt_for_target_tokens,
    require_usable_demotion_target_for_dry_run,
    resolve_demotion_target_resolution,
)


PROCESSED_RESULTS_DIR = Path("results/processed")
DEFAULT_OUTPUT_JSON = PROCESSED_RESULTS_DIR / "vram_budget_demotion_sweep.json"
DEFAULT_OUTPUT_MD = PROCESSED_RESULTS_DIR / "vram_budget_demotion_sweep.md"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GLOBAL_WORKING_KV_BUDGET_MIB_LIST = "32,64,128"
DEFAULT_TARGET_CONCURRENT_REQUESTS_LIST = "1,2,4"


def ensure_results_dir() -> None:
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_float_list(csv_text: str) -> list[float]:
    values = [part.strip() for part in csv_text.split(",") if part.strip()]
    if not values:
        raise ValueError("No float values were provided")

    parsed = [float(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("All float values must be > 0")
    return parsed


def parse_int_list(csv_text: str) -> list[int]:
    values = [part.strip() for part in csv_text.split(",") if part.strip()]
    if not values:
        raise ValueError("No integer values were provided")

    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("All integer values must be > 0")
    return parsed


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.9f}"


def make_case_name(
    *,
    global_working_kv_budget_mib: float,
    target_concurrent_requests: int,
) -> str:
    budget_text = f"{global_working_kv_budget_mib:g}".replace(".", "p")
    return f"budget_{budget_text}_concurrency_{target_concurrent_requests}"


def make_markdown_table(case_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| case_name | global_working_kv_budget_mib | target_concurrent_requests | request_working_kv_budget_mib | derived_target_keep_blocks | keep_block_ids | drop_block_ids | working_ratio | mean_abs_diff | max_abs_diff | fallback_reason | error |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]

    for case in case_summaries:
        lines.append(
            f"| {case['case_name']} "
            f"| {case['global_working_kv_budget_mib']} "
            f"| {case['target_concurrent_requests']} "
            f"| {format_float(case['request_working_kv_budget_mib'])} "
            f"| {case['derived_target_keep_blocks']} "
            f"| {case['keep_block_ids']} "
            f"| {case['drop_block_ids']} "
            f"| {format_float(case['working_ratio'])} "
            f"| {format_float(case['mean_abs_diff'])} "
            f"| {format_float(case['max_abs_diff'])} "
            f"| {case['fallback_reason'] or ''} "
            f"| {case['error'] or ''} |"
        )

    return "\n".join(lines)


def make_error_case_summary(
    *,
    case_name: str,
    global_working_kv_budget_mib: float,
    target_concurrent_requests: int,
    error: Exception,
    vram_budget_decision: dict[str, Any] | None = None,
    demotion_target_resolution: dict[str, Any] | None = None,
    demotion_policy_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "global_working_kv_budget_mib": global_working_kv_budget_mib,
        "target_concurrent_requests": target_concurrent_requests,
        "request_working_kv_budget_mib": (
            vram_budget_decision.get("request_working_kv_budget_mib")
            if vram_budget_decision is not None
            else None
        ),
        "derived_target_keep_blocks": (
            vram_budget_decision.get("derived_target_keep_blocks")
            if vram_budget_decision is not None
            else None
        ),
        "keep_block_ids": [],
        "drop_block_ids": [],
        "working_k_len": None,
        "working_ratio": None,
        "mean_abs_diff": None,
        "max_abs_diff": None,
        "fallback_reason": (
            demotion_policy_decision.get("fallback_reason")
            if demotion_policy_decision is not None
            else (
                demotion_target_resolution.get("fallback_reason")
                if demotion_target_resolution is not None
                else (
                    vram_budget_decision.get("fallback_reason")
                    if vram_budget_decision is not None
                    else None
                )
            )
        ),
        "error": str(error),
        "vram_budget_decision": vram_budget_decision,
        "demotion_target_resolution": demotion_target_resolution,
        "demotion_policy_decision": demotion_policy_decision,
        "candidate_kv": None,
        "attention_compare": None,
    }


def run_case(
    *,
    layers: Any,
    layer_idx: int,
    block_size: int,
    global_working_kv_budget_mib: float,
    target_concurrent_requests: int,
    global_residual_vram_mib: float | None,
    kv_dtype_bytes: int,
    demotion_recent_blocks: int,
    protect_boundary_blocks: int,
    protect_prefix_blocks: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    case_name = make_case_name(
        global_working_kv_budget_mib=global_working_kv_budget_mib,
        target_concurrent_requests=target_concurrent_requests,
    )
    full_k = layers[layer_idx].keys
    full_v = layers[layer_idx].values
    query = full_k[:, :, -1:, :]
    seq_len_actual = int(full_k.shape[2])

    full_segment = ColdSegment(
        layer_idx=layer_idx,
        start=0,
        end=seq_len_actual,
        k=full_k,
        v=full_v,
    )
    full_blocks = full_segment.to_blocks(block_size=block_size)
    total_blocks = len(full_blocks)

    vram_budget_decision_obj = build_vram_budget_decision(
        global_residual_vram_mib=global_residual_vram_mib,
        global_working_kv_budget_mib=global_working_kv_budget_mib,
        target_concurrent_requests=target_concurrent_requests,
        allocation_policy="equal_share",
        kv_dtype_bytes=kv_dtype_bytes,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    vram_budget_decision = vram_budget_decision_obj.summary()

    demotion_target_resolution = resolve_demotion_target_resolution(
        demotion_policy_mode="dry_run",
        explicit_target_keep_blocks=None,
        vram_budget_decision=vram_budget_decision_obj,
    )
    require_usable_demotion_target_for_dry_run(
        demotion_policy_mode="dry_run",
        demotion_target_resolution=demotion_target_resolution,
    )

    demotion_policy_decision_obj = build_demotion_decision(
        total_blocks=total_blocks,
        target_keep_blocks=demotion_target_resolution["effective_target_keep_blocks"],
        recent_blocks=demotion_recent_blocks,
        protect_boundary_blocks=protect_boundary_blocks,
        protect_prefix_blocks=protect_prefix_blocks,
        demotion_strategy="oldest",
    )
    demotion_policy_decision = demotion_policy_decision_obj.summary()
    keep_block_ids = demotion_policy_decision_obj.keep_block_ids
    drop_block_ids = demotion_policy_decision_obj.drop_block_ids

    if not keep_block_ids:
        raise ValueError(f"Case '{case_name}' produced no keep blocks")

    retrieved = retrieve_blocks_by_ids(
        full_blocks,
        layer_idx=layer_idx,
        block_ids=keep_block_ids,
    )
    candidate_kv = build_candidate_kv(retrieved)
    attention_result = compare_attention_outputs(
        query=query,
        full_k=full_k,
        full_v=full_v,
        approx_k=candidate_kv.k,
        approx_v=candidate_kv.v,
    )

    working_k_len = int(candidate_kv.k.shape[2])
    working_ratio = working_k_len / seq_len_actual if seq_len_actual > 0 else 0.0

    return {
        "case_name": case_name,
        "global_working_kv_budget_mib": global_working_kv_budget_mib,
        "target_concurrent_requests": target_concurrent_requests,
        "request_working_kv_budget_mib": vram_budget_decision_obj.request_working_kv_budget_mib,
        "derived_target_keep_blocks": vram_budget_decision_obj.derived_target_keep_blocks,
        "keep_block_ids": keep_block_ids,
        "drop_block_ids": drop_block_ids,
        "working_k_len": working_k_len,
        "working_ratio": working_ratio,
        "mean_abs_diff": attention_result.mean_abs_diff,
        "max_abs_diff": attention_result.max_abs_diff,
        "fallback_reason": (
            demotion_policy_decision_obj.fallback_reason
            or vram_budget_decision_obj.fallback_reason
        ),
        "error": None,
        "vram_budget_decision": vram_budget_decision,
        "demotion_target_resolution": demotion_target_resolution,
        "demotion_policy_decision": demotion_policy_decision,
        "candidate_kv": candidate_kv.summary(),
        "attention_compare": attention_result.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a RelayKV VRAM-budget-to-demotion dry-run sweep."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Target input sequence length",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Full-KV block size",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Layer index for attention comparison",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="structured",
        help="Prompt type: repetitive, prose, structured",
    )
    parser.add_argument(
        "--global-working-kv-budget-mib-list",
        type=str,
        default=DEFAULT_GLOBAL_WORKING_KV_BUDGET_MIB_LIST,
        help="Comma-separated global working KV budgets in MiB",
    )
    parser.add_argument(
        "--target-concurrent-requests-list",
        type=str,
        default=DEFAULT_TARGET_CONCURRENT_REQUESTS_LIST,
        help="Comma-separated target concurrency values",
    )
    parser.add_argument(
        "--global-residual-vram-mib",
        type=float,
        default=None,
        help="Optional residual global VRAM ceiling in MiB",
    )
    parser.add_argument(
        "--kv-dtype-bytes",
        type=int,
        default=2,
        help="KV dtype size in bytes used for VRAM budget derivation",
    )
    parser.add_argument(
        "--demotion-recent-blocks",
        type=int,
        default=0,
        help="Recent block count protected by the demotion dry-run policy",
    )
    parser.add_argument(
        "--protect-boundary-blocks",
        type=int,
        default=1,
        help="Boundary block count before the recent window protected from demotion",
    )
    parser.add_argument(
        "--protect-prefix-blocks",
        type=int,
        default=0,
        help="Prefix block count protected from demotion",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path under results/processed/",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Output Markdown path under results/processed/",
    )
    args = parser.parse_args()

    budget_list = parse_float_list(args.global_working_kv_budget_mib_list)
    concurrency_list = parse_int_list(args.target_concurrent_requests_list)

    ensure_results_dir()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model(args.model)
    prompt = make_prompt_for_target_tokens(args.seq_len, args.prompt_type)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.seq_len,
    )
    if device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    layers = outputs.past_key_values.layers
    seq_len_actual = int(layers[0].keys.shape[2])
    inferred_num_layers = len(layers)
    inferred_num_kv_heads = int(layers[0].keys.shape[1])
    inferred_head_dim = int(layers[0].keys.shape[-1])
    total_blocks = (seq_len_actual + args.block_size - 1) // args.block_size

    case_summaries: list[dict[str, Any]] = []
    for global_working_kv_budget_mib in budget_list:
        for target_concurrent_requests in concurrency_list:
            case_name = make_case_name(
                global_working_kv_budget_mib=global_working_kv_budget_mib,
                target_concurrent_requests=target_concurrent_requests,
            )
            try:
                case_summaries.append(
                    run_case(
                        layers=layers,
                        layer_idx=args.layer_idx,
                        block_size=args.block_size,
                        global_working_kv_budget_mib=global_working_kv_budget_mib,
                        target_concurrent_requests=target_concurrent_requests,
                        global_residual_vram_mib=args.global_residual_vram_mib,
                        kv_dtype_bytes=args.kv_dtype_bytes,
                        demotion_recent_blocks=args.demotion_recent_blocks,
                        protect_boundary_blocks=args.protect_boundary_blocks,
                        protect_prefix_blocks=args.protect_prefix_blocks,
                        num_layers=inferred_num_layers,
                        num_kv_heads=inferred_num_kv_heads,
                        head_dim=inferred_head_dim,
                    )
                )
            except Exception as exc:
                partial_vram_budget_decision: dict[str, Any] | None = None
                partial_demotion_target_resolution: dict[str, Any] | None = None
                try:
                    vram_budget_decision_obj = build_vram_budget_decision(
                        global_residual_vram_mib=args.global_residual_vram_mib,
                        global_working_kv_budget_mib=global_working_kv_budget_mib,
                        target_concurrent_requests=target_concurrent_requests,
                        allocation_policy="equal_share",
                        kv_dtype_bytes=args.kv_dtype_bytes,
                        num_layers=inferred_num_layers,
                        num_kv_heads=inferred_num_kv_heads,
                        head_dim=inferred_head_dim,
                        block_size=args.block_size,
                    )
                    partial_vram_budget_decision = vram_budget_decision_obj.summary()
                    partial_demotion_target_resolution = (
                        resolve_demotion_target_resolution(
                            demotion_policy_mode="dry_run",
                            explicit_target_keep_blocks=None,
                            vram_budget_decision=vram_budget_decision_obj,
                        )
                    )
                except Exception:
                    pass

                case_summaries.append(
                    make_error_case_summary(
                        case_name=case_name,
                        global_working_kv_budget_mib=global_working_kv_budget_mib,
                        target_concurrent_requests=target_concurrent_requests,
                        error=exc,
                        vram_budget_decision=partial_vram_budget_decision,
                        demotion_target_resolution=partial_demotion_target_resolution,
                    )
                )

    summary = {
        "model": args.model,
        "device": device,
        "seq_len": args.seq_len,
        "seq_len_actual": seq_len_actual,
        "block_size": args.block_size,
        "layer_idx": args.layer_idx,
        "prompt_type": args.prompt_type,
        "global_residual_vram_mib": args.global_residual_vram_mib,
        "kv_dtype_bytes": args.kv_dtype_bytes,
        "demotion_recent_blocks": args.demotion_recent_blocks,
        "protect_boundary_blocks": args.protect_boundary_blocks,
        "protect_prefix_blocks": args.protect_prefix_blocks,
        "num_layers": inferred_num_layers,
        "num_kv_heads": inferred_num_kv_heads,
        "head_dim": inferred_head_dim,
        "total_blocks": total_blocks,
        "cases": case_summaries,
    }

    markdown = make_markdown_table(case_summaries)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with args.output_md.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved json: {args.output_json}")
    print(f"saved md: {args.output_md}")

    if any(case.get("error") for case in case_summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
