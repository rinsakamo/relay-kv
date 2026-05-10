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
    compare_attention_outputs,
    retrieve_blocks_by_ids,
)
from scripts.run_relaykv_pipeline import load_model, make_prompt_for_target_tokens


PROCESSED_RESULTS_DIR = Path("results/processed")
DEFAULT_OUTPUT_JSON = PROCESSED_RESULTS_DIR / "drop_policy_sweep.json"
DEFAULT_OUTPUT_MD = PROCESSED_RESULTS_DIR / "drop_policy_sweep.md"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CASE_ORDER = [
    "full",
    "drop_oldest_1",
    "drop_oldest_2",
    "drop_middle_2",
    "drop_boundary_before_recent_1",
    "keep_recent4",
    "keep_recent4_boundary1",
    "keep_recent4_anchor1_boundary1",
]


def ensure_results_dir() -> None:
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_cases(cases_arg: str | None) -> list[str]:
    if cases_arg is None:
        return list(DEFAULT_CASE_ORDER)

    case_names = [part.strip() for part in cases_arg.split(",") if part.strip()]
    if not case_names:
        raise ValueError("No case names were provided to --cases")

    unknown = [name for name in case_names if name not in DEFAULT_CASE_ORDER]
    if unknown:
        raise ValueError(f"Unsupported case names: {unknown}")

    return case_names


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.9f}"


def dedupe_in_order(block_ids: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for block_id in block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        deduped.append(block_id)
    return deduped


def sort_block_ids(block_ids: list[int]) -> list[int]:
    return sorted(block_ids)


def get_recent_block_ids(total_blocks: int, recent_blocks_for_cases: int) -> list[int]:
    recent_count = max(0, min(total_blocks, recent_blocks_for_cases))
    return list(range(total_blocks - recent_count, total_blocks))


def get_boundary_before_recent_block_id(
    total_blocks: int,
    recent_blocks_for_cases: int,
) -> int | None:
    recent_block_ids = get_recent_block_ids(total_blocks, recent_blocks_for_cases)
    if not recent_block_ids:
        return None

    boundary_block_id = recent_block_ids[0] - 1
    if boundary_block_id < 0:
        return None
    return boundary_block_id


def get_middle_drop_block_ids(candidate_block_ids: list[int]) -> list[int]:
    if len(candidate_block_ids) < 2:
        return []

    middle_start = max(0, (len(candidate_block_ids) - 2) // 2)
    middle_end = middle_start + 2
    return candidate_block_ids[middle_start:middle_end]


def build_case_decision(
    case_name: str,
    total_blocks: int,
    recent_blocks_for_cases: int,
) -> tuple[list[int], list[int], dict[str, Any], str | None]:
    all_block_ids = list(range(total_blocks))
    recent_block_ids = get_recent_block_ids(total_blocks, recent_blocks_for_cases)
    recent_protected = set(recent_block_ids)
    boundary_block_id = get_boundary_before_recent_block_id(
        total_blocks=total_blocks,
        recent_blocks_for_cases=recent_blocks_for_cases,
    )
    eviction_candidate_block_ids = [
        block_id for block_id in all_block_ids if block_id not in recent_protected
    ]
    eviction_excluded_block_ids = list(recent_block_ids)

    fallback_reasons: list[str] = []

    if case_name == "full":
        keep_block_ids = list(all_block_ids)
        drop_block_ids: list[int] = []
    elif case_name == "drop_oldest_1":
        drop_block_ids = [0] if total_blocks >= 1 else []
        if total_blocks < 1:
            fallback_reasons.append("no_blocks_available")
        keep_block_ids = [block_id for block_id in all_block_ids if block_id not in set(drop_block_ids)]
    elif case_name == "drop_oldest_2":
        drop_block_ids = [block_id for block_id in [0, 1] if block_id < total_blocks]
        if total_blocks < 2:
            fallback_reasons.append("insufficient_blocks_for_drop_oldest_2")
        keep_block_ids = [block_id for block_id in all_block_ids if block_id not in set(drop_block_ids)]
    elif case_name == "drop_middle_2":
        drop_block_ids = get_middle_drop_block_ids(eviction_candidate_block_ids)
        if len(drop_block_ids) < 2:
            fallback_reasons.append("insufficient_blocks_for_drop_middle_2")
        keep_block_ids = [block_id for block_id in all_block_ids if block_id not in set(drop_block_ids)]
    elif case_name == "drop_boundary_before_recent_1":
        if boundary_block_id is None:
            drop_block_ids = []
            fallback_reasons.append("no_boundary_before_recent_block")
        else:
            drop_block_ids = [boundary_block_id]
        keep_block_ids = [block_id for block_id in all_block_ids if block_id not in set(drop_block_ids)]
    elif case_name == "keep_recent4":
        keep_block_ids = list(recent_block_ids)
        drop_block_ids = [block_id for block_id in all_block_ids if block_id not in set(keep_block_ids)]
        if len(recent_block_ids) < min(4, total_blocks):
            fallback_reasons.append("recent_window_truncated_by_total_blocks")
    elif case_name == "keep_recent4_boundary1":
        keep_block_ids = list(recent_block_ids)
        if boundary_block_id is None:
            fallback_reasons.append("no_boundary_before_recent_block")
        else:
            keep_block_ids.append(boundary_block_id)
        keep_block_ids = dedupe_in_order(keep_block_ids)
        drop_block_ids = [block_id for block_id in all_block_ids if block_id not in set(keep_block_ids)]
    elif case_name == "keep_recent4_anchor1_boundary1":
        keep_block_ids = list(recent_block_ids)
        if total_blocks > 0:
            keep_block_ids.insert(0, 0)
        else:
            fallback_reasons.append("no_blocks_available")
        if boundary_block_id is None:
            fallback_reasons.append("no_boundary_before_recent_block")
        else:
            keep_block_ids.append(boundary_block_id)
        keep_block_ids = dedupe_in_order(keep_block_ids)
        drop_block_ids = [block_id for block_id in all_block_ids if block_id not in set(keep_block_ids)]
    else:
        raise ValueError(f"Unsupported case_name: {case_name}")

    keep_block_ids = sort_block_ids(
        dedupe_in_order(
            [block_id for block_id in keep_block_ids if 0 <= block_id < total_blocks]
        )
    )
    drop_block_ids = sort_block_ids(
        dedupe_in_order(
            [block_id for block_id in drop_block_ids if 0 <= block_id < total_blocks]
        )
    )

    reason_labels_by_block: dict[str, list[str]] = {}
    for block_id in all_block_ids:
        labels = ["FULLKV_REFERENCE"]
        if block_id in recent_protected:
            labels.append("RECENT_PROTECTED")
        if boundary_block_id is not None and block_id == boundary_block_id:
            labels.append("BOUNDARY_NEAR_RECENT")
        if block_id in eviction_candidate_block_ids:
            labels.append("PREFIX_CANDIDATE")
        if block_id in keep_block_ids and case_name.startswith("keep_"):
            labels.append("KEEP_BY_CASE")
        if case_name == "drop_oldest_1" and block_id in drop_block_ids:
            labels.append("DROP_OLDEST")
        if case_name == "drop_oldest_2" and block_id in drop_block_ids:
            labels.append("DROP_OLDEST")
        if case_name == "drop_middle_2" and block_id in drop_block_ids:
            labels.append("DROP_MIDDLE")
        reason_labels_by_block[str(block_id)] = labels

    decision_summary = {
        "eviction_excluded_block_ids": sort_block_ids(eviction_excluded_block_ids),
        "eviction_candidate_block_ids": sort_block_ids(eviction_candidate_block_ids),
        "demoted_block_ids": sort_block_ids(drop_block_ids),
        "reason_labels_by_block": reason_labels_by_block,
    }

    fallback_reason = ",".join(fallback_reasons) if fallback_reasons else None
    return keep_block_ids, drop_block_ids, decision_summary, fallback_reason


def run_case(
    *,
    case_name: str,
    layers: Any,
    layer_idx: int,
    block_size: int,
    recent_blocks_for_cases: int,
) -> dict[str, Any]:
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

    keep_block_ids, drop_block_ids, decision_summary, fallback_reason = build_case_decision(
        case_name=case_name,
        total_blocks=total_blocks,
        recent_blocks_for_cases=recent_blocks_for_cases,
    )

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
        "total_blocks": total_blocks,
        "keep_block_ids": keep_block_ids,
        "drop_block_ids": drop_block_ids,
        "working_k_len": working_k_len,
        "working_ratio": working_ratio,
        "mean_abs_diff": attention_result.mean_abs_diff,
        "max_abs_diff": attention_result.max_abs_diff,
        "fallback_reason": fallback_reason,
        "error": None,
        "decision_summary": decision_summary,
        "candidate_kv": candidate_kv.summary(),
        "attention_compare": attention_result.summary(),
    }


def make_error_case_summary(
    *,
    case_name: str,
    total_blocks: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "total_blocks": total_blocks,
        "keep_block_ids": [],
        "drop_block_ids": [],
        "working_k_len": None,
        "working_ratio": None,
        "mean_abs_diff": None,
        "max_abs_diff": None,
        "fallback_reason": None,
        "error": str(error),
        "decision_summary": {
            "eviction_excluded_block_ids": [],
            "eviction_candidate_block_ids": [],
            "demoted_block_ids": [],
            "reason_labels_by_block": {},
        },
    }


def make_markdown_table(case_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| case | keep_blocks | drop_blocks | working_ratio | mean_abs_diff | max_abs_diff | fallback_reason | error |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]

    for case in case_summaries:
        lines.append(
            f"| {case['case_name']} "
            f"| {case['keep_block_ids']} "
            f"| {case['drop_block_ids']} "
            f"| {format_float(case['working_ratio'])} "
            f"| {format_float(case['mean_abs_diff'])} "
            f"| {format_float(case['max_abs_diff'])} "
            f"| {case['fallback_reason'] or ''} "
            f"| {case['error'] or ''} |"
        )

    return "\n".join(lines)


def make_markdown_diagnostics(case_summaries: list[dict[str, Any]]) -> str:
    lines = ["", "Diagnostics:"]
    for case in case_summaries:
        decision_summary = case["decision_summary"]
        lines.append(
            f"- {case['case_name']}: "
            f"excluded={decision_summary['eviction_excluded_block_ids']}, "
            f"candidates={decision_summary['eviction_candidate_block_ids']}, "
            f"demoted={decision_summary['demoted_block_ids']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a RelayKV full-KV drop policy dry-run sweep."
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
        default=1024,
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
        default=14,
        help="Layer index for attention comparison",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="structured",
        help="Prompt type: repetitive, prose, structured",
    )
    parser.add_argument(
        "--recent-blocks-for-cases",
        type=int,
        default=4,
        help="Recent block count used by built-in keep/drop case definitions",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated subset of cases to run",
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
    total_blocks = (seq_len_actual + args.block_size - 1) // args.block_size

    case_names = parse_cases(args.cases)
    case_summaries: list[dict[str, Any]] = []
    for case_name in case_names:
        try:
            case_summaries.append(
                run_case(
                    case_name=case_name,
                    layers=layers,
                    layer_idx=args.layer_idx,
                    block_size=args.block_size,
                    recent_blocks_for_cases=args.recent_blocks_for_cases,
                )
            )
        except Exception as exc:
            case_summaries.append(
                make_error_case_summary(
                    case_name=case_name,
                    total_blocks=total_blocks,
                    error=exc,
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
        "recent_blocks_for_cases": args.recent_blocks_for_cases,
        "cases": case_summaries,
    }

    markdown = make_markdown_table(case_summaries) + make_markdown_diagnostics(
        case_summaries
    )

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with args.output_md.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved json: {args.output_json}")
    print(f"saved md: {args.output_md}")


if __name__ == "__main__":
    main()
