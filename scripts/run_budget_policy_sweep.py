#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_relaykv_pipeline import run_pipeline


PROCESSED_RESULTS_DIR = Path("results/processed")
DEFAULT_OUTPUT_JSON = PROCESSED_RESULTS_DIR / "budget_policy_sweep.json"
DEFAULT_OUTPUT_MD = PROCESSED_RESULTS_DIR / "budget_policy_sweep.md"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SCORING_VARIANT = "mean_only"
DEFAULT_CASE_ORDER = [
    "recent_only",
    "anchor_recent",
    "anchor_retrieval_recent",
    "retrieval_recent",
]
DEFAULT_CASE_SPECS = {
    "recent_only": {
        "working": 3,
        "recent": 3,
        "anchor": 0,
        "retrieval": 0,
    },
    "anchor_recent": {
        "working": 3,
        "recent": 2,
        "anchor": 1,
        "retrieval": 0,
    },
    "anchor_retrieval_recent": {
        "working": 3,
        "recent": 1,
        "anchor": 1,
        "retrieval": 1,
    },
    "retrieval_recent": {
        "working": 3,
        "recent": 1,
        "anchor": 0,
        "retrieval": 2,
    },
}


def ensure_results_dir() -> None:
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_cases(cases_arg: str | None) -> list[str]:
    if cases_arg is None:
        return list(DEFAULT_CASE_ORDER)

    case_names = [part.strip() for part in cases_arg.split(",") if part.strip()]
    if not case_names:
        raise ValueError("No case names were provided to --cases")

    unknown = [name for name in case_names if name not in DEFAULT_CASE_SPECS]
    if unknown:
        raise ValueError(f"Unsupported case names: {unknown}")

    return case_names


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.9f}"


def normalize_case_budgets(
    case_name: str,
    base_case_budgets: dict[str, int],
    working_override: int | None,
) -> dict[str, int]:
    case_budgets = dict(base_case_budgets)
    if working_override is None:
        validate_case_budgets(case_name, case_budgets)
        return case_budgets

    case_budgets["working"] = working_override

    sub_budget_total = (
        int(case_budgets["recent"])
        + int(case_budgets["anchor"])
        + int(case_budgets["retrieval"])
    )
    working_budget = int(case_budgets["working"])

    if working_budget < sub_budget_total:
        validate_case_budgets(case_name, case_budgets)

    extra_slots = working_budget - sub_budget_total
    if extra_slots > 0:
        if case_name in {"recent_only", "anchor_recent"}:
            allocation_key = "recent"
        elif case_name in {"retrieval_recent", "anchor_retrieval_recent"}:
            allocation_key = "retrieval"
        elif "retrieval" in case_budgets:
            allocation_key = "retrieval"
        else:
            allocation_key = "recent"

        case_budgets[allocation_key] = int(case_budgets[allocation_key]) + extra_slots

    validate_case_budgets(case_name, case_budgets)
    return case_budgets


def validate_case_budgets(case_name: str, case_budgets: dict[str, int]) -> None:
    sub_budget_total = (
        int(case_budgets["recent"])
        + int(case_budgets["anchor"])
        + int(case_budgets["retrieval"])
    )
    working_budget = int(case_budgets["working"])

    if working_budget < sub_budget_total:
        raise ValueError(
            f"Invalid budget override for case '{case_name}': "
            f"working={working_budget}, "
            f"recent={case_budgets['recent']}, "
            f"anchor={case_budgets['anchor']}, "
            f"retrieval={case_budgets['retrieval']}, "
            f"sub_budget_total={sub_budget_total}"
        )


def make_case_summary(
    case_name: str,
    case_budgets: dict[str, int],
    payload: dict[str, Any],
    retrieval_exclude_tail_blocks: int,
) -> dict[str, Any]:
    activation_policy_decision = payload.get("activation_policy_decision")
    budget_policy_decision = payload.get("budget_policy_decision")
    if budget_policy_decision is None:
        working_block_ids = []
        recent_block_ids = []
        anchor_block_ids = []
        retrieved_block_ids = []
    else:
        selected = budget_policy_decision["selected"]
        working_block_ids = list(selected["working_block_ids"])
        recent_block_ids = list(selected["recent_block_ids"])
        anchor_block_ids = list(selected["anchor_block_ids"])
        retrieved_block_ids = list(selected["retrieved_block_ids"])
    retrieval_excluded_block_ids = list(payload.get("retrieval_excluded_block_ids", []))
    excluded_tail_set = set(retrieval_excluded_block_ids)
    recent_set = set(recent_block_ids)

    if len(working_block_ids) > int(case_budgets["working"]):
        raise ValueError(
            f"Case '{case_name}' exceeded working budget: "
            f"{len(working_block_ids)} > {case_budgets['working']}"
        )

    attention_compare = payload.get("attention_compare", {})

    return {
        "case_name": case_name,
        "budgets": dict(case_budgets),
        "activation_policy_decision": activation_policy_decision,
        "budget_policy_decision": budget_policy_decision,
        "budget_ok": (
            bool(budget_policy_decision["budget_ok"])
            if budget_policy_decision is not None
            else None
        ),
        "selected": {
            "working_block_ids": working_block_ids,
            "recent_block_ids": recent_block_ids,
            "anchor_block_ids": anchor_block_ids,
            "retrieved_block_ids": retrieved_block_ids,
        },
        "retrieval_exclude_tail_blocks": retrieval_exclude_tail_blocks,
        "retrieval_excluded_block_ids": retrieval_excluded_block_ids,
        "retrieved_overlap_with_excluded_tail": sum(
            1 for block_id in retrieved_block_ids if block_id in excluded_tail_set
        ),
        "retrieved_overlap_with_recent_blocks": sum(
            1 for block_id in retrieved_block_ids if block_id in recent_set
        ),
        "effective_retrieved_block_count": len(retrieved_block_ids),
        "working_k_len": payload.get("working_k_len"),
        "working_ratio": payload.get("working_ratio"),
        "coverage_ratio": payload.get("coverage_ratio"),
        "mean_abs_diff": attention_compare.get("mean_abs_diff"),
        "max_abs_diff": attention_compare.get("max_abs_diff"),
        "cosine_similarity": attention_compare.get("cosine_similarity"),
        "fallback_reason": (
            budget_policy_decision.get("fallback_reason")
            if budget_policy_decision is not None
            else None
        ),
        "run_identifier": (
            f"{case_name}_seq{payload.get('seq_len_actual')}_"
            f"layer{payload.get('layer_idx')}_block{payload.get('block_size')}"
        ),
    }


def make_markdown_table(case_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| case | working | recent | anchor | retrieval | budget_ok | working_blocks | mean_abs_diff | max_abs_diff | working_ratio | fallback_reason |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]

    for case in case_summaries:
        budgets = case["budgets"]
        lines.append(
            f"| {case['case_name']} "
            f"| {budgets['working']} "
            f"| {budgets['recent']} "
            f"| {budgets['anchor']} "
            f"| {budgets['retrieval']} "
            f"| {str(case['budget_ok']).lower()} "
            f"| {case['selected']['working_block_ids']} "
            f"| {format_float(case['mean_abs_diff'])} "
            f"| {format_float(case['max_abs_diff'])} "
            f"| {format_float(case['working_ratio'])} "
            f"| {case['fallback_reason'] or ''} |"
        )

    return "\n".join(lines)


def make_markdown_diagnostics(case_summaries: list[dict[str, Any]]) -> str:
    lines = ["", "Diagnostics:"]
    for case in case_summaries:
        lines.append(
            f"- {case['case_name']}: "
            f"retrieval_exclude_tail_blocks={case['retrieval_exclude_tail_blocks']}, "
            f"retrieval_excluded_block_ids={case['retrieval_excluded_block_ids']}, "
            f"retrieved_overlap_with_excluded_tail={case['retrieved_overlap_with_excluded_tail']}, "
            f"retrieved_overlap_with_recent_blocks={case['retrieved_overlap_with_recent_blocks']}, "
            f"effective_retrieved_block_count={case['effective_retrieved_block_count']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight RelayKV budget policy comparison sweep."
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
        default=128,
        help="Target input sequence length",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Cold block size",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Layer index for scoring and attention comparison",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="repetitive",
        help="Prompt type: repetitive, prose, structured",
    )
    parser.add_argument(
        "--scoring-variant",
        type=str,
        default=DEFAULT_SCORING_VARIANT,
        help="Scoring variant",
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
    parser.add_argument(
        "--working-budget-blocks",
        type=int,
        default=None,
        help="Override the shared working budget for all built-in cases",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated subset of cases to run",
    )
    parser.add_argument(
        "--retrieval-exclude-tail-blocks",
        type=int,
        default=0,
        help="Exclude the last N full-sequence block ids from retrieval candidates",
    )
    parser.add_argument(
        "--activation-mode",
        type=str,
        choices=("diagnostic", "practical"),
        default="diagnostic",
        help="Whether to force RelayKV diagnostics or gate it for practical mode.",
    )
    parser.add_argument(
        "--min-relaykv-seq-len",
        type=int,
        default=None,
        help="Disable RelayKV in practical mode below this sequence length.",
    )
    parser.add_argument(
        "--disable-relaykv-below-budget",
        action="store_true",
        help=(
            "In practical mode, keep FullKV active when the sequence length "
            "already fits within the working budget."
        ),
    )
    args = parser.parse_args()

    ensure_results_dir()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    case_names = parse_cases(args.cases)
    case_summaries: list[dict[str, Any]] = []

    for case_name in case_names:
        case_budgets = normalize_case_budgets(
            case_name,
            DEFAULT_CASE_SPECS[case_name],
            args.working_budget_blocks,
        )

        payload = run_pipeline(
            model_name=args.model,
            seq_len_target=args.seq_len,
            hot_window=case_budgets["recent"] * args.block_size,
            block_size=args.block_size,
            top_k=max(1, case_budgets["retrieval"]),
            layer_idx=args.layer_idx,
            scoring_variant=args.scoring_variant,
            prompt_type=args.prompt_type,
            anchor_blocks=case_budgets["anchor"],
            working_budget_blocks=case_budgets["working"],
            recent_budget_blocks=case_budgets["recent"],
            anchor_budget_blocks=case_budgets["anchor"],
            retrieval_budget_blocks=case_budgets["retrieval"],
            retrieval_exclude_tail_blocks=args.retrieval_exclude_tail_blocks,
            activation_mode=args.activation_mode,
            min_relaykv_seq_len=args.min_relaykv_seq_len,
            disable_relaykv_below_budget=args.disable_relaykv_below_budget,
        )

        case_summaries.append(
            make_case_summary(
                case_name=case_name,
                case_budgets=case_budgets,
                payload=payload,
                retrieval_exclude_tail_blocks=args.retrieval_exclude_tail_blocks,
            )
        )

    summary = {
        "model": args.model,
        "seq_len": args.seq_len,
        "block_size": args.block_size,
        "layer_idx": args.layer_idx,
        "prompt_type": args.prompt_type,
        "scoring_variant": args.scoring_variant,
        "retrieval_exclude_tail_blocks": args.retrieval_exclude_tail_blocks,
        "activation_mode": args.activation_mode,
        "min_relaykv_seq_len": args.min_relaykv_seq_len,
        "disable_relaykv_below_budget": args.disable_relaykv_below_budget,
        "cases": case_summaries,
    }

    markdown = make_markdown_table(case_summaries)
    markdown += make_markdown_diagnostics(case_summaries)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    args.output_md.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"\nsaved json: {args.output_json}")
    print(f"saved markdown: {args.output_md}")


if __name__ == "__main__":
    main()
