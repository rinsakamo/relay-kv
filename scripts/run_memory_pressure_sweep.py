#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv.memory_pressure import decide_memory_pressure_state
from relaykv.memory_pressure_summary import summarize_memory_pressure_decisions


DEFAULT_OUTPUT_JSON = Path(
    "results/raw/memory_pressure_sweep/memory_pressure_sweep.json"
)


def build_synthetic_cases(
    *,
    min_seq_len_for_relaykv: int,
    min_selection_stability_ratio: float,
    min_estimated_net_benefit_ms: float,
) -> list[dict[str, Any]]:
    active_seq_len = max(min_seq_len_for_relaykv, 1)
    cases: list[dict[str, Any]] = []

    if min_seq_len_for_relaykv > 0:
        cases.append(
            {
                "case_name": "short_context",
                "inputs": {
                    "seq_len": max(0, min_seq_len_for_relaykv - 1),
                },
                "expected_state": "disabled_short_context",
            }
        )

    cases.extend(
        [
            {
            "case_name": "fullkv_within_budget_by_bytes",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 4096,
                "residual_vram_budget_bytes": 8192,
            },
            "expected_state": "fullkv_within_budget",
        },
        {
            "case_name": "fullkv_within_budget_by_fallback_reason",
            "inputs": {
                "seq_len": active_seq_len,
                "fallback_reason": "fullkv_within_budget",
            },
            "expected_state": "fullkv_within_budget",
        },
        {
            "case_name": "labels_not_ready",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 16384,
                "residual_vram_budget_bytes": 8192,
                "labels_ready": False,
                "host_backup_available": True,
            },
            "expected_state": "shadow_only_warmup",
        },
        {
            "case_name": "host_backup_unavailable",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 16384,
                "residual_vram_budget_bytes": 8192,
                "labels_ready": True,
                "host_backup_available": False,
            },
            "expected_state": "shadow_only_warmup",
        },
        {
            "case_name": "shadow_compare_not_ready",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 16384,
                "residual_vram_budget_bytes": 8192,
                "labels_ready": True,
                "host_backup_available": True,
                "shadow_compare_passed": None,
            },
            "expected_state": "shadow_only_warmup",
        },
        {
            "case_name": "shadow_compare_failed",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 16384,
                "residual_vram_budget_bytes": 8192,
                "labels_ready": True,
                "host_backup_available": True,
                "shadow_compare_passed": False,
            },
            "expected_state": "fallback_required",
        },
        {
            "case_name": "net_benefit_too_low",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 16384,
                "residual_vram_budget_bytes": 8192,
                "labels_ready": True,
                "host_backup_available": True,
                "shadow_compare_passed": True,
                "selection_stability_ratio": min_selection_stability_ratio,
                "estimated_net_benefit_ms": min_estimated_net_benefit_ms - 0.1,
            },
            "expected_state": "shadow_only_warmup",
        },
        {
            "case_name": "routed_ready_under_pressure",
            "inputs": {
                "seq_len": active_seq_len,
                "projected_fullkv_bytes": 16384,
                "residual_vram_budget_bytes": 8192,
                "labels_ready": True,
                "host_backup_available": True,
                "shadow_compare_passed": True,
                "selection_stability_ratio": min_selection_stability_ratio,
                "estimated_net_benefit_ms": max(
                    min_estimated_net_benefit_ms,
                    min_estimated_net_benefit_ms + 1.0,
                ),
            },
            "expected_state": "relaykv_routed_ready",
        },
        {
            "case_name": "explicit_non_fullkv_fallback",
            "inputs": {
                "seq_len": active_seq_len,
                "fallback_reason": "insufficient_eviction_candidates",
            },
            "expected_state": "fallback_required",
        },
        ]
    )

    if min_selection_stability_ratio > 0.0:
        unstable_ratio = max(0.0, min_selection_stability_ratio - 0.1)
        cases.insert(
            len(cases) - 3,
            {
                "case_name": "selection_unstable",
                "inputs": {
                    "seq_len": active_seq_len,
                    "projected_fullkv_bytes": 16384,
                    "residual_vram_budget_bytes": 8192,
                    "labels_ready": True,
                    "host_backup_available": True,
                    "shadow_compare_passed": True,
                    "selection_stability_ratio": unstable_ratio,
                },
                "expected_state": "shadow_only_warmup",
            },
        )

    return cases


def run_memory_pressure_sweep(
    *,
    output: Path = DEFAULT_OUTPUT_JSON,
    min_seq_len_for_relaykv: int = 2048,
    min_selection_stability_ratio: float = 0.8,
    min_estimated_net_benefit_ms: float = 0.0,
) -> dict[str, Any]:
    cases = build_synthetic_cases(
        min_seq_len_for_relaykv=min_seq_len_for_relaykv,
        min_selection_stability_ratio=min_selection_stability_ratio,
        min_estimated_net_benefit_ms=min_estimated_net_benefit_ms,
    )

    decisions: list[dict[str, Any]] = []
    decision_summaries: list[dict[str, Any]] = []
    for case in cases:
        inputs = dict(case["inputs"])
        decision = decide_memory_pressure_state(
            min_seq_len_for_relaykv=min_seq_len_for_relaykv,
            min_selection_stability_ratio=min_selection_stability_ratio,
            min_estimated_net_benefit_ms=min_estimated_net_benefit_ms,
            **inputs,
        )
        decision_summary = decision.summary()
        decision_summaries.append(decision_summary)
        decisions.append(
            {
                "case_name": case["case_name"],
                "decision": decision_summary,
            }
        )

    payload = {
        "metadata": {
            "script": "run_memory_pressure_sweep.py",
            "schema_version": 1,
            "min_seq_len_for_relaykv": min_seq_len_for_relaykv,
            "min_selection_stability_ratio": min_selection_stability_ratio,
            "min_estimated_net_benefit_ms": min_estimated_net_benefit_ms,
        },
        "cases": cases,
        "decisions": decisions,
        "summary": summarize_memory_pressure_decisions(decision_summaries),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic RelayKV memory-pressure sweep."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-seq-len-for-relaykv",
        type=int,
        default=2048,
        help="Minimum sequence length before RelayKV is considered",
    )
    parser.add_argument(
        "--min-selection-stability-ratio",
        type=float,
        default=0.8,
        help="Minimum selection stability ratio for routed readiness",
    )
    parser.add_argument(
        "--min-estimated-net-benefit-ms",
        type=float,
        default=0.0,
        help="Minimum estimated net benefit for routed readiness",
    )
    args = parser.parse_args()

    payload = run_memory_pressure_sweep(
        output=args.output,
        min_seq_len_for_relaykv=args.min_seq_len_for_relaykv,
        min_selection_stability_ratio=args.min_selection_stability_ratio,
        min_estimated_net_benefit_ms=args.min_estimated_net_benefit_ms,
    )
    print(f"saved json: {args.output}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
