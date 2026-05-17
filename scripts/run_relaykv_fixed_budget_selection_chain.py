#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayKVBlockCandidate,
    RelayKVFixedBudgetConfig,
    build_relaykv_fixed_budget_block_selection_decision,
    build_relaykv_fixed_budget_working_set_decision,
    export_pipeline_candidates_from_json_file,
)


def _build_block_candidates(
    candidates_payload: list[dict],
) -> list[RelayKVBlockCandidate]:
    return [
        RelayKVBlockCandidate(
            block_id=int(candidate["block_id"]),
            token_start=int(candidate["token_start"]),
            token_end=int(candidate["token_end"]),
            score=(
                float(candidate["score"])
                if candidate.get("score") is not None
                else None
            ),
            is_recent=bool(candidate.get("is_recent", False)),
            is_anchor=bool(candidate.get("is_anchor", False)),
            is_retrieval_candidate=bool(
                candidate.get("is_retrieval_candidate", True)
            ),
            layer_id=(
                int(candidate["layer_id"])
                if candidate.get("layer_id") is not None
                else None
            ),
            kv_head_group=(
                int(candidate["kv_head_group"])
                if candidate.get("kv_head_group") is not None
                else None
            ),
        )
        for candidate in candidates_payload
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 11-D no-model/no-GPU RelayKV fixed-budget selection chain: "
            "pipeline/scoring artifact -> candidates-json -> fixed-budget block selection "
            "-> chain_summary."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-working-budget-tokens", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--input-key",
        type=str,
        default="auto",
        choices=("auto", "top_scores", "block_scores", "candidates", "top_blocks"),
    )
    parser.add_argument("--recent-budget-tokens", type=int, default=None)
    parser.add_argument("--anchor-budget-tokens", type=int, default=None)
    parser.add_argument("--retrieved-budget-tokens", type=int, default=None)
    parser.add_argument("--transient-budget-tokens", type=int, default=0)
    parser.add_argument("--recent-ratio", type=float, default=0.50)
    parser.add_argument("--anchor-ratio", type=float, default=0.10)
    parser.add_argument("--retrieved-ratio", type=float, default=0.40)
    parser.add_argument("--min-recent-tokens", type=int, default=128)
    parser.add_argument("--min-anchor-tokens", type=int, default=0)
    parser.add_argument("--mark-recent-tail-blocks", type=int, default=0)
    parser.add_argument("--mark-anchor-head-blocks", type=int, default=0)
    parser.add_argument("--default-layer-id", type=int, default=None)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_artifact = output_dir / "relaykv_candidates.json"
    selection_artifact = output_dir / "relaykv_fixed_budget_block_selection.json"
    chain_summary_artifact = output_dir / "chain_summary.json"

    candidates_payload = export_pipeline_candidates_from_json_file(
        input_path,
        block_size=args.block_size,
        input_key=args.input_key,
        default_layer_id=args.default_layer_id,
        mark_recent_tail_blocks=args.mark_recent_tail_blocks,
        mark_anchor_head_blocks=args.mark_anchor_head_blocks,
    )
    candidate_artifact.write_text(
        json.dumps(candidates_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fixed_budget_decision = build_relaykv_fixed_budget_working_set_decision(
        config=RelayKVFixedBudgetConfig(
            total_working_budget_tokens=args.total_working_budget_tokens,
            recent_budget_tokens=args.recent_budget_tokens,
            anchor_budget_tokens=args.anchor_budget_tokens,
            retrieved_budget_tokens=args.retrieved_budget_tokens,
            transient_budget_tokens=args.transient_budget_tokens,
            recent_ratio=args.recent_ratio,
            anchor_ratio=args.anchor_ratio,
            retrieved_ratio=args.retrieved_ratio,
            min_recent_tokens=args.min_recent_tokens,
            min_anchor_tokens=args.min_anchor_tokens,
            block_size=args.block_size,
        )
    )
    selection_decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=fixed_budget_decision,
        candidates=_build_block_candidates(candidates_payload),
        block_size=args.block_size,
    )
    selection_summary = selection_decision.summary()
    selection_artifact.write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    chain_summary = {
        "phase": "11-D",
        "dry_run_only": True,
        "no_model_loading": True,
        "no_kv_materialization": True,
        "no_attention_connection": True,
        "no_runtime_adapter": True,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "candidate_artifact": str(candidate_artifact),
        "selection_artifact": str(selection_artifact),
        "candidate_count": len(candidates_payload),
        "selection_decision_state": selection_decision.decision_state,
        "selected_block_count_by_class": dict(
            selection_decision.selected_block_count_by_class
        ),
        "materialized_working_tokens": selection_decision.materialized_working_tokens,
        "estimated_working_tokens": selection_decision.estimated_working_tokens,
        "total_working_budget_tokens": (
            fixed_budget_decision.total_working_budget_tokens
        ),
        "rejected_block_count": len(selection_decision.rejected_block_ids),
        "overflow_block_count": len(selection_decision.overflow_block_ids),
        "notes": list(selection_decision.notes),
    }
    chain_summary_artifact.write_text(
        json.dumps(chain_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(chain_summary_artifact),
                "candidate_count": chain_summary["candidate_count"],
                "selection_decision_state": chain_summary[
                    "selection_decision_state"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
