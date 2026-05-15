#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMEpisodeKind,
    RelayMEMEpisodeRecord,
    RelayMEMProfileKind,
    RelayMEMProfileRecord,
    RelayMEMRetrievalMode,
    RelayMEMStructuredRecord,
    RelayMEMSummaryKind,
    RelayMEMSummaryRecord,
    build_relaymem_prompt_preview_plan,
    search_relaymem_fast_recall,
)


DEFAULT_OUTPUT = Path("results/processed/relaymem_prompt_preview_smoke.json")
DEFAULT_QUERY = "RelayStack low VRAM Fast Recall memory planning"
DEFAULT_TOKEN_BUDGET = 160


def build_synthetic_records() -> list[object]:
    return [
        RelayMEMProfileRecord(
            profile_id="profile:relay-project",
            profile_kind=RelayMEMProfileKind.PROJECT,
            display_name="RelayKV / RelayStack project",
            facts={
                "definition": "RelayStack coordinates RelayMEM, RelayKV, VRAM reservation, and fallback policy.",
                "relaymem": "Fast Recall should stay stdlib-only and feed a prompt preview before deeper paths.",
            },
            preferences={
                "workflow": "Keep planning artifacts torch-free, no-GPU, and smoke-testable first.",
            },
            relationships={},
            importance=0.95,
            source_refs=["doc://current_status"],
            updated_at="2026-05-15",
        ),
        RelayMEMEpisodeRecord(
            episode_id="episode:phase-6-review",
            episode_kind=RelayMEMEpisodeKind.PROJECT_EVENT,
            title="Phase 6 Fast Recall review",
            episode_summary=(
                "The review accepted Fast Recall as a lightweight memory recall step and "
                "requested a prompt preview before any deeper fallback planning."
            ),
            participants=["user", "assistant"],
            key_events=["merged PR #50", "queued Phase 6.5 planning", "kept runtime paths untouched"],
            source_refs=["pr://50"],
            importance=0.88,
            ended_at="2026-05-15",
        ),
        RelayMEMSummaryRecord(
            summary_id="summary:vtuber-planning",
            summary_kind=RelayMEMSummaryKind.LONG_TERM_SUMMARY,
            text=(
                "Open-LLM-VTuber planning keeps RelayMEM as a hierarchical memory layer where "
                "Fast Recall previews safe context before deeper recall, runtime KV work, or model loading."
            ),
            covered_source_refs=["doc://vtuber-plan"],
            token_estimate=28,
            importance=0.9,
            updated_at="2026-05-15",
        ),
        RelayMEMStructuredRecord(
            record_id="structured:phase-6-5",
            namespace="relay.phase",
            key="current",
            value="Phase 6.5 implements RelayMEM prompt preview and user-gated fallback planning.",
            attributes={"kind": "phase_plan"},
            source_refs=["doc://current_status"],
            confidence=0.96,
            updated_at="2026-05-15",
        ),
    ]


def make_summary(
    retrieval_results: list[Any],
    preview_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "retrieval_result_count": len(retrieval_results),
        "preview_item_count": len(preview_plan["preview_items"]),
        "dropped_memory_count": len(preview_plan["dropped_memory_ids"]),
        "approval_required": preview_plan["approval_required"],
        "can_apply_without_user_approval": preview_plan[
            "can_apply_without_user_approval"
        ],
    }


def run_relaymem_prompt_preview_smoke(
    *,
    query: str = DEFAULT_QUERY,
    output: Path = DEFAULT_OUTPUT,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_results: int = 4,
    approval_required: bool = True,
) -> dict[str, Any]:
    records = build_synthetic_records()
    retrieval_results = search_relaymem_fast_recall(
        query=query,
        records=records,
        max_results=max_results,
    )
    approval_reason = (
        "prompt preview should be confirmed before insertion"
        if approval_required
        else None
    )
    fallback_if_denied = (
        RelayMEMRetrievalMode.FAST_RECALL
        if approval_required
        else None
    )
    preview_plan_obj = build_relaymem_prompt_preview_plan(
        query=query,
        retrieval_results=retrieval_results,
        retrieval_mode=RelayMEMRetrievalMode.FAST_RECALL,
        token_budget=token_budget,
        approval_required=approval_required,
        approval_reason=approval_reason,
        fallback_if_denied=fallback_if_denied,
        max_preview_chars=120,
    )
    preview_plan = preview_plan_obj.summary()
    payload = {
        "metadata": {
            "script": "run_relaymem_prompt_preview_smoke.py",
            "schema_version": 1,
            "synthetic": True,
            "stdlib_only": True,
            "external_backend_called": False,
            "notes": (
                "Offline RelayMEM prompt preview smoke path. Uses Fast Recall retrieval "
                "results plus preview planning only. No model, GPU, runtime, or KV apply path is called."
            ),
        },
        "query": query,
        "retrieval_results": [result.summary() for result in retrieval_results],
        "preview_plan": preview_plan,
        "summary": make_summary(retrieval_results, preview_plan),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--max-results", type=int, default=4)
    parser.add_argument(
        "--approval-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require approval before applying preview items.",
    )
    args = parser.parse_args()

    payload = run_relaymem_prompt_preview_smoke(
        query=args.query,
        output=args.output,
        token_budget=args.token_budget,
        max_results=args.max_results,
        approval_required=args.approval_required,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "preview_item_count": payload["summary"]["preview_item_count"],
                "dropped_memory_count": payload["summary"]["dropped_memory_count"],
                "approval_required": payload["summary"]["approval_required"],
                "can_apply_without_user_approval": payload["summary"][
                    "can_apply_without_user_approval"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
