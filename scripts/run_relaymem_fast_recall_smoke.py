#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMBackendKind,
    RelayMEMEpisodeKind,
    RelayMEMEpisodeRecord,
    RelayMEMProfileKind,
    RelayMEMProfileRecord,
    RelayMEMRecordStatus,
    RelayMEMRetrievalMode,
    RelayMEMStructuredRecord,
    RelayMEMSummaryKind,
    RelayMEMSummaryRecord,
    build_default_fast_recall_backend_capabilities,
    build_relaymem_context_assembly_plan,
    search_relaymem_fast_recall,
)

DEFAULT_OUTPUT = Path("results/processed/relaymem_fast_recall_smoke.json")
DEFAULT_QUERY = "RelayStack low VRAM Open-LLM-VTuber memory planning"
DEFAULT_TOKEN_BUDGET = 96


def build_synthetic_records() -> list[object]:
    return [
        RelayMEMProfileRecord(
            profile_id="profile:relay-project",
            profile_kind=RelayMEMProfileKind.PROJECT,
            display_name="RelayKV / RelayStack project",
            facts={
                "definition": "RelayStack combines RelayMEM, RelayKV, VRAM reservation, and fallback policy.",
                "relaykv": "RelayKV controls decode-time KV working sets under fixed VRAM.",
            },
            preferences={
                "workflow": "Keep early phases stdlib-only, smoke-testable, and no-GPU.",
            },
            relationships={},
            importance=0.95,
            source_refs=["doc://current_status"],
            updated_at="2026-05-15",
        ),
        RelayMEMEpisodeRecord(
            episode_id="episode:relaystack-doc-pr",
            episode_kind=RelayMEMEpisodeKind.PROJECT_EVENT,
            title="RelayStack fixed-budget design PR",
            episode_summary=(
                "Documentation reframed RelayKV as a fixed-VRAM-budget working-set "
                "controller and RelayStack as memory/context/KV-budget orchestration."
            ),
            participants=["user", "assistant"],
            key_events=["merged PR #49", "updated phase plan", "selected Fast Recall as Phase 6"],
            source_refs=["pr://49"],
            importance=0.9,
            ended_at="2026-05-15",
        ),
        RelayMEMSummaryRecord(
            summary_id="summary:open-llm-vtuber-target",
            summary_kind=RelayMEMSummaryKind.LONG_TERM_SUMMARY,
            text=(
                "Open-LLM-VTuber is the low-VRAM target where model weights, TTS, "
                "ASR, avatar, safety margin, and residual KV budget must be planned together."
            ),
            covered_source_refs=["doc://open_llm_vtuber_target"],
            token_estimate=30,
            importance=0.86,
            updated_at="2026-05-15",
        ),
        RelayMEMStructuredRecord(
            record_id="structured:phase-next",
            namespace="relaystack.phase",
            key="next_phase",
            value="Phase 6 is RelayMEM Fast Recall backend.",
            attributes={"after": "RelayStack design document consolidation"},
            source_refs=["doc://current_status#revised-phase-direction"],
            confidence=0.93,
            updated_at="2026-05-15",
        ),
        RelayMEMSummaryRecord(
            summary_id="summary:archived-old-plan",
            summary_kind=RelayMEMSummaryKind.SESSION_SUMMARY,
            text="Archived plan: immediately restart SGLang runtime adapter work.",
            covered_source_refs=["old://sglang-plan"],
            token_estimate=12,
            importance=0.2,
            status=RelayMEMRecordStatus.ARCHIVED,
        ),
    ]


def make_summary(results: list[Any], plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_result_count": len(results),
        "selected_item_count": len(plan["selected_items"]),
        "dropped_memory_count": len(plan["dropped_memory_ids"]),
        "top_memory_ids": [result.memory_id for result in results[:3]],
        "backend_kind": RelayMEMBackendKind.BM25.value,
        "external_backend_called": False,
    }


def run_relaymem_fast_recall_smoke(
    *,
    query: str = DEFAULT_QUERY,
    output: Path = DEFAULT_OUTPUT,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_results: int = 4,
) -> dict[str, Any]:
    records = build_synthetic_records()
    backend_capabilities = build_default_fast_recall_backend_capabilities()
    retrieval_results = search_relaymem_fast_recall(
        query=query,
        records=records,
        max_results=max_results,
    )
    plan_obj = build_relaymem_context_assembly_plan(
        query=query,
        retrieval_mode=RelayMEMRetrievalMode.FAST_RECALL,
        backend_kind=RelayMEMBackendKind.BM25,
        retrieval_results=retrieval_results,
        token_budget=token_budget,
        approval_required=False,
    )
    plan = plan_obj.summary()
    payload = {
        "metadata": {
            "script": "run_relaymem_fast_recall_smoke.py",
            "schema_version": 1,
            "synthetic": True,
            "stdlib_only": True,
            "external_backend_called": False,
            "backend_capabilities": backend_capabilities.summary(),
            "notes": (
                "Offline RelayMEM Fast Recall smoke path. Uses synthetic records, "
                "keyword/token overlap scoring, and existing RelayMEM context assembly."
            ),
        },
        "query": query,
        "retrieval_results": [result.summary() for result in retrieval_results],
        "plan": plan,
        "summary": make_summary(retrieval_results, plan),
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
    args = parser.parse_args()

    payload = run_relaymem_fast_recall_smoke(
        query=args.query,
        output=args.output,
        token_budget=args.token_budget,
        max_results=args.max_results,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "retrieval_result_count": payload["summary"]["retrieval_result_count"],
                "selected_item_count": payload["summary"]["selected_item_count"],
                "top_memory_ids": payload["summary"]["top_memory_ids"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
