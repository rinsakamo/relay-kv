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
    RelayMEMMemorySource,
    RelayMEMRetrievalMode,
    RelayMEMRetrievalResult,
    build_relaymem_context_assembly_plan,
)


DEFAULT_OUTPUT = Path("results/processed/relaymem_context_assembly_smoke.json")
DEFAULT_TOKEN_BUDGET = 140


def build_synthetic_retrieval_results() -> list[RelayMEMRetrievalResult]:
    return [
        RelayMEMRetrievalResult(
            memory_id="profile:character-core",
            memory_source=RelayMEMMemorySource.PROFILE,
            text=(
                "Character profile: speaks politely, prefers concise answers, "
                "and keeps a calm on-stream tone."
            ),
            score=0.92,
            evidence=[],
            source_refs=["profile://character/core"],
            estimated_tokens=24,
            retrieval_backend=RelayMEMBackendKind.VECTOR,
            confidence=0.95,
        ),
        RelayMEMRetrievalResult(
            memory_id="episode:festival-promise",
            memory_source=RelayMEMMemorySource.EPISODE,
            text=(
                "Prior episode: the character promised to revisit the summer "
                "festival story with the user next stream."
            ),
            score=7.4,
            evidence=["stream transcript fragment", "episode memory note"],
            source_refs=["episode://2026-05-10/festival-promise"],
            estimated_tokens=28,
            retrieval_backend=RelayMEMBackendKind.BM25,
            confidence=None,
        ),
        RelayMEMRetrievalResult(
            memory_id="summary:recent-arc",
            memory_source=RelayMEMMemorySource.SUMMARY,
            text=(
                "Recent summary: the last three sessions focused on character "
                "continuity, older memory recall, and low-VRAM runtime tradeoffs."
            ),
            score=0.81,
            evidence=[],
            source_refs=["summary://recent/arc"],
            estimated_tokens=26,
            retrieval_backend=RelayMEMBackendKind.HYBRID,
            confidence=0.88,
        ),
        RelayMEMRetrievalResult(
            memory_id="rag:open-llm-vtuber-budget",
            memory_source=RelayMEMMemorySource.RAG_CHUNK,
            text=(
                "Retrieved note: the Open-LLM-VTuber demo target budgets VRAM "
                "after model weights, TTS, ASR, avatar, and safety margin."
            ),
            score=-0.35,
            evidence=["docs/open_llm_vtuber_target.md"],
            source_refs=["doc://open-llm-vtuber-target"],
            estimated_tokens=23,
            retrieval_backend=RelayMEMBackendKind.VECTOR,
            confidence=None,
        ),
        RelayMEMRetrievalResult(
            memory_id="rag:deep-recall-evidence-chain",
            memory_source=RelayMEMMemorySource.RAG_CHUNK,
            text=(
                "Deep recall candidate: a chained evidence path links the user's "
                "earlier request, a summary note, and a supporting design memo."
            ),
            score=3.6,
            evidence=[
                "user asked to preserve profile memory naming",
                "summary memory captured the naming shift",
                "design memo linked the change to AI Vtuber use cases",
            ],
            source_refs=[
                "note://devlog/relaystack-target",
                "summary://naming/profile-memory",
                "memo://ai-vtuber/use-case-link",
            ],
            estimated_tokens=34,
            retrieval_backend=RelayMEMBackendKind.EVIDENCE_CHAIN,
            confidence=0.84,
        ),
    ]


def make_summary(
    retrieval_results: list[RelayMEMRetrievalResult],
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    for result in retrieval_results:
        source_key = result.memory_source.value
        backend_key = result.retrieval_backend.value
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        backend_counts[backend_key] = backend_counts.get(backend_key, 0) + 1

    return {
        "retrieval_result_count": len(retrieval_results),
        "selected_item_count": len(plan["selected_items"]),
        "dropped_memory_count": len(plan["dropped_memory_ids"]),
        "source_counts": source_counts,
        "backend_counts": backend_counts,
        "evidence_chain_present": any(
            result.retrieval_backend is RelayMEMBackendKind.EVIDENCE_CHAIN
            for result in retrieval_results
        ),
    }


def run_relaymem_context_assembly_smoke(
    *,
    output: Path = DEFAULT_OUTPUT,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, Any]:
    retrieval_results = build_synthetic_retrieval_results()
    plan_obj = build_relaymem_context_assembly_plan(
        query="Assemble active context for a low-VRAM AI character follow-up reply.",
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        backend_kind=RelayMEMBackendKind.EVIDENCE_CHAIN,
        retrieval_results=retrieval_results,
        token_budget=token_budget,
        approval_required=True,
        approval_reason="deep recall may add latency",
        user_visible_message="May I search older memory for supporting details?",
    )
    plan = plan_obj.summary()
    payload = {
        "metadata": {
            "script": "run_relaymem_context_assembly_smoke.py",
            "schema_version": 1,
            "synthetic": True,
            "external_backend_called": False,
            "notes": (
                "Offline RelayMEM smoke path using synthetic retrieval results "
                "only. No model loading and no concrete RAG backend calls."
            ),
        },
        "retrieval_results": [result.summary() for result in retrieval_results],
        "plan": plan,
        "summary": make_summary(retrieval_results, plan),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the smoke JSON payload.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Token budget for RelayMEM context assembly.",
    )
    args = parser.parse_args()

    payload = run_relaymem_context_assembly_smoke(
        output=args.output,
        token_budget=args.token_budget,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "selected_item_count": payload["summary"]["selected_item_count"],
                "dropped_memory_count": payload["summary"]["dropped_memory_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
