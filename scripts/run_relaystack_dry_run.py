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
    RelayKVVramReservation,
    RelayKVVramReservationStatus,
    build_relaymem_context_assembly_plan,
    build_default_fast_recall_backend_capabilities,
    build_relaymem_prompt_preview_plan,
    build_vram_reservation_budget_decision,
    decide_memory_pressure_state,
    decide_relaystack_final_routing,
    summarize_memory_pressure_decisions,
)


MIB_BYTES = 1024 * 1024


def build_synthetic_retrieval_results() -> list[RelayMEMRetrievalResult]:
    return [
        RelayMEMRetrievalResult(
            memory_id="profile:character-core",
            memory_source=RelayMEMMemorySource.PROFILE,
            text=(
                "Character profile: calm speaking style, polite Japanese, "
                "and preference for concise responses during live interaction."
            ),
            score=0.91,
            evidence=[],
            source_refs=["profile://character/core"],
            estimated_tokens=28,
            retrieval_backend=RelayMEMBackendKind.VECTOR,
            confidence=0.96,
        ),
        RelayMEMRetrievalResult(
            memory_id="episode:viewer-promise",
            memory_source=RelayMEMMemorySource.EPISODE,
            text=(
                "Prior episode: the character promised to revisit the user's "
                "festival story in a later stream."
            ),
            score=6.5,
            evidence=["episode note", "stream transcript fragment"],
            source_refs=["episode://2026-05-12/festival-promise"],
            estimated_tokens=26,
            retrieval_backend=RelayMEMBackendKind.BM25,
            confidence=None,
        ),
        RelayMEMRetrievalResult(
            memory_id="summary:recent-arc",
            memory_source=RelayMEMMemorySource.SUMMARY,
            text=(
                "Recent summary: the last sessions focused on RelayMEM, "
                "RelayKV, low-VRAM tradeoffs, and long-term continuity."
            ),
            score=0.84,
            evidence=[],
            source_refs=["summary://recent/arc"],
            estimated_tokens=25,
            retrieval_backend=RelayMEMBackendKind.HYBRID,
            confidence=0.89,
        ),
        RelayMEMRetrievalResult(
            memory_id="rag:open-llm-vtuber-budget",
            memory_source=RelayMEMMemorySource.RAG_CHUNK,
            text=(
                "Retrieved note: VRAM should be budgeted after model weights, "
                "TTS, ASR, avatar rendering, and a safety margin."
            ),
            score=-0.2,
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
                "Deep recall candidate: chained evidence links an older user "
                "request, a rolling summary, and a design note about memory policy."
            ),
            score=3.4,
            evidence=[
                "older user asked to preserve continuity",
                "rolling summary captured the recurring topic",
                "design note linked it to live fallback policy",
            ],
            source_refs=[
                "episode://older/request",
                "summary://rolling/continuity",
                "memo://fallback/policy-link",
            ],
            estimated_tokens=34,
            retrieval_backend=RelayMEMBackendKind.EVIDENCE_CHAIN,
            confidence=0.86,
        ),
    ]


def run_relaystack_dry_run(
    *,
    output: Path,
    token_budget: int = 180,
    total_vram_mib: int = 12288,
    model_weights_reserved_mib: int = 6144,
    tts_reserved_mib: int = 1536,
    asr_reserved_mib: int = 1024,
    avatar_reserved_mib: int = 512,
    safety_margin_mib: int = 1024,
    other_reserved_mib: int = 0,
    min_working_kv_budget_mib: int = 512,
    disable_approval_gate: bool = False,
) -> dict[str, Any]:
    retrieval_results = build_synthetic_retrieval_results()
    backend_capabilities = build_default_fast_recall_backend_capabilities()

    approval_required = not disable_approval_gate
    approval_reason = (
        "deep recall may add latency in live_low_latency mode"
        if approval_required
        else None
    )
    context_user_visible_message = (
        "Deep Recall may add latency in live-low-latency mode. Apply these deeper memories to active context?"
        if approval_required
        else None
    )
    proposed_retrieval_mode = (
        RelayMEMRetrievalMode.DEEP_RECALL
        if approval_required
        else None
    )
    retrieval_fallback_if_denied = (
        RelayMEMRetrievalMode.FAST_RECALL
        if approval_required
        else None
    )

    context_assembly_plan = build_relaymem_context_assembly_plan(
        query="Plan a live low-latency AI Vtuber response with optional deeper memory recall.",
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        backend_kind=RelayMEMBackendKind.EVIDENCE_CHAIN,
        retrieval_results=retrieval_results,
        token_budget=token_budget,
        approval_required=approval_required,
        approval_reason=approval_reason,
        user_visible_message=context_user_visible_message,
        proposed_retrieval_mode=proposed_retrieval_mode,
        fallback_if_denied=retrieval_fallback_if_denied,
    )
    prompt_preview_plan = build_relaymem_prompt_preview_plan(
        query="Plan a live low-latency AI Vtuber response with optional deeper memory recall.",
        retrieval_results=retrieval_results,
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        backend_kind=RelayMEMBackendKind.EVIDENCE_CHAIN,
        token_budget=token_budget,
        approval_required=approval_required,
        approval_reason=approval_reason,
        fallback_if_denied=retrieval_fallback_if_denied,
    )

    reservation = RelayKVVramReservation(
        total_vram_mib=total_vram_mib,
        model_weights_reserved_mib=model_weights_reserved_mib,
        tts_reserved_mib=tts_reserved_mib,
        asr_reserved_mib=asr_reserved_mib,
        avatar_reserved_mib=avatar_reserved_mib,
        safety_margin_mib=safety_margin_mib,
        other_reserved_mib=other_reserved_mib,
    )
    vram_reservation_decision = build_vram_reservation_budget_decision(
        reservation,
        min_working_kv_budget_mib=min_working_kv_budget_mib,
    )

    relaykv_routing_allowed = (
        vram_reservation_decision.status == RelayKVVramReservationStatus.OK
    )
    if not relaykv_routing_allowed:
        memory_pressure_decision = None
        memory_pressure_summary = None
        memory_pressure_note = "skipped: vram reservation status is not ok"
    else:
        projected_fullkv_bytes = (reservation.total_vram_mib * MIB_BYTES) // 2
        residual_vram_budget_bytes = (
            vram_reservation_decision.available_working_kv_budget_mib * MIB_BYTES
        )
        try:
            memory_pressure_decision_obj = decide_memory_pressure_state(
                seq_len=8192,
                min_seq_len_for_relaykv=2048,
                projected_fullkv_bytes=projected_fullkv_bytes,
                residual_vram_budget_bytes=residual_vram_budget_bytes,
                labels_ready=True,
                host_backup_available=True,
                shadow_compare_passed=True,
                selection_stability_ratio=0.92,
                estimated_net_benefit_ms=3.5,
                min_estimated_net_benefit_ms=0.0,
            )
            memory_pressure_decision = memory_pressure_decision_obj.summary()
            memory_pressure_summary = summarize_memory_pressure_decisions(
                [memory_pressure_decision_obj]
            )
        except Exception:
            memory_pressure_decision = None
            memory_pressure_summary = None
            memory_pressure_note = "skipped: no lightweight safe helper selected"
        else:
            memory_pressure_note = "included: lightweight memory pressure helper"

    final_routing_decision = decide_relaystack_final_routing(
        prompt_preview_plan=prompt_preview_plan,
        vram_reservation_decision=vram_reservation_decision,
        memory_pressure_decision=memory_pressure_decision,
        relaykv_routing_allowed=relaykv_routing_allowed,
        approval_gate_enabled=not disable_approval_gate,
    )

    user_gated_fallback = {
        "approval_required": prompt_preview_plan.approval_required,
        "approval_reason": prompt_preview_plan.approval_reason,
        "proposed_retrieval_mode": (
            proposed_retrieval_mode.value
            if proposed_retrieval_mode is not None
            else None
        ),
        "fallback_if_denied": (
            prompt_preview_plan.fallback_if_denied.value
            if prompt_preview_plan.fallback_if_denied is not None
            else None
        ),
        "fallback_reason": prompt_preview_plan.fallback_reason,
        "user_visible_message": prompt_preview_plan.user_visible_message,
        "can_apply_without_user_approval": (
            prompt_preview_plan.can_apply_without_user_approval
        ),
    }

    payload = {
        "metadata": {
            "script_name": "run_relaystack_dry_run.py",
            "schema_version": "relaystack_dry_run_v1",
            "scenario": "local_ai_vtuber_12gb",
            "model_family": "llm-jp-4-8b-4bit-placeholder",
            "no_model_loaded": True,
            "no_gpu_inspection": True,
        },
        "runtime_policy": {
            "mode": "live_low_latency",
            "latency_sensitive": True,
            "approval_gate_enabled": not disable_approval_gate,
        },
        "relaymem": {
            "fast_recall_fallback_backend_capabilities": (
                backend_capabilities.summary()
            ),
            "retrieval_results": [item.summary() for item in retrieval_results],
            "context_assembly_plan": context_assembly_plan.summary(),
            "prompt_preview_plan": prompt_preview_plan.summary(),
        },
        "relaykv": {
            "vram_reservation": reservation.summary(),
            "vram_reservation_decision": vram_reservation_decision.summary(),
            "relaykv_routing_allowed": relaykv_routing_allowed,
            "memory_pressure_decision": memory_pressure_decision,
            "memory_pressure_summary": memory_pressure_summary,
            "memory_pressure_note": memory_pressure_note,
        },
        "relaystack": {
            "final_routing_decision": final_routing_decision.summary(),
        },
        "user_gated_fallback": user_gated_fallback,
        "summary": {
            "retrieval_result_count": len(retrieval_results),
            "selected_item_count": len(context_assembly_plan.selected_items),
            "dropped_memory_count": len(context_assembly_plan.dropped_memory_ids),
            "preview_item_count": len(prompt_preview_plan.preview_items),
            "prompt_preview_dropped_memory_count": len(
                prompt_preview_plan.dropped_memory_ids
            ),
            "prompt_preview_approval_required": prompt_preview_plan.approval_required,
            "prompt_preview_can_apply_without_user_approval": (
                prompt_preview_plan.can_apply_without_user_approval
            ),
            "prompt_preview_fallback_reason": prompt_preview_plan.fallback_reason,
            "available_working_kv_budget_mib": (
                vram_reservation_decision.available_working_kv_budget_mib
            ),
            "relaykv_routing_allowed": relaykv_routing_allowed,
            "memory_pressure_included": memory_pressure_decision is not None,
            "final_routing_state": final_routing_decision.state.value,
            "relaymem_apply_allowed": final_routing_decision.relaymem_apply_allowed,
            "final_relaykv_routing_allowed": (
                final_routing_decision.relaykv_routing_allowed
            ),
            "final_blocking_reasons": list(final_routing_decision.blocking_reasons),
            "final_fallback_reason": final_routing_decision.fallback_reason,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=180)
    parser.add_argument("--total-vram-mib", type=int, default=12288)
    parser.add_argument("--model-weights-reserved-mib", type=int, default=6144)
    parser.add_argument("--tts-reserved-mib", type=int, default=1536)
    parser.add_argument("--asr-reserved-mib", type=int, default=1024)
    parser.add_argument("--avatar-reserved-mib", type=int, default=512)
    parser.add_argument("--safety-margin-mib", type=int, default=1024)
    parser.add_argument("--other-reserved-mib", type=int, default=0)
    parser.add_argument("--min-working-kv-budget-mib", type=int, default=512)
    parser.add_argument("--disable-approval-gate", action="store_true")
    args = parser.parse_args()

    payload = run_relaystack_dry_run(
        output=args.output,
        token_budget=args.token_budget,
        total_vram_mib=args.total_vram_mib,
        model_weights_reserved_mib=args.model_weights_reserved_mib,
        tts_reserved_mib=args.tts_reserved_mib,
        asr_reserved_mib=args.asr_reserved_mib,
        avatar_reserved_mib=args.avatar_reserved_mib,
        safety_margin_mib=args.safety_margin_mib,
        other_reserved_mib=args.other_reserved_mib,
        min_working_kv_budget_mib=args.min_working_kv_budget_mib,
        disable_approval_gate=args.disable_approval_gate,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "approval_required": payload["user_gated_fallback"]["approval_required"],
                "preview_item_count": payload["summary"]["preview_item_count"],
                "can_apply_without_user_approval": payload["summary"][
                    "prompt_preview_can_apply_without_user_approval"
                ],
                "prompt_preview_fallback_reason": payload["summary"][
                    "prompt_preview_fallback_reason"
                ],
                "available_working_kv_budget_mib": payload["summary"][
                    "available_working_kv_budget_mib"
                ],
                "final_routing_state": payload["summary"]["final_routing_state"],
                "relaymem_apply_allowed": payload["summary"][
                    "relaymem_apply_allowed"
                ],
                "final_relaykv_routing_allowed": payload["summary"][
                    "final_relaykv_routing_allowed"
                ],
                "final_blocking_reasons": payload["summary"][
                    "final_blocking_reasons"
                ],
                "final_fallback_reason": payload["summary"][
                    "final_fallback_reason"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
