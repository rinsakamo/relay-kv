from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):
    FULL_ATTENTION = "full_attention"
    FULLKV_GPU = "fullkv_gpu"
    RELAYKV_ROUTED = "relaykv_routed"
    SHADOW_COMPARE = "shadow_compare"
    SHADOW_ONLY = "shadow_only"
    PROPOSE_FALLBACK = "propose_fallback"
    WAIT_USER_APPROVAL = "wait_user_approval"
    APPLY_RAM_BACKED = "apply_ram_backed"
    FALLBACK_FULLKV_RAM = "fallback_fullkv_ram"
    FALLBACK_FULLKV_TIERED = "fallback_fullkv_tiered"
    FALLBACK_RECENT_ANCHOR = "fallback_recent_anchor"


@dataclass(frozen=True)
class RelayKVDecision:
    execution_mode: ExecutionMode
    selected_active_block_ids: list[int]
    protected_block_ids: list[int]
    demotion_candidate_block_ids: list[int]
    demoted_block_ids: list[int]
    retrieved_block_ids: list[int]
    prefetched_block_ids: list[int]
    reused_block_ids: list[int]
    newly_retrieved_block_ids: list[int]
    estimated_working_kv_bytes: int | None
    estimated_ram_swap_bytes: int | None
    estimated_ssd_read_bytes: int | None
    estimated_materialization_latency_ms: float | None
    estimated_policy_compute_ms: float | None
    estimated_attention_tokens_saved: int | None
    estimated_net_benefit_ms: float | None
    fallback_reason: str | None
    apply_blocked_reason: str | None
    shadow_compare_passed: bool | None
    selection_stability_ratio: float | None
    approval_required: bool = False
    approval_reason: str | None = None
    proposed_fallback_mode: ExecutionMode | None = None
    user_visible_message: str | None = None
    fallback_if_denied: ExecutionMode | None = None

    def summary(self) -> dict:
        return {
            "execution_mode": self.execution_mode.value,
            "selected_active_block_ids": list(self.selected_active_block_ids),
            "protected_block_ids": list(self.protected_block_ids),
            "demotion_candidate_block_ids": list(self.demotion_candidate_block_ids),
            "demoted_block_ids": list(self.demoted_block_ids),
            "retrieved_block_ids": list(self.retrieved_block_ids),
            "prefetched_block_ids": list(self.prefetched_block_ids),
            "reused_block_ids": list(self.reused_block_ids),
            "newly_retrieved_block_ids": list(self.newly_retrieved_block_ids),
            "estimated_working_kv_bytes": self.estimated_working_kv_bytes,
            "estimated_ram_swap_bytes": self.estimated_ram_swap_bytes,
            "estimated_ssd_read_bytes": self.estimated_ssd_read_bytes,
            "estimated_materialization_latency_ms": self.estimated_materialization_latency_ms,
            "estimated_policy_compute_ms": self.estimated_policy_compute_ms,
            "estimated_attention_tokens_saved": self.estimated_attention_tokens_saved,
            "estimated_net_benefit_ms": self.estimated_net_benefit_ms,
            "fallback_reason": self.fallback_reason,
            "apply_blocked_reason": self.apply_blocked_reason,
            "shadow_compare_passed": self.shadow_compare_passed,
            "selection_stability_ratio": self.selection_stability_ratio,
            "approval_required": self.approval_required,
            "approval_reason": self.approval_reason,
            "proposed_fallback_mode": (
                self.proposed_fallback_mode.value
                if self.proposed_fallback_mode is not None
                else None
            ),
            "user_visible_message": self.user_visible_message,
            "fallback_if_denied": (
                self.fallback_if_denied.value
                if self.fallback_if_denied is not None
                else None
            ),
        }
