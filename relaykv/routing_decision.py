from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):
    FULL_ATTENTION = "full_attention"
    RELAYKV_ROUTED = "relaykv_routed"
    SHADOW_COMPARE = "shadow_compare"


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
    estimated_working_kv_bytes: int
    estimated_ram_swap_bytes: int
    estimated_ssd_read_bytes: int
    estimated_materialization_latency_ms: float
    estimated_policy_compute_ms: float
    estimated_attention_tokens_saved: int
    estimated_net_benefit_ms: float
    fallback_reason: str | None
    apply_blocked_reason: str | None
    shadow_compare_passed: bool | None
    selection_stability_ratio: float | None

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
        }
