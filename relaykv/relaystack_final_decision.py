from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .memory_pressure import RelayKVMemoryPressureDecision, RelayKVMemoryPressureState
from .relaymem import RelayMEMRetrievalMode
from .relaymem_prompt_preview import RelayMEMPromptPreviewPlan
from .vram_reservation import RelayKVVramBudgetDecision, RelayKVVramReservationStatus


class RelayStackFinalRoutingState(str, Enum):
    WAITING_FOR_USER_APPROVAL = "waiting_for_user_approval"
    FULL_CONTEXT_RECENT_ONLY = "full_context_recent_only"
    RELAYMEM_ONLY = "relaymem_only"
    RELAYKV_SHADOW_ONLY = "relaykv_shadow_only"
    RELAYMEM_AND_RELAYKV_READY = "relaymem_and_relaykv_ready"
    BLOCKED_NO_KV_BUDGET = "blocked_no_kv_budget"
    FALLBACK_SAFE_BASELINE = "fallback_safe_baseline"


def _memory_pressure_state_value(
    memory_pressure_decision: RelayKVMemoryPressureDecision | dict[str, Any] | None,
) -> str | None:
    if memory_pressure_decision is None:
        return None
    if isinstance(memory_pressure_decision, RelayKVMemoryPressureDecision):
        return memory_pressure_decision.state.value
    state = memory_pressure_decision.get("state")
    if isinstance(state, RelayKVMemoryPressureState):
        return state.value
    if state is None:
        return None
    return str(state)


@dataclass(frozen=True)
class RelayStackFinalRoutingDecision:
    state: RelayStackFinalRoutingState
    relaymem_apply_allowed: bool
    relaykv_routing_allowed: bool
    approval_required: bool
    can_apply_without_user_approval: bool
    fallback_reason: str | None
    blocking_reasons: list[str]
    selected_retrieval_mode: RelayMEMRetrievalMode | None
    proposed_retrieval_mode: RelayMEMRetrievalMode | None
    fallback_if_denied: RelayMEMRetrievalMode | None
    vram_status: str
    memory_pressure_state: str | None
    user_visible_message: str | None

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "relaymem_apply_allowed": self.relaymem_apply_allowed,
            "relaykv_routing_allowed": self.relaykv_routing_allowed,
            "approval_required": self.approval_required,
            "can_apply_without_user_approval": self.can_apply_without_user_approval,
            "fallback_reason": self.fallback_reason,
            "blocking_reasons": list(self.blocking_reasons),
            "selected_retrieval_mode": (
                self.selected_retrieval_mode.value
                if self.selected_retrieval_mode is not None
                else None
            ),
            "proposed_retrieval_mode": (
                self.proposed_retrieval_mode.value
                if self.proposed_retrieval_mode is not None
                else None
            ),
            "fallback_if_denied": (
                self.fallback_if_denied.value
                if self.fallback_if_denied is not None
                else None
            ),
            "vram_status": self.vram_status,
            "memory_pressure_state": self.memory_pressure_state,
            "user_visible_message": self.user_visible_message,
        }


def decide_relaystack_final_routing(
    *,
    prompt_preview_plan: RelayMEMPromptPreviewPlan,
    vram_reservation_decision: RelayKVVramBudgetDecision,
    memory_pressure_decision: RelayKVMemoryPressureDecision | dict[str, Any] | None,
    relaykv_routing_allowed: bool,
    approval_gate_enabled: bool,
) -> RelayStackFinalRoutingDecision:
    vram_status = vram_reservation_decision.status.value
    memory_pressure_state = _memory_pressure_state_value(memory_pressure_decision)
    proposed_retrieval_mode = (
        prompt_preview_plan.retrieval_mode
        if prompt_preview_plan.approval_required
        else None
    )
    fallback_reason = prompt_preview_plan.fallback_reason
    blocking_reasons: list[str] = []

    if vram_reservation_decision.status is not RelayKVVramReservationStatus.OK:
        blocking_reasons.append(f"vram_status:{vram_status}")
        relaymem_apply_allowed = (
            prompt_preview_plan.can_apply_without_user_approval
            if not prompt_preview_plan.approval_required
            else False
        )
        return RelayStackFinalRoutingDecision(
            state=RelayStackFinalRoutingState.BLOCKED_NO_KV_BUDGET,
            relaymem_apply_allowed=relaymem_apply_allowed,
            relaykv_routing_allowed=False,
            approval_required=prompt_preview_plan.approval_required,
            can_apply_without_user_approval=(
                prompt_preview_plan.can_apply_without_user_approval
            ),
            fallback_reason=fallback_reason,
            blocking_reasons=blocking_reasons,
            selected_retrieval_mode=(
                prompt_preview_plan.retrieval_mode if relaymem_apply_allowed else None
            ),
            proposed_retrieval_mode=proposed_retrieval_mode,
            fallback_if_denied=prompt_preview_plan.fallback_if_denied,
            vram_status=vram_status,
            memory_pressure_state=memory_pressure_state,
            user_visible_message=prompt_preview_plan.user_visible_message,
        )

    if prompt_preview_plan.approval_required:
        blocking_reasons.append("user_approval_required")
        return RelayStackFinalRoutingDecision(
            state=RelayStackFinalRoutingState.WAITING_FOR_USER_APPROVAL,
            relaymem_apply_allowed=False,
            relaykv_routing_allowed=False,
            approval_required=True,
            can_apply_without_user_approval=False,
            fallback_reason=fallback_reason,
            blocking_reasons=blocking_reasons,
            selected_retrieval_mode=None,
            proposed_retrieval_mode=proposed_retrieval_mode,
            fallback_if_denied=prompt_preview_plan.fallback_if_denied,
            vram_status=vram_status,
            memory_pressure_state=memory_pressure_state,
            user_visible_message=prompt_preview_plan.user_visible_message,
        )

    if fallback_reason is not None:
        blocking_reasons.append(fallback_reason)
        return RelayStackFinalRoutingDecision(
            state=RelayStackFinalRoutingState.FALLBACK_SAFE_BASELINE,
            relaymem_apply_allowed=False,
            relaykv_routing_allowed=False,
            approval_required=False,
            can_apply_without_user_approval=(
                prompt_preview_plan.can_apply_without_user_approval
            ),
            fallback_reason=fallback_reason,
            blocking_reasons=blocking_reasons,
            selected_retrieval_mode=None,
            proposed_retrieval_mode=None,
            fallback_if_denied=prompt_preview_plan.fallback_if_denied,
            vram_status=vram_status,
            memory_pressure_state=memory_pressure_state,
            user_visible_message=prompt_preview_plan.user_visible_message,
        )

    if (
        relaykv_routing_allowed
        and memory_pressure_state == RelayKVMemoryPressureState.RELAYKV_ROUTED_READY.value
        and prompt_preview_plan.can_apply_without_user_approval
    ):
        return RelayStackFinalRoutingDecision(
            state=RelayStackFinalRoutingState.RELAYMEM_AND_RELAYKV_READY,
            relaymem_apply_allowed=True,
            relaykv_routing_allowed=True,
            approval_required=False,
            can_apply_without_user_approval=True,
            fallback_reason=None,
            blocking_reasons=[],
            selected_retrieval_mode=prompt_preview_plan.retrieval_mode,
            proposed_retrieval_mode=None,
            fallback_if_denied=prompt_preview_plan.fallback_if_denied,
            vram_status=vram_status,
            memory_pressure_state=memory_pressure_state,
            user_visible_message=prompt_preview_plan.user_visible_message,
        )

    if prompt_preview_plan.can_apply_without_user_approval:
        if approval_gate_enabled:
            blocking_reasons.append("approval_gate_enabled_without_required_approval")
        return RelayStackFinalRoutingDecision(
            state=RelayStackFinalRoutingState.RELAYMEM_ONLY,
            relaymem_apply_allowed=True,
            relaykv_routing_allowed=False,
            approval_required=False,
            can_apply_without_user_approval=True,
            fallback_reason=None,
            blocking_reasons=blocking_reasons,
            selected_retrieval_mode=prompt_preview_plan.retrieval_mode,
            proposed_retrieval_mode=None,
            fallback_if_denied=prompt_preview_plan.fallback_if_denied,
            vram_status=vram_status,
            memory_pressure_state=memory_pressure_state,
            user_visible_message=prompt_preview_plan.user_visible_message,
        )

    blocking_reasons.append("no_applicable_relaymem_or_relaykv_path")
    return RelayStackFinalRoutingDecision(
        state=RelayStackFinalRoutingState.FALLBACK_SAFE_BASELINE,
        relaymem_apply_allowed=False,
        relaykv_routing_allowed=False,
        approval_required=prompt_preview_plan.approval_required,
        can_apply_without_user_approval=(
            prompt_preview_plan.can_apply_without_user_approval
        ),
        fallback_reason=fallback_reason,
        blocking_reasons=blocking_reasons,
        selected_retrieval_mode=None,
        proposed_retrieval_mode=proposed_retrieval_mode,
        fallback_if_denied=prompt_preview_plan.fallback_if_denied,
        vram_status=vram_status,
        memory_pressure_state=memory_pressure_state,
        user_visible_message=prompt_preview_plan.user_visible_message,
    )
