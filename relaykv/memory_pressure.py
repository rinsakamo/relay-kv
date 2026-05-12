from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .routing_decision import ExecutionMode


class RelayKVMemoryPressureState(str, Enum):
    DISABLED_SHORT_CONTEXT = "disabled_short_context"
    FULLKV_WITHIN_BUDGET = "fullkv_within_budget"
    SHADOW_ONLY_WARMUP = "shadow_only_warmup"
    BUDGET_PRESSURE = "budget_pressure"
    RELAYKV_ROUTED_READY = "relaykv_routed_ready"
    FALLBACK_REQUIRED = "fallback_required"


@dataclass(frozen=True)
class RelayKVMemoryPressureDecision:
    state: RelayKVMemoryPressureState
    execution_mode: ExecutionMode
    apply_blocked_reason: str | None = None
    fallback_reason: str | None = None
    budget_pressure: bool = False
    short_context: bool = False
    labels_ready: bool = False
    host_backup_available: bool = False
    shadow_compare_passed: bool | None = None
    selection_stability_ratio: float | None = None
    estimated_net_benefit_ms: float | None = None
    projected_fullkv_bytes: int | None = None
    residual_vram_budget_bytes: int | None = None
    min_seq_len_for_relaykv: int | None = None
    seq_len: int | None = None

    def __post_init__(self) -> None:
        if self.seq_len is not None and self.seq_len < 0:
            raise ValueError("seq_len must be >= 0")
        if (
            self.projected_fullkv_bytes is not None
            and self.projected_fullkv_bytes < 0
        ):
            raise ValueError("projected_fullkv_bytes must be >= 0")
        if (
            self.residual_vram_budget_bytes is not None
            and self.residual_vram_budget_bytes < 0
        ):
            raise ValueError("residual_vram_budget_bytes must be >= 0")
        if (
            self.min_seq_len_for_relaykv is not None
            and self.min_seq_len_for_relaykv < 0
        ):
            raise ValueError("min_seq_len_for_relaykv must be >= 0")
        if self.selection_stability_ratio is not None and not (
            0.0 <= self.selection_stability_ratio <= 1.0
        ):
            raise ValueError("selection_stability_ratio must be between 0.0 and 1.0")

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "execution_mode": self.execution_mode.value,
            "apply_blocked_reason": self.apply_blocked_reason,
            "fallback_reason": self.fallback_reason,
            "budget_pressure": self.budget_pressure,
            "short_context": self.short_context,
            "labels_ready": self.labels_ready,
            "host_backup_available": self.host_backup_available,
            "shadow_compare_passed": self.shadow_compare_passed,
            "selection_stability_ratio": self.selection_stability_ratio,
            "estimated_net_benefit_ms": self.estimated_net_benefit_ms,
            "projected_fullkv_bytes": self.projected_fullkv_bytes,
            "residual_vram_budget_bytes": self.residual_vram_budget_bytes,
            "min_seq_len_for_relaykv": self.min_seq_len_for_relaykv,
            "seq_len": self.seq_len,
        }


def decide_memory_pressure_state(
    *,
    seq_len: int,
    min_seq_len_for_relaykv: int = 0,
    projected_fullkv_bytes: int | None = None,
    residual_vram_budget_bytes: int | None = None,
    labels_ready: bool = False,
    host_backup_available: bool = False,
    shadow_compare_passed: bool | None = None,
    selection_stability_ratio: float | None = None,
    min_selection_stability_ratio: float = 0.8,
    estimated_net_benefit_ms: float | None = None,
    min_estimated_net_benefit_ms: float = 0.0,
    fallback_reason: str | None = None,
) -> RelayKVMemoryPressureDecision:
    budget_pressure = False
    fullkv_within_budget = False
    if (
        projected_fullkv_bytes is not None
        and residual_vram_budget_bytes is not None
    ):
        fullkv_within_budget = projected_fullkv_bytes <= residual_vram_budget_bytes
        budget_pressure = projected_fullkv_bytes > residual_vram_budget_bytes

    decision_kwargs = {
        "labels_ready": labels_ready,
        "host_backup_available": host_backup_available,
        "shadow_compare_passed": shadow_compare_passed,
        "selection_stability_ratio": selection_stability_ratio,
        "estimated_net_benefit_ms": estimated_net_benefit_ms,
        "projected_fullkv_bytes": projected_fullkv_bytes,
        "residual_vram_budget_bytes": residual_vram_budget_bytes,
        "min_seq_len_for_relaykv": min_seq_len_for_relaykv,
        "seq_len": seq_len,
    }

    if seq_len < min_seq_len_for_relaykv:
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.DISABLED_SHORT_CONTEXT,
            execution_mode=ExecutionMode.FULL_ATTENTION,
            apply_blocked_reason="short_context",
            short_context=True,
            budget_pressure=budget_pressure,
            fallback_reason=fallback_reason,
            **decision_kwargs,
        )

    if fallback_reason == "fullkv_within_budget":
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.FULLKV_WITHIN_BUDGET,
            execution_mode=ExecutionMode.FULLKV_GPU,
            fallback_reason=fallback_reason,
            budget_pressure=False,
            **decision_kwargs,
        )

    if fallback_reason is not None:
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.FALLBACK_REQUIRED,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason=fallback_reason,
            fallback_reason=fallback_reason,
            budget_pressure=budget_pressure,
            **decision_kwargs,
        )

    if fullkv_within_budget:
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.FULLKV_WITHIN_BUDGET,
            execution_mode=ExecutionMode.FULLKV_GPU,
            budget_pressure=False,
            **decision_kwargs,
        )

    if not labels_ready:
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="labels_not_ready",
            budget_pressure=budget_pressure,
            **decision_kwargs,
        )

    if not host_backup_available:
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="host_backup_unavailable",
            budget_pressure=budget_pressure,
            **decision_kwargs,
        )

    if shadow_compare_passed is False:
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.FALLBACK_REQUIRED,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="shadow_compare_failed",
            fallback_reason="shadow_compare_failed",
            budget_pressure=budget_pressure,
            **decision_kwargs,
        )

    if (
        selection_stability_ratio is not None
        and selection_stability_ratio < min_selection_stability_ratio
    ):
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="selection_unstable",
            budget_pressure=budget_pressure,
            **decision_kwargs,
        )

    if (
        estimated_net_benefit_ms is not None
        and estimated_net_benefit_ms < min_estimated_net_benefit_ms
    ):
        return RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="net_benefit_too_low",
            budget_pressure=budget_pressure,
            **decision_kwargs,
        )

    return RelayKVMemoryPressureDecision(
        state=RelayKVMemoryPressureState.RELAYKV_ROUTED_READY,
        execution_mode=ExecutionMode.RELAYKV_ROUTED,
        budget_pressure=True if budget_pressure else False,
        **decision_kwargs,
    )
