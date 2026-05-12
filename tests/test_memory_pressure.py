import json

import pytest

from relaykv import (
    ExecutionMode,
    RelayKVMemoryPressureDecision,
    RelayKVMemoryPressureState,
    decide_memory_pressure_state,
)


def test_memory_pressure_short_context_disables_relaykv() -> None:
    decision = decide_memory_pressure_state(
        seq_len=127,
        min_seq_len_for_relaykv=128,
    )

    assert decision.state is RelayKVMemoryPressureState.DISABLED_SHORT_CONTEXT
    assert decision.execution_mode is ExecutionMode.FULL_ATTENTION
    assert decision.short_context is True
    assert decision.apply_blocked_reason == "short_context"


def test_memory_pressure_fullkv_within_budget_uses_fullkv_gpu() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        projected_fullkv_bytes=4096,
        residual_vram_budget_bytes=8192,
    )

    assert decision.state is RelayKVMemoryPressureState.FULLKV_WITHIN_BUDGET
    assert decision.execution_mode is ExecutionMode.FULLKV_GPU
    assert decision.budget_pressure is False
    assert decision.apply_blocked_reason is None


def test_memory_pressure_fullkv_within_budget_fallback_uses_fullkv_gpu() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        fallback_reason="fullkv_within_budget",
    )

    assert decision.state is RelayKVMemoryPressureState.FULLKV_WITHIN_BUDGET
    assert decision.execution_mode is ExecutionMode.FULLKV_GPU
    assert decision.fallback_reason == "fullkv_within_budget"
    assert decision.apply_blocked_reason is None
    assert decision.budget_pressure is False


def test_memory_pressure_fallback_reason_requires_fallback() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        fallback_reason="insufficient_eviction_candidates",
    )

    assert decision.state is RelayKVMemoryPressureState.FALLBACK_REQUIRED
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.fallback_reason == "insufficient_eviction_candidates"
    assert decision.apply_blocked_reason == "insufficient_eviction_candidates"


def test_memory_pressure_labels_not_ready_uses_shadow_warmup() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=False,
        host_backup_available=True,
    )

    assert decision.state is RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "labels_not_ready"


def test_memory_pressure_host_backup_unavailable_uses_shadow_warmup() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=True,
        host_backup_available=False,
    )

    assert decision.state is RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "host_backup_unavailable"


def test_memory_pressure_shadow_compare_failure_requires_fallback() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=False,
    )

    assert decision.state is RelayKVMemoryPressureState.FALLBACK_REQUIRED
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.fallback_reason == "shadow_compare_failed"
    assert decision.apply_blocked_reason == "shadow_compare_failed"


def test_memory_pressure_shadow_compare_not_ready_uses_shadow_warmup() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=None,
    )

    assert decision.state is RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "shadow_compare_not_ready"
    assert decision.fallback_reason is None


def test_memory_pressure_selection_instability_uses_shadow_warmup() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=True,
        selection_stability_ratio=0.5,
        min_selection_stability_ratio=0.8,
    )

    assert decision.state is RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "selection_unstable"


def test_memory_pressure_low_net_benefit_uses_shadow_warmup() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=True,
        selection_stability_ratio=0.95,
        estimated_net_benefit_ms=-0.1,
        min_estimated_net_benefit_ms=0.0,
    )

    assert decision.state is RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "net_benefit_too_low"


def test_memory_pressure_ready_under_budget_pressure_is_routed_ready() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        projected_fullkv_bytes=16384,
        residual_vram_budget_bytes=8192,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=True,
        selection_stability_ratio=0.95,
        estimated_net_benefit_ms=1.2,
    )

    assert decision.state is RelayKVMemoryPressureState.RELAYKV_ROUTED_READY
    assert decision.execution_mode is ExecutionMode.RELAYKV_ROUTED
    assert decision.budget_pressure is True
    assert decision.apply_blocked_reason is None


def test_memory_pressure_missing_budget_bytes_can_still_be_routed_ready() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=True,
        selection_stability_ratio=0.95,
        estimated_net_benefit_ms=1.2,
    )

    assert decision.state is RelayKVMemoryPressureState.RELAYKV_ROUTED_READY
    assert decision.execution_mode is ExecutionMode.RELAYKV_ROUTED
    assert decision.budget_pressure is False


def test_memory_pressure_summary_is_json_serializable() -> None:
    decision = decide_memory_pressure_state(
        seq_len=512,
        min_seq_len_for_relaykv=128,
        projected_fullkv_bytes=16384,
        residual_vram_budget_bytes=8192,
        labels_ready=True,
        host_backup_available=True,
        shadow_compare_passed=True,
        selection_stability_ratio=0.95,
        estimated_net_benefit_ms=1.2,
    )

    summary = decision.summary()
    assert summary["state"] == "relaykv_routed_ready"
    assert summary["execution_mode"] == "relaykv_routed"
    assert summary["budget_pressure"] is True
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_memory_pressure_invalid_selection_stability_ratio_raises(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="selection_stability_ratio"):
        RelayKVMemoryPressureDecision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            selection_stability_ratio=value,
        )
