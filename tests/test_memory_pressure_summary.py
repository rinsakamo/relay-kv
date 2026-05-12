import json

from relaykv import (
    ExecutionMode,
    RelayKVMemoryPressureDecision,
    RelayKVMemoryPressureState,
    summarize_memory_pressure_decisions,
)


def make_decision(
    *,
    state: RelayKVMemoryPressureState | str,
    execution_mode: ExecutionMode | str,
    apply_blocked_reason: str | None = None,
    fallback_reason: str | None = None,
    short_context: bool = False,
    budget_pressure: bool = False,
    seq_len: int | None = None,
    selection_stability_ratio: float | None = None,
    estimated_net_benefit_ms: float | None = None,
) -> RelayKVMemoryPressureDecision:
    return RelayKVMemoryPressureDecision(
        state=state,
        execution_mode=execution_mode,
        apply_blocked_reason=apply_blocked_reason,
        fallback_reason=fallback_reason,
        short_context=short_context,
        budget_pressure=budget_pressure,
        seq_len=seq_len,
        selection_stability_ratio=selection_stability_ratio,
        estimated_net_benefit_ms=estimated_net_benefit_ms,
    )


def test_memory_pressure_summary_empty_input_returns_zero_counts() -> None:
    summary = summarize_memory_pressure_decisions([])

    assert summary["total_decisions"] == 0
    assert summary["state_counts"] == {}
    assert summary["execution_mode_counts"] == {}
    assert summary["apply_blocked_reason_counts"] == {}
    assert summary["fallback_reason_counts"] == {}
    assert summary["short_context_count"] == 0
    assert summary["budget_pressure_count"] == 0


def test_memory_pressure_summary_summarizes_decision_objects() -> None:
    decisions = [
        make_decision(
            state=RelayKVMemoryPressureState.DISABLED_SHORT_CONTEXT,
            execution_mode=ExecutionMode.FULL_ATTENTION,
            apply_blocked_reason="short_context",
            short_context=True,
            seq_len=64,
        ),
        make_decision(
            state=RelayKVMemoryPressureState.RELAYKV_ROUTED_READY,
            execution_mode=ExecutionMode.RELAYKV_ROUTED,
            budget_pressure=True,
            seq_len=512,
            selection_stability_ratio=0.9,
            estimated_net_benefit_ms=1.2,
        ),
        make_decision(
            state=RelayKVMemoryPressureState.FULLKV_WITHIN_BUDGET,
            execution_mode=ExecutionMode.FULLKV_GPU,
            fallback_reason="fullkv_within_budget",
            seq_len=256,
        ),
    ]

    summary = summarize_memory_pressure_decisions(decisions)

    assert summary["total_decisions"] == 3
    assert summary["state_counts"] == {
        "disabled_short_context": 1,
        "fullkv_within_budget": 1,
        "relaykv_routed_ready": 1,
    }
    assert summary["execution_mode_counts"] == {
        "full_attention": 1,
        "fullkv_gpu": 1,
        "relaykv_routed": 1,
    }
    assert summary["short_context_count"] == 1
    assert summary["budget_pressure_count"] == 1
    assert summary["routed_ready_count"] == 1
    assert summary["fullkv_within_budget_count"] == 1
    assert summary["min_seq_len"] == 64.0
    assert summary["max_seq_len"] == 512.0
    assert summary["avg_seq_len"] == 832.0 / 3.0


def test_memory_pressure_summary_summarizes_summary_dicts() -> None:
    decisions = [
        make_decision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="labels_not_ready",
            seq_len=128,
        ).summary(),
        make_decision(
            state=RelayKVMemoryPressureState.FALLBACK_REQUIRED,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="shadow_compare_failed",
            fallback_reason="shadow_compare_failed",
            seq_len=256,
        ).summary(),
    ]

    summary = summarize_memory_pressure_decisions(decisions)

    assert summary["total_decisions"] == 2
    assert summary["state_counts"]["shadow_only_warmup"] == 1
    assert summary["state_counts"]["fallback_required"] == 1
    assert summary["execution_mode_counts"]["shadow_only"] == 2
    assert summary["labels_not_ready_count"] == 1
    assert summary["shadow_compare_failed_count"] == 1


def test_memory_pressure_summary_mixed_object_and_dict_input_works() -> None:
    decisions = [
        make_decision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="host_backup_unavailable",
        ),
        make_decision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="shadow_compare_not_ready",
        ).summary(),
    ]

    summary = summarize_memory_pressure_decisions(decisions)

    assert summary["total_decisions"] == 2
    assert summary["shadow_warmup_count"] == 2
    assert summary["host_backup_unavailable_count"] == 1
    assert summary["shadow_compare_not_ready_count"] == 1


def test_memory_pressure_summary_counts_reasons_and_special_states() -> None:
    decisions = [
        make_decision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="selection_unstable",
            seq_len=512,
            selection_stability_ratio=0.4,
        ),
        make_decision(
            state=RelayKVMemoryPressureState.SHADOW_ONLY_WARMUP,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="net_benefit_too_low",
            seq_len=768,
            estimated_net_benefit_ms=-0.5,
        ),
        make_decision(
            state=RelayKVMemoryPressureState.FALLBACK_REQUIRED,
            execution_mode=ExecutionMode.SHADOW_ONLY,
            apply_blocked_reason="custom_blocker",
            fallback_reason="arbitrary_fallback",
        ),
    ]

    summary = summarize_memory_pressure_decisions(decisions)

    assert summary["selection_unstable_count"] == 1
    assert summary["net_benefit_too_low_count"] == 1
    assert summary["fallback_reason_counts"]["arbitrary_fallback"] == 1
    assert summary["fallback_required_count"] == 1
    assert summary["min_selection_stability_ratio"] == 0.4
    assert summary["max_selection_stability_ratio"] == 0.4
    assert summary["avg_selection_stability_ratio"] == 0.4
    assert summary["min_estimated_net_benefit_ms"] == -0.5
    assert summary["max_estimated_net_benefit_ms"] == -0.5
    assert summary["avg_estimated_net_benefit_ms"] == -0.5


def test_memory_pressure_summary_unknown_state_and_reason_do_not_crash() -> None:
    decisions = [
        {
            "state": "custom_state",
            "execution_mode": "custom_mode",
            "apply_blocked_reason": "custom_reason",
            "fallback_reason": "custom_fallback",
            "short_context": False,
            "budget_pressure": False,
        }
    ]

    summary = summarize_memory_pressure_decisions(decisions)

    assert summary["state_counts"]["custom_state"] == 1
    assert summary["execution_mode_counts"]["custom_mode"] == 1
    assert summary["apply_blocked_reason_counts"]["custom_reason"] == 1
    assert summary["fallback_reason_counts"]["custom_fallback"] == 1


def test_memory_pressure_summary_output_is_json_serializable() -> None:
    decisions = [
        make_decision(
            state=RelayKVMemoryPressureState.RELAYKV_ROUTED_READY,
            execution_mode=ExecutionMode.RELAYKV_ROUTED,
            budget_pressure=True,
            seq_len=1024,
            selection_stability_ratio=0.95,
            estimated_net_benefit_ms=2.0,
        )
    ]

    summary = summarize_memory_pressure_decisions(decisions)
    assert json.loads(json.dumps(summary)) == summary
