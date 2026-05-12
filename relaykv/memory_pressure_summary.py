from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .memory_pressure import RelayKVMemoryPressureDecision


def _normalize_decision(
    decision: RelayKVMemoryPressureDecision | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(decision, RelayKVMemoryPressureDecision):
        return decision.summary()
    return dict(decision)


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _normalize_string_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return str(value)


def _compute_numeric_stats(values: list[float | int]) -> tuple[float, float, float] | None:
    if not values:
        return None
    numeric_values = [float(value) for value in values]
    return (
        min(numeric_values),
        max(numeric_values),
        sum(numeric_values) / len(numeric_values),
    )


def summarize_memory_pressure_decisions(
    decisions: Iterable[RelayKVMemoryPressureDecision | dict],
) -> dict:
    normalized_decisions = [_normalize_decision(decision) for decision in decisions]

    state_counts: Counter[str] = Counter()
    execution_mode_counts: Counter[str] = Counter()
    apply_blocked_reason_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()

    short_context_count = 0
    budget_pressure_count = 0
    seq_lens: list[int] = []
    selection_stability_ratios: list[float] = []
    estimated_net_benefits_ms: list[float] = []

    for decision in normalized_decisions:
        state = _normalize_string_value(decision.get("state"))
        if state is not None:
            state_counts[state] += 1

        execution_mode = _normalize_string_value(decision.get("execution_mode"))
        if execution_mode is not None:
            execution_mode_counts[execution_mode] += 1

        apply_blocked_reason = _normalize_string_value(
            decision.get("apply_blocked_reason")
        )
        if apply_blocked_reason is not None:
            apply_blocked_reason_counts[apply_blocked_reason] += 1

        fallback_reason = _normalize_string_value(decision.get("fallback_reason"))
        if fallback_reason is not None:
            fallback_reason_counts[fallback_reason] += 1

        if bool(decision.get("short_context")):
            short_context_count += 1
        if bool(decision.get("budget_pressure")):
            budget_pressure_count += 1

        seq_len = decision.get("seq_len")
        if seq_len is not None:
            seq_lens.append(seq_len)

        selection_stability_ratio = decision.get("selection_stability_ratio")
        if selection_stability_ratio is not None:
            selection_stability_ratios.append(selection_stability_ratio)

        estimated_net_benefit_ms = decision.get("estimated_net_benefit_ms")
        if estimated_net_benefit_ms is not None:
            estimated_net_benefits_ms.append(estimated_net_benefit_ms)

    seq_len_stats = _compute_numeric_stats(seq_lens)
    selection_stability_stats = _compute_numeric_stats(selection_stability_ratios)
    estimated_net_benefit_stats = _compute_numeric_stats(estimated_net_benefits_ms)

    return {
        "total_decisions": len(normalized_decisions),
        "state_counts": _counter_to_dict(state_counts),
        "execution_mode_counts": _counter_to_dict(execution_mode_counts),
        "apply_blocked_reason_counts": _counter_to_dict(apply_blocked_reason_counts),
        "fallback_reason_counts": _counter_to_dict(fallback_reason_counts),
        "short_context_count": short_context_count,
        "budget_pressure_count": budget_pressure_count,
        "routed_ready_count": state_counts.get("relaykv_routed_ready", 0),
        "shadow_warmup_count": state_counts.get("shadow_only_warmup", 0),
        "fallback_required_count": state_counts.get("fallback_required", 0),
        "fullkv_within_budget_count": state_counts.get("fullkv_within_budget", 0),
        "shadow_compare_not_ready_count": apply_blocked_reason_counts.get(
            "shadow_compare_not_ready", 0
        ),
        "shadow_compare_failed_count": fallback_reason_counts.get(
            "shadow_compare_failed", 0
        ),
        "labels_not_ready_count": apply_blocked_reason_counts.get(
            "labels_not_ready", 0
        ),
        "host_backup_unavailable_count": apply_blocked_reason_counts.get(
            "host_backup_unavailable", 0
        ),
        "selection_unstable_count": apply_blocked_reason_counts.get(
            "selection_unstable", 0
        ),
        "net_benefit_too_low_count": apply_blocked_reason_counts.get(
            "net_benefit_too_low", 0
        ),
        "min_seq_len": seq_len_stats[0] if seq_len_stats is not None else None,
        "max_seq_len": seq_len_stats[1] if seq_len_stats is not None else None,
        "avg_seq_len": seq_len_stats[2] if seq_len_stats is not None else None,
        "min_selection_stability_ratio": (
            selection_stability_stats[0]
            if selection_stability_stats is not None
            else None
        ),
        "max_selection_stability_ratio": (
            selection_stability_stats[1]
            if selection_stability_stats is not None
            else None
        ),
        "avg_selection_stability_ratio": (
            selection_stability_stats[2]
            if selection_stability_stats is not None
            else None
        ),
        "min_estimated_net_benefit_ms": (
            estimated_net_benefit_stats[0]
            if estimated_net_benefit_stats is not None
            else None
        ),
        "max_estimated_net_benefit_ms": (
            estimated_net_benefit_stats[1]
            if estimated_net_benefit_stats is not None
            else None
        ),
        "avg_estimated_net_benefit_ms": (
            estimated_net_benefit_stats[2]
            if estimated_net_benefit_stats is not None
            else None
        ),
    }
