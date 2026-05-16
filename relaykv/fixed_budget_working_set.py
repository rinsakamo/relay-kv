from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any


_CLASS_NAMES = ("recent", "anchor", "retrieved")
_BLOCK_REMAINDER_PRIORITY = ("recent", "retrieved", "anchor")


@dataclass(frozen=True)
class RelayKVFixedBudgetConfig:
    total_working_budget_tokens: int
    recent_budget_tokens: int | None = None
    anchor_budget_tokens: int | None = None
    retrieved_budget_tokens: int | None = None
    transient_budget_tokens: int = 0
    recent_ratio: float = 0.50
    anchor_ratio: float = 0.10
    retrieved_ratio: float = 0.40
    min_recent_tokens: int = 128
    min_anchor_tokens: int = 0
    block_size: int = 64

    def __post_init__(self) -> None:
        if self.total_working_budget_tokens <= 0:
            raise ValueError("total_working_budget_tokens must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.transient_budget_tokens < 0:
            raise ValueError("transient_budget_tokens must be >= 0")
        if self.min_recent_tokens < 0:
            raise ValueError("min_recent_tokens must be >= 0")
        if self.min_anchor_tokens < 0:
            raise ValueError("min_anchor_tokens must be >= 0")
        for name, value in (
            ("recent_ratio", self.recent_ratio),
            ("anchor_ratio", self.anchor_ratio),
            ("retrieved_ratio", self.retrieved_ratio),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name, value in (
            ("recent_budget_tokens", self.recent_budget_tokens),
            ("anchor_budget_tokens", self.anchor_budget_tokens),
            ("retrieved_budget_tokens", self.retrieved_budget_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class RelayKVClassBudget:
    class_name: str
    budget_tokens: int
    budget_blocks: int
    source: str

    def summary(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "budget_tokens": self.budget_tokens,
            "budget_blocks": self.budget_blocks,
            "source": self.source,
        }


@dataclass(frozen=True)
class RelayKVFixedBudgetWorkingSetDecision:
    total_working_budget_tokens: int
    transient_budget_tokens: int
    block_size: int
    class_budgets: dict[str, RelayKVClassBudget]
    selected_block_plan: dict[str, int]
    estimated_working_tokens: int
    materialized_working_tokens: int
    decision_state: str
    fallback_reason: str | None
    notes: list[str]
    dry_run_only: bool = True
    policy_name: str = "relaykv_fixed_budget_working_set_dry_run_v0"

    def summary(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_working_budget_tokens": self.total_working_budget_tokens,
            "transient_budget_tokens": self.transient_budget_tokens,
            "class_budgets": {
                name: budget.summary()
                for name, budget in self.class_budgets.items()
            },
            "selected_block_plan": dict(self.selected_block_plan),
            "estimated_working_tokens": self.estimated_working_tokens,
            "materialized_working_tokens": self.materialized_working_tokens,
            "block_size": self.block_size,
            "decision_state": self.decision_state,
            "fallback_reason": self.fallback_reason,
            "notes": list(self.notes),
            "dry_run_only": self.dry_run_only,
        }


def _floor_blocks(tokens: int, block_size: int) -> int:
    if tokens <= 0:
        return 0
    return tokens // block_size


def _build_block_aligned_plan(
    *,
    config: RelayKVFixedBudgetConfig,
    budgets: dict[str, int],
    sources: dict[str, str],
    notes: list[str],
) -> tuple[dict[str, RelayKVClassBudget], dict[str, int], int, int]:
    available_physical_tokens = max(
        0,
        config.total_working_budget_tokens - config.transient_budget_tokens,
    )
    available_total_budget_blocks = available_physical_tokens // config.block_size
    desired_total_budget_blocks = min(
        available_total_budget_blocks,
        sum(budgets[name] for name in _CLASS_NAMES) // config.block_size,
    )

    base_block_counts = {
        name: _floor_blocks(budgets[name], config.block_size)
        for name in _CLASS_NAMES
    }
    for name in _CLASS_NAMES:
        if 0 < budgets[name] < config.block_size:
            notes.append(f"{name}_budget_below_one_block_not_materialized")

    allocated_blocks = sum(base_block_counts.values())
    if allocated_blocks > desired_total_budget_blocks:
        trim_order = list(reversed(_BLOCK_REMAINDER_PRIORITY))
        overflow_blocks = allocated_blocks - desired_total_budget_blocks
        for name in trim_order:
            if overflow_blocks <= 0:
                break
            removable = min(base_block_counts[name], overflow_blocks)
            if removable <= 0:
                continue
            base_block_counts[name] -= removable
            overflow_blocks -= removable
        notes.append(
            "block_plan_trimmed_to_total_budget_priority:anchor,retrieved,recent"
        )
        allocated_blocks = sum(base_block_counts.values())

    remaining_blocks = max(0, desired_total_budget_blocks - allocated_blocks)
    remainders = {
        name: budgets[name] % config.block_size
        for name in _CLASS_NAMES
    }
    if remaining_blocks > 0:
        ordered_names = sorted(
            _CLASS_NAMES,
            key=lambda name: (
                -remainders[name],
                _BLOCK_REMAINDER_PRIORITY.index(name),
            ),
        )
        distributed = 0
        for name in ordered_names:
            if distributed >= remaining_blocks:
                break
            if remainders[name] <= 0:
                continue
            base_block_counts[name] += 1
            distributed += 1
        if distributed > 0:
            notes.append(
                "block_remainder_priority_applied:recent,retrieved,anchor"
            )

    class_budgets = {
        name: RelayKVClassBudget(
            class_name=name,
            budget_tokens=budgets[name],
            budget_blocks=(
                base_block_counts[name]
                if name in _CLASS_NAMES
                else 0
            ),
            source=sources[name],
        )
        for name in (*_CLASS_NAMES, "transient")
    }
    total_selected_blocks = sum(base_block_counts.values())
    materialized_working_tokens = total_selected_blocks * config.block_size
    estimated_working_tokens = (
        materialized_working_tokens + config.transient_budget_tokens
    )
    selected_block_plan = {
        "total_budget_blocks_available": available_total_budget_blocks,
        "total_selected_blocks": total_selected_blocks,
        "recent_block_count": class_budgets["recent"].budget_blocks,
        "anchor_block_count": class_budgets["anchor"].budget_blocks,
        "retrieved_block_count": class_budgets["retrieved"].budget_blocks,
        "transient_block_count": class_budgets["transient"].budget_blocks,
        "estimated_physical_working_tokens": materialized_working_tokens,
        "estimated_working_tokens": estimated_working_tokens,
        "materialized_working_tokens": materialized_working_tokens,
        "unallocated_tokens": max(
            0,
            config.total_working_budget_tokens - estimated_working_tokens,
        ),
    }
    return (
        class_budgets,
        selected_block_plan,
        materialized_working_tokens,
        estimated_working_tokens,
    )


def _build_invalid_decision(
    *,
    config: RelayKVFixedBudgetConfig,
    budgets: dict[str, int],
    sources: dict[str, str],
    fallback_reason: str,
    notes: list[str],
) -> RelayKVFixedBudgetWorkingSetDecision:
    (
        class_budgets,
        selected_block_plan,
        materialized_working_tokens,
        estimated_working_tokens,
    ) = (
        _build_block_aligned_plan(
            config=config,
            budgets=budgets,
            sources=sources,
            notes=notes,
        )
    )
    return RelayKVFixedBudgetWorkingSetDecision(
        total_working_budget_tokens=config.total_working_budget_tokens,
        transient_budget_tokens=config.transient_budget_tokens,
        block_size=config.block_size,
        class_budgets=class_budgets,
        selected_block_plan=selected_block_plan,
        estimated_working_tokens=estimated_working_tokens,
        materialized_working_tokens=materialized_working_tokens,
        decision_state="invalid_budget",
        fallback_reason=fallback_reason,
        notes=notes,
    )


def build_relaykv_fixed_budget_working_set_decision(
    *,
    config: RelayKVFixedBudgetConfig,
) -> RelayKVFixedBudgetWorkingSetDecision:
    notes: list[str] = []
    sources = {name: "ratio" for name in _CLASS_NAMES}
    sources["transient"] = (
        "explicit" if config.transient_budget_tokens > 0 else "fixed_zero"
    )
    budgets = {
        "recent": config.recent_budget_tokens or 0,
        "anchor": config.anchor_budget_tokens or 0,
        "retrieved": config.retrieved_budget_tokens or 0,
        "transient": config.transient_budget_tokens,
    }

    explicit_flags = {
        "recent": config.recent_budget_tokens is not None,
        "anchor": config.anchor_budget_tokens is not None,
        "retrieved": config.retrieved_budget_tokens is not None,
    }
    for name, is_explicit in explicit_flags.items():
        if is_explicit:
            sources[name] = "explicit"

    physical_budget_tokens = (
        config.total_working_budget_tokens - config.transient_budget_tokens
    )
    if physical_budget_tokens < 0:
        notes.append("transient_budget_tokens_exceed_total_working_budget_tokens")
        return _build_invalid_decision(
            config=config,
            budgets=budgets,
            sources=sources,
            fallback_reason="transient_budget_tokens_exceed_total_working_budget_tokens",
            notes=notes,
        )

    explicit_sum = sum(budgets[name] for name in _CLASS_NAMES if explicit_flags[name])
    if explicit_sum > physical_budget_tokens:
        notes.append("explicit_class_budgets_exceed_total_working_budget_tokens")
        return _build_invalid_decision(
            config=config,
            budgets=budgets,
            sources=sources,
            fallback_reason="explicit_class_budgets_exceed_total_working_budget_tokens",
            notes=notes,
        )

    remaining_tokens = physical_budget_tokens - explicit_sum
    derived_classes = [name for name in _CLASS_NAMES if not explicit_flags[name]]
    ratio_map = {
        "recent": config.recent_ratio,
        "anchor": config.anchor_ratio,
        "retrieved": config.retrieved_ratio,
    }
    if derived_classes and remaining_tokens > 0:
        remaining_ratio = sum(ratio_map[name] for name in derived_classes)
        if remaining_ratio <= 0:
            notes.append("remaining_ratio_sum_must_be_positive")
            return _build_invalid_decision(
                config=config,
                budgets=budgets,
                sources=sources,
                fallback_reason="remaining_ratio_sum_must_be_positive",
                notes=notes,
            )
        assigned_tokens = 0
        last_index = len(derived_classes) - 1
        for index, name in enumerate(derived_classes):
            if index == last_index:
                allocation = remaining_tokens - assigned_tokens
            else:
                allocation = floor(
                    remaining_tokens * ratio_map[name] / remaining_ratio
                )
                assigned_tokens += allocation
            budgets[name] = allocation

    minimums = {
        "recent": config.min_recent_tokens,
        "anchor": config.min_anchor_tokens,
    }
    for target in ("recent", "anchor"):
        minimum = minimums[target]
        if explicit_flags.get(target, False) or budgets[target] >= minimum:
            continue
        deficit = minimum - budgets[target]
        donor_names = [
            name
            for name in _CLASS_NAMES
            if name != target and not explicit_flags[name]
        ]
        donor_names.sort(key=lambda name: budgets[name], reverse=True)
        for donor_name in donor_names:
            donor_floor = minimums.get(donor_name, 0)
            available = max(0, budgets[donor_name] - donor_floor)
            transfer = min(deficit, available)
            if transfer <= 0:
                continue
            budgets[donor_name] -= transfer
            budgets[target] += transfer
            deficit -= transfer
            notes.append(f"rebalanced_tokens_to_{target}_from_{donor_name}")
            if deficit == 0:
                break
        if deficit > 0:
            notes.append(f"minimum_budget_unsatisfied:{target}")
            return _build_invalid_decision(
                config=config,
                budgets=budgets,
                sources=sources,
                fallback_reason="minimum_budgets_exceed_total_working_budget_tokens",
                notes=notes,
            )

    estimated_physical_tokens = sum(budgets[name] for name in _CLASS_NAMES)
    estimated_working_tokens = estimated_physical_tokens + budgets["transient"]
    if estimated_working_tokens > config.total_working_budget_tokens:
        notes.append("estimated_working_tokens_exceed_total_working_budget_tokens")
        return _build_invalid_decision(
            config=config,
            budgets=budgets,
            sources=sources,
            fallback_reason="estimated_working_tokens_exceed_total_working_budget_tokens",
            notes=notes,
        )

    if budgets["transient"] > 0:
        notes.append("transient_budget_tokens_reserved_but_not_materialized")
    if estimated_working_tokens < config.total_working_budget_tokens:
        notes.append("working_budget_tokens_left_unallocated")
    for name in ("recent", "anchor"):
        minimum = minimums[name]
        if explicit_flags.get(name, False) and budgets[name] < minimum:
            notes.append(f"explicit_{name}_budget_below_minimum")

    (
        class_budgets,
        selected_block_plan,
        materialized_working_tokens,
        estimated_working_tokens_block_aligned,
    ) = (
        _build_block_aligned_plan(
            config=config,
            budgets=budgets,
            sources=sources,
            notes=notes,
        )
    )
    if estimated_working_tokens_block_aligned > config.total_working_budget_tokens:
        notes.append("block_aligned_plan_exceeds_total_budget")
        return _build_invalid_decision(
            config=config,
            budgets=budgets,
            sources=sources,
            fallback_reason="block_aligned_plan_exceeds_total_budget",
            notes=notes,
        )
    return RelayKVFixedBudgetWorkingSetDecision(
        total_working_budget_tokens=config.total_working_budget_tokens,
        transient_budget_tokens=config.transient_budget_tokens,
        block_size=config.block_size,
        class_budgets=class_budgets,
        selected_block_plan=selected_block_plan,
        estimated_working_tokens=estimated_working_tokens_block_aligned,
        materialized_working_tokens=materialized_working_tokens,
        decision_state="dry_run_ready",
        fallback_reason=None,
        notes=notes,
    )
