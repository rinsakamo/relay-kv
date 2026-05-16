from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fixed_budget_working_set import RelayKVFixedBudgetWorkingSetDecision


@dataclass(frozen=True)
class RelayKVBlockCandidate:
    block_id: int
    token_start: int
    token_end: int
    score: float | None
    is_recent: bool = False
    is_anchor: bool = False
    is_retrieval_candidate: bool = True
    layer_id: int | None = None
    kv_head_group: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "score": self.score,
            "is_recent": self.is_recent,
            "is_anchor": self.is_anchor,
            "is_retrieval_candidate": self.is_retrieval_candidate,
            "layer_id": self.layer_id,
            "kv_head_group": self.kv_head_group,
        }


@dataclass(frozen=True)
class RelayKVClassBlockSelection:
    class_name: str
    selected_block_ids: list[int]
    overflow_block_ids: list[int]

    def summary(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "selected_block_ids": list(self.selected_block_ids),
            "overflow_block_ids": list(self.overflow_block_ids),
        }


@dataclass(frozen=True)
class RelayKVFixedBudgetBlockSelectionDecision:
    decision_state: str
    dry_run_only: bool
    fixed_budget_decision_summary: dict[str, Any]
    selected_block_ids_by_class: dict[str, list[int]]
    selected_block_count_by_class: dict[str, int]
    selected_token_estimate_by_class: dict[str, int]
    materialized_working_tokens: int
    transient_budget_tokens: int
    estimated_working_tokens: int
    rejected_block_ids: list[int]
    overflow_block_ids: list[int]
    rejection_reason_counts: dict[str, int]
    notes: list[str]
    no_kv_materialization: bool = True
    selection_by_class: dict[str, RelayKVClassBlockSelection] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "decision_state": self.decision_state,
            "dry_run_only": self.dry_run_only,
            "fixed_budget_decision_summary": dict(self.fixed_budget_decision_summary),
            "selected_block_ids_by_class": {
                name: list(block_ids)
                for name, block_ids in self.selected_block_ids_by_class.items()
            },
            "selected_block_count_by_class": dict(self.selected_block_count_by_class),
            "selected_token_estimate_by_class": dict(
                self.selected_token_estimate_by_class
            ),
            "materialized_working_tokens": self.materialized_working_tokens,
            "transient_budget_tokens": self.transient_budget_tokens,
            "estimated_working_tokens": self.estimated_working_tokens,
            "rejected_block_ids": list(self.rejected_block_ids),
            "overflow_block_ids": list(self.overflow_block_ids),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "notes": list(self.notes),
            "no_kv_materialization": self.no_kv_materialization,
            "selection_by_class": {
                name: selection.summary()
                for name, selection in (self.selection_by_class or {}).items()
            },
        }


def build_synthetic_block_candidates(
    *,
    num_blocks: int,
    block_size: int,
) -> list[RelayKVBlockCandidate]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    recent_threshold = max(1, min(8, num_blocks // 8 or 1))
    anchor_threshold = max(1, min(2, num_blocks))
    candidates: list[RelayKVBlockCandidate] = []
    for block_id in range(num_blocks):
        token_start = block_id * block_size
        token_end = token_start + block_size
        is_recent = block_id >= num_blocks - recent_threshold
        is_anchor = block_id < anchor_threshold
        score = float(num_blocks - block_id)
        candidates.append(
            RelayKVBlockCandidate(
                block_id=block_id,
                token_start=token_start,
                token_end=token_end,
                score=score,
                is_recent=is_recent,
                is_anchor=is_anchor,
                is_retrieval_candidate=not is_recent,
                layer_id=0,
                kv_head_group=0,
            )
        )
    return candidates


def _score_value(candidate: RelayKVBlockCandidate) -> float:
    return candidate.score if candidate.score is not None else float("-inf")


def _select_recent_candidates(
    candidates: list[RelayKVBlockCandidate],
    selected_ids: set[int],
    budget_blocks: int,
) -> RelayKVClassBlockSelection:
    eligible = sorted(
        [candidate for candidate in candidates if candidate.block_id not in selected_ids],
        key=lambda candidate: (
            not candidate.is_recent,
            -candidate.token_end,
            candidate.block_id,
        ),
    )
    selected = eligible[:budget_blocks]
    overflow = eligible[budget_blocks:]
    return RelayKVClassBlockSelection(
        class_name="recent",
        selected_block_ids=[candidate.block_id for candidate in selected],
        overflow_block_ids=[candidate.block_id for candidate in overflow],
    )


def _select_anchor_candidates(
    candidates: list[RelayKVBlockCandidate],
    selected_ids: set[int],
    budget_blocks: int,
) -> RelayKVClassBlockSelection:
    eligible = sorted(
        [candidate for candidate in candidates if candidate.block_id not in selected_ids],
        key=lambda candidate: (
            not candidate.is_anchor,
            -_score_value(candidate),
            candidate.block_id,
        ),
    )
    selected = eligible[:budget_blocks]
    overflow = eligible[budget_blocks:]
    return RelayKVClassBlockSelection(
        class_name="anchor",
        selected_block_ids=[candidate.block_id for candidate in selected],
        overflow_block_ids=[candidate.block_id for candidate in overflow],
    )


def _select_retrieved_candidates(
    candidates: list[RelayKVBlockCandidate],
    selected_ids: set[int],
    budget_blocks: int,
) -> RelayKVClassBlockSelection:
    eligible = sorted(
        [
            candidate
            for candidate in candidates
            if candidate.block_id not in selected_ids
            and candidate.is_retrieval_candidate
        ],
        key=lambda candidate: (-_score_value(candidate), candidate.block_id),
    )
    selected = eligible[:budget_blocks]
    overflow = eligible[budget_blocks:]
    return RelayKVClassBlockSelection(
        class_name="retrieved",
        selected_block_ids=[candidate.block_id for candidate in selected],
        overflow_block_ids=[candidate.block_id for candidate in overflow],
    )


def build_relaykv_fixed_budget_block_selection_decision(
    *,
    fixed_budget_decision: RelayKVFixedBudgetWorkingSetDecision,
    candidates: list[RelayKVBlockCandidate],
    block_size: int,
) -> RelayKVFixedBudgetBlockSelectionDecision:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    notes = [
        "selection_priority:recent,anchor,retrieved",
        "block_selection_dry_run_only",
    ]
    selected_ids: set[int] = set()
    budget_summary = fixed_budget_decision.summary()
    class_budget_counts = {
        "recent": int(
            budget_summary["selected_block_plan"].get("recent_block_count", 0)
        ),
        "anchor": int(
            budget_summary["selected_block_plan"].get("anchor_block_count", 0)
        ),
        "retrieved": int(
            budget_summary["selected_block_plan"].get("retrieved_block_count", 0)
        ),
    }

    recent_selection = _select_recent_candidates(
        candidates,
        selected_ids,
        class_budget_counts["recent"],
    )
    selected_ids.update(recent_selection.selected_block_ids)

    anchor_selection = _select_anchor_candidates(
        candidates,
        selected_ids,
        class_budget_counts["anchor"],
    )
    selected_ids.update(anchor_selection.selected_block_ids)

    retrieved_selection = _select_retrieved_candidates(
        candidates,
        selected_ids,
        class_budget_counts["retrieved"],
    )
    selected_ids.update(retrieved_selection.selected_block_ids)

    selection_by_class = {
        "recent": recent_selection,
        "anchor": anchor_selection,
        "retrieved": retrieved_selection,
    }
    selected_block_ids_by_class = {
        name: selection.selected_block_ids
        for name, selection in selection_by_class.items()
    }
    selected_block_count_by_class = {
        name: len(selection.selected_block_ids)
        for name, selection in selection_by_class.items()
    }
    selected_token_estimate_by_class = {
        name: len(selection.selected_block_ids) * block_size
        for name, selection in selection_by_class.items()
    }

    overflow_reason_by_id: dict[int, str] = {}
    for class_name, selection in selection_by_class.items():
        for block_id in selection.overflow_block_ids:
            overflow_reason_by_id.setdefault(block_id, f"overflow_{class_name}_budget")

    selected_all_ids = {
        block_id
        for block_ids in selected_block_ids_by_class.values()
        for block_id in block_ids
    }
    rejected_block_ids = [
        candidate.block_id
        for candidate in candidates
        if candidate.block_id not in selected_all_ids
    ]
    overflow_block_ids = sorted(overflow_reason_by_id.keys())

    rejection_reason_counts: dict[str, int] = {}
    for candidate in candidates:
        if candidate.block_id in selected_all_ids:
            continue
        reason = overflow_reason_by_id.get(candidate.block_id)
        if reason is None:
            reason = "not_selected_by_priority"
        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1

    materialized_working_tokens = sum(selected_token_estimate_by_class.values())
    estimated_working_tokens = (
        materialized_working_tokens + fixed_budget_decision.transient_budget_tokens
    )
    if estimated_working_tokens > fixed_budget_decision.total_working_budget_tokens:
        decision_state = "invalid_budget"
        notes.append("selected_blocks_exceed_fixed_budget_decision")
    else:
        decision_state = fixed_budget_decision.decision_state

    if len(selected_all_ids) != sum(selected_block_count_by_class.values()):
        notes.append("duplicate_block_ids_detected_and_removed")

    return RelayKVFixedBudgetBlockSelectionDecision(
        decision_state=decision_state,
        dry_run_only=True,
        fixed_budget_decision_summary=budget_summary,
        selected_block_ids_by_class=selected_block_ids_by_class,
        selected_block_count_by_class=selected_block_count_by_class,
        selected_token_estimate_by_class=selected_token_estimate_by_class,
        materialized_working_tokens=materialized_working_tokens,
        transient_budget_tokens=fixed_budget_decision.transient_budget_tokens,
        estimated_working_tokens=estimated_working_tokens,
        rejected_block_ids=rejected_block_ids,
        overflow_block_ids=overflow_block_ids,
        rejection_reason_counts=rejection_reason_counts,
        notes=notes,
        no_kv_materialization=True,
        selection_by_class=selection_by_class,
    )
