from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .block_scoring import BlockScore


@dataclass(frozen=True)
class WorkingBlockBudgets:
    total_working_blocks: int
    recent_blocks: int
    anchor_blocks: int
    retrieval_blocks: int

    def summary(self) -> dict:
        return {
            "total_working_blocks": self.total_working_blocks,
            "recent_blocks": self.recent_blocks,
            "anchor_blocks": self.anchor_blocks,
            "retrieval_blocks": self.retrieval_blocks,
        }


@dataclass(frozen=True)
class WorkingBlockSelection:
    recent_block_ids: list[int]
    anchor_block_ids: list[int]
    retrieved_block_ids: list[int]
    working_block_ids: list[int]

    def summary(self) -> dict:
        return {
            "recent_block_ids": self.recent_block_ids,
            "anchor_block_ids": self.anchor_block_ids,
            "retrieved_block_ids": self.retrieved_block_ids,
            "working_block_ids": self.working_block_ids,
        }


@dataclass(frozen=True)
class WorkingBlockBudgetDecision:
    seq_len: int
    block_size: int
    budgets: WorkingBlockBudgets
    selected: WorkingBlockSelection
    budget_ok: bool
    fallback_reason: str | None
    policy_name: str = "relaykv_block_budget_mvp_v0"

    def summary(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "seq_len": self.seq_len,
            "block_size": self.block_size,
            "budgets": self.budgets.summary(),
            "selected": self.selected.summary(),
            "budget_ok": self.budget_ok,
            "fallback_reason": self.fallback_reason,
        }


def _validate_budget(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _append_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def _dedupe_in_order(block_ids: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for block_id in block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        deduped.append(block_id)
    return deduped


def _take_tail(block_ids: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(block_ids):
        return list(block_ids)
    return block_ids[len(block_ids) - count:]


def _take_head(block_ids: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(block_ids):
        return list(block_ids)
    return block_ids[:count]


def build_working_block_budget_decision(
    *,
    seq_len: int,
    block_size: int,
    total_working_blocks: int,
    recent_blocks: int,
    anchor_blocks: int,
    retrieval_blocks: int,
    scored_blocks: list[BlockScore],
    anchor_block_ids: list[int] | None = None,
    retrieval_exclude_block_ids: list[int] | None = None,
) -> WorkingBlockBudgetDecision:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    _validate_budget("total_working_blocks", total_working_blocks)
    _validate_budget("recent_blocks", recent_blocks)
    _validate_budget("anchor_blocks", anchor_blocks)
    _validate_budget("retrieval_blocks", retrieval_blocks)

    issues: list[str] = []
    total_sequence_blocks = ceil(seq_len / block_size)

    requested_recent = list(
        range(
            max(0, total_sequence_blocks - recent_blocks),
            total_sequence_blocks,
        )
    )
    if len(requested_recent) < recent_blocks:
        _append_issue(issues, "recent_budget_exceeds_available_blocks")

    recent_reserved = set(requested_recent)

    requested_anchor: list[int] = []
    if anchor_blocks == 0:
        requested_anchor = []
    elif anchor_block_ids is not None:
        for block_id in anchor_block_ids:
            if block_id < 0 or block_id >= total_sequence_blocks:
                _append_issue(issues, "anchor_block_id_out_of_range")
                continue
            if block_id in recent_reserved:
                _append_issue(issues, "anchor_blocks_overlap_recent")
                continue
            requested_anchor.append(block_id)
            if len(requested_anchor) == anchor_blocks:
                break
        if len(requested_anchor) < anchor_blocks:
            _append_issue(issues, "insufficient_anchor_block_ids")
    else:
        for block_id in range(total_sequence_blocks):
            if block_id in recent_reserved:
                _append_issue(issues, "anchor_blocks_overlap_recent")
                continue
            requested_anchor.append(block_id)
            if len(requested_anchor) == anchor_blocks:
                break
        if len(requested_anchor) < anchor_blocks:
            _append_issue(issues, "insufficient_distinct_anchor_blocks")

    requested_anchor = _dedupe_in_order(requested_anchor)
    reserved_cold = set(requested_anchor) | recent_reserved
    retrieval_excluded = set(retrieval_exclude_block_ids or [])

    requested_retrieved: list[int] = []
    if retrieval_blocks > 0:
        for score in scored_blocks:
            block_id = score.block_id
            if block_id in reserved_cold:
                continue
            if block_id in retrieval_excluded:
                continue
            requested_retrieved.append(block_id)
            reserved_cold.add(block_id)
            if len(requested_retrieved) == retrieval_blocks:
                break

    if len(requested_retrieved) < retrieval_blocks:
        _append_issue(issues, "insufficient_retrieval_candidates")

    if recent_blocks + anchor_blocks + retrieval_blocks > total_working_blocks:
        _append_issue(issues, "requested_budgets_exceed_total_working_blocks")

    remaining_slots = total_working_blocks

    final_recent = _take_tail(requested_recent, remaining_slots)
    remaining_slots -= len(final_recent)

    final_anchor = _take_head(requested_anchor, remaining_slots)
    remaining_slots -= len(final_anchor)

    final_retrieved = _take_head(requested_retrieved, remaining_slots)

    if len(final_recent) < len(requested_recent):
        _append_issue(issues, "recent_budget_trimmed_by_total")
    if len(final_anchor) < len(requested_anchor):
        _append_issue(issues, "anchor_budget_trimmed_by_total")
    if len(final_retrieved) < len(requested_retrieved):
        _append_issue(issues, "retrieval_budget_trimmed_by_total")

    working_block_ids = sorted(
        set(final_recent) | set(final_anchor) | set(final_retrieved)
    )

    budget_ok = (
        len(final_recent) == recent_blocks
        and len(final_anchor) == anchor_blocks
        and len(final_retrieved) == retrieval_blocks
        and len(working_block_ids) <= total_working_blocks
    )

    fallback_reason = ",".join(issues) if issues else None

    return WorkingBlockBudgetDecision(
        seq_len=seq_len,
        block_size=block_size,
        budgets=WorkingBlockBudgets(
            total_working_blocks=total_working_blocks,
            recent_blocks=recent_blocks,
            anchor_blocks=anchor_blocks,
            retrieval_blocks=retrieval_blocks,
        ),
        selected=WorkingBlockSelection(
            recent_block_ids=final_recent,
            anchor_block_ids=final_anchor,
            retrieved_block_ids=final_retrieved,
            working_block_ids=working_block_ids,
        ),
        budget_ok=budget_ok,
        fallback_reason=fallback_reason,
    )
