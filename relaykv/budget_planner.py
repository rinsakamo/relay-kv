from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class BudgetPlan:
    available_kv_budget_mib: float
    kv_working_budget_tokens: int
    kv_working_budget_source: str
    recent_window_tokens: int
    anchor_blocks: int
    budget_block_size: int
    anchor_budget_tokens: int
    retrieval_budget_tokens: int
    retrieval_block_budget: int
    retrieval_top_k_requested: int
    retrieval_top_k_effective: int
    budget_overflow: bool
    budget_policy_reason: str

    def summary(self) -> dict:
        return asdict(self)


def plan_budget(
    *,
    available_kv_budget_mib: float = 0.0,
    kv_working_budget_tokens: int = 0,
    kv_bytes_per_token: Optional[int] = None,
    recent_window_tokens: int = 0,
    anchor_blocks: int = 0,
    budget_block_size: int = 128,
    retrieval_top_k_requested: int = 0,
    fallback_working_budget_tokens: int = 0,
) -> BudgetPlan:
    """Plan RelayKV working-set budget metadata without changing retrieval behavior."""

    if kv_working_budget_tokens > 0:
        working_budget = int(kv_working_budget_tokens)
        source = "explicit_working_budget_tokens"
    elif available_kv_budget_mib > 0:
        if kv_bytes_per_token and kv_bytes_per_token > 0:
            budget_bytes = int(available_kv_budget_mib * 1024 * 1024)
            working_budget = budget_bytes // kv_bytes_per_token
            source = "estimated_from_available_kv_budget_mib"
        elif fallback_working_budget_tokens > 0:
            working_budget = int(fallback_working_budget_tokens)
            source = "fallback_working_budget_tokens_missing_kv_bytes_per_token"
        else:
            working_budget = 0
            source = "missing_kv_bytes_per_token_for_available_kv_budget_mib"
    else:
        working_budget = int(fallback_working_budget_tokens)
        source = "fallback_working_budget_tokens"

    working_budget = max(working_budget, 0)
    block_size = max(int(budget_block_size), 1)
    recent = min(max(int(recent_window_tokens), 0), working_budget)
    anchor_requested = max(int(anchor_blocks), 0) * block_size
    anchor_budget = min(anchor_requested, max(working_budget - recent, 0))
    retrieval_budget = max(working_budget - recent - anchor_budget, 0)
    retrieval_block_budget = retrieval_budget // block_size
    retrieval_top_k = max(int(retrieval_top_k_requested), 0)
    retrieval_top_k_effective = min(retrieval_top_k, retrieval_block_budget)

    overflow = False
    reason = source
    if working_budget <= 0:
        overflow = True
        reason = source
    elif int(recent_window_tokens) > recent:
        overflow = True
        reason = "recent_window_clipped_to_working_budget"
    elif anchor_requested > anchor_budget:
        overflow = True
        reason = "anchor_budget_clipped_after_recent_window"
    elif retrieval_top_k > 0 and retrieval_block_budget <= 0:
        overflow = True
        reason = "no_retrieval_room_after_recent_and_anchor"
    elif retrieval_top_k > retrieval_top_k_effective:
        overflow = True
        reason = "retrieval_top_k_clipped_to_remaining_budget"

    return BudgetPlan(
        available_kv_budget_mib=float(available_kv_budget_mib),
        kv_working_budget_tokens=working_budget,
        kv_working_budget_source=source,
        recent_window_tokens=recent,
        anchor_blocks=max(int(anchor_blocks), 0),
        budget_block_size=block_size,
        anchor_budget_tokens=anchor_budget,
        retrieval_budget_tokens=retrieval_budget,
        retrieval_block_budget=retrieval_block_budget,
        retrieval_top_k_requested=retrieval_top_k,
        retrieval_top_k_effective=retrieval_top_k_effective,
        budget_overflow=overflow,
        budget_policy_reason=reason,
    )
