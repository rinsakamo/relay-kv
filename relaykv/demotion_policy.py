from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelayKVDemotionDecision:
    keep_block_ids: list[int]
    drop_block_ids: list[int]
    eviction_excluded_block_ids: list[int]
    eviction_candidate_block_ids: list[int]
    demoted_block_ids: list[int]
    reason_labels_by_block: dict[str, list[str]]
    fallback_reason: str | None
    budget_ok: bool
    target_keep_blocks: int | None
    total_blocks: int
    demotion_strategy: str
    dry_run_only: bool = True
    demotion_applied: bool = False

    def summary(self) -> dict:
        return {
            "keep_block_ids": self.keep_block_ids,
            "drop_block_ids": self.drop_block_ids,
            "eviction_excluded_block_ids": self.eviction_excluded_block_ids,
            "eviction_candidate_block_ids": self.eviction_candidate_block_ids,
            "demoted_block_ids": self.demoted_block_ids,
            "reason_labels_by_block": self.reason_labels_by_block,
            "fallback_reason": self.fallback_reason,
            "budget_ok": self.budget_ok,
            "target_keep_blocks": self.target_keep_blocks,
            "total_blocks": self.total_blocks,
            "demotion_strategy": self.demotion_strategy,
            "dry_run_only": self.dry_run_only,
            "demotion_applied": self.demotion_applied,
        }


def _dedupe_in_order(block_ids: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for block_id in block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        deduped.append(block_id)
    return deduped


def _sort_unique(block_ids: list[int]) -> list[int]:
    return sorted(_dedupe_in_order(block_ids))


def build_demotion_decision(
    *,
    total_blocks: int,
    target_keep_blocks: int | None,
    recent_blocks: int,
    protect_boundary_blocks: int = 1,
    protect_prefix_blocks: int = 0,
    demotion_strategy: str = "oldest",
    allow_drop_recent: bool = False,
) -> RelayKVDemotionDecision:
    if total_blocks < 0:
        raise ValueError("total_blocks must be >= 0")
    if target_keep_blocks is not None and target_keep_blocks < 0:
        raise ValueError("target_keep_blocks must be >= 0")
    if recent_blocks < 0:
        raise ValueError("recent_blocks must be >= 0")
    if protect_boundary_blocks < 0:
        raise ValueError("protect_boundary_blocks must be >= 0")
    if protect_prefix_blocks < 0:
        raise ValueError("protect_prefix_blocks must be >= 0")
    if demotion_strategy != "oldest":
        raise ValueError(f"Unsupported demotion_strategy: {demotion_strategy}")

    all_block_ids = list(range(total_blocks))
    recent_count = min(total_blocks, recent_blocks)
    recent_block_ids = (
        list(range(total_blocks - recent_count, total_blocks))
        if recent_count > 0
        else []
    )
    protected_block_ids: list[int] = []
    if not allow_drop_recent:
        protected_block_ids.extend(recent_block_ids)

    boundary_block_ids: list[int] = []
    if protect_boundary_blocks > 0 and recent_block_ids:
        recent_start = recent_block_ids[0]
        boundary_start = max(0, recent_start - protect_boundary_blocks)
        boundary_block_ids = list(range(boundary_start, recent_start))
        protected_block_ids.extend(boundary_block_ids)

    prefix_block_ids = list(range(min(total_blocks, protect_prefix_blocks)))
    protected_block_ids.extend(prefix_block_ids)
    eviction_excluded_block_ids = _sort_unique(protected_block_ids)
    eviction_excluded_set = set(eviction_excluded_block_ids)
    eviction_candidate_block_ids = [
        block_id
        for block_id in all_block_ids
        if block_id not in eviction_excluded_set
    ]

    keep_block_ids: list[int]
    drop_block_ids: list[int]
    demoted_block_ids: list[int]
    budget_ok = True
    fallback_reason: str | None = None

    if target_keep_blocks is None:
        keep_block_ids = list(all_block_ids)
        drop_block_ids = []
        demoted_block_ids = []
    elif total_blocks <= target_keep_blocks:
        keep_block_ids = list(all_block_ids)
        drop_block_ids = []
        demoted_block_ids = []
        fallback_reason = "fullkv_within_budget"
    else:
        drop_target = total_blocks - target_keep_blocks
        demoted_block_ids = eviction_candidate_block_ids[:drop_target]
        if len(demoted_block_ids) < drop_target:
            fallback_reason = "insufficient_eviction_candidates"
            budget_ok = False
        drop_block_ids = list(demoted_block_ids)
        keep_block_ids = [
            block_id
            for block_id in all_block_ids
            if block_id not in set(drop_block_ids)
        ]
        if len(keep_block_ids) != target_keep_blocks:
            budget_ok = False

    keep_block_ids = _sort_unique(keep_block_ids)
    drop_block_ids = _sort_unique(drop_block_ids)
    demoted_block_ids = _sort_unique(demoted_block_ids)
    recent_block_ids_set = set(recent_block_ids)
    boundary_block_ids_set = set(boundary_block_ids)
    prefix_block_ids_set = set(prefix_block_ids)
    keep_block_ids_set = set(keep_block_ids)
    demoted_block_ids_set = set(demoted_block_ids)
    eviction_candidate_set = set(eviction_candidate_block_ids)

    reason_labels_by_block: dict[str, list[str]] = {}
    for block_id in all_block_ids:
        labels: list[str] = []
        if block_id in recent_block_ids_set and not allow_drop_recent:
            labels.append("RECENT_PROTECTED")
        if block_id in boundary_block_ids_set:
            labels.append("BOUNDARY_NEAR_RECENT")
        if block_id in prefix_block_ids_set:
            labels.append("PREFIX_PROTECTED")
        if block_id in eviction_candidate_set:
            labels.append("EVICTION_CANDIDATE")
        if block_id in demoted_block_ids_set:
            labels.append("DEMOTE_OLDEST")
        if block_id in keep_block_ids_set:
            if block_id in eviction_excluded_set:
                labels.append("KEEP_BY_PROTECTION")
            else:
                labels.append("KEEP_BY_BUDGET")
        if (
            fallback_reason == "insufficient_eviction_candidates"
            and block_id in keep_block_ids_set
            and block_id in eviction_candidate_set
            and block_id not in demoted_block_ids_set
        ):
            labels.append("FALLBACK_INSUFFICIENT_CANDIDATES")
        reason_labels_by_block[str(block_id)] = labels

    return RelayKVDemotionDecision(
        keep_block_ids=keep_block_ids,
        drop_block_ids=drop_block_ids,
        eviction_excluded_block_ids=eviction_excluded_block_ids,
        eviction_candidate_block_ids=_sort_unique(eviction_candidate_block_ids),
        demoted_block_ids=demoted_block_ids,
        reason_labels_by_block=reason_labels_by_block,
        fallback_reason=fallback_reason,
        budget_ok=budget_ok,
        target_keep_blocks=target_keep_blocks,
        total_blocks=total_blocks,
        demotion_strategy=demotion_strategy,
    )
