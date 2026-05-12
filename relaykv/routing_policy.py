from __future__ import annotations

from .routing_decision import ExecutionMode, RelayKVDecision


def build_routing_decision_from_demotion(
    *,
    total_blocks: int,
    target_keep_blocks: int | None,
    keep_block_ids: list[int],
    drop_block_ids: list[int],
    fallback_reason: str | None,
    eviction_excluded_block_ids: list[int] | None = None,
    eviction_candidate_block_ids: list[int] | None = None,
    dry_run_only: bool = True,
    estimated_working_kv_bytes: int | None = None,
    estimated_ram_swap_bytes: int | None = None,
    estimated_ssd_read_bytes: int | None = None,
) -> RelayKVDecision:
    if total_blocks < 0:
        raise ValueError("total_blocks must be >= 0")
    if target_keep_blocks is not None and target_keep_blocks < 0:
        raise ValueError("target_keep_blocks must be >= 0")

    all_block_ids = list(range(total_blocks))
    keep_block_ids = list(keep_block_ids)
    drop_block_ids = list(drop_block_ids)
    protected_block_ids = (
        list(eviction_excluded_block_ids)
        if eviction_excluded_block_ids is not None
        else []
    )
    protected_block_id_set = set(protected_block_ids)
    if eviction_candidate_block_ids is not None:
        demotion_candidate_block_ids = [
            block_id
            for block_id in eviction_candidate_block_ids
            if block_id not in protected_block_id_set
        ]
    else:
        demotion_candidate_block_ids = [
            block_id
            for block_id in all_block_ids
            if block_id not in protected_block_id_set
        ]

    if fallback_reason == "fullkv_within_budget":
        execution_mode = ExecutionMode.FULLKV_GPU
    else:
        execution_mode = ExecutionMode.SHADOW_ONLY

    apply_blocked_reason: str | None = None
    if fallback_reason is not None and fallback_reason != "fullkv_within_budget":
        apply_blocked_reason = fallback_reason
    elif fallback_reason == "fullkv_within_budget":
        apply_blocked_reason = None
    elif dry_run_only:
        apply_blocked_reason = "dry_run_only"

    return RelayKVDecision(
        execution_mode=execution_mode,
        selected_active_block_ids=keep_block_ids,
        protected_block_ids=protected_block_ids,
        demotion_candidate_block_ids=demotion_candidate_block_ids,
        demoted_block_ids=drop_block_ids,
        retrieved_block_ids=[],
        prefetched_block_ids=[],
        reused_block_ids=[],
        newly_retrieved_block_ids=[],
        estimated_working_kv_bytes=estimated_working_kv_bytes,
        estimated_ram_swap_bytes=estimated_ram_swap_bytes,
        estimated_ssd_read_bytes=estimated_ssd_read_bytes,
        estimated_materialization_latency_ms=None,
        estimated_policy_compute_ms=None,
        estimated_attention_tokens_saved=None,
        estimated_net_benefit_ms=None,
        fallback_reason=fallback_reason,
        apply_blocked_reason=apply_blocked_reason,
        shadow_compare_passed=None,
        selection_stability_ratio=None,
    )
