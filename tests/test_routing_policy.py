import json

from relaykv import (
    ExecutionMode,
    build_demotion_decision,
    build_routing_decision_from_demotion,
)


def test_routing_policy_maps_keep_and_drop_blocks_from_demotion_dry_run() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=6,
        target_keep_blocks=4,
        keep_block_ids=[1, 2, 4, 5],
        drop_block_ids=[0, 3],
        fallback_reason=None,
        eviction_excluded_block_ids=None,
        eviction_candidate_block_ids=None,
        estimated_working_kv_bytes=8192,
        estimated_ram_swap_bytes=0,
        estimated_ssd_read_bytes=0,
    )

    assert decision.selected_active_block_ids == [1, 2, 4, 5]
    assert decision.demotion_candidate_block_ids == [0, 1, 2, 3, 4, 5]
    assert decision.protected_block_ids == []
    assert decision.demoted_block_ids == [0, 3]
    assert decision.retrieved_block_ids == []
    assert decision.prefetched_block_ids == []
    assert decision.reused_block_ids == []
    assert decision.newly_retrieved_block_ids == []
    assert decision.estimated_working_kv_bytes == 8192
    assert decision.estimated_ram_swap_bytes == 0
    assert decision.estimated_ssd_read_bytes == 0


def test_routing_policy_uses_apply_vram_working_for_non_dry_run_demotion() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=4,
        target_keep_blocks=2,
        keep_block_ids=[2, 3],
        drop_block_ids=[0, 1],
        fallback_reason=None,
        dry_run_only=False,
    )

    assert decision.execution_mode is ExecutionMode.APPLY_VRAM_WORKING
    assert decision.apply_blocked_reason is None


def test_routing_policy_uses_shadow_only_for_dry_run_demotion() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=4,
        target_keep_blocks=2,
        keep_block_ids=[2, 3],
        drop_block_ids=[0, 1],
        fallback_reason=None,
        dry_run_only=True,
    )

    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "dry_run_only"


def test_routing_policy_preserves_protected_blocks_and_excludes_them_from_candidates() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=6,
        target_keep_blocks=3,
        keep_block_ids=[2, 4, 5],
        drop_block_ids=[0, 1, 3],
        fallback_reason=None,
        eviction_excluded_block_ids=[4, 5],
        eviction_candidate_block_ids=[0, 1, 2, 3],
    )

    assert decision.protected_block_ids == [4, 5]
    assert decision.demotion_candidate_block_ids == [0, 1, 2, 3]


def test_routing_policy_fullkv_within_budget_maps_to_fullkv_gpu() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=3,
        target_keep_blocks=4,
        keep_block_ids=[0, 1, 2],
        drop_block_ids=[],
        fallback_reason="fullkv_within_budget",
        dry_run_only=False,
    )

    assert decision.execution_mode is ExecutionMode.FULLKV_GPU
    assert decision.apply_blocked_reason is None


def test_routing_policy_propagates_non_fullkv_fallback_reason() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=5,
        target_keep_blocks=1,
        keep_block_ids=[2, 3, 4],
        drop_block_ids=[0, 1],
        fallback_reason="insufficient_eviction_candidates",
        dry_run_only=True,
    )

    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
    assert decision.apply_blocked_reason == "insufficient_eviction_candidates"
    assert decision.fallback_reason == "insufficient_eviction_candidates"


def test_routing_policy_summary_is_json_serializable() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=3,
        target_keep_blocks=1,
        keep_block_ids=[1],
        drop_block_ids=[0],
        fallback_reason=None,
        eviction_excluded_block_ids=[2],
    )

    summary = decision.summary()
    assert summary["execution_mode"] == "shadow_only"
    assert summary["protected_block_ids"] == [2]
    assert summary["demotion_candidate_block_ids"] == [0, 1]
    assert summary["estimated_materialization_latency_ms"] is None
    assert summary["estimated_policy_compute_ms"] is None
    assert summary["estimated_attention_tokens_saved"] is None
    assert summary["estimated_net_benefit_ms"] is None
    assert json.loads(json.dumps(summary)) == summary


def test_routing_policy_fullkv_within_budget_clears_blocker_even_in_dry_run() -> None:
    decision = build_routing_decision_from_demotion(
        total_blocks=3,
        target_keep_blocks=4,
        keep_block_ids=[0, 1, 2],
        drop_block_ids=[],
        fallback_reason="fullkv_within_budget",
        dry_run_only=True,
    )

    assert decision.execution_mode is ExecutionMode.FULLKV_GPU
    assert decision.apply_blocked_reason is None


def test_routing_policy_wraps_existing_demotion_decision() -> None:
    demotion = build_demotion_decision(
        total_blocks=6,
        target_keep_blocks=4,
        recent_blocks=2,
        protect_boundary_blocks=1,
        protect_prefix_blocks=0,
        demotion_strategy="oldest",
    )

    decision = build_routing_decision_from_demotion(
        total_blocks=demotion.total_blocks,
        target_keep_blocks=demotion.target_keep_blocks,
        keep_block_ids=demotion.keep_block_ids,
        drop_block_ids=demotion.drop_block_ids,
        fallback_reason=demotion.fallback_reason,
        eviction_excluded_block_ids=demotion.eviction_excluded_block_ids,
        eviction_candidate_block_ids=demotion.eviction_candidate_block_ids,
        dry_run_only=demotion.dry_run_only,
    )

    assert decision.selected_active_block_ids == demotion.keep_block_ids
    assert decision.protected_block_ids == demotion.eviction_excluded_block_ids
    assert decision.demotion_candidate_block_ids == demotion.eviction_candidate_block_ids
    assert decision.demoted_block_ids == demotion.drop_block_ids
    assert decision.execution_mode is ExecutionMode.SHADOW_ONLY
