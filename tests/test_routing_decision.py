from relaykv import ExecutionMode, RelayKVDecision


def test_routing_decision_summary_is_json_serializable_shape() -> None:
    decision = RelayKVDecision(
        execution_mode=ExecutionMode.RELAYKV_ROUTED,
        selected_active_block_ids=[1, 3, 5],
        protected_block_ids=[1],
        demotion_candidate_block_ids=[7, 9],
        demoted_block_ids=[9],
        retrieved_block_ids=[3, 5],
        prefetched_block_ids=[11],
        reused_block_ids=[1, 3],
        newly_retrieved_block_ids=[5],
        estimated_working_kv_bytes=8192,
        estimated_ram_swap_bytes=4096,
        estimated_ssd_read_bytes=2048,
        estimated_materialization_latency_ms=2.5,
        estimated_policy_compute_ms=0.4,
        estimated_attention_tokens_saved=128,
        estimated_net_benefit_ms=1.1,
        fallback_reason=None,
        apply_blocked_reason=None,
        shadow_compare_passed=True,
        selection_stability_ratio=0.875,
    )

    assert decision.summary() == {
        "execution_mode": "relaykv_routed",
        "selected_active_block_ids": [1, 3, 5],
        "protected_block_ids": [1],
        "demotion_candidate_block_ids": [7, 9],
        "demoted_block_ids": [9],
        "retrieved_block_ids": [3, 5],
        "prefetched_block_ids": [11],
        "reused_block_ids": [1, 3],
        "newly_retrieved_block_ids": [5],
        "estimated_working_kv_bytes": 8192,
        "estimated_ram_swap_bytes": 4096,
        "estimated_ssd_read_bytes": 2048,
        "estimated_materialization_latency_ms": 2.5,
        "estimated_policy_compute_ms": 0.4,
        "estimated_attention_tokens_saved": 128,
        "estimated_net_benefit_ms": 1.1,
        "fallback_reason": None,
        "apply_blocked_reason": None,
        "shadow_compare_passed": True,
        "selection_stability_ratio": 0.875,
    }
