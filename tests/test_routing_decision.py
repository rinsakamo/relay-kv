import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
        "approval_required": False,
        "approval_reason": None,
        "proposed_fallback_mode": None,
        "user_visible_message": None,
        "fallback_if_denied": None,
    }


def test_routing_decision_serializes_user_gated_fallback_modes() -> None:
    decision = RelayKVDecision(
        execution_mode=ExecutionMode.PROPOSE_FALLBACK,
        selected_active_block_ids=[],
        protected_block_ids=[],
        demotion_candidate_block_ids=[],
        demoted_block_ids=[],
        retrieved_block_ids=[],
        prefetched_block_ids=[],
        reused_block_ids=[],
        newly_retrieved_block_ids=[],
        estimated_working_kv_bytes=None,
        estimated_ram_swap_bytes=None,
        estimated_ssd_read_bytes=None,
        estimated_materialization_latency_ms=None,
        estimated_policy_compute_ms=None,
        estimated_attention_tokens_saved=None,
        estimated_net_benefit_ms=None,
        fallback_reason="need_approval_for_ram_backed_mode",
        apply_blocked_reason="approval_required",
        shadow_compare_passed=None,
        selection_stability_ratio=None,
        approval_required=True,
        approval_reason="ram-backed fallback requires approval",
        proposed_fallback_mode=ExecutionMode.FALLBACK_FULLKV_RAM,
        user_visible_message="May I switch to a slower RAM-backed mode?",
        fallback_if_denied=ExecutionMode.FALLBACK_RECENT_ANCHOR,
    )

    assert decision.summary()["execution_mode"] == "propose_fallback"
    assert decision.summary()["approval_required"] is True
    assert decision.summary()["proposed_fallback_mode"] == "fallback_fullkv_ram"
    assert decision.summary()["fallback_if_denied"] == "fallback_recent_anchor"
