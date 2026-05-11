from relaykv import RelayKVVramBudgetDecision
from scripts.run_relaykv_pipeline import resolve_effective_target_keep_blocks


def make_vram_budget_decision(
    *,
    derived_target_keep_blocks: int = 18,
    budget_ok: bool = True,
) -> RelayKVVramBudgetDecision:
    return RelayKVVramBudgetDecision(
        global_residual_vram_mib=None,
        global_working_kv_budget_mib=128.0,
        target_concurrent_requests=2,
        request_working_kv_budget_mib=64.0,
        allocation_policy="equal_share",
        kv_dtype_bytes=2,
        num_layers=28,
        num_kv_heads=2,
        head_dim=128,
        block_size=128,
        kv_bytes_per_token=28672,
        kv_bytes_per_block=3670016,
        derived_target_keep_blocks=derived_target_keep_blocks,
        budget_ok=budget_ok,
        fallback_reason=None if budget_ok else "request_budget_smaller_than_one_block",
        dry_run_only=True,
    )


def test_explicit_target_keep_blocks_wins() -> None:
    resolution = resolve_effective_target_keep_blocks(
        explicit_target_keep_blocks=3,
        vram_budget_decision=make_vram_budget_decision(derived_target_keep_blocks=18),
    )

    assert resolution == {
        "effective_target_keep_blocks": 3,
        "target_keep_blocks_source": "explicit_cli",
        "fallback_reason": None,
        "vram_budget_to_demotion_connected": False,
    }


def test_vram_budget_target_is_used_when_explicit_is_absent() -> None:
    resolution = resolve_effective_target_keep_blocks(
        explicit_target_keep_blocks=None,
        vram_budget_decision=make_vram_budget_decision(derived_target_keep_blocks=18),
    )

    assert resolution == {
        "effective_target_keep_blocks": 18,
        "target_keep_blocks_source": "vram_budget",
        "fallback_reason": None,
        "vram_budget_to_demotion_connected": True,
    }


def test_resolution_unset_when_vram_budget_is_not_enabled() -> None:
    resolution = resolve_effective_target_keep_blocks(
        explicit_target_keep_blocks=None,
        vram_budget_decision=None,
    )

    assert resolution == {
        "effective_target_keep_blocks": None,
        "target_keep_blocks_source": "unset",
        "fallback_reason": "vram_budget_not_enabled",
        "vram_budget_to_demotion_connected": False,
    }


def test_resolution_unset_when_vram_budget_is_not_ok() -> None:
    resolution = resolve_effective_target_keep_blocks(
        explicit_target_keep_blocks=None,
        vram_budget_decision=make_vram_budget_decision(
            derived_target_keep_blocks=0,
            budget_ok=False,
        ),
    )

    assert resolution == {
        "effective_target_keep_blocks": None,
        "target_keep_blocks_source": "unset",
        "fallback_reason": "vram_budget_not_ok",
        "vram_budget_to_demotion_connected": False,
    }
