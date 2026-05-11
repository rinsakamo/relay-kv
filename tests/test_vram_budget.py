import pytest

from relaykv import build_vram_budget_decision


def test_build_vram_budget_decision_equal_share_ok() -> None:
    decision = build_vram_budget_decision(
        global_working_kv_budget_mib=128.0,
        target_concurrent_requests=2,
        allocation_policy="equal_share",
        kv_dtype_bytes=2,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        block_size=128,
        global_residual_vram_mib=256.0,
    )

    assert decision.request_working_kv_budget_mib == 64.0
    assert decision.kv_bytes_per_token == 12288
    assert decision.kv_bytes_per_block == 1572864
    assert decision.derived_target_keep_blocks == 42
    assert decision.budget_ok is True
    assert decision.fallback_reason is None
    assert decision.dry_run_only is True
    assert decision.summary()["derived_target_keep_blocks"] == 42


def test_build_vram_budget_decision_falls_back_when_request_budget_too_small() -> None:
    decision = build_vram_budget_decision(
        global_working_kv_budget_mib=0.5,
        target_concurrent_requests=2,
        allocation_policy="equal_share",
        kv_dtype_bytes=2,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        block_size=128,
    )

    assert decision.derived_target_keep_blocks == 0
    assert decision.budget_ok is False
    assert decision.fallback_reason == "request_budget_smaller_than_one_block"


def test_build_vram_budget_decision_falls_back_when_budget_exceeds_residual() -> None:
    decision = build_vram_budget_decision(
        global_working_kv_budget_mib=128.0,
        target_concurrent_requests=2,
        allocation_policy="equal_share",
        kv_dtype_bytes=2,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        block_size=128,
        global_residual_vram_mib=64.0,
    )

    assert decision.budget_ok is False
    assert decision.fallback_reason == "working_budget_exceeds_residual_vram"
    assert decision.derived_target_keep_blocks == 42


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("allocation_policy", {"allocation_policy": "weighted"}),
        ("global_working_kv_budget_mib", {"global_working_kv_budget_mib": 0.0}),
        ("global_residual_vram_mib", {"global_residual_vram_mib": 0.0}),
        ("target_concurrent_requests", {"target_concurrent_requests": 0}),
        ("kv_dtype_bytes", {"kv_dtype_bytes": 0}),
        ("num_layers", {"num_layers": 0}),
        ("num_kv_heads", {"num_kv_heads": 0}),
        ("head_dim", {"head_dim": 0}),
        ("block_size", {"block_size": 0}),
    ],
)
def test_build_vram_budget_decision_rejects_invalid_inputs(
    field_name: str,
    kwargs: dict,
) -> None:
    base_kwargs = {
        "global_working_kv_budget_mib": 128.0,
        "target_concurrent_requests": 2,
        "allocation_policy": "equal_share",
        "kv_dtype_bytes": 2,
        "num_layers": 24,
        "num_kv_heads": 2,
        "head_dim": 64,
        "block_size": 128,
        "global_residual_vram_mib": 256.0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=field_name):
        build_vram_budget_decision(**base_kwargs)
