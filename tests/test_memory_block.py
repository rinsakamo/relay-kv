import pytest

from relaykv import (
    CachePositionPolicy,
    KVClass,
    PrecisionLevel,
    RelayKVMemoryBlock,
    ResidencyLevel,
    RopeStatus,
)


def test_memory_block_summary_is_json_serializable_shape() -> None:
    block = RelayKVMemoryBlock(
        logical_block_id=7,
        logical_token_span=(128, 256),
        kv_class=KVClass.SHARED_PREFIX,
        residency_level=ResidencyLevel.RAM,
        precision_level=PrecisionLevel.BF16,
        rope_status=RopeStatus.NATIVE,
        cache_position_policy=CachePositionPolicy.PRESERVE,
        protected=True,
        protection_reason="active_prefix",
        protection_ttl=3,
        demotion_priority=0.2,
        retrieval_priority=0.9,
        reuse_eligible=True,
        score_skip_reason=None,
        materialization_cost_estimate=1.5,
        last_retrieved_step=42,
        retrieval_reuse_count=5,
    )

    assert block.summary() == {
        "logical_block_id": 7,
        "logical_token_span": [128, 256],
        "kv_class": "shared_prefix",
        "residency_level": "ram",
        "precision_level": "bf16",
        "rope_status": "native",
        "cache_position_policy": "preserve",
        "protected": True,
        "protection_reason": "active_prefix",
        "protection_ttl": 3,
        "demotion_priority": 0.2,
        "retrieval_priority": 0.9,
        "reuse_eligible": True,
        "score_skip_reason": None,
        "materialization_cost_estimate": 1.5,
        "last_retrieved_step": 42,
        "retrieval_reuse_count": 5,
    }


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("logical_block_id", {"logical_block_id": -1}),
        ("logical_token_span start", {"logical_token_span": (-1, 8)}),
        ("logical_token_span end", {"logical_token_span": (8, 8)}),
        ("retrieval_reuse_count", {"retrieval_reuse_count": -1}),
        ("protection_ttl", {"protection_ttl": -1}),
    ],
)
def test_memory_block_rejects_invalid_inputs(
    field_name: str,
    kwargs: dict,
) -> None:
    base_kwargs = {
        "logical_block_id": 1,
        "logical_token_span": (0, 16),
        "kv_class": KVClass.PREFILL,
        "residency_level": ResidencyLevel.GPU,
        "precision_level": PrecisionLevel.FP16,
        "rope_status": RopeStatus.UNKNOWN,
        "cache_position_policy": CachePositionPolicy.RECOMPUTE,
        "protected": False,
        "protection_reason": None,
        "protection_ttl": None,
        "demotion_priority": 0.5,
        "retrieval_priority": 0.5,
        "reuse_eligible": True,
        "score_skip_reason": None,
        "materialization_cost_estimate": 0.25,
        "last_retrieved_step": None,
        "retrieval_reuse_count": 0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=field_name):
        RelayKVMemoryBlock(**base_kwargs)
