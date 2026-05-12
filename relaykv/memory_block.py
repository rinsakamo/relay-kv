from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KVClass(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    SHARED_PREFIX = "shared_prefix"
    EPHEMERAL = "ephemeral"


class ResidencyLevel(str, Enum):
    GPU = "gpu"
    RAM = "ram"
    SSD = "ssd"


class PrecisionLevel(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


class RopeStatus(str, Enum):
    NATIVE = "native"
    REBASED = "rebased"
    UNKNOWN = "unknown"


class CachePositionPolicy(str, Enum):
    PRESERVE = "preserve"
    RECOMPUTE = "recompute"
    REBASE = "rebase"


@dataclass(frozen=True)
class RelayKVMemoryBlock:
    logical_block_id: int
    logical_token_span: tuple[int, int]
    kv_class: KVClass
    residency_level: ResidencyLevel
    precision_level: PrecisionLevel
    rope_status: RopeStatus
    cache_position_policy: CachePositionPolicy
    protected: bool
    protection_reason: str | None
    protection_ttl: int | None
    demotion_priority: float
    retrieval_priority: float
    reuse_eligible: bool
    score_skip_reason: str | None
    materialization_cost_estimate: float
    last_retrieved_step: int | None
    retrieval_reuse_count: int

    def __post_init__(self) -> None:
        if self.logical_block_id < 0:
            raise ValueError("logical_block_id must be >= 0")

        if len(self.logical_token_span) != 2:
            raise ValueError("logical_token_span must contain exactly two integers")
        start, end = self.logical_token_span
        if start < 0:
            raise ValueError("logical_token_span start must be >= 0")
        if end <= start:
            raise ValueError("logical_token_span end must be > start")

        if self.retrieval_reuse_count < 0:
            raise ValueError("retrieval_reuse_count must be >= 0")

        if self.protection_ttl is not None and self.protection_ttl < 0:
            raise ValueError("protection_ttl must be >= 0")

    def summary(self) -> dict:
        return {
            "logical_block_id": self.logical_block_id,
            "logical_token_span": list(self.logical_token_span),
            "kv_class": self.kv_class.value,
            "residency_level": self.residency_level.value,
            "precision_level": self.precision_level.value,
            "rope_status": self.rope_status.value,
            "cache_position_policy": self.cache_position_policy.value,
            "protected": self.protected,
            "protection_reason": self.protection_reason,
            "protection_ttl": self.protection_ttl,
            "demotion_priority": self.demotion_priority,
            "retrieval_priority": self.retrieval_priority,
            "reuse_eligible": self.reuse_eligible,
            "score_skip_reason": self.score_skip_reason,
            "materialization_cost_estimate": self.materialization_cost_estimate,
            "last_retrieved_step": self.last_retrieved_step,
            "retrieval_reuse_count": self.retrieval_reuse_count,
        }
