from __future__ import annotations

from dataclasses import dataclass
from math import floor


MIB_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RelayKVVramBudgetDecision:
    global_residual_vram_mib: float | None
    global_working_kv_budget_mib: float
    target_concurrent_requests: int
    request_working_kv_budget_mib: float
    allocation_policy: str
    kv_dtype_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    kv_bytes_per_token: int
    kv_bytes_per_block: int
    derived_target_keep_blocks: int
    budget_ok: bool
    fallback_reason: str | None
    dry_run_only: bool = True

    def summary(self) -> dict:
        return {
            "global_residual_vram_mib": self.global_residual_vram_mib,
            "global_working_kv_budget_mib": self.global_working_kv_budget_mib,
            "target_concurrent_requests": self.target_concurrent_requests,
            "request_working_kv_budget_mib": self.request_working_kv_budget_mib,
            "allocation_policy": self.allocation_policy,
            "kv_dtype_bytes": self.kv_dtype_bytes,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_bytes_per_block": self.kv_bytes_per_block,
            "derived_target_keep_blocks": self.derived_target_keep_blocks,
            "budget_ok": self.budget_ok,
            "fallback_reason": self.fallback_reason,
            "dry_run_only": self.dry_run_only,
        }


def build_vram_budget_decision(
    *,
    global_working_kv_budget_mib: float,
    target_concurrent_requests: int,
    allocation_policy: str,
    kv_dtype_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    global_residual_vram_mib: float | None = None,
) -> RelayKVVramBudgetDecision:
    if allocation_policy != "equal_share":
        raise ValueError(f"Unsupported allocation_policy: {allocation_policy}")
    if global_working_kv_budget_mib <= 0:
        raise ValueError("global_working_kv_budget_mib must be > 0")
    if global_residual_vram_mib is not None and global_residual_vram_mib <= 0:
        raise ValueError("global_residual_vram_mib must be > 0")
    if target_concurrent_requests <= 0:
        raise ValueError("target_concurrent_requests must be > 0")
    if kv_dtype_bytes <= 0:
        raise ValueError("kv_dtype_bytes must be > 0")
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be > 0")
    if head_dim <= 0:
        raise ValueError("head_dim must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    request_working_kv_budget_mib = (
        global_working_kv_budget_mib / target_concurrent_requests
    )
    kv_bytes_per_token = (
        2 * kv_dtype_bytes * num_layers * num_kv_heads * head_dim
    )
    kv_bytes_per_block = kv_bytes_per_token * block_size
    request_budget_bytes = request_working_kv_budget_mib * MIB_BYTES
    derived_target_keep_blocks = floor(request_budget_bytes / kv_bytes_per_block)

    budget_ok = True
    fallback_reason: str | None = None
    if (
        global_residual_vram_mib is not None
        and global_working_kv_budget_mib > global_residual_vram_mib
    ):
        budget_ok = False
        fallback_reason = "working_budget_exceeds_residual_vram"
    elif derived_target_keep_blocks < 1:
        budget_ok = False
        fallback_reason = "request_budget_smaller_than_one_block"

    return RelayKVVramBudgetDecision(
        global_residual_vram_mib=global_residual_vram_mib,
        global_working_kv_budget_mib=global_working_kv_budget_mib,
        target_concurrent_requests=target_concurrent_requests,
        request_working_kv_budget_mib=request_working_kv_budget_mib,
        allocation_policy=allocation_policy,
        kv_dtype_bytes=kv_dtype_bytes,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_bytes_per_block=kv_bytes_per_block,
        derived_target_keep_blocks=derived_target_keep_blocks,
        budget_ok=budget_ok,
        fallback_reason=fallback_reason,
    )
