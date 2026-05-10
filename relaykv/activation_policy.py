from __future__ import annotations

from dataclasses import dataclass


DISABLED_SHORT_CONTEXT = "DISABLED_SHORT_CONTEXT"
FULLKV_ACTIVE = "FULLKV_ACTIVE"
SHADOW_ONLY = "SHADOW_ONLY"
DEMOTION_CANDIDATE = "DEMOTION_CANDIDATE"
APPLY = "APPLY"
FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class RelayKVActivationDecision:
    activation_state: str
    relaykv_enabled: bool
    diagnostic_mode: bool
    seq_len: int
    min_relaykv_seq_len: int | None
    working_budget_tokens: int | None
    fullkv_within_budget: bool | None
    reason: str | None

    def summary(self) -> dict:
        return {
            "activation_state": self.activation_state,
            "relaykv_enabled": self.relaykv_enabled,
            "diagnostic_mode": self.diagnostic_mode,
            "seq_len": self.seq_len,
            "min_relaykv_seq_len": self.min_relaykv_seq_len,
            "working_budget_tokens": self.working_budget_tokens,
            "fullkv_within_budget": self.fullkv_within_budget,
            "reason": self.reason,
        }


def build_activation_decision(
    *,
    activation_mode: str,
    seq_len: int,
    min_relaykv_seq_len: int | None,
    working_budget_tokens: int | None,
    disable_relaykv_below_budget: bool,
) -> RelayKVActivationDecision:
    if activation_mode not in {"diagnostic", "practical"}:
        raise ValueError(f"Unsupported activation_mode: {activation_mode}")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if min_relaykv_seq_len is not None and min_relaykv_seq_len < 0:
        raise ValueError("min_relaykv_seq_len must be >= 0")
    if working_budget_tokens is not None and working_budget_tokens < 0:
        raise ValueError("working_budget_tokens must be >= 0")

    fullkv_within_budget = (
        seq_len <= working_budget_tokens
        if working_budget_tokens is not None
        else None
    )

    if activation_mode == "diagnostic":
        activation_state = (
            DEMOTION_CANDIDATE
            if working_budget_tokens is not None
            else SHADOW_ONLY
        )
        return RelayKVActivationDecision(
            activation_state=activation_state,
            relaykv_enabled=True,
            diagnostic_mode=True,
            seq_len=seq_len,
            min_relaykv_seq_len=min_relaykv_seq_len,
            working_budget_tokens=working_budget_tokens,
            fullkv_within_budget=fullkv_within_budget,
            reason="diagnostic_mode_forces_relaykv_path",
        )

    if min_relaykv_seq_len is not None and seq_len < min_relaykv_seq_len:
        return RelayKVActivationDecision(
            activation_state=DISABLED_SHORT_CONTEXT,
            relaykv_enabled=False,
            diagnostic_mode=False,
            seq_len=seq_len,
            min_relaykv_seq_len=min_relaykv_seq_len,
            working_budget_tokens=working_budget_tokens,
            fullkv_within_budget=fullkv_within_budget,
            reason="seq_len_below_min_relaykv_seq_len",
        )

    if disable_relaykv_below_budget and fullkv_within_budget is True:
        return RelayKVActivationDecision(
            activation_state=FULLKV_ACTIVE,
            relaykv_enabled=False,
            diagnostic_mode=False,
            seq_len=seq_len,
            min_relaykv_seq_len=min_relaykv_seq_len,
            working_budget_tokens=working_budget_tokens,
            fullkv_within_budget=fullkv_within_budget,
            reason="fullkv_within_working_budget",
        )

    activation_state = (
        DEMOTION_CANDIDATE
        if working_budget_tokens is not None
        else SHADOW_ONLY
    )
    return RelayKVActivationDecision(
        activation_state=activation_state,
        relaykv_enabled=True,
        diagnostic_mode=False,
        seq_len=seq_len,
        min_relaykv_seq_len=min_relaykv_seq_len,
        working_budget_tokens=working_budget_tokens,
        fullkv_within_budget=fullkv_within_budget,
        reason="practical_mode_activation_threshold_passed",
    )
