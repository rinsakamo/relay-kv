from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RelayKVVramReservationStatus(str, Enum):
    OK = "ok"
    OVER_BUDGET = "over_budget"
    NO_KV_BUDGET = "no_kv_budget"


def _require_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


@dataclass(frozen=True)
class RelayKVVramReservation:
    total_vram_mib: int
    model_weights_reserved_mib: int
    tts_reserved_mib: int = 0
    asr_reserved_mib: int = 0
    avatar_reserved_mib: int = 0
    safety_margin_mib: int = 0
    other_reserved_mib: int = 0

    def __post_init__(self) -> None:
        if self.total_vram_mib <= 0:
            raise ValueError("total_vram_mib must be > 0")
        _require_non_negative(
            self.model_weights_reserved_mib,
            "model_weights_reserved_mib",
        )
        _require_non_negative(self.tts_reserved_mib, "tts_reserved_mib")
        _require_non_negative(self.asr_reserved_mib, "asr_reserved_mib")
        _require_non_negative(self.avatar_reserved_mib, "avatar_reserved_mib")
        _require_non_negative(self.safety_margin_mib, "safety_margin_mib")
        _require_non_negative(self.other_reserved_mib, "other_reserved_mib")

    @property
    def total_reserved_mib(self) -> int:
        return (
            self.model_weights_reserved_mib
            + self.tts_reserved_mib
            + self.asr_reserved_mib
            + self.avatar_reserved_mib
            + self.safety_margin_mib
            + self.other_reserved_mib
        )

    @property
    def residual_vram_mib(self) -> int:
        return self.total_vram_mib - self.total_reserved_mib

    def summary(self) -> dict:
        return {
            "total_vram_mib": self.total_vram_mib,
            "model_weights_reserved_mib": self.model_weights_reserved_mib,
            "tts_reserved_mib": self.tts_reserved_mib,
            "asr_reserved_mib": self.asr_reserved_mib,
            "avatar_reserved_mib": self.avatar_reserved_mib,
            "safety_margin_mib": self.safety_margin_mib,
            "other_reserved_mib": self.other_reserved_mib,
            "total_reserved_mib": self.total_reserved_mib,
            "residual_vram_mib": self.residual_vram_mib,
        }


@dataclass(frozen=True)
class RelayKVVramBudgetDecision:
    reservation: RelayKVVramReservation
    status: RelayKVVramReservationStatus
    available_working_kv_budget_mib: int
    total_reserved_mib: int
    residual_vram_mib: int
    warning: str | None = None

    def summary(self) -> dict:
        return {
            "reservation": self.reservation.summary(),
            "status": self.status.value,
            "available_working_kv_budget_mib": self.available_working_kv_budget_mib,
            "total_reserved_mib": self.total_reserved_mib,
            "residual_vram_mib": self.residual_vram_mib,
            "warning": self.warning,
        }


def build_vram_budget_decision(
    reservation: RelayKVVramReservation,
    *,
    min_working_kv_budget_mib: int = 0,
) -> RelayKVVramBudgetDecision:
    _require_non_negative(min_working_kv_budget_mib, "min_working_kv_budget_mib")

    total_reserved_mib = reservation.total_reserved_mib
    residual_vram_mib = reservation.residual_vram_mib
    available_working_kv_budget_mib = max(residual_vram_mib, 0)
    warning: str | None = None

    if total_reserved_mib > reservation.total_vram_mib:
        status = RelayKVVramReservationStatus.OVER_BUDGET
        available_working_kv_budget_mib = 0
        warning = "reserved VRAM exceeds total VRAM"
    elif available_working_kv_budget_mib <= min_working_kv_budget_mib:
        status = RelayKVVramReservationStatus.NO_KV_BUDGET
        warning = (
            "residual VRAM is at or below minimum working KV budget"
        )
    else:
        status = RelayKVVramReservationStatus.OK

    return RelayKVVramBudgetDecision(
        reservation=reservation,
        status=status,
        available_working_kv_budget_mib=available_working_kv_budget_mib,
        total_reserved_mib=total_reserved_mib,
        residual_vram_mib=residual_vram_mib,
        warning=warning,
    )
