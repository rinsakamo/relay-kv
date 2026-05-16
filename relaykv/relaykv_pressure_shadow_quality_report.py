from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class RelayKVShadowQualitySummary:
    mean_abs_diff: float | None
    max_abs_diff: float | None
    coverage_ratio: float | None
    working_ratio: float | None
    seq_len: int | None
    layer_idx: int | None
    prompt_type: str | None
    candidate_k_len: int | None
    cold_k_len: int | None
    working_k_len: int | None
    full_k_len: int | None

    def summary(self) -> dict[str, Any]:
        return {
            "mean_abs_diff": self.mean_abs_diff,
            "max_abs_diff": self.max_abs_diff,
            "coverage_ratio": self.coverage_ratio,
            "working_ratio": self.working_ratio,
            "seq_len": self.seq_len,
            "layer_idx": self.layer_idx,
            "prompt_type": self.prompt_type,
            "candidate_k_len": self.candidate_k_len,
            "cold_k_len": self.cold_k_len,
            "working_k_len": self.working_k_len,
            "full_k_len": self.full_k_len,
        }


@dataclass(frozen=True)
class RelayKVPressureShadowQualityReport:
    shadow_quality_test_recommended: bool
    pressure_reason: str | None
    quality_summary: dict[str, Any] | None
    quality_status: str
    mean_abs_diff_threshold: float
    max_abs_diff_threshold: float
    notes: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "shadow_quality_test_recommended": self.shadow_quality_test_recommended,
            "pressure_reason": self.pressure_reason,
            "quality_summary": (
                dict(self.quality_summary)
                if isinstance(self.quality_summary, dict)
                else None
            ),
            "quality_status": self.quality_status,
            "mean_abs_diff_threshold": self.mean_abs_diff_threshold,
            "max_abs_diff_threshold": self.max_abs_diff_threshold,
            "notes": list(self.notes),
        }


def _build_quality_summary(
    relaykv_pipeline_payload: dict[str, Any],
) -> RelayKVShadowQualitySummary | None:
    attention_compare = relaykv_pipeline_payload.get("attention_compare")
    if not isinstance(attention_compare, dict):
        return None

    return RelayKVShadowQualitySummary(
        mean_abs_diff=_as_float(attention_compare.get("mean_abs_diff")),
        max_abs_diff=_as_float(attention_compare.get("max_abs_diff")),
        coverage_ratio=_as_float(relaykv_pipeline_payload.get("coverage_ratio")),
        working_ratio=_as_float(relaykv_pipeline_payload.get("working_ratio")),
        seq_len=_as_int(relaykv_pipeline_payload.get("seq_len_actual")),
        layer_idx=_as_int(relaykv_pipeline_payload.get("layer_idx")),
        prompt_type=(
            str(relaykv_pipeline_payload.get("prompt_type"))
            if relaykv_pipeline_payload.get("prompt_type") is not None
            else None
        ),
        candidate_k_len=_as_int(relaykv_pipeline_payload.get("candidate_k_len")),
        cold_k_len=_as_int(relaykv_pipeline_payload.get("cold_k_len")),
        working_k_len=_as_int(relaykv_pipeline_payload.get("working_k_len")),
        full_k_len=_as_int(relaykv_pipeline_payload.get("full_k_len")),
    )


def _expected_pressure_context_tokens(
    relaystack_hf_report_payload: dict[str, Any],
) -> int | None:
    shadow_inputs = relaystack_hf_report_payload.get(
        "relaykv_shadow_quality_test_inputs"
    )
    if not isinstance(shadow_inputs, dict):
        shadow_inputs = {}
    return (
        _as_int(relaystack_hf_report_payload.get("first_failed_context_tokens"))
        or _as_int(shadow_inputs.get("first_failed_context_tokens"))
        or _as_int(relaystack_hf_report_payload.get("max_ok_context_tokens"))
        or _as_int(shadow_inputs.get("max_ok_context_tokens"))
        or _as_int(relaystack_hf_report_payload.get("observed_max_context_tokens"))
        or _as_int(shadow_inputs.get("observed_max_context_tokens"))
    )


def build_relaykv_pressure_shadow_quality_report(
    *,
    relaystack_hf_report_payload: dict[str, Any],
    relaykv_pipeline_payload: dict[str, Any],
    mean_abs_diff_threshold: float = 0.01,
    max_abs_diff_threshold: float = 0.10,
) -> RelayKVPressureShadowQualityReport:
    recommended = bool(
        relaystack_hf_report_payload.get("relaykv_shadow_quality_test_recommended")
    )
    pressure_reason = relaystack_hf_report_payload.get(
        "relaykv_shadow_quality_test_reason"
    )
    pressure_reason = str(pressure_reason) if pressure_reason is not None else None
    expected_context_tokens = _expected_pressure_context_tokens(
        relaystack_hf_report_payload
    )

    quality = _build_quality_summary(relaykv_pipeline_payload)
    quality_summary = quality.summary() if quality is not None else None

    notes: list[str] = []
    if recommended:
        notes.append("shadow_quality_test_recommended_from_pressure_signals")
    else:
        notes.append("shadow_quality_test_not_recommended")

    if quality is None:
        notes.append("relaykv_pipeline_attention_compare_missing")
        quality_status = (
            "recommended_quality_unknown"
            if recommended
            else "not_recommended"
        )
    elif not recommended:
        notes.append("quality_metrics_present_but_pressure_not_recommended")
        quality_status = "not_recommended"
    elif (
        expected_context_tokens is not None
        and quality.seq_len is not None
        and expected_context_tokens != quality.seq_len
    ):
        notes.append(
            "pressure_context_mismatch:"
            f"expected={expected_context_tokens}:observed={quality.seq_len}"
        )
        quality_status = "recommended_quality_context_mismatch"
    elif quality.mean_abs_diff is None or quality.max_abs_diff is None:
        notes.append("quality_metrics_incomplete")
        quality_status = "recommended_quality_observed"
    elif (
        quality.mean_abs_diff <= mean_abs_diff_threshold
        and quality.max_abs_diff <= max_abs_diff_threshold
    ):
        notes.append("quality_within_threshold")
        quality_status = "recommended_quality_within_threshold"
    else:
        notes.append("quality_exceeds_threshold")
        quality_status = "recommended_quality_exceeds_threshold"

    return RelayKVPressureShadowQualityReport(
        shadow_quality_test_recommended=recommended,
        pressure_reason=pressure_reason,
        quality_summary=quality_summary,
        quality_status=quality_status,
        mean_abs_diff_threshold=mean_abs_diff_threshold,
        max_abs_diff_threshold=max_abs_diff_threshold,
        notes=notes,
    )
