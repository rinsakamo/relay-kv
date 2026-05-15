from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelayStackHFContextRowSummary:
    target_context_tokens: int | None
    input_tokens: int | None
    ok: bool
    error_type: str | None
    error_message: str | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    tokens_per_sec_new: float | None
    new_tokens: int | None
    elapsed_sec: float | None

    def summary(self) -> dict[str, Any]:
        return {
            "target_context_tokens": self.target_context_tokens,
            "input_tokens": self.input_tokens,
            "ok": self.ok,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "peak_allocated_mib": self.peak_allocated_mib,
            "peak_reserved_mib": self.peak_reserved_mib,
            "tokens_per_sec_new": self.tokens_per_sec_new,
            "new_tokens": self.new_tokens,
            "elapsed_sec": self.elapsed_sec,
        }


@dataclass(frozen=True)
class RelayStackHFSmokeReport:
    hf_script: str | None
    hf_model: str | None
    hf_load_ok: bool
    hf_env_torch: str | None
    hf_env_cuda_version_torch: str | None
    hf_env_transformers: str | None
    hf_model_config: dict[str, Any] | None
    context_rows: list[RelayStackHFContextRowSummary]
    final_routing_state: str | None
    relaymem_apply_allowed: bool
    relaykv_routing_allowed: bool
    final_fallback_reason: str | None
    final_blocking_reasons: list[str]
    relaykv_vram_status: str | None
    relaykv_available_working_kv_budget_mib: int | None
    summary_available_working_kv_budget_mib: int | None
    total_vram_mib: int | None
    max_ok_context_tokens: int | None
    first_failed_context_tokens: int | None
    any_oom: bool
    max_peak_reserved_mib: float | None
    max_peak_allocated_mib: float | None
    measured_vram_pressure_level: str
    relaystack_predicted_kv_budget_ok: bool
    hf_all_lengths_ok: bool
    observed_max_context_tokens: int | None
    report_notes: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "hf_script": self.hf_script,
            "hf_model": self.hf_model,
            "hf_load_ok": self.hf_load_ok,
            "hf_env_torch": self.hf_env_torch,
            "hf_env_cuda_version_torch": self.hf_env_cuda_version_torch,
            "hf_env_transformers": self.hf_env_transformers,
            "hf_model_config": self.hf_model_config,
            "context_rows": [row.summary() for row in self.context_rows],
            "final_routing_state": self.final_routing_state,
            "relaymem_apply_allowed": self.relaymem_apply_allowed,
            "relaykv_routing_allowed": self.relaykv_routing_allowed,
            "final_fallback_reason": self.final_fallback_reason,
            "final_blocking_reasons": list(self.final_blocking_reasons),
            "relaykv_vram_status": self.relaykv_vram_status,
            "relaykv_available_working_kv_budget_mib": (
                self.relaykv_available_working_kv_budget_mib
            ),
            "summary_available_working_kv_budget_mib": (
                self.summary_available_working_kv_budget_mib
            ),
            "total_vram_mib": self.total_vram_mib,
            "max_ok_context_tokens": self.max_ok_context_tokens,
            "first_failed_context_tokens": self.first_failed_context_tokens,
            "any_oom": self.any_oom,
            "max_peak_reserved_mib": self.max_peak_reserved_mib,
            "max_peak_allocated_mib": self.max_peak_allocated_mib,
            "measured_vram_pressure_level": self.measured_vram_pressure_level,
            "relaystack_predicted_kv_budget_ok": (
                self.relaystack_predicted_kv_budget_ok
            ),
            "hf_all_lengths_ok": self.hf_all_lengths_ok,
            "observed_max_context_tokens": self.observed_max_context_tokens,
            "report_notes": list(self.report_notes),
        }


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


def _extract_error(row: dict[str, Any]) -> tuple[str | None, str | None]:
    error = row.get("error")
    if not isinstance(error, dict):
        return None, None
    error_type = error.get("type")
    error_message = error.get("message")
    return (
        str(error_type) if error_type is not None else None,
        str(error_message) if error_message is not None else None,
    )


def _is_oom_error(error_payload: dict[str, Any] | None) -> bool:
    if not isinstance(error_payload, dict):
        return False
    error_type = error_payload.get("type")
    error_message = error_payload.get("message")
    error_type_text = str(error_type).lower() if error_type is not None else ""
    error_message_text = (
        str(error_message).lower() if error_message is not None else ""
    )
    if "torch.cuda.outofmemoryerror" in error_type_text:
        return True
    if "outofmemoryerror" in error_type_text:
        return True
    return "oom" in error_type_text or "oom" in error_message_text


def _extract_cuda_peak_mib(snapshot: dict[str, Any]) -> tuple[float | None, float | None]:
    if not isinstance(snapshot, dict):
        return None, None
    cuda = snapshot.get("cuda")
    if not isinstance(cuda, dict):
        cuda = snapshot
    if not isinstance(cuda, dict):
        return None, None
    return (
        _as_float(cuda.get("peak_allocated_mib")),
        _as_float(cuda.get("peak_reserved_mib")),
    )


def _extract_cuda_metrics(row: dict[str, Any]) -> tuple[float | None, float | None]:
    after = row.get("after")
    if not isinstance(after, dict):
        after = row.get("after_error")
    return _extract_cuda_peak_mib(after)


def _build_context_rows(rows: list[dict[str, Any]]) -> list[RelayStackHFContextRowSummary]:
    summaries: list[RelayStackHFContextRowSummary] = []
    for row in rows:
        error_type, error_message = _extract_error(row)
        peak_allocated_mib, peak_reserved_mib = _extract_cuda_metrics(row)
        summaries.append(
            RelayStackHFContextRowSummary(
                target_context_tokens=_as_int(row.get("target_context_tokens")),
                input_tokens=_as_int(row.get("input_tokens")),
                ok=bool(row.get("ok")),
                error_type=error_type,
                error_message=error_message,
                peak_allocated_mib=peak_allocated_mib,
                peak_reserved_mib=peak_reserved_mib,
                tokens_per_sec_new=_as_float(row.get("tokens_per_sec_new")),
                new_tokens=_as_int(row.get("new_tokens")),
                elapsed_sec=_as_float(row.get("elapsed_sec")),
            )
        )
    return summaries


def _measure_vram_pressure_level(
    *,
    any_oom: bool,
    max_peak_reserved_mib: float | None,
    total_vram_mib: int | None,
) -> str:
    if max_peak_reserved_mib is None:
        return "no_cuda_measurement"
    if any_oom:
        return "oom_observed"
    if total_vram_mib is not None and total_vram_mib > 0:
        if max_peak_reserved_mib >= 0.9 * float(total_vram_mib):
            return "near_capacity"
        if max_peak_reserved_mib >= 0.7 * float(total_vram_mib):
            return "moderate"
    return "low"


def build_relaystack_hf_smoke_report(
    *,
    hf_smoke_payload: dict[str, Any],
    relaystack_payload: dict[str, Any],
) -> RelayStackHFSmokeReport:
    hf_env = hf_smoke_payload.get("env")
    if not isinstance(hf_env, dict):
        hf_env = {}
    hf_load = hf_smoke_payload.get("load")
    if not isinstance(hf_load, dict):
        hf_load = {}
    hf_rows = hf_smoke_payload.get("results")
    if not isinstance(hf_rows, list):
        hf_rows = []
    context_rows = _build_context_rows(
        [row for row in hf_rows if isinstance(row, dict)]
    )

    relaystack_section = relaystack_payload.get("relaystack")
    if not isinstance(relaystack_section, dict):
        relaystack_section = {}
    final_routing_decision = relaystack_section.get("final_routing_decision")
    if not isinstance(final_routing_decision, dict):
        final_routing_decision = {}

    relaykv_section = relaystack_payload.get("relaykv")
    if not isinstance(relaykv_section, dict):
        relaykv_section = {}
    vram_reservation_decision = relaykv_section.get("vram_reservation_decision")
    if not isinstance(vram_reservation_decision, dict):
        vram_reservation_decision = {}
    vram_reservation = relaykv_section.get("vram_reservation")
    if not isinstance(vram_reservation, dict):
        vram_reservation = {}

    summary = relaystack_payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    ok_contexts = [
        row.target_context_tokens
        for row in context_rows
        if row.ok and row.target_context_tokens is not None
    ]
    failed_contexts = [
        row.target_context_tokens
        for row in context_rows
        if (not row.ok) and row.target_context_tokens is not None
    ]
    peak_reserved_values = [
        row.peak_reserved_mib
        for row in context_rows
        if row.peak_reserved_mib is not None
    ]
    peak_allocated_values = [
        row.peak_allocated_mib
        for row in context_rows
        if row.peak_allocated_mib is not None
    ]
    load_peak_allocated_mib, load_peak_reserved_mib = _extract_cuda_peak_mib(hf_load)
    if load_peak_reserved_mib is not None:
        peak_reserved_values.append(load_peak_reserved_mib)
    if load_peak_allocated_mib is not None:
        peak_allocated_values.append(load_peak_allocated_mib)
    any_oom = _is_oom_error(hf_load.get("error")) or any(
        _is_oom_error({"type": row.error_type, "message": row.error_message})
        for row in context_rows
    )
    hf_load_ok = bool(hf_load.get("ok"))
    max_ok_context_tokens = max(ok_contexts) if ok_contexts else None
    first_failed_context_tokens = min(failed_contexts) if failed_contexts else None
    max_peak_reserved_mib = max(peak_reserved_values) if peak_reserved_values else None
    max_peak_allocated_mib = (
        max(peak_allocated_values) if peak_allocated_values else None
    )
    total_vram_mib = _as_int(vram_reservation.get("total_vram_mib"))
    measured_vram_pressure_level = _measure_vram_pressure_level(
        any_oom=any_oom,
        max_peak_reserved_mib=max_peak_reserved_mib,
        total_vram_mib=total_vram_mib,
    )
    observed_max_context_tokens = max(
        (row.target_context_tokens for row in context_rows if row.target_context_tokens),
        default=None,
    )
    relaykv_vram_status = vram_reservation_decision.get("status")
    relaykv_vram_status = (
        str(relaykv_vram_status) if relaykv_vram_status is not None else None
    )
    final_routing_state = final_routing_decision.get("state")
    final_routing_state = (
        str(final_routing_state) if final_routing_state is not None else None
    )
    relaykv_available_working_kv_budget_mib = _as_int(
        vram_reservation_decision.get("available_working_kv_budget_mib")
    )
    summary_available_working_kv_budget_mib = _as_int(
        summary.get("available_working_kv_budget_mib")
    )
    hf_all_lengths_ok = hf_load_ok and bool(context_rows) and all(
        row.ok for row in context_rows
    )

    report_notes: list[str] = []
    if not hf_load_ok:
        report_notes.append("hf_load_failed")
    if any_oom:
        report_notes.append("hf_oom_observed")
    if relaykv_vram_status is not None and relaykv_vram_status != "ok":
        report_notes.append("relaystack_no_kv_budget")
    if final_routing_state == "waiting_for_user_approval":
        report_notes.append("relaystack_waiting_for_user_approval")
    if (
        final_routing_state == "relaymem_only"
        and not bool(final_routing_decision.get("relaykv_routing_allowed"))
    ):
        report_notes.append("relaymem_only_possible_without_relaykv")
    if measured_vram_pressure_level == "near_capacity":
        report_notes.append("hf_peak_reserved_near_total_vram")

    return RelayStackHFSmokeReport(
        hf_script=(
            str(hf_smoke_payload.get("script"))
            if hf_smoke_payload.get("script") is not None
            else None
        ),
        hf_model=(
            str(hf_smoke_payload.get("model"))
            if hf_smoke_payload.get("model") is not None
            else None
        ),
        hf_load_ok=hf_load_ok,
        hf_env_torch=(
            str(hf_env.get("torch")) if hf_env.get("torch") is not None else None
        ),
        hf_env_cuda_version_torch=(
            str(hf_env.get("cuda_version_torch"))
            if hf_env.get("cuda_version_torch") is not None
            else None
        ),
        hf_env_transformers=(
            str(hf_env.get("transformers"))
            if hf_env.get("transformers") is not None
            else None
        ),
        hf_model_config=(
            hf_load.get("config") if isinstance(hf_load.get("config"), dict) else None
        ),
        context_rows=context_rows,
        final_routing_state=(
            str(summary.get("final_routing_state"))
            if summary.get("final_routing_state") is not None
            else final_routing_state
        ),
        relaymem_apply_allowed=bool(final_routing_decision.get("relaymem_apply_allowed")),
        relaykv_routing_allowed=bool(final_routing_decision.get("relaykv_routing_allowed")),
        final_fallback_reason=(
            str(final_routing_decision.get("fallback_reason"))
            if final_routing_decision.get("fallback_reason") is not None
            else None
        ),
        final_blocking_reasons=[
            str(reason)
            for reason in final_routing_decision.get("blocking_reasons", [])
            if reason is not None
        ],
        relaykv_vram_status=relaykv_vram_status,
        relaykv_available_working_kv_budget_mib=(
            relaykv_available_working_kv_budget_mib
        ),
        summary_available_working_kv_budget_mib=(
            summary_available_working_kv_budget_mib
        ),
        total_vram_mib=total_vram_mib,
        max_ok_context_tokens=max_ok_context_tokens,
        first_failed_context_tokens=first_failed_context_tokens,
        any_oom=any_oom,
        max_peak_reserved_mib=max_peak_reserved_mib,
        max_peak_allocated_mib=max_peak_allocated_mib,
        measured_vram_pressure_level=measured_vram_pressure_level,
        relaystack_predicted_kv_budget_ok=relaykv_vram_status == "ok",
        hf_all_lengths_ok=hf_all_lengths_ok,
        observed_max_context_tokens=observed_max_context_tokens,
        report_notes=report_notes,
    )
