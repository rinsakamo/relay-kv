#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path, artifact_name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} at {path} must be a JSON object, not non-object JSON")
    return payload


def _as_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _add_reason(reasons: list[str], message: str) -> None:
    if message not in reasons:
        reasons.append(message)


def _check_expected_field(
    *,
    errors: list[str],
    artifact_name: str,
    field_name: str,
    observed: object,
    expected: object,
) -> None:
    if observed != expected:
        _add_reason(
            errors,
            f"{artifact_name}.{field_name} must be {expected!r} (observed {observed!r})",
        )


def _matches_optional(expected: object, observed: object) -> bool:
    if expected is None or observed is None:
        return True
    return expected == observed


def _resolved_path_string(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def _model_ref_consistent(
    adapter_model_ref: dict[str, object],
    tokenizer_model_ref: dict[str, object],
    engine_model_ref: dict[str, object],
    probe_model_ref: dict[str, object],
) -> bool:
    adapter_model_id = adapter_model_ref.get("model_id")
    tokenizer_model_id = tokenizer_model_ref.get("model_id")
    engine_model_id = engine_model_ref.get("model_id")
    probe_model_id = probe_model_ref.get("model_id")
    if not (
        isinstance(adapter_model_id, str)
        and adapter_model_id == tokenizer_model_id == engine_model_id == probe_model_id
    ):
        return False

    adapter_revision = adapter_model_ref.get("model_revision")
    engine_revision = engine_model_ref.get("model_revision")
    tokenizer_revision = tokenizer_model_ref.get("model_revision")
    probe_revision = probe_model_ref.get("model_revision")
    if adapter_revision != engine_revision:
        return False
    if tokenizer_revision is not None and tokenizer_revision != adapter_revision:
        return False
    if probe_revision is not None and probe_revision != adapter_revision:
        return False

    adapter_local_path = adapter_model_ref.get("local_path")
    engine_local_path = engine_model_ref.get("local_path")
    tokenizer_local_path = tokenizer_model_ref.get("local_path")
    probe_local_path = probe_model_ref.get("local_path")
    if (
        adapter_local_path is not None
        and engine_local_path is not None
        and adapter_local_path != engine_local_path
    ):
        return False
    for local_path in (tokenizer_local_path, probe_local_path):
        if local_path is None:
            continue
        if adapter_local_path is not None and local_path != adapter_local_path:
            return False
        if engine_local_path is not None and local_path != engine_local_path:
            return False
    return True


def _tokenizer_ref_consistent(
    adapter_tokenizer_ref: dict[str, object],
    tokenizer_tokenizer_ref: dict[str, object],
    engine_tokenizer_ref: dict[str, object],
    probe_tokenizer_ref: dict[str, object],
) -> bool:
    adapter_name = adapter_tokenizer_ref.get("tokenizer_name_or_path")
    tokenizer_name = tokenizer_tokenizer_ref.get("tokenizer_name_or_path")
    engine_name = engine_tokenizer_ref.get("tokenizer_name_or_path")
    probe_name = probe_tokenizer_ref.get("tokenizer_name_or_path")
    if not (
        isinstance(adapter_name, str)
        and adapter_name == tokenizer_name == engine_name == probe_name
    ):
        return False

    for field_name in ("tokenizer_revision", "tokenizer_config_hash"):
        values = [
            adapter_tokenizer_ref.get(field_name),
            tokenizer_tokenizer_ref.get(field_name),
            engine_tokenizer_ref.get(field_name),
            probe_tokenizer_ref.get(field_name),
        ]
        baseline = values[0]
        if any(value != baseline for value in values[1:]):
            return False
    return True


def _tokenizer_config_probe_accepted(
    probe_data: dict[str, object],
    errors: list[str],
    warnings: list[str],
) -> bool:
    summary = _as_mapping(probe_data.get("summary"))
    tokenizer_probe = _as_mapping(probe_data.get("tokenizer_probe"))
    config_probe = _as_mapping(probe_data.get("config_probe"))
    safety_scope = _as_mapping(probe_data.get("safety_scope"))

    if summary.get("ok") is True:
        return True

    blocked_reason = summary.get("blocked_reason")
    if not isinstance(blocked_reason, str) or not blocked_reason:
        _add_reason(errors, "tokenizer_config_probe summary.ok is false without blocked_reason")
        return False

    skipped = (
        tokenizer_probe.get("skipped") is True
        or config_probe.get("skipped") is True
    )
    metadata_only = (
        safety_scope.get("model_loaded") is False
        and safety_scope.get("no_model_loading") is True
        and safety_scope.get("no_gpu_inspection") is True
        and safety_scope.get("no_kv_materialization") is True
        and safety_scope.get("no_attention_connection") is True
        and safety_scope.get("no_scheduler_change") is True
        and safety_scope.get("no_runtime_apply") is True
    )
    non_runtime_reason = any(
        marker in blocked_reason.lower()
        for marker in ("skip", "skipped", "unavailable", "metadata", "local-files-only")
    )
    if skipped and metadata_only and non_runtime_reason:
        _add_reason(
            warnings,
            f"tokenizer_config_probe accepted with blocked metadata-only reason: {blocked_reason}",
        )
        return True

    _add_reason(
        errors,
        f"tokenizer_config_probe not accepted: {blocked_reason}",
    )
    return False


def _selected_artifact_identity_scope_ok(
    *,
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
    probe_data: dict[str, object],
    errors: list[str],
) -> tuple[bool, bool, bool]:
    adapter_kinds = [
        adapter_data.get("adapter_kind"),
        tokenizer_data.get("adapter_kind"),
        engine_data.get("adapter_kind"),
    ]
    probe_adapter_kind = probe_data.get("adapter_kind")
    if probe_adapter_kind is not None:
        adapter_kinds.append(probe_adapter_kind)
    adapter_identity_consistent = all(kind == "hf_prototype" for kind in adapter_kinds)
    if not adapter_identity_consistent:
        _add_reason(errors, "selected artifact adapter_kind mismatch across phase12 chain")

    runtime_targets = [
        adapter_data.get("runtime_target"),
        tokenizer_data.get("runtime_target"),
        engine_data.get("runtime_target"),
    ]
    probe_runtime_target = probe_data.get("runtime_target")
    if probe_runtime_target is not None:
        runtime_targets.append(probe_runtime_target)
    runtime_target_consistent = all(target == "huggingface_transformers" for target in runtime_targets)
    if adapter_data.get("runtime_target") != "huggingface_transformers":
        _add_reason(errors, "selected adapter capabilities runtime_target must be 'huggingface_transformers'")
    elif not runtime_target_consistent:
        _add_reason(errors, "selected artifact runtime_target mismatch across phase12 chain")

    adapter_names = [
        value
        for value in (
            adapter_data.get("adapter_name"),
            engine_data.get("adapter_name"),
            probe_data.get("adapter_name"),
        )
        if value is not None
    ]
    adapter_name_consistent = len(set(adapter_names)) <= 1
    if not adapter_name_consistent:
        _add_reason(errors, "selected artifact adapter_name mismatch across phase12 chain")

    selected_artifact_identity_scope_ok = (
        adapter_identity_consistent and runtime_target_consistent and adapter_name_consistent
    )
    if not selected_artifact_identity_scope_ok:
        _add_reason(errors, "selected artifact identity scope does not match readiness HF boundary")
    return (
        adapter_identity_consistent,
        runtime_target_consistent,
        selected_artifact_identity_scope_ok,
    )


def _adapter_capability_safety_ok(
    adapter_data: dict[str, object],
    errors: list[str],
) -> bool:
    adapter_summary = _as_mapping(adapter_data.get("summary"))
    adapter_capabilities = _as_mapping(adapter_data.get("capabilities"))
    adapter_safety_scope = _as_mapping(adapter_data.get("safety_scope"))

    ok = True
    if adapter_capabilities.get("supports_materialization") is not False or adapter_capabilities.get("supports_apply") is not False:
        _add_reason(errors, "selected adapter capabilities artifact enables materialization/apply capability")
        ok = False
    if not (
        adapter_summary.get("ok") is True
        and adapter_capabilities.get("supports_tokenizer_span_probe") is True
        and adapter_capabilities.get("supports_engine_metadata_probe") is True
        and adapter_safety_scope.get("dry_run_only") is True
        and adapter_safety_scope.get("no_model_loading_required") is True
        and adapter_safety_scope.get("no_kv_materialization") is True
        and adapter_safety_scope.get("no_attention_connection") is True
        and adapter_safety_scope.get("no_scheduler_change") is True
        and adapter_safety_scope.get("no_runtime_apply") is True
    ):
        _add_reason(errors, "selected adapter capabilities artifact does not satisfy readiness safety invariants")
        ok = False
    return ok


def _tokenizer_span_safety_ok(
    tokenizer_data: dict[str, object],
    errors: list[str],
) -> bool:
    tokenizer_summary = _as_mapping(tokenizer_data.get("summary"))
    tokenizer_safety_scope = _as_mapping(tokenizer_data.get("safety_scope"))
    spans = _as_list(tokenizer_data.get("spans"))

    ok = True
    if not (
        tokenizer_summary.get("ok") is True
        and tokenizer_summary.get("no_model_loaded") is True
        and tokenizer_summary.get("no_tokenizer_loaded") is True
        and tokenizer_summary.get("no_gpu_inspection") is True
        and tokenizer_safety_scope.get("dry_run_only") is True
        and tokenizer_safety_scope.get("no_model_loading_required") is True
        and tokenizer_safety_scope.get("no_tokenizer_loading_required") is True
        and tokenizer_safety_scope.get("no_gpu_inspection") is True
        and tokenizer_safety_scope.get("no_kv_materialization") is True
        and tokenizer_safety_scope.get("no_attention_connection") is True
        and tokenizer_safety_scope.get("no_scheduler_change") is True
        and tokenizer_safety_scope.get("no_runtime_apply") is True
        and len(spans) >= 1
    ):
        _add_reason(errors, "selected tokenizer span probe artifact is not metadata-only")
        ok = False
    for span in spans:
        span_mapping = _as_mapping(span)
        lineage = _as_mapping(span_mapping.get("lineage"))
        if span_mapping.get("token_span_is_estimated") is not True or span_mapping.get("tokenizer_scoped") is not True:
            _add_reason(errors, "selected tokenizer span probe artifact is not metadata-only")
            ok = False
            break
        if lineage.get("engine_block_ref") is not None:
            _add_reason(errors, "selected tokenizer span probe contains engine block refs")
            ok = False
            break
    return ok


def _engine_metadata_safety_ok(
    engine_data: dict[str, object],
    errors: list[str],
) -> bool:
    engine_summary = _as_mapping(engine_data.get("summary"))
    engine_safety_scope = _as_mapping(engine_data.get("safety_scope"))
    engine_metadata = _as_mapping(engine_data.get("engine_metadata"))
    engine_capabilities = _as_mapping(engine_data.get("capabilities_snapshot"))

    ok = True
    if (
        engine_metadata.get("model_loaded") is True
        or engine_metadata.get("tokenizer_loaded") is True
        or engine_metadata.get("gpu_inspected") is True
    ):
        _add_reason(errors, "selected engine metadata probe reports model/tokenizer/GPU loaded state")
        ok = False
    attention_type_hint = engine_metadata.get("attention_type_hint")
    kv_head_group_count = engine_metadata.get("kv_head_group_count")
    if attention_type_hint == "mha" and kv_head_group_count != 1:
        _add_reason(errors, "selected engine metadata probe is not metadata-only")
        ok = False
    if attention_type_hint == "gqa" and not isinstance(kv_head_group_count, int):
        _add_reason(errors, "selected engine metadata probe is not metadata-only")
        ok = False
    if attention_type_hint not in ("mha", "gqa", "unknown"):
        _add_reason(errors, "selected engine metadata probe is not metadata-only")
        ok = False
    if not (
        engine_summary.get("ok") is True
        and engine_summary.get("no_model_loaded") is True
        and engine_summary.get("no_tokenizer_loaded") is True
        and engine_summary.get("no_gpu_inspection") is True
        and engine_capabilities.get("supports_tokenizer_span_probe") is True
        and engine_capabilities.get("supports_engine_metadata_probe") is True
        and engine_capabilities.get("supports_materialization") is False
        and engine_capabilities.get("supports_apply") is False
        and engine_safety_scope.get("dry_run_only") is True
        and engine_safety_scope.get("no_model_loading_required") is True
        and engine_safety_scope.get("no_tokenizer_loading_required") is True
        and engine_safety_scope.get("no_gpu_inspection") is True
        and engine_safety_scope.get("no_kv_materialization") is True
        and engine_safety_scope.get("no_attention_connection") is True
        and engine_safety_scope.get("no_scheduler_change") is True
        and engine_safety_scope.get("no_runtime_apply") is True
    ):
        _add_reason(errors, "selected engine metadata probe is not metadata-only")
        ok = False
    return ok


def build_hf_phase12_chain_acceptance_report_payload(
    *,
    adapter_capabilities_path: Path,
    tokenizer_span_probe_path: Path,
    engine_metadata_probe_path: Path,
    readiness_report_path: Path,
    tokenizer_config_probe_path: Path,
    output_path: Path,
) -> tuple[dict[str, object], int]:
    adapter_data = _load_json(adapter_capabilities_path, "adapter capabilities")
    tokenizer_data = _load_json(tokenizer_span_probe_path, "tokenizer span probe")
    engine_data = _load_json(engine_metadata_probe_path, "engine metadata probe")
    readiness_data = _load_json(readiness_report_path, "readiness report")
    probe_data = _load_json(tokenizer_config_probe_path, "tokenizer config probe")

    errors: list[str] = []
    warnings: list[str] = []

    _check_expected_field(
        errors=errors,
        artifact_name="adapter_capabilities",
        field_name="schema_version",
        observed=adapter_data.get("schema_version"),
        expected="relaystack.adapter_capabilities.v0.1",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="adapter_capabilities",
        field_name="phase",
        observed=adapter_data.get("phase"),
        expected="12-C",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="tokenizer_span_probe",
        field_name="schema_version",
        observed=tokenizer_data.get("schema_version"),
        expected="relaystack.tokenizer_span_probe.v0.1",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="tokenizer_span_probe",
        field_name="phase",
        observed=tokenizer_data.get("phase"),
        expected="12-E",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="tokenizer_span_probe",
        field_name="artifact_kind",
        observed=tokenizer_data.get("artifact_kind"),
        expected="hf_tokenizer_span_probe",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="engine_metadata_probe",
        field_name="schema_version",
        observed=engine_data.get("schema_version"),
        expected="relaystack.engine_metadata_probe.v0.1",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="engine_metadata_probe",
        field_name="phase",
        observed=engine_data.get("phase"),
        expected="12-F",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="engine_metadata_probe",
        field_name="artifact_kind",
        observed=engine_data.get("artifact_kind"),
        expected="hf_engine_metadata_probe",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="readiness_report",
        field_name="schema_version",
        observed=readiness_data.get("schema_version"),
        expected="relaystack.hf_adapter_readiness_report.v0.1",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="readiness_report",
        field_name="phase",
        observed=readiness_data.get("phase"),
        expected="12-H",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="readiness_report",
        field_name="artifact_kind",
        observed=readiness_data.get("artifact_kind"),
        expected="hf_adapter_readiness_report",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="tokenizer_config_probe",
        field_name="schema_version",
        observed=probe_data.get("schema_version"),
        expected="relaystack.hf_tokenizer_config_probe.v0.1",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="tokenizer_config_probe",
        field_name="phase",
        observed=probe_data.get("phase"),
        expected="12-I",
    )
    _check_expected_field(
        errors=errors,
        artifact_name="tokenizer_config_probe",
        field_name="artifact_kind",
        observed=probe_data.get("artifact_kind"),
        expected="hf_tokenizer_config_probe",
    )

    expected_artifacts = [
        "relaystack_adapter_capabilities.json",
        "relaystack_tokenizer_span_probe.json",
        "relaystack_engine_metadata_probe.json",
        "relaystack_hf_adapter_readiness_report.json",
        "relaystack_hf_tokenizer_config_probe.json",
    ]
    present_artifacts = [
        adapter_capabilities_path.name,
        tokenizer_span_probe_path.name,
        engine_metadata_probe_path.name,
        readiness_report_path.name,
        tokenizer_config_probe_path.name,
    ]
    missing_artifacts = [name for name in expected_artifacts if name not in present_artifacts]
    chain_complete = len(missing_artifacts) == 0
    if not chain_complete:
        _add_reason(errors, f"missing phase12 artifacts: {missing_artifacts}")

    adapter_model_ref = _as_mapping(adapter_data.get("model_ref"))
    tokenizer_model_ref = _as_mapping(tokenizer_data.get("model_ref"))
    engine_model_ref = _as_mapping(engine_data.get("model_ref"))
    probe_model_ref = _as_mapping(probe_data.get("model_ref"))
    model_ref_consistent = _model_ref_consistent(
        adapter_model_ref,
        tokenizer_model_ref,
        engine_model_ref,
        probe_model_ref,
    )
    if not model_ref_consistent:
        _add_reason(errors, "model_ref mismatch across phase12 artifact chain")

    adapter_tokenizer_ref = _as_mapping(adapter_data.get("tokenizer_ref"))
    tokenizer_tokenizer_ref = _as_mapping(tokenizer_data.get("tokenizer_ref"))
    engine_tokenizer_ref = _as_mapping(engine_data.get("tokenizer_ref"))
    probe_tokenizer_ref = _as_mapping(probe_data.get("tokenizer_ref"))
    tokenizer_ref_consistent = _tokenizer_ref_consistent(
        adapter_tokenizer_ref,
        tokenizer_tokenizer_ref,
        engine_tokenizer_ref,
        probe_tokenizer_ref,
    )
    if not tokenizer_ref_consistent:
        _add_reason(errors, "tokenizer_ref mismatch across phase12 artifact chain")

    readiness_summary = _as_mapping(readiness_data.get("summary"))
    readiness_gate_ok = readiness_summary.get("ok") is True
    if not readiness_gate_ok:
        _add_reason(errors, "readiness report summary.ok must be true")
    readiness_input_refs = _as_mapping(readiness_data.get("input_refs"))
    expected_readiness_input_paths = {
        "adapter_capabilities_path": str(adapter_capabilities_path.expanduser().resolve(strict=False)),
        "tokenizer_span_probe_path": str(tokenizer_span_probe_path.expanduser().resolve(strict=False)),
        "engine_metadata_probe_path": str(engine_metadata_probe_path.expanduser().resolve(strict=False)),
    }
    readiness_input_ref_messages = {
        "adapter_capabilities_path": (
            "readiness report input_refs.adapter_capabilities_path does not match selected "
            "adapter capabilities path"
        ),
        "tokenizer_span_probe_path": (
            "readiness report input_refs.tokenizer_span_probe_path does not match selected "
            "tokenizer span probe path"
        ),
        "engine_metadata_probe_path": (
            "readiness report input_refs.engine_metadata_probe_path does not match selected "
            "engine metadata probe path"
        ),
    }
    readiness_input_refs_match = True
    for field_name, expected_path in expected_readiness_input_paths.items():
        observed_path = _resolved_path_string(readiness_input_refs.get(field_name))
        if observed_path != expected_path:
            readiness_input_refs_match = False
            _add_reason(errors, readiness_input_ref_messages[field_name])
    probe_input_refs = _as_mapping(probe_data.get("input_refs"))
    expected_probe_input_paths = {
        "readiness_report_path": str(readiness_report_path.expanduser().resolve(strict=False)),
        "adapter_capabilities_path": expected_readiness_input_paths["adapter_capabilities_path"],
        "tokenizer_span_probe_path": expected_readiness_input_paths["tokenizer_span_probe_path"],
        "engine_metadata_probe_path": expected_readiness_input_paths["engine_metadata_probe_path"],
    }
    tokenizer_config_probe_input_refs_match = True
    for field_name, expected_path in expected_probe_input_paths.items():
        observed_path = _resolved_path_string(probe_input_refs.get(field_name))
        if observed_path != expected_path:
            tokenizer_config_probe_input_refs_match = False
            if field_name == "readiness_report_path":
                _add_reason(
                    errors,
                    "tokenizer_config_probe input_refs.readiness_report_path does not match selected readiness report path",
                )
            elif field_name == "adapter_capabilities_path":
                _add_reason(
                    errors,
                    "tokenizer_config_probe input_refs.adapter_capabilities_path does not match selected adapter capabilities path",
                )
            elif field_name == "tokenizer_span_probe_path":
                _add_reason(
                    errors,
                    "tokenizer_config_probe input_refs.tokenizer_span_probe_path does not match selected tokenizer span probe path",
                )
            elif field_name == "engine_metadata_probe_path":
                _add_reason(
                    errors,
                    "tokenizer_config_probe input_refs.engine_metadata_probe_path does not match selected engine metadata probe path",
                )
    probe_readiness_ref = _as_mapping(probe_data.get("readiness_ref"))
    tokenizer_config_probe_readiness_ref_match = (
        _resolved_path_string(probe_readiness_ref.get("path"))
        == expected_probe_input_paths["readiness_report_path"]
    )
    if not tokenizer_config_probe_readiness_ref_match:
        _add_reason(
            errors,
            "tokenizer_config_probe readiness_ref.path does not match selected readiness report path",
        )

    tokenizer_config_probe_accepted = _tokenizer_config_probe_accepted(
        probe_data,
        errors,
        warnings,
    )
    (
        adapter_identity_consistent,
        runtime_target_consistent,
        selected_artifact_identity_scope_ok,
    ) = _selected_artifact_identity_scope_ok(
        adapter_data=adapter_data,
        tokenizer_data=tokenizer_data,
        engine_data=engine_data,
        probe_data=probe_data,
        errors=errors,
    )
    adapter_capability_safety_ok = _adapter_capability_safety_ok(adapter_data, errors)
    tokenizer_span_safety_ok = _tokenizer_span_safety_ok(tokenizer_data, errors)
    engine_metadata_safety_ok = _engine_metadata_safety_ok(engine_data, errors)
    upstream_artifact_content_safety_match = (
        adapter_capability_safety_ok
        and tokenizer_span_safety_ok
        and engine_metadata_safety_ok
    )

    readiness_readiness = _as_mapping(readiness_data.get("readiness"))
    if readiness_readiness.get("ready_for_materialization") is not False:
        _add_reason(errors, "readiness report must keep ready_for_materialization false")
    if readiness_readiness.get("ready_for_apply") is not False:
        _add_reason(errors, "readiness report must keep ready_for_apply false")

    probe_summary = _as_mapping(probe_data.get("summary"))
    if probe_summary.get("ready_for_materialization") is not False:
        _add_reason(errors, "tokenizer_config_probe must keep ready_for_materialization false")
    if probe_summary.get("ready_for_apply") is not False:
        _add_reason(errors, "tokenizer_config_probe must keep ready_for_apply false")

    adapter_summary = _as_mapping(adapter_data.get("summary"))
    tokenizer_summary = _as_mapping(tokenizer_data.get("summary"))
    engine_summary = _as_mapping(engine_data.get("summary"))
    probe_safety_scope = _as_mapping(probe_data.get("safety_scope"))
    readiness_safety_scope = _as_mapping(readiness_data.get("safety_scope"))
    engine_metadata = _as_mapping(engine_data.get("engine_metadata"))

    metadata_report_only = (
        adapter_summary.get("no_model_loaded") is True
        and adapter_summary.get("no_gpu_inspection") is True
        and tokenizer_summary.get("no_model_loaded") is True
        and tokenizer_summary.get("no_tokenizer_loaded") is True
        and tokenizer_summary.get("no_gpu_inspection") is True
        and engine_summary.get("no_model_loaded") is True
        and engine_summary.get("no_tokenizer_loaded") is True
        and engine_summary.get("no_gpu_inspection") is True
        and readiness_safety_scope.get("dry_run_only") is True
        and readiness_safety_scope.get("no_model_loading_required") is True
        and readiness_safety_scope.get("no_tokenizer_loading_required") is True
        and readiness_safety_scope.get("no_gpu_inspection") is True
        and readiness_safety_scope.get("no_kv_materialization") is True
        and readiness_safety_scope.get("no_attention_connection") is True
        and readiness_safety_scope.get("no_scheduler_change") is True
        and readiness_safety_scope.get("no_runtime_apply") is True
        and probe_safety_scope.get("no_model_loading") is True
        and probe_safety_scope.get("no_gpu_inspection") is True
        and probe_safety_scope.get("no_kv_materialization") is True
        and probe_safety_scope.get("no_attention_connection") is True
        and probe_safety_scope.get("no_scheduler_change") is True
        and probe_safety_scope.get("no_runtime_apply") is True
    )
    if not metadata_report_only:
        _add_reason(errors, "phase12 chain must remain metadata/report-only")

    model_loaded = (
        engine_metadata.get("model_loaded") is True or probe_safety_scope.get("model_loaded") is True
    )
    gpu_inspected = (
        engine_metadata.get("gpu_inspected") is True
        or probe_safety_scope.get("no_gpu_inspection") is not True
        or readiness_safety_scope.get("no_gpu_inspection") is not True
    )
    kv_materialized = (
        probe_safety_scope.get("no_kv_materialization") is not True
        or readiness_safety_scope.get("no_kv_materialization") is not True
    )
    attention_connected = (
        probe_safety_scope.get("no_attention_connection") is not True
        or readiness_safety_scope.get("no_attention_connection") is not True
    )
    scheduler_changed = (
        probe_safety_scope.get("no_scheduler_change") is not True
        or readiness_safety_scope.get("no_scheduler_change") is not True
    )
    runtime_apply = (
        probe_safety_scope.get("no_runtime_apply") is not True
        or readiness_safety_scope.get("no_runtime_apply") is not True
    )
    for message, violated in (
        ("model_loaded must remain false", model_loaded),
        ("gpu_inspected must remain false", gpu_inspected),
        ("kv_materialized must remain false", kv_materialized),
        ("attention_connected must remain false", attention_connected),
        ("scheduler_changed must remain false", scheduler_changed),
        ("runtime_apply must remain false", runtime_apply),
    ):
        if violated:
            _add_reason(errors, message)

    accepted = (
        chain_complete
        and model_ref_consistent
        and tokenizer_ref_consistent
        and selected_artifact_identity_scope_ok
        and readiness_gate_ok
        and readiness_input_refs_match
        and upstream_artifact_content_safety_match
        and tokenizer_config_probe_input_refs_match
        and tokenizer_config_probe_readiness_ref_match
        and tokenizer_config_probe_accepted
        and metadata_report_only
        and not model_loaded
        and not gpu_inspected
        and not kv_materialized
        and not attention_connected
        and not scheduler_changed
        and not runtime_apply
        and len(errors) == 0
    )

    payload = {
        "schema_version": "relaystack.hf_phase12_chain_acceptance_report.v0.1",
        "phase": "12-J",
        "artifact_kind": "hf_phase12_chain_acceptance_report",
        "input_refs": {
            "adapter_capabilities_path": str(adapter_capabilities_path),
            "tokenizer_span_probe_path": str(tokenizer_span_probe_path),
            "engine_metadata_probe_path": str(engine_metadata_probe_path),
            "readiness_report_path": str(readiness_report_path),
            "tokenizer_config_probe_path": str(tokenizer_config_probe_path),
        },
        "chain": {
            "expected_artifacts": expected_artifacts,
            "present_artifacts": present_artifacts,
            "missing_artifacts": missing_artifacts,
            "complete": chain_complete,
        },
        "consistency": {
            "model_ref_consistent": model_ref_consistent,
            "tokenizer_ref_consistent": tokenizer_ref_consistent,
            "adapter_identity_consistent": adapter_identity_consistent,
            "runtime_target_consistent": runtime_target_consistent,
            "selected_artifact_identity_scope_ok": selected_artifact_identity_scope_ok,
            "readiness_gate_ok": readiness_gate_ok,
            "readiness_input_refs_match": readiness_input_refs_match,
            "adapter_capability_safety_ok": adapter_capability_safety_ok,
            "tokenizer_span_safety_ok": tokenizer_span_safety_ok,
            "engine_metadata_safety_ok": engine_metadata_safety_ok,
            "upstream_artifact_content_safety_match": upstream_artifact_content_safety_match,
            "tokenizer_config_probe_input_refs_match": tokenizer_config_probe_input_refs_match,
            "tokenizer_config_probe_readiness_ref_match": tokenizer_config_probe_readiness_ref_match,
            "tokenizer_config_probe_accepted": tokenizer_config_probe_accepted,
        },
        "safety_scope": {
            "metadata_report_only": metadata_report_only,
            "model_loaded": model_loaded,
            "gpu_inspected": gpu_inspected,
            "kv_materialized": kv_materialized,
            "attention_connected": attention_connected,
            "scheduler_changed": scheduler_changed,
            "runtime_apply": runtime_apply,
        },
        "acceptance": {
            "accepted": accepted,
            "ready_for_next_metadata_step": accepted,
            "ready_for_materialization": False,
            "ready_for_apply": False,
            "blocking_reasons": errors,
            "warning_reasons": warnings,
        },
        "summary": {
            "ok": accepted,
            "artifact_kind": "hf_phase12_chain_acceptance_report",
            "output_path": str(output_path),
        },
    }
    return payload, (0 if accepted else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-capabilities", type=Path, required=True)
    parser.add_argument("--tokenizer-span-probe", type=Path, required=True)
    parser.add_argument("--engine-metadata-probe", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--tokenizer-config-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()
    try:
        payload, exit_code = build_hf_phase12_chain_acceptance_report_payload(
            adapter_capabilities_path=args.adapter_capabilities.expanduser().resolve(),
            tokenizer_span_probe_path=args.tokenizer_span_probe.expanduser().resolve(),
            engine_metadata_probe_path=args.engine_metadata_probe.expanduser().resolve(),
            readiness_report_path=args.readiness_report.expanduser().resolve(),
            tokenizer_config_probe_path=args.tokenizer_config_probe.expanduser().resolve(),
            output_path=output_path,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{exc.__class__.__name__}: {exc}", flush=True)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
