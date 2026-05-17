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


def _add_check(
    checks: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    severity: str,
    message: str,
    observed: object | None = None,
) -> None:
    check = {
        "name": name,
        "passed": passed,
        "severity": severity,
        "message": message,
    }
    if observed is not None:
        check["observed"] = observed
    checks.append(check)


def _validate_schema_versions(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    _add_check(
        checks,
        name="adapter_capabilities_schema_version",
        passed=adapter_data.get("schema_version") == "relaystack.adapter_capabilities.v0.1",
        severity="error",
        message="Adapter capabilities schema_version must be relaystack.adapter_capabilities.v0.1",
        observed=adapter_data.get("schema_version"),
    )
    _add_check(
        checks,
        name="tokenizer_span_probe_schema_version",
        passed=tokenizer_data.get("schema_version") == "relaystack.tokenizer_span_probe.v0.1",
        severity="error",
        message="Tokenizer span probe schema_version must be relaystack.tokenizer_span_probe.v0.1",
        observed=tokenizer_data.get("schema_version"),
    )
    _add_check(
        checks,
        name="engine_metadata_probe_schema_version",
        passed=engine_data.get("schema_version") == "relaystack.engine_metadata_probe.v0.1",
        severity="error",
        message="Engine metadata probe schema_version must be relaystack.engine_metadata_probe.v0.1",
        observed=engine_data.get("schema_version"),
    )


def _validate_artifact_identity(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    _add_check(
        checks,
        name="adapter_capabilities_shape",
        passed=(
            isinstance(adapter_data.get("capabilities"), dict)
            and adapter_data.get("schema_version") == "relaystack.adapter_capabilities.v0.1"
        ),
        severity="error",
        message="Adapter capabilities artifact must include schema_version and capabilities mapping",
    )
    _add_check(
        checks,
        name="tokenizer_span_probe_artifact_kind",
        passed=tokenizer_data.get("artifact_kind") == "hf_tokenizer_span_probe",
        severity="error",
        message="Tokenizer span probe artifact_kind must be hf_tokenizer_span_probe",
        observed=tokenizer_data.get("artifact_kind"),
    )
    _add_check(
        checks,
        name="engine_metadata_probe_artifact_kind",
        passed=engine_data.get("artifact_kind") == "hf_engine_metadata_probe",
        severity="error",
        message="Engine metadata probe artifact_kind must be hf_engine_metadata_probe",
        observed=engine_data.get("artifact_kind"),
    )


def _validate_adapter_and_runtime_consistency(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    adapter_kinds = [
        adapter_data.get("adapter_kind"),
        tokenizer_data.get("adapter_kind"),
        engine_data.get("adapter_kind"),
    ]
    _add_check(
        checks,
        name="adapter_kind_consistency",
        passed=all(kind == "hf_prototype" for kind in adapter_kinds),
        severity="error",
        message="All artifacts must use adapter_kind hf_prototype",
        observed=adapter_kinds,
    )
    runtime_targets = [
        adapter_data.get("runtime_target"),
        tokenizer_data.get("runtime_target"),
        engine_data.get("runtime_target"),
    ]
    _add_check(
        checks,
        name="runtime_target_consistency",
        passed=all(target == "huggingface_transformers" for target in runtime_targets),
        severity="error",
        message="All artifacts must use runtime_target huggingface_transformers",
        observed=runtime_targets,
    )


def _validate_model_and_tokenizer_consistency(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    adapter_model_ref = adapter_data.get("model_ref") or {}
    tokenizer_model_ref = tokenizer_data.get("model_ref") or {}
    engine_model_ref = engine_data.get("model_ref") or {}
    adapter_model_id = adapter_model_ref.get("model_id")
    tokenizer_model_id = tokenizer_model_ref.get("model_id")
    engine_model_id = engine_model_ref.get("model_id")
    model_ids = [adapter_model_id, tokenizer_model_id, engine_model_id]
    _add_check(
        checks,
        name="model_ref_consistency",
        passed=(
            adapter_model_id is not None
            and adapter_model_id == tokenizer_model_id == engine_model_id
        ),
        severity="error",
        message="model_ref.model_id must match across adapter, tokenizer, and engine artifacts",
        observed=model_ids,
    )
    model_revisions = {
        "adapter": adapter_model_ref.get("model_revision"),
        "tokenizer": tokenizer_model_ref.get("model_revision"),
        "engine": engine_model_ref.get("model_revision"),
    }
    _add_check(
        checks,
        name="model_ref_model_revision_consistency",
        passed=(
            model_revisions["adapter"] == model_revisions["engine"]
            and (
                model_revisions["tokenizer"] is None
                or model_revisions["tokenizer"] == model_revisions["adapter"]
            )
        ),
        severity="error",
        message="model_ref.model_revision must match across adapter and engine artifacts; tokenizer probe may omit it but must match if present",
        observed=model_revisions,
    )
    local_paths = {
        "adapter": adapter_model_ref.get("local_path"),
        "tokenizer": tokenizer_model_ref.get("local_path"),
        "engine": engine_model_ref.get("local_path"),
    }
    _add_check(
        checks,
        name="model_ref_local_path_observed",
        passed=True,
        severity="info",
        message="model_ref.local_path is observed for reference only in Phase 12-H",
        observed=local_paths,
    )
    adapter_tokenizer = ((adapter_data.get("tokenizer_ref") or {}).get("tokenizer_name_or_path"))
    tokenizer_tokenizer = ((tokenizer_data.get("tokenizer_ref") or {}).get("tokenizer_name_or_path"))
    engine_tokenizer = ((engine_data.get("tokenizer_ref") or {}).get("tokenizer_name_or_path"))
    tokenizer_names = [adapter_tokenizer, tokenizer_tokenizer, engine_tokenizer]
    _add_check(
        checks,
        name="tokenizer_ref_consistency",
        passed=(
            adapter_tokenizer is not None
            and adapter_tokenizer == tokenizer_tokenizer == engine_tokenizer
        ),
        severity="error",
        message="tokenizer_ref.tokenizer_name_or_path must match across adapter, tokenizer, and engine artifacts",
        observed=tokenizer_names,
    )
    adapter_tokenizer_ref = adapter_data.get("tokenizer_ref") or {}
    tokenizer_tokenizer_ref = tokenizer_data.get("tokenizer_ref") or {}
    engine_tokenizer_ref = engine_data.get("tokenizer_ref") or {}
    for field_name in (
        "tokenizer_revision",
        "tokenizer_config_hash",
        "tokenizer_family",
    ):
        observed = {
            "adapter": adapter_tokenizer_ref.get(field_name),
            "tokenizer": tokenizer_tokenizer_ref.get(field_name),
            "engine": engine_tokenizer_ref.get(field_name),
        }
        _add_check(
            checks,
            name=f"tokenizer_ref_{field_name}_consistency",
            passed=(
                observed["adapter"]
                == observed["tokenizer"]
                == observed["engine"]
            ),
            severity="error",
            message=f"tokenizer_ref.{field_name} must match across adapter, tokenizer, and engine artifacts",
            observed=observed,
        )


def _validate_safety_scope(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    required_common_flags = (
        "dry_run_only",
        "no_model_loading_required",
        "no_gpu_inspection",
        "no_kv_materialization",
        "no_attention_connection",
        "no_scheduler_change",
        "no_runtime_apply",
    )
    for flag_name in required_common_flags:
        adapter_observed = (adapter_data.get("safety_scope") or {}).get(flag_name)
        if flag_name == "no_gpu_inspection" and adapter_observed is None:
            adapter_observed = (adapter_data.get("summary") or {}).get(flag_name)
        observed = {
            "adapter": adapter_observed,
            "tokenizer": (tokenizer_data.get("safety_scope") or {}).get(flag_name),
            "engine": (engine_data.get("safety_scope") or {}).get(flag_name),
        }
        _add_check(
            checks,
            name=f"safety_scope_{flag_name}",
            passed=all(value is True for value in observed.values()),
            severity="error",
            message=f"All artifacts must set safety_scope.{flag_name} to true",
            observed=observed,
        )
    for artifact_name, artifact_data in (
        ("tokenizer", tokenizer_data),
        ("engine", engine_data),
    ):
        observed = (artifact_data.get("safety_scope") or {}).get(
            "no_tokenizer_loading_required"
        )
        _add_check(
            checks,
            name=f"{artifact_name}_no_tokenizer_loading_required",
            passed=observed is True,
            severity="error",
            message=f"{artifact_name} artifact must set safety_scope.no_tokenizer_loading_required to true",
            observed=observed,
        )


def _validate_no_apply_materialization(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    adapter_capabilities = adapter_data.get("capabilities") or {}
    engine_capabilities = engine_data.get("capabilities_snapshot") or {}
    _add_check(
        checks,
        name="adapter_supports_materialization_false",
        passed=adapter_capabilities.get("supports_materialization") is False,
        severity="error",
        message="Adapter capability artifact must keep supports_materialization false",
        observed=adapter_capabilities.get("supports_materialization"),
    )
    _add_check(
        checks,
        name="adapter_supports_apply_false",
        passed=adapter_capabilities.get("supports_apply") is False,
        severity="error",
        message="Adapter capability artifact must keep supports_apply false",
        observed=adapter_capabilities.get("supports_apply"),
    )
    _add_check(
        checks,
        name="engine_supports_materialization_false",
        passed=engine_capabilities.get("supports_materialization") is False,
        severity="error",
        message="Engine metadata probe must keep supports_materialization false",
        observed=engine_capabilities.get("supports_materialization"),
    )
    _add_check(
        checks,
        name="engine_supports_apply_false",
        passed=engine_capabilities.get("supports_apply") is False,
        severity="error",
        message="Engine metadata probe must keep supports_apply false",
        observed=engine_capabilities.get("supports_apply"),
    )


def _validate_tokenizer_spans(
    checks: list[dict[str, object]],
    adapter_data: dict[str, object],
    tokenizer_data: dict[str, object],
    engine_data: dict[str, object],
) -> None:
    spans = tokenizer_data.get("spans")
    adapter_tokenizer_ref = adapter_data.get("tokenizer_ref") or {}
    tokenizer_tokenizer_ref = tokenizer_data.get("tokenizer_ref") or {}
    engine_tokenizer_ref = engine_data.get("tokenizer_ref") or {}
    _add_check(
        checks,
        name="tokenizer_span_count",
        passed=isinstance(spans, list) and len(spans) >= 1,
        severity="error",
        message="Tokenizer span probe must include at least one span",
        observed=None if not isinstance(spans, list) else len(spans),
    )
    if not isinstance(spans, list):
        return
    for index, span in enumerate(spans):
        _add_check(
            checks,
            name=f"tokenizer_span_object_{index}",
            passed=isinstance(span, dict),
            severity="error",
            message="Each tokenizer span entry must be an object/dict",
            observed=span,
        )
        if not isinstance(span, dict):
            continue
        token_start = span.get("token_start")
        token_end = span.get("token_end")
        _add_check(
            checks,
            name=f"tokenizer_span_bounds_{index}",
            passed=(
                isinstance(token_start, int)
                and isinstance(token_end, int)
                and token_start >= 0
                and token_end > token_start
            ),
            severity="error",
            message="Each tokenizer span must satisfy token_start >= 0 and token_end > token_start",
            observed={"token_start": token_start, "token_end": token_end},
        )
        _add_check(
            checks,
            name=f"tokenizer_span_estimated_{index}",
            passed=span.get("token_span_is_estimated") is True,
            severity="error",
            message="Each tokenizer span must remain estimated in Phase 12-H",
            observed=span.get("token_span_is_estimated"),
        )
        _add_check(
            checks,
            name=f"tokenizer_span_scoped_{index}",
            passed=span.get("tokenizer_scoped") is True,
            severity="error",
            message="Each tokenizer span must remain tokenizer_scoped",
            observed=span.get("tokenizer_scoped"),
        )
        lineage = span.get("lineage")
        _add_check(
            checks,
            name=f"tokenizer_span_lineage_{index}",
            passed=isinstance(lineage, dict),
            severity="error",
            message="Each tokenizer span must include lineage mapping",
            observed=lineage,
        )
        engine_block_ref = None if not isinstance(lineage, dict) else lineage.get("engine_block_ref")
        _add_check(
            checks,
            name=f"tokenizer_span_engine_block_ref_{index}",
            passed=engine_block_ref is None,
            severity="error",
            message="Tokenizer span lineage.engine_block_ref must remain null in Phase 12-H",
            observed=engine_block_ref,
        )
        span_tokenizer_ref = span.get("tokenizer_ref")
        _add_check(
            checks,
            name=f"tokenizer_span_tokenizer_ref_present_{index}",
            passed=isinstance(span_tokenizer_ref, dict),
            severity="error",
            message="Each tokenizer span must include tokenizer_ref",
            observed=span_tokenizer_ref,
        )
        if not isinstance(span_tokenizer_ref, dict):
            continue
        for field_name in (
            "tokenizer_name_or_path",
            "tokenizer_revision",
            "tokenizer_config_hash",
            "tokenizer_family",
        ):
            observed = {
                "adapter": adapter_tokenizer_ref.get(field_name),
                "tokenizer_top_level": tokenizer_tokenizer_ref.get(field_name),
                "engine": engine_tokenizer_ref.get(field_name),
                "span": span_tokenizer_ref.get(field_name),
            }
            _add_check(
                checks,
                name=f"tokenizer_span_{field_name}_consistency_{index}",
                passed=(
                    observed["adapter"]
                    == observed["tokenizer_top_level"]
                    == observed["engine"]
                    == observed["span"]
                ),
                severity="error",
                message=f"Tokenizer span tokenizer_ref.{field_name} must match adapter, tokenizer top-level, and engine tokenizer_ref",
                observed=observed,
            )


def _validate_engine_metadata(
    checks: list[dict[str, object]],
    engine_data: dict[str, object],
) -> None:
    engine_metadata = engine_data.get("engine_metadata") or {}
    for field_name in (
        "model_loaded",
        "tokenizer_loaded",
        "gpu_inspected",
        "model_config_loaded",
    ):
        observed = engine_metadata.get(field_name)
        _add_check(
            checks,
            name=f"engine_metadata_{field_name}",
            passed=observed is False,
            severity="error",
            message=f"Engine metadata must keep {field_name} false in Phase 12-H",
            observed=observed,
        )
    attention_type_hint = engine_metadata.get("attention_type_hint")
    kv_head_group_count = engine_metadata.get("kv_head_group_count")
    _add_check(
        checks,
        name="engine_metadata_attention_type_hint",
        passed=attention_type_hint in {"mha", "gqa", "unknown"},
        severity="error",
        message="Engine metadata attention_type_hint must be one of mha/gqa/unknown",
        observed=attention_type_hint,
    )
    if attention_type_hint == "gqa":
        passed = isinstance(kv_head_group_count, int) and kv_head_group_count > 0
        message = "GQA engine metadata must expose positive kv_head_group_count"
    elif attention_type_hint == "mha":
        passed = kv_head_group_count == 1
        message = "MHA engine metadata must expose kv_head_group_count == 1"
    else:
        passed = kv_head_group_count is None
        message = "Unknown attention metadata should leave kv_head_group_count null"
    _add_check(
        checks,
        name="engine_metadata_kv_head_group_count",
        passed=passed,
        severity="error",
        message=message,
        observed=kv_head_group_count,
    )


def build_hf_adapter_readiness_report(
    *,
    adapter_capabilities_path: Path,
    tokenizer_span_probe_path: Path,
    engine_metadata_probe_path: Path,
    output_path: Path,
    strict: bool,
) -> tuple[dict[str, object], int]:
    adapter_data = _load_json(adapter_capabilities_path, "adapter capabilities artifact")
    tokenizer_data = _load_json(tokenizer_span_probe_path, "tokenizer span probe artifact")
    engine_data = _load_json(engine_metadata_probe_path, "engine metadata probe artifact")

    checks: list[dict[str, object]] = []
    _validate_schema_versions(checks, adapter_data, tokenizer_data, engine_data)
    _validate_artifact_identity(checks, adapter_data, tokenizer_data, engine_data)
    _validate_adapter_and_runtime_consistency(checks, adapter_data, tokenizer_data, engine_data)
    _validate_model_and_tokenizer_consistency(checks, adapter_data, tokenizer_data, engine_data)
    _validate_safety_scope(checks, adapter_data, tokenizer_data, engine_data)
    _validate_no_apply_materialization(checks, adapter_data, engine_data)
    _validate_tokenizer_spans(checks, adapter_data, tokenizer_data, engine_data)
    _validate_engine_metadata(checks, engine_data)

    failed_error_checks = [
        check for check in checks if (not check["passed"] and check["severity"] == "error")
    ]
    failed_warning_checks = [
        check for check in checks if (not check["passed"] and check["severity"] == "warning")
    ]
    blocking_reasons = [str(check["message"]) for check in failed_error_checks]
    warning_reasons = [str(check["message"]) for check in failed_warning_checks]
    ready_for_next_metadata_step = len(failed_error_checks) == 0

    report = {
        "schema_version": "relaystack.hf_adapter_readiness_report.v0.1",
        "phase": "12-H",
        "artifact_kind": "hf_adapter_readiness_report",
        "input_refs": {
            "adapter_capabilities_path": str(adapter_capabilities_path),
            "tokenizer_span_probe_path": str(tokenizer_span_probe_path),
            "engine_metadata_probe_path": str(engine_metadata_probe_path),
        },
        "readiness": {
            "ready_for_next_metadata_step": ready_for_next_metadata_step,
            "ready_for_real_tokenizer_probe": ready_for_next_metadata_step,
            "ready_for_model_config_probe": ready_for_next_metadata_step,
            "ready_for_materialization": False,
            "ready_for_apply": False,
            "blocking_reasons": blocking_reasons,
            "warning_reasons": warning_reasons,
        },
        "checks": checks,
        "safety_scope": {
            "dry_run_only": True,
            "no_model_loading_required": True,
            "no_tokenizer_loading_required": True,
            "no_gpu_inspection": True,
            "no_kv_materialization": True,
            "no_attention_connection": True,
            "no_scheduler_change": True,
            "no_runtime_apply": True,
        },
        "notes": [
            "phase12_h_hf_adapter_readiness_report",
            "consistency_and_safety_only",
            "no_model_load",
            "no_tokenizer_load",
            "no_gpu_inspection",
        ],
    }
    report["summary"] = {
        "ok": ready_for_next_metadata_step,
        "artifact_kind": "hf_adapter_readiness_report",
        "check_count": len(checks),
        "passed_check_count": sum(1 for check in checks if check["passed"]),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
        "ready_for_next_metadata_step": ready_for_next_metadata_step,
        "ready_for_real_tokenizer_probe": ready_for_next_metadata_step,
        "ready_for_model_config_probe": ready_for_next_metadata_step,
        "ready_for_materialization": False,
        "ready_for_apply": False,
        "output_path": str(output_path),
        "strict": strict,
    }
    return report, (0 if ready_for_next_metadata_step else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-capabilities", type=Path, required=True)
    parser.add_argument("--tokenizer-span-probe", type=Path, required=True)
    parser.add_argument("--engine-metadata-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    adapter_capabilities_path = args.adapter_capabilities.expanduser().resolve()
    tokenizer_span_probe_path = args.tokenizer_span_probe.expanduser().resolve()
    engine_metadata_probe_path = args.engine_metadata_probe.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    try:
        report, exit_code = build_hf_adapter_readiness_report(
            adapter_capabilities_path=adapter_capabilities_path,
            tokenizer_span_probe_path=tokenizer_span_probe_path,
            engine_metadata_probe_path=engine_metadata_probe_path,
            output_path=output_path,
            strict=args.strict,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "artifact_kind": "hf_adapter_readiness_report",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
