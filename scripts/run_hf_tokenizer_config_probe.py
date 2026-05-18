#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


DEFAULT_SAMPLE_TEXT = "RelayStack tokenizer config probe."


def _load_json(path: Path, artifact_name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} at {path} must be a JSON object, not non-object JSON")
    return payload


def _as_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _compact_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _load_upstream_artifact(readiness_data: dict[str, object], key: str) -> dict[str, object] | None:
    input_refs = _as_mapping(readiness_data.get("input_refs"))
    ref_path = input_refs.get(key)
    if not isinstance(ref_path, str) or not ref_path:
        return None
    path = Path(ref_path).expanduser()
    if not path.exists():
        return None
    try:
        return _load_json(path, key)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _resolve_from_sources(
    cli_value: str | None,
    *candidate_values: object,
) -> str | None:
    if cli_value is not None:
        return cli_value
    for value in candidate_values:
        if isinstance(value, str):
            return value
    return None


def _derive_head_dim(hidden_size: object, num_attention_heads: object) -> int | None:
    if not isinstance(hidden_size, int) or not isinstance(num_attention_heads, int):
        return None
    if hidden_size <= 0 or num_attention_heads <= 0:
        return None
    if hidden_size % num_attention_heads != 0:
        return None
    return hidden_size // num_attention_heads


def _resolve_attention_metadata(
    num_attention_heads: object,
    num_key_value_heads: object,
) -> tuple[str, int | None, str | None]:
    if not isinstance(num_attention_heads, int) or not isinstance(num_key_value_heads, int):
        return "unknown", None, None
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        return "unknown", None, "attention head metadata must be positive"
    if num_attention_heads == num_key_value_heads:
        return "mha", 1, None
    if num_key_value_heads > num_attention_heads:
        return (
            "unknown",
            None,
            "num_key_value_heads must not exceed num_attention_heads",
        )
    if num_attention_heads % num_key_value_heads != 0:
        return (
            "unknown",
            None,
            "num_attention_heads must be divisible by num_key_value_heads",
        )
    return "gqa", num_attention_heads // num_key_value_heads, None


def _reasonable_context_window(value: object) -> int | None:
    if not isinstance(value, int):
        return None
    if value <= 0 or value >= 1_000_000_000:
        return None
    return value


def _special_tokens_map(tokenizer: object) -> dict[str, object] | None:
    mapping = getattr(tokenizer, "special_tokens_map", None)
    return mapping if isinstance(mapping, dict) else None


def _tokenizer_family(tokenizer: object) -> str | None:
    class_name = tokenizer.__class__.__name__
    return class_name if isinstance(class_name, str) and class_name else None


def _call_tokenizer(tokenizer: object, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        token_ids = encode(text, add_special_tokens=False)
        if isinstance(token_ids, list):
            return token_ids
    call = getattr(tokenizer, "__call__", None)
    if callable(call):
        payload = call(text, add_special_tokens=False)
        if isinstance(payload, dict):
            token_ids = payload.get("input_ids")
            if isinstance(token_ids, list):
                return token_ids
    raise ValueError("tokenizer did not return input_ids list")


def _stringify_dtype(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _compact_quantization_config(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        compact = to_dict()
        if isinstance(compact, dict):
            return compact
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if isinstance(key, str) and not key.startswith("_")
        }
    return str(value)


def _import_transformers():
    module = importlib.import_module("transformers")
    return module.AutoTokenizer, module.AutoConfig


def _load_tokenizer_probe(
    *,
    tokenizer_name_or_path: str | None,
    tokenizer_revision: str | None,
    local_path: str | None,
    sample_text: str,
    local_files_only: bool,
    skip_load: bool,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    warnings: list[str] = []
    if skip_load:
        return (
            {
                "attempted": False,
                "loaded": False,
                "skipped": True,
                "tokenizer_loading_error": None,
            },
            {
                "tokenizer_name_or_path": tokenizer_name_or_path,
                "tokenizer_revision": tokenizer_revision,
                "tokenizer_config_hash": None,
                "tokenizer_family": None,
            },
            warnings,
        )
    if tokenizer_name_or_path is None and local_path is None:
        return (
            {
                "attempted": False,
                "loaded": False,
                "skipped": False,
                "tokenizer_loading_error": "Tokenizer reference is missing",
            },
            {
                "tokenizer_name_or_path": tokenizer_name_or_path,
                "tokenizer_revision": tokenizer_revision,
                "tokenizer_config_hash": None,
                "tokenizer_family": None,
            },
            warnings,
        )
    try:
        AutoTokenizer, _ = _import_transformers()
        target = local_path or tokenizer_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            target,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
        )
        token_ids = _call_tokenizer(tokenizer, sample_text)
        special_tokens = _special_tokens_map(tokenizer)
        tokenizer_ref = {
            "tokenizer_name_or_path": tokenizer_name_or_path or target,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_config_hash": None,
            "tokenizer_family": _tokenizer_family(tokenizer),
        }
        probe = {
            "attempted": True,
            "loaded": True,
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "sample_text": sample_text,
            "sample_token_count": len(token_ids),
            "sample_token_ids_preview": token_ids[:16],
            "special_tokens_map": special_tokens,
            "tokenizer_loading_error": None,
        }
        return probe, tokenizer_ref, warnings
    except Exception as exc:
        return (
            {
                "attempted": True,
                "loaded": False,
                "skipped": False,
                "tokenizer_loading_error": _compact_error(exc),
            },
            {
                "tokenizer_name_or_path": tokenizer_name_or_path,
                "tokenizer_revision": tokenizer_revision,
                "tokenizer_config_hash": None,
                "tokenizer_family": None,
            },
            warnings,
        )


def _load_config_probe(
    *,
    model_id: str | None,
    model_revision: str | None,
    local_path: str | None,
    local_files_only: bool,
    skip_load: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if skip_load:
        return (
            {
                "attempted": False,
                "loaded": False,
                "skipped": True,
                "config_loading_error": None,
            },
            errors,
            warnings,
        )
    if model_id is None and local_path is None:
        return (
            {
                "attempted": False,
                "loaded": False,
                "skipped": False,
                "config_loading_error": "Model reference is missing",
            },
            ["Model reference is missing"],
            warnings,
        )
    try:
        _, AutoConfig = _import_transformers()
        target = local_path or model_id
        config = AutoConfig.from_pretrained(
            target,
            revision=model_revision,
            local_files_only=local_files_only,
        )
        num_attention_heads = getattr(config, "num_attention_heads", None)
        num_key_value_heads = getattr(config, "num_key_value_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = _derive_head_dim(hidden_size, num_attention_heads)
        probe = {
            "attempted": True,
            "loaded": True,
            "config_class": config.__class__.__name__,
            "model_type": getattr(config, "model_type", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "intermediate_size": getattr(config, "intermediate_size", None),
            "torch_dtype": _stringify_dtype(getattr(config, "torch_dtype", None)),
            "quantization_config": _compact_quantization_config(
                getattr(config, "quantization_config", None)
            ),
            "config_loading_error": None,
        }
        attention_type_hint, kv_head_group_count, attention_error = _resolve_attention_metadata(
            num_attention_heads,
            num_key_value_heads,
        )
        if attention_error is not None:
            errors.append(attention_error)
        probe["attention_type_hint"] = attention_type_hint
        probe["kv_head_group_count"] = kv_head_group_count
        return probe, errors, warnings
    except Exception as exc:
        return (
            {
                "attempted": True,
                "loaded": False,
                "skipped": False,
                "config_loading_error": _compact_error(exc),
            },
            [_compact_error(exc)],
            warnings,
        )


def _build_probe_artifact(
    *,
    output_path: Path,
    readiness_path: Path,
    readiness_data: dict[str, object],
    adapter_data: dict[str, object] | None,
    tokenizer_data: dict[str, object] | None,
    engine_data: dict[str, object] | None,
    model_ref: dict[str, object],
    tokenizer_ref: dict[str, object],
    tokenizer_probe: dict[str, object],
    config_probe: dict[str, object],
    safety_scope: dict[str, object],
    errors: list[str],
    warnings: list[str],
    blocked_reason: str | None,
) -> dict[str, object]:
    readiness_summary = _as_mapping(readiness_data.get("summary"))
    readiness_ref = {
        "path": str(readiness_path),
        "ok": readiness_summary.get("ok"),
        "ready_for_next_metadata_step": _as_mapping(readiness_data.get("readiness")).get(
            "ready_for_next_metadata_step"
        ),
        "ready_for_real_tokenizer_probe": _as_mapping(readiness_data.get("readiness")).get(
            "ready_for_real_tokenizer_probe"
        ),
        "ready_for_model_config_probe": _as_mapping(readiness_data.get("readiness")).get(
            "ready_for_model_config_probe"
        ),
    }
    adapter_model_ref = _as_mapping((adapter_data or {}).get("model_ref"))
    adapter_tokenizer_ref = _as_mapping((adapter_data or {}).get("tokenizer_ref"))
    tokenizer_span_ref = _as_mapping((tokenizer_data or {}).get("tokenizer_ref"))
    span_list = (tokenizer_data or {}).get("spans")
    sample_span_token_count_available = (
        isinstance(span_list, list)
        and len(span_list) >= 1
        and isinstance(span_list[0], dict)
        and isinstance(span_list[0].get("estimated_token_count"), int)
        and span_list[0]["estimated_token_count"] > 0
    )
    tokenizer_model_max_length = tokenizer_probe.get("model_max_length")
    config_max_position_embeddings = config_probe.get("max_position_embeddings")
    context_window_candidate = _reasonable_context_window(config_max_position_embeddings)
    if context_window_candidate is None:
        context_window_candidate = _reasonable_context_window(tokenizer_model_max_length)
    consistency = {
        "readiness_model_ref_match": (
            model_ref.get("model_id") == adapter_model_ref.get("model_id")
            and (
                model_ref.get("model_revision") is None
                or model_ref.get("model_revision") == adapter_model_ref.get("model_revision")
            )
        ),
        "readiness_tokenizer_ref_match": (
            tokenizer_ref.get("tokenizer_name_or_path")
            in {
                adapter_tokenizer_ref.get("tokenizer_name_or_path"),
                tokenizer_span_ref.get("tokenizer_name_or_path"),
            }
        ),
        "sample_span_token_count_available": sample_span_token_count_available,
        "config_attention_type_hint": config_probe.get("attention_type_hint", "unknown"),
        "kv_head_group_count": config_probe.get("kv_head_group_count"),
        "context_window_candidate": context_window_candidate,
    }
    ok = len(errors) == 0
    readiness_flags = _as_mapping(readiness_data.get("readiness"))
    return {
        "schema_version": "relaystack.hf_tokenizer_config_probe.v0.1",
        "phase": "12-I",
        "artifact_kind": "hf_tokenizer_config_probe",
        "input_refs": {
            "readiness_report_path": str(readiness_path),
            "adapter_capabilities_path": _as_mapping(readiness_data.get("input_refs")).get(
                "adapter_capabilities_path"
            ),
            "tokenizer_span_probe_path": _as_mapping(readiness_data.get("input_refs")).get(
                "tokenizer_span_probe_path"
            ),
            "engine_metadata_probe_path": _as_mapping(readiness_data.get("input_refs")).get(
                "engine_metadata_probe_path"
            ),
        },
        "model_ref": model_ref,
        "tokenizer_ref": tokenizer_ref,
        "readiness_ref": readiness_ref,
        "tokenizer_probe": tokenizer_probe,
        "config_probe": config_probe,
        "consistency": consistency,
        "safety_scope": safety_scope,
        "summary": {
            "ok": ok,
            "artifact_kind": "hf_tokenizer_config_probe",
            "tokenizer_loaded": tokenizer_probe.get("loaded") is True,
            "config_loaded": config_probe.get("loaded") is True,
            "ready_for_next_metadata_step": ok,
            "ready_for_materialization": False,
            "ready_for_apply": False,
            "blocked_reason": blocked_reason,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "output_path": str(output_path),
        },
        "notes": [
            "phase12_i_hf_tokenizer_config_probe",
            "tokenizer_config_metadata_only",
            "no_model_load",
            "no_gpu_inspection",
            f"readiness_ready_for_real_tokenizer_probe={readiness_flags.get('ready_for_real_tokenizer_probe')}",
            f"readiness_ready_for_model_config_probe={readiness_flags.get('ready_for_model_config_probe')}",
        ]
        + warnings
        + errors,
    }


def build_hf_tokenizer_config_probe_payload(
    *,
    readiness_path: Path,
    output_path: Path,
    model_id: str | None,
    tokenizer_name_or_path: str | None,
    model_revision: str | None,
    tokenizer_revision: str | None,
    local_path: str | None,
    sample_text: str,
    local_files_only: bool,
    allow_network: bool,
    skip_tokenizer_load: bool,
    skip_config_load: bool,
) -> tuple[dict[str, object] | None, int]:
    readiness_data = _load_json(readiness_path, "readiness report")
    readiness_summary = _as_mapping(readiness_data.get("summary"))
    if readiness_summary.get("ok") is not True:
        blocked_reason = "readiness summary.ok is not true"
        artifact = _build_probe_artifact(
            output_path=output_path,
            readiness_path=readiness_path,
            readiness_data=readiness_data,
            adapter_data=None,
            tokenizer_data=None,
            engine_data=None,
            model_ref={
                "model_id": model_id,
                "model_revision": model_revision,
                "local_path": local_path,
            },
            tokenizer_ref={
                "tokenizer_name_or_path": tokenizer_name_or_path,
                "tokenizer_revision": tokenizer_revision,
                "tokenizer_config_hash": None,
                "tokenizer_family": None,
            },
            tokenizer_probe={
                "attempted": False,
                "loaded": False,
                "skipped": skip_tokenizer_load,
                "tokenizer_loading_error": None,
            },
            config_probe={
                "attempted": False,
                "loaded": False,
                "skipped": skip_config_load,
                "config_loading_error": None,
            },
            safety_scope={
                "dry_run_only": True,
                "model_loaded": False,
                "tokenizer_loaded": False,
                "config_loaded": False,
                "no_model_loading": True,
                "no_gpu_inspection": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
                "local_files_only": local_files_only,
                "allow_network": allow_network,
            },
            errors=[blocked_reason],
            warnings=[],
            blocked_reason=blocked_reason,
        )
        return artifact, 1

    readiness_flags = _as_mapping(readiness_data.get("readiness"))
    if not skip_tokenizer_load and readiness_flags.get("ready_for_real_tokenizer_probe") is not True:
        blocked_reason = "readiness report does not allow real tokenizer probe"
    elif not skip_config_load and readiness_flags.get("ready_for_model_config_probe") is not True:
        blocked_reason = "readiness report does not allow model config probe"
    else:
        blocked_reason = None

    adapter_data = _load_upstream_artifact(readiness_data, "adapter_capabilities_path")
    tokenizer_data = _load_upstream_artifact(readiness_data, "tokenizer_span_probe_path")
    engine_data = _load_upstream_artifact(readiness_data, "engine_metadata_probe_path")

    adapter_model_ref = _as_mapping((adapter_data or {}).get("model_ref"))
    tokenizer_model_ref = _as_mapping((tokenizer_data or {}).get("model_ref"))
    engine_model_ref = _as_mapping((engine_data or {}).get("model_ref"))
    adapter_tokenizer_ref = _as_mapping((adapter_data or {}).get("tokenizer_ref"))
    tokenizer_top_ref = _as_mapping((tokenizer_data or {}).get("tokenizer_ref"))
    engine_tokenizer_ref = _as_mapping((engine_data or {}).get("tokenizer_ref"))

    resolved_model_id = _resolve_from_sources(
        model_id,
        adapter_model_ref.get("model_id"),
        tokenizer_model_ref.get("model_id"),
        engine_model_ref.get("model_id"),
    )
    resolved_model_revision = _resolve_from_sources(
        model_revision,
        adapter_model_ref.get("model_revision"),
        engine_model_ref.get("model_revision"),
        tokenizer_model_ref.get("model_revision"),
    )
    resolved_local_path = _resolve_from_sources(
        local_path,
        adapter_model_ref.get("local_path"),
        engine_model_ref.get("local_path"),
        tokenizer_model_ref.get("local_path"),
    )
    resolved_tokenizer_name = _resolve_from_sources(
        tokenizer_name_or_path,
        adapter_tokenizer_ref.get("tokenizer_name_or_path"),
        tokenizer_top_ref.get("tokenizer_name_or_path"),
        engine_tokenizer_ref.get("tokenizer_name_or_path"),
        resolved_model_id,
    )
    resolved_tokenizer_revision = _resolve_from_sources(
        tokenizer_revision,
        adapter_tokenizer_ref.get("tokenizer_revision"),
        tokenizer_top_ref.get("tokenizer_revision"),
        engine_tokenizer_ref.get("tokenizer_revision"),
    )

    errors: list[str] = []
    warnings: list[str] = []

    if blocked_reason is not None:
        errors.append(blocked_reason)
        artifact = _build_probe_artifact(
            output_path=output_path,
            readiness_path=readiness_path,
            readiness_data=readiness_data,
            adapter_data=adapter_data,
            tokenizer_data=tokenizer_data,
            engine_data=engine_data,
            model_ref={
                "model_id": resolved_model_id,
                "model_revision": resolved_model_revision,
                "local_path": resolved_local_path,
            },
            tokenizer_ref={
                "tokenizer_name_or_path": resolved_tokenizer_name,
                "tokenizer_revision": resolved_tokenizer_revision,
                "tokenizer_config_hash": None,
                "tokenizer_family": None,
            },
            tokenizer_probe={
                "attempted": False,
                "loaded": False,
                "skipped": skip_tokenizer_load,
                "tokenizer_loading_error": None,
            },
            config_probe={
                "attempted": False,
                "loaded": False,
                "skipped": skip_config_load,
                "config_loading_error": None,
            },
            safety_scope={
                "dry_run_only": True,
                "model_loaded": False,
                "tokenizer_loaded": False,
                "config_loaded": False,
                "no_model_loading": True,
                "no_gpu_inspection": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
                "local_files_only": local_files_only,
                "allow_network": allow_network,
            },
            errors=errors,
            warnings=warnings,
            blocked_reason=blocked_reason,
        )
        return artifact, 1

    tokenizer_probe, probed_tokenizer_ref, tokenizer_warnings = _load_tokenizer_probe(
        tokenizer_name_or_path=resolved_tokenizer_name,
        tokenizer_revision=resolved_tokenizer_revision,
        local_path=resolved_local_path,
        sample_text=sample_text,
        local_files_only=local_files_only,
        skip_load=skip_tokenizer_load,
    )
    warnings.extend(tokenizer_warnings)
    if tokenizer_probe.get("attempted") is True and tokenizer_probe.get("loaded") is not True:
        error = tokenizer_probe.get("tokenizer_loading_error")
        errors.append(str(error))

    config_probe, config_errors, config_warnings = _load_config_probe(
        model_id=resolved_model_id,
        model_revision=resolved_model_revision,
        local_path=resolved_local_path,
        local_files_only=local_files_only,
        skip_load=skip_config_load,
    )
    errors.extend(config_errors)
    warnings.extend(config_warnings)
    if config_probe.get("attempted") is True and config_probe.get("loaded") is not True:
        error = config_probe.get("config_loading_error")
        if error is not None and str(error) not in errors:
            errors.append(str(error))

    tokenizer_ref_payload = {
        "tokenizer_name_or_path": probed_tokenizer_ref.get("tokenizer_name_or_path")
        or resolved_tokenizer_name,
        "tokenizer_revision": probed_tokenizer_ref.get("tokenizer_revision")
        if probed_tokenizer_ref.get("tokenizer_revision") is not None
        else resolved_tokenizer_revision,
        "tokenizer_config_hash": None,
        "tokenizer_family": probed_tokenizer_ref.get("tokenizer_family")
        or adapter_tokenizer_ref.get("tokenizer_family")
        or tokenizer_top_ref.get("tokenizer_family")
        or engine_tokenizer_ref.get("tokenizer_family"),
    }

    artifact = _build_probe_artifact(
        output_path=output_path,
        readiness_path=readiness_path,
        readiness_data=readiness_data,
        adapter_data=adapter_data,
        tokenizer_data=tokenizer_data,
        engine_data=engine_data,
        model_ref={
            "model_id": resolved_model_id,
            "model_revision": resolved_model_revision,
            "local_path": resolved_local_path,
        },
        tokenizer_ref=tokenizer_ref_payload,
        tokenizer_probe=tokenizer_probe,
        config_probe=config_probe,
        safety_scope={
            "dry_run_only": True,
            "model_loaded": False,
            "tokenizer_loaded": tokenizer_probe.get("loaded") is True,
            "config_loaded": config_probe.get("loaded") is True,
            "no_model_loading": True,
            "no_gpu_inspection": True,
            "no_kv_materialization": True,
            "no_attention_connection": True,
            "no_scheduler_change": True,
            "no_runtime_apply": True,
            "local_files_only": local_files_only,
            "allow_network": allow_network,
        },
        errors=errors,
        warnings=warnings,
        blocked_reason=None,
    )
    return artifact, (0 if artifact["summary"]["ok"] is True else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--sample-text", default=DEFAULT_SAMPLE_TEXT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--skip-tokenizer-load", action="store_true")
    parser.add_argument("--skip-config-load", action="store_true")
    args = parser.parse_args()

    readiness_path = args.readiness_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    local_files_only = (not args.allow_network) or args.local_files_only

    try:
        artifact, exit_code = build_hf_tokenizer_config_probe_payload(
            readiness_path=readiness_path,
            output_path=output_path,
            model_id=args.model_id,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            local_path=args.local_path,
            sample_text=args.sample_text,
            local_files_only=local_files_only,
            allow_network=args.allow_network,
            skip_tokenizer_load=args.skip_tokenizer_load,
            skip_config_load=args.skip_config_load,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "artifact_kind": "hf_tokenizer_config_probe",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2

    if artifact is None:
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
