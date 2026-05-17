#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"


def _validate_positive_optional_int(
    parser: argparse.ArgumentParser,
    flag_name: str,
    value: int | None,
) -> None:
    if value is not None and value <= 0:
        parser.error(f"{flag_name} must be positive")


def _resolve_attention_metadata(
    parser: argparse.ArgumentParser,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
) -> tuple[str, int | None]:
    if num_attention_heads is None or num_key_value_heads is None:
        return "unknown", None
    if num_key_value_heads == num_attention_heads:
        return "mha", 1
    if num_key_value_heads > num_attention_heads:
        parser.error("--num-key-value-heads must not exceed --num-attention-heads")
    if num_attention_heads % num_key_value_heads != 0:
        parser.error(
            "--num-attention-heads must be divisible by --num-key-value-heads"
        )
    return "gqa", num_attention_heads // num_key_value_heads


def build_hf_engine_metadata_probe_payload(
    *,
    parser: argparse.ArgumentParser,
    output: Path,
    model_id: str,
    tokenizer_name_or_path: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    local_path: str | None,
    quantization_hint: str | None,
    dtype_hint: str | None,
    device_target: str,
    runtime_target: str,
    adapter_name: str,
    max_model_context_tokens: int | None,
    configured_context_tokens: int | None,
    context_window_source: str,
    num_hidden_layers: int | None,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
    head_dim: int | None,
    hidden_size: int | None,
) -> dict[str, object]:
    attention_type_hint, kv_head_group_count = _resolve_attention_metadata(
        parser,
        num_attention_heads,
        num_key_value_heads,
    )
    payload = {
        "schema_version": "relaystack.engine_metadata_probe.v0.1",
        "phase": "12-F",
        "artifact_kind": "hf_engine_metadata_probe",
        "adapter_kind": "hf_prototype",
        "runtime_target": runtime_target,
        "adapter_name": adapter_name,
        "model_ref": {
            "model_id": model_id,
            "model_revision": model_revision,
            "local_path": local_path,
            "quantization_hint": quantization_hint,
            "dtype_hint": dtype_hint,
        },
        "tokenizer_ref": {
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_config_hash": None,
            "tokenizer_family": "hf_auto_tokenizer",
        },
        "context_window_hint": {
            "max_model_context_tokens": max_model_context_tokens,
            "configured_context_tokens": configured_context_tokens,
            "source": context_window_source,
        },
        "engine_metadata": {
            "device_target": device_target,
            "config_source": "cli_or_unknown",
            "model_config_loaded": False,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "gpu_inspected": False,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "attention_type_hint": attention_type_hint,
            "kv_head_group_count": kv_head_group_count,
        },
        "capabilities_snapshot": {
            "supports_tokenizer_span_probe": True,
            "supports_engine_metadata_probe": True,
            "supports_fullkv_reference": True,
            "supports_shadow_compare": False,
            "supports_materialization": False,
            "supports_apply": False,
            "supports_safe_degrade": False,
            "supports_context_reduction_request": True,
        },
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
            "phase12_f_engine_metadata_probe",
            "cli_or_unknown_metadata_only",
            "no_model_load",
            "no_tokenizer_load",
            "no_gpu_inspection",
        ],
    }
    payload["summary"] = {
        "ok": True,
        "artifact_kind": "hf_engine_metadata_probe",
        "no_model_loaded": True,
        "no_tokenizer_loaded": True,
        "no_gpu_inspection": True,
        "supports_apply": False,
        "supports_materialization": False,
        "output_path": str(output),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-name-or-path")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--quantization-hint", default="awq")
    parser.add_argument("--dtype-hint", default=None)
    parser.add_argument("--device-target", default="cuda")
    parser.add_argument("--runtime-target", default="huggingface_transformers")
    parser.add_argument("--adapter-name", default="hf_local_prototype")
    parser.add_argument("--max-model-context-tokens", type=int, default=None)
    parser.add_argument("--configured-context-tokens", type=int, default=None)
    parser.add_argument("--context-window-source", default="user_override_or_unknown")
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    args = parser.parse_args()

    _validate_positive_optional_int(
        parser,
        "--max-model-context-tokens",
        args.max_model_context_tokens,
    )
    _validate_positive_optional_int(
        parser,
        "--configured-context-tokens",
        args.configured_context_tokens,
    )
    _validate_positive_optional_int(parser, "--num-hidden-layers", args.num_hidden_layers)
    _validate_positive_optional_int(
        parser,
        "--num-attention-heads",
        args.num_attention_heads,
    )
    _validate_positive_optional_int(
        parser,
        "--num-key-value-heads",
        args.num_key_value_heads,
    )
    _validate_positive_optional_int(parser, "--head-dim", args.head_dim)
    _validate_positive_optional_int(parser, "--hidden-size", args.hidden_size)

    output = args.output.expanduser().resolve()
    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path is not None
        else args.model_id
    )
    payload = build_hf_engine_metadata_probe_payload(
        parser=parser,
        output=output,
        model_id=args.model_id,
        tokenizer_name_or_path=tokenizer_name_or_path,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        local_path=args.local_path,
        quantization_hint=args.quantization_hint,
        dtype_hint=args.dtype_hint,
        device_target=args.device_target,
        runtime_target=args.runtime_target,
        adapter_name=args.adapter_name,
        max_model_context_tokens=args.max_model_context_tokens,
        configured_context_tokens=args.configured_context_tokens,
        context_window_source=args.context_window_source,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
