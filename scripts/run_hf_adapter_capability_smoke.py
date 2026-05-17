#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"


def build_hf_adapter_capability_smoke_payload(
    *,
    output: Path,
    model_id: str,
    tokenizer_name_or_path: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    local_path: str | None,
    quantization_hint: str | None,
    dtype_hint: str | None,
    max_model_context_tokens: int | None,
    configured_context_tokens: int | None,
    context_window_source: str,
) -> dict[str, object]:
    payload = {
        "schema_version": "relaystack.adapter_capabilities.v0.1",
        "phase": "12-C",
        "adapter_kind": "hf_prototype",
        "adapter_name": "hf_local_prototype",
        "runtime_target": "huggingface_transformers",
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
        "capabilities": {
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
            "no_kv_materialization": True,
            "no_attention_connection": True,
            "no_scheduler_change": True,
            "no_runtime_apply": True,
        },
        "notes": [
            "phase12_c_metadata_smoke_only",
            "no_model_loading",
            "no_gpu_inspection",
            "no_runtime_apply",
        ],
    }
    payload["summary"] = {
        "ok": True,
        "artifact_kind": "hf_adapter_capability_smoke",
        "no_model_loaded": True,
        "no_gpu_inspection": True,
        "supports_apply": payload["capabilities"]["supports_apply"],
        "supports_materialization": payload["capabilities"]["supports_materialization"],
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
    parser.add_argument("--max-model-context-tokens", type=int, default=None)
    parser.add_argument("--configured-context-tokens", type=int, default=None)
    parser.add_argument("--context-window-source", default="user_override_or_unknown")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path is not None
        else args.model_id
    )
    payload = build_hf_adapter_capability_smoke_payload(
        output=output,
        model_id=args.model_id,
        tokenizer_name_or_path=tokenizer_name_or_path,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        local_path=args.local_path,
        quantization_hint=args.quantization_hint,
        dtype_hint=args.dtype_hint,
        max_model_context_tokens=args.max_model_context_tokens,
        configured_context_tokens=args.configured_context_tokens,
        context_window_source=args.context_window_source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
