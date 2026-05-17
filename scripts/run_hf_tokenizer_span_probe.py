#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
DEFAULT_TEXT = "RelayStack tokenizer span probe."


def _estimate_token_count(text: str, estimated_token_count: int | None) -> int:
    if estimated_token_count is not None:
        return estimated_token_count
    return max(1, len(text.split()))


def build_hf_tokenizer_span_probe_payload(
    *,
    output: Path,
    model_id: str,
    tokenizer_name_or_path: str,
    tokenizer_revision: str | None,
    tokenizer_config_hash: str | None,
    tokenizer_family: str,
    text: str,
    source_item_id: str,
    span_kind: str,
    estimated_token_count: int | None,
) -> dict[str, object]:
    token_end = _estimate_token_count(text, estimated_token_count)
    tokenizer_ref = {
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_config_hash": tokenizer_config_hash,
        "tokenizer_family": tokenizer_family,
    }
    payload = {
        "schema_version": "relaystack.tokenizer_span_probe.v0.1",
        "phase": "12-E",
        "artifact_kind": "hf_tokenizer_span_probe",
        "adapter_kind": "hf_prototype",
        "runtime_target": "huggingface_transformers",
        "model_ref": {
            "model_id": model_id,
            "model_revision": None,
            "local_path": None,
        },
        "tokenizer_ref": tokenizer_ref,
        "input": {
            "text": text,
            "source_item_id": source_item_id,
        },
        "spans": [
            {
                "source_item_id": source_item_id,
                "span_kind": span_kind,
                "token_start": 0,
                "token_end": token_end,
                "estimated_token_count": token_end,
                "token_span_is_estimated": True,
                "tokenizer_scoped": True,
                "tokenizer_ref": tokenizer_ref,
                "span_confidence": "synthetic_estimate",
                "span_source": "phase12_e_cli_estimate",
                "lineage": {
                    "relaymem_item_id": source_item_id,
                    "relayctx_source_item_id": source_item_id,
                    "logical_block_id": None,
                    "engine_block_ref": None,
                },
            }
        ],
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
            "phase12_e_tokenizer_span_probe",
            "token_span_is_estimated",
            "tokenizer_scoped_metadata_only",
            "no_tokenizer_load",
        ],
    }
    payload["summary"] = {
        "ok": True,
        "artifact_kind": "hf_tokenizer_span_probe",
        "no_model_loaded": True,
        "no_tokenizer_loaded": True,
        "no_gpu_inspection": True,
        "span_count": 1,
        "token_span_is_estimated": True,
        "tokenizer_scoped": True,
        "output_path": str(output),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-name-or-path")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--tokenizer-config-hash", default=None)
    parser.add_argument("--tokenizer-family", default="hf_auto_tokenizer")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--source-item-id",
        default="synthetic:phase12-tokenizer-span-probe",
    )
    parser.add_argument("--span-kind", default="prompt_text")
    parser.add_argument("--estimated-token-count", type=int, default=None)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path is not None
        else args.model_id
    )
    payload = build_hf_tokenizer_span_probe_payload(
        output=output,
        model_id=args.model_id,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer_config_hash=args.tokenizer_config_hash,
        tokenizer_family=args.tokenizer_family,
        text=args.text,
        source_item_id=args.source_item_id,
        span_kind=args.span_kind,
        estimated_token_count=args.estimated_token_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
