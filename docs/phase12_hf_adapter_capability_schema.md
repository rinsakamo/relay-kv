# Phase 12-B HF Adapter Capability Schema

Design status: 2026-05-17 JST.

This document records the Phase 12-B HF adapter capability schema and skeleton plan for RelayStack V0.1. It is a docs/planning artifact only. It does not implement a runtime adapter, load a model, materialize KV, connect attention, mutate an engine KV pool, or change scheduler behavior.

## Goal

Phase 12-A selected the HF prototype adapter as the first RelayStack V0.1 validation target. Phase 12-B defines the minimum JSON-safe capability and metadata schema that an HF adapter should emit before any runtime integration or RelayKV apply work begins.

The immediate target is a capability artifact, not execution control:

```text
HF environment / model metadata
  -> HF adapter capability schema
  -> relaystack_adapter_capabilities.json
  -> later metadata smoke artifacts
```

## Non-goals

Phase 12-B does not implement:

- model loading
- generation
- KV materialization
- attention comparison
- RelayKV apply
- scheduler or KV-pool mutation
- SGLang or vLLM adapter code
- Open-LLM-VTuber integration
- tool execution or approval UX

## Capability artifact

The first HF adapter artifact should be JSON-safe and stable enough for offline replay.

Suggested filename:

```text
relaystack_adapter_capabilities.json
```

Suggested top-level shape:

```json
{
  "schema_version": "relaystack.adapter_capabilities.v0.1",
  "phase": "12-B",
  "adapter_kind": "hf_prototype",
  "adapter_name": "hf_local_prototype",
  "runtime_target": "huggingface_transformers",
  "model_ref": {
    "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "model_revision": null,
    "local_path": null,
    "quantization_hint": "awq",
    "dtype_hint": null
  },
  "tokenizer_ref": {
    "tokenizer_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "tokenizer_revision": null,
    "tokenizer_config_hash": null,
    "tokenizer_family": "hf_auto_tokenizer"
  },
  "context_window_hint": {
    "max_model_context_tokens": null,
    "configured_context_tokens": null,
    "source": "model_config_or_user_override"
  },
  "capabilities": {
    "supports_tokenizer_span_probe": true,
    "supports_engine_metadata_probe": true,
    "supports_fullkv_reference": true,
    "supports_shadow_compare": false,
    "supports_materialization": false,
    "supports_apply": false,
    "supports_safe_degrade": false,
    "supports_context_reduction_request": true
  },
  "safety_scope": {
    "dry_run_only": true,
    "no_model_loading_required": true,
    "no_kv_materialization": true,
    "no_attention_connection": true,
    "no_scheduler_change": true,
    "no_runtime_apply": true
  },
  "notes": []
}
```

## Required fields

The capability artifact must include:

- `schema_version`
- `phase`
- `adapter_kind`
- `adapter_name`
- `runtime_target`
- `model_ref`
- `tokenizer_ref`
- `context_window_hint`
- `capabilities`
- `safety_scope`

## Capability semantics

### supports_tokenizer_span_probe

Whether the adapter can produce tokenizer-scoped token-span information for RelayCTX / RelayKV lineage validation.

HF V0.1 target:

```text
true
```

### supports_engine_metadata_probe

Whether the adapter can emit engine/model metadata without mutating runtime state.

HF V0.1 target:

```text
true
```

### supports_fullkv_reference

Whether the adapter can use the ordinary full-context path as a reference for later shadow comparison.

HF V0.1 target:

```text
true
```

This does not mean FullKV is always available under VRAM pressure after RelayKV apply. It only means the HF prototype path can provide a reference during safe validation phases.

### supports_shadow_compare

Whether the adapter can run RelayKV decisions in shadow while FullKV output remains active.

HF Phase 12-B target:

```text
false
```

This can become true only after a later shadow-compare implementation exists.

### supports_materialization

Whether the adapter can materialize selected KV blocks into an engine working set.

HF Phase 12-B target:

```text
false
```

### supports_apply

Whether the adapter can make RelayKV the active required runtime path.

HF Phase 12-B target:

```text
false
```

### supports_safe_degrade

Whether the adapter can execute a post-apply safe-degrade path.

HF Phase 12-B target:

```text
false
```

### supports_context_reduction_request

Whether the adapter/Core boundary can emit a request for reduced context rather than continuing unsafe execution.

HF Phase 12-B target:

```text
true
```

This is a planning/reporting capability. The App / Agent layer owns the user-facing UX.

## Model reference fields

`model_ref` should identify the model without requiring model loading.

Recommended fields:

- `model_id`
- `model_revision`
- `local_path`
- `quantization_hint`
- `dtype_hint`

The first HF adapter may infer these from CLI arguments or a smoke artifact. It should not require a live model load just to emit a capability artifact.

## Tokenizer reference fields

`tokenizer_ref` should be compatible with the RelayCTX token-span contract.

Recommended fields:

- `tokenizer_name_or_path`
- `tokenizer_revision`
- `tokenizer_config_hash`
- `tokenizer_family`

Token spans must remain tokenizer-scoped. A token span produced with one tokenizer configuration must not be treated as portable across another tokenizer configuration.

## Context window hint

`context_window_hint` is a hint, not an unconditional guarantee.

Recommended fields:

- `max_model_context_tokens`
- `configured_context_tokens`
- `source`

The source should distinguish values coming from model config, user override, smoke observation, or unknown defaults.

## Safety scope

The Phase 12-B artifact must make its non-runtime nature explicit:

```json
{
  "dry_run_only": true,
  "no_model_loading_required": true,
  "no_kv_materialization": true,
  "no_attention_connection": true,
  "no_scheduler_change": true,
  "no_runtime_apply": true
}
```

## Relationship to existing contracts

This schema should align with:

- `docs/relaystack_data_contract.md`
- `docs/relaystack_adapter_contracts.md`
- `docs/relaystack_runtime_modes.md`
- `docs/relaystack_eval_plan.md`
- `docs/phase12_runtime_target_selection.md`

The HF adapter capability artifact should not replace those contracts. It is a target-specific declaration of what the HF prototype adapter can safely provide.

## Phase 12-C handoff

Phase 12-C can implement or document the first no-apply smoke artifact around this schema.

Recommended Phase 12-C artifacts:

```text
relaystack_adapter_capabilities.json
relaystack_tokenizer_span_probe.json
relaystack_engine_metadata_probe.json
```

The smoke path should remain metadata-only until a later phase explicitly starts safe materialization or shadow attention compare.

Phase 12-C smoke CLI:

```bash
python scripts/run_hf_adapter_capability_smoke.py \
  --output /tmp/relaystack_adapter_capabilities.json
```

This artifact remains no-model/no-GPU and no-apply.
