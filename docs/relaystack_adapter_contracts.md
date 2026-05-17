# RelayStack Adapter Contracts

Design status: 2026-05-17 JST.

This document records the design-level adapter contract for RelayStack Core. It is a design/contract document only. It does not claim that the described runtime integration is implemented.

## Core rule

RelayStack Core decides and describes. Adapters execute, store, and translate.

```text
RelayStack Core:
  decision, metadata, budget, lineage, trace

Adapters:
  execution, storage, translation, product/runtime integration
```

RelayStack Core should remain engine-agnostic and product-agnostic.

## MemoryBackendAdapter

Purpose:

- expose memory records and retrieval candidates to RelayMEM
- hide DB, vector index, GraphRAG, GBrain, and backend implementation details

Inputs / outputs:

- query, request, session, or scope metadata
- retrieval constraints and privacy constraints
- returns RelayMEMContextItem-compatible records

Must provide:

- stable `item_id`
- `memory_type`
- `content_ref` or `content_text`
- priority, confidence, and evidence metadata
- privacy and scope metadata
- `source_uri` or external reference when available

Must not:

- mutate RelayKV state
- execute tools directly
- require engine-specific KV objects
- hide lineage or source identity

Example output:

```json
{
  "item_id": "mem:summary:17",
  "memory_type": "summary",
  "content_ref": "sqlite://relaymem/summary/17",
  "priority": 0.74,
  "confidence": 0.88,
  "evidence_role": "project_state",
  "privacy_scope": "workspace_visible",
  "source_uri": "sqlite://relaymem/summary/17",
  "retrieval_reason": "tag_match:relaystack_contract"
}
```

## TokenizerAdapter

Purpose:

- convert RelayCTX context items and spans into tokenizer-scoped token spans
- isolate model tokenizer differences from RelayStack Core

Must provide:

- `tokenizer_ref` and tokenizer version identity
- encode/decode or span-mapping behavior
- half-open token spans `[token_start, token_end)`
- stable mapping from context items to token spans

Must not:

- decide memory retrieval
- decide KV selection
- execute tools
- silently sever lineage

Example output:

```json
{
  "span_id": "ctx:span:12",
  "source_item_id": "mem:summary:17",
  "prompt_layout_role": "memory_context",
  "token_start": 320,
  "token_end": 352,
  "token_count": 32,
  "tokenizer_ref": "Qwen/Qwen2.5-1.5B-Instruct@tokenizer:v1",
  "lineage_status": "preserved"
}
```

## EngineAdapter

Purpose:

- expose engine-safe logical and physical KV metadata to RelayKV
- translate `logical_block_id` to `engine_block_ref` at the adapter boundary
- later execute materialization/apply only after safe phases allow it

Must provide:

- `engine_request_id`
- `logical_sequence_id`
- token-span or cache-position mapping
- `logical_block_id`
- optional `engine_block_ref`
- residency and precision metadata when available
- explicit capability flags:
  - `supports_shadow_compare`
  - `supports_materialization`
  - `supports_apply`
  - `supports_safe_degrade`
  - `supports_context_reduction_request`

Must not:

- let RelayStack Core mutate the KV pool directly
- expose scheduler internals as Core-owned state
- imply FullKV fallback after RelayKV apply under VRAM pressure
- mix `kv_class` with `residency_level` or `precision_level`

Future targets:

- HF prototype adapter
- SGLang adapter
- vLLM adapter
- llama.cpp or other engine adapters

Example output:

```json
{
  "engine_request_id": "engine:req:001",
  "logical_sequence_id": "seq:demo:01",
  "logical_block_id": 8,
  "token_start": 512,
  "token_end": 576,
  "engine_block_ref": "engine:block:gpu:8",
  "residency_level": "gpu",
  "precision_level": "fp16",
  "supports_shadow_compare": true,
  "supports_materialization": false,
  "supports_apply": false,
  "supports_safe_degrade": true,
  "supports_context_reduction_request": true
}
```

## ObservabilityAdapter

Purpose:

- receive RelayStackTraceEvent-compatible events
- export them to JSONL, OpenTelemetry, Langfuse, or dashboards

Must provide:

- append-only event sink
- JSON-safe trace event serialization
- request/session correlation
- budget snapshot and decision-state recording

Must not:

- change runtime decisions
- execute tools
- own product UI approval logic

Example input:

```json
{
  "trace_id": "trace:relaystack:0101",
  "request_id": "req:abc123",
  "phase": "11.5-B",
  "component": "RelayKV",
  "event_type": "adapter_contract_recorded",
  "input_refs": [
    "doc:relaystack_adapter_contracts"
  ],
  "output_refs": [
    "artifact:/tmp/relaykv_fixed_budget_block_selection.json"
  ],
  "budget_snapshot": {
    "total_working_budget_tokens": 512
  },
  "decision_state": "dry_run_ready",
  "timestamp": "2026-05-17T18:00:00+09:00"
}
```

## App / Agent boundary

Purpose:

- tool execution
- approval
- UI
- product workflow orchestration
- user-facing fallback or context-reduction request flow

Must own:

- tool side effects
- auth scopes
- retry and rollback
- user approval
- persona and product prompts
- final UX for `BLOCKED_NO_SAFE_KV_PATH` or `REQUEST_CONTEXT_REDUCTION`

RelayStack Core may emit recommendations and events, but it must not execute tools.

## Cross-adapter invariants

- preserve lineage from RelayMEM `item_id` to RelayCTX span to RelayKV `logical_block_id` to `RelayStackTraceEvent`
- use JSON-safe data exchange across adapter boundaries
- adapter outputs must be deterministic enough for offline replay where possible
- explicit capability flags are required before Core assumes shadow, apply, or materialization support
- Core should degrade gracefully when an adapter capability is missing
- no adapter should erase fallback, degrade, block, or context-reduction semantics
- token spans must be tokenizer-scoped
- `engine_block_ref` must not leak into Core as a portable identity

## Minimal contract table

| Adapter | Owned by | Core consumes | Core emits | Forbidden responsibilities |
| --- | --- | --- | --- | --- |
| `MemoryBackendAdapter` | external memory backend / retrieval service | RelayMEMContextItem-compatible records | retrieval query, scope, constraints | KV mutation, tool execution, engine KV ownership |
| `TokenizerAdapter` | model/tokenizer integration layer | token-span mappings, tokenizer identity | context items/spans to encode | retrieval decisions, KV decisions, tool execution |
| `EngineAdapter` | inference engine integration layer | logical/physical KV metadata, capability flags | logical block queries and budget decisions | Core-owned scheduler state, forced FullKV fallback semantics |
| `ObservabilityAdapter` | logging / telemetry sink | append-only trace event acknowledgements or sink status | RelayStackTraceEvent-compatible payloads | runtime decision control, tool execution, approval logic |
| `App / Agent layer` | product/application orchestration | recommendations, events, block/context-reduction states | approvals, tool calls, user-facing actions | pretending tool execution belongs to RelayStack Core |

## Near-term usage

Phase 11.5-B is contract consolidation only.

This document does not add:

- runtime adapter behavior
- KV materialization
- attention connection
- scheduler changes
- tool execution inside RelayStack Core

The next follow-on documents should remain split:

- `docs/relaystack_architecture.md`
- `docs/relaystack_data_contract.md`
- `docs/relaystack_adapter_contracts.md`
- `docs/relaystack_runtime_modes.md`

Capability flags and post-apply safety semantics should be interpreted together with [relaystack_runtime_modes.md](relaystack_runtime_modes.md).
