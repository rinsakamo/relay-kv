# RelayStack Data Contract

Design status: 2026-05-17 JST.

This document records the design-level data contract for RelayStack Core. It is a design/contract document only. It does not claim that the described runtime integration is implemented.

## Core invariant

RelayStack Core decides and describes. External adapters execute and store.

```text
RelayStack Core:
  decision, metadata, budget, lineage, trace

External adapters:
  execution, storage, UI, approval, tool execution,
  heavy RAG, model-worker execution, engine runtime internals
```

The near-term contract chain is:

```text
RelayMEMContextItem
  → RelayCTXContextSpan
  → RelayKVBlockMeta
  → RelayStackTraceEvent
```

## Scope boundary

The following remain outside RelayStack Core:

- tool execution
- approval flow implementation
- UI/product workflow
- heavy RAG implementation
- embedding / reranking / summarization worker execution
- engine-specific runtime internals
- concrete storage backends

Core contracts may reference external records, but they should not embed tool execution or application workflow state.

## RelayMEMContextItem

RelayMEMContextItem is the contract for memory proposals entering RelayCTX.

Required fields:

- `item_id`
- `memory_type`
- `content_ref` or `content_text`
- `priority`
- `confidence`
- `evidence_role`
- `privacy_scope`
- `retrieval_reason`

Optional fields:

- `source_uri`
- `created_at`
- `updated_at`
- `budget_hint_tokens`

Notes:

- `memory_type` should align with RelayMEM source families such as profile, episode, summary, structured, rag_chunk, or checkpoint metadata.
- `content_ref` should be preferred when the content body is large or lives in external storage.
- `priority` and `confidence` should remain logically separate.
- `privacy_scope` must be explicit because RelayStack Core should be able to propagate policy metadata without implementing the policy itself.

Example:

```json
{
  "item_id": "mem:episode:42",
  "memory_type": "episode",
  "content_text": "User prefers concise Japanese summaries for design notes.",
  "priority": 0.82,
  "confidence": 0.91,
  "evidence_role": "preference",
  "privacy_scope": "session_visible",
  "source_uri": "sqlite://relaymem/episode/42",
  "created_at": "2026-05-16T12:00:00+09:00",
  "updated_at": "2026-05-17T09:10:00+09:00",
  "retrieval_reason": "keyword_match:user_preference",
  "budget_hint_tokens": 24
}
```

## RelayCTXContextSpan

RelayCTXContextSpan is the contract for traceable prompt/context spans produced after packing and token-budget fitting.

Required fields:

- `span_id`
- `source_item_id`
- `source_item_type`
- `prompt_layout_role`
- `token_start`
- `token_end`
- `token_count`
- `compression_status`
- `attribution`
- `tokenizer_ref`
- `lineage_status`

Optional fields:

- `compression_plan_id`

Notes:

- RelayCTX owns token-budget fitting, prompt layout, compression plan metadata, source attribution, and token/span mapping.
- `tokenizer_ref` must scope every token span to the tokenizer/version used to produce it.
- `lineage_status` should explain whether lineage is preserved, merged, truncated, or intentionally severed.

Example:

```json
{
  "span_id": "ctx:span:7",
  "source_item_id": "mem:episode:42",
  "source_item_type": "episode",
  "prompt_layout_role": "memory_context",
  "token_start": 256,
  "token_end": 288,
  "token_count": 32,
  "compression_status": "uncompressed",
  "compression_plan_id": null,
  "attribution": {
    "source_refs": [
      "sqlite://relaymem/episode/42"
    ],
    "evidence_role": "preference"
  },
  "tokenizer_ref": "Qwen/Qwen2.5-1.5B-Instruct@tokenizer:v1",
  "lineage_status": "preserved"
}
```

## RelayKVBlockMeta

RelayKVBlockMeta is the contract for logical KV working-set metadata before engine-specific materialization.

Required fields:

- `logical_block_id`
- `token_start`
- `token_end`
- `block_size`
- `kv_class`
- `decision_state`
- `source_span_ids`

Optional fields:

- `anchor_score`
- `retrieval_score`
- `residency_level`
- `precision_level`
- `engine_block_ref`
- `fallback_reason`

Notes:

- `kv_class`, `precision_level`, and `residency_level` are separate axes.
- `engine_block_ref` exists only at the adapter boundary and should not be treated as a core identifier.
- `decision_state` should distinguish selected, rejected, overflow, shadow, apply, degrade, blocked, or context-reduction-related states as needed by the current phase.
- RelayKV remains a fixed-VRAM-budget decode-time KV working-set controller.

Example:

```json
{
  "logical_block_id": 12,
  "token_start": 768,
  "token_end": 832,
  "block_size": 64,
  "kv_class": "retrieved",
  "decision_state": "selected",
  "source_span_ids": [
    "ctx:span:7"
  ],
  "anchor_score": null,
  "retrieval_score": 0.83,
  "residency_level": "gpu",
  "precision_level": "fp16",
  "engine_block_ref": null,
  "fallback_reason": null
}
```

## RelayStackTraceEvent

RelayStackTraceEvent is the append-only, JSON-safe trace contract for decisions and state transitions.

Required fields:

- `trace_id`
- `request_id`
- `phase`
- `component`
- `event_type`
- `input_refs`
- `output_refs`
- `budget_snapshot`
- `decision_state`
- `timestamp`

Optional fields:

- `session_id`
- `fallback_or_degrade_reason`

Notes:

- trace events should be append-only and JSON-safe
- trace events should describe decisions, not execute workflows
- input/output refs should point to upstream/downstream contract objects rather than inline huge payloads

Example:

```json
{
  "trace_id": "trace:relaystack:0001",
  "request_id": "req:abc123",
  "session_id": "session:demo",
  "phase": "11.5-A",
  "component": "RelayKV",
  "event_type": "fixed_budget_selection_ready",
  "input_refs": [
    "artifact:/tmp/relaykv_candidates.json"
  ],
  "output_refs": [
    "artifact:/tmp/relaykv_fixed_budget_block_selection.json"
  ],
  "budget_snapshot": {
    "total_working_budget_tokens": 512,
    "materialized_working_tokens": 256,
    "estimated_working_tokens": 256
  },
  "decision_state": "dry_run_ready",
  "fallback_or_degrade_reason": null,
  "timestamp": "2026-05-17T13:40:00+09:00"
}
```

## Lineage chain

The required lineage chain is:

```text
RelayMEM.item_id
  → RelayCTX.source_item_id
  → RelayCTX.token_span
  → RelayKV.logical_block_id
  → engine_block_ref (adapter boundary only)
  → KV decision / trace event
```

This lineage should be preserved unless an explicit contract field records why it was severed or merged.

## Invariants

- token spans are half-open: `[token_start, token_end)`
- token spans must be tokenizer-version scoped
- source lineage should be preserved unless explicitly severed
- `kv_class`, `precision_level`, and `residency_level` are separate concepts
- `engine_block_ref` is adapter-boundary only
- trace events should be append-only and JSON-safe
- Core contracts must not contain tool execution fields, except references to external tool-produced records
- FullKV fallback semantics must distinguish:
  - pre-apply / shadow fallback still available
  - post-apply fallback generally unavailable
- after RelayKV apply under VRAM pressure, the safe paths are:
  - degrade
  - block
  - request context reduction

## Near-term usage

Phase 11.5-A is contract consolidation only.

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
