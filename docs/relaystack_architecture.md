# RelayStack Architecture Boundary

Design status: 2026-05-17 JST.

This document records the design-level boundary for RelayStack after the RelayMEM / RelayCTX / RelayKV split. It is a design document only. It does not claim that the described runtime integration is implemented.

## Core definition

RelayStack Core is the memory, context, token-span, KV, budget, and trace control layer.

```text
RelayStack Core
├─ RelayMEM
├─ RelayCTX
├─ RelayKV
├─ RelayPolicy
└─ Trace Schema
```

The core should decide and describe. External adapters, services, workers, applications, or inference engines should execute and store.

```text
RelayStack Core:
  decision, metadata, budget, lineage, trace

External systems:
  execution, storage, UI, approval, model-specific runtime behavior
```

## Layer responsibilities

### RelayMEM

RelayMEM decides what memories, records, summaries, retrieved facts, or structured state should be proposed for active context.

RelayMEM owns:

- memory record schemas
- retrieval result schemas
- memory ranking and budgeting policy
- context item proposal
- memory lifecycle state metadata

RelayMEM does not own:

- KV cache routing
- tokenizer internals
- inference-engine runtime state
- tool execution
- product-specific user interface or approval flows
- concrete heavy RAG, embedding, reranking, summarization, or graph extraction implementations

### RelayCTX

RelayCTX transforms selected context items into model-input context and traceable token spans.

RelayCTX owns:

- context packing
- token budget fitting
- prompt layout at the context-item level
- token compression plan metadata
- source attribution
- token span mapping

RelayCTX does not own:

- tool execution
- workflow orchestration
- product-specific persona or UI prompt templates
- inference-engine KV routing
- long-term memory backend implementation

A useful internal split is:

```text
RelayCTX
├─ ContextPacker
├─ TokenCompressor
├─ SourceAttributor
└─ TokenizerAdapter
```

RelayCTX may emit reference-only context entries, but it should not call external tools itself.

### RelayKV

RelayKV controls the decode-time active-context KV working set under a fixed residual VRAM budget.

RelayKV owns:

- logical KV block metadata
- KV role classes such as RECENT, ANCHOR, RETRIEVED, TRANSIENT, and COLD_CANDIDATE
- working-set budget decisions
- selected and rejected logical block IDs
- pressure-triggered shadow and apply decisions
- degrade / block / context-reduction recommendations when FullKV is no longer available

RelayKV does not own:

- long-term memory retrieval
- token compression
- concrete inference-engine scheduler internals
- SGLang / vLLM / Hugging Face runtime objects outside adapter boundaries
- tool execution or approval

## Externalized responsibilities

The following responsibilities should stay outside RelayStack Core:

```text
App / Agent Layer:
  Tool execution
  Approval
  UI
  user interaction
  product-specific prompts
  workflow orchestration

Memory Backend:
  vector DB
  GraphRAG
  GBrain
  SQLite / Postgres
  embedding index
  reranker

Model Worker:
  summarization
  compression model execution
  embedding generation
  graph extraction

Engine Adapter:
  SGLang
  vLLM
  Hugging Face
  llama.cpp or other engines

Observability Adapter:
  Langfuse
  OpenTelemetry
  dashboards
  metrics storage
```

RelayStack may consume records produced by tools or external workers, but RelayStack Core should not execute tools. Tool execution, approval, side effects, rollback, retry, and authentication scope belong to the application or agent orchestration layer.

## Adapter boundary

RelayStack Core should remain engine-agnostic and product-agnostic. External dependencies enter through adapter contracts.

```text
RelayMEM
  ↓ MemoryBackendAdapter
  SQLite / GBrain / Qdrant / GraphRAG / external memory service

RelayCTX
  ↓ TokenizerAdapter
  model tokenizer / HF tokenizer / SGLang tokenizer / vLLM tokenizer

RelayKV
  ↓ EngineAdapter
  HF prototype / SGLang / vLLM / llama.cpp

Trace Schema
  ↓ ObservabilityAdapter
  local JSONL / OpenTelemetry / Langfuse / dashboard
```

The most important cross-adapter invariant is lineage preservation:

```text
RelayMEM item_id
  ↓
RelayCTX source_item_id
  ↓
token_span
  ↓
logical_block_id
  ↓
engine_block_ref
  ↓
KV decision
```

Every transformed context should preserve lineage back to its source item or explicitly record why lineage was intentionally severed.

## Data contract priority

The next schema work should prioritize these contracts:

```text
RelayMEMContextItem
  source item, memory type, priority, evidence role, confidence, privacy/scope metadata

RelayCTXContextSpan
  source item, compression status, token span, prompt layout role, attribution metadata

RelayKVBlockMeta
  logical block, token span, KV class, budget role, score hints, source lineage

RelayStackTraceEvent
  request/session, layer, decision state, input/output refs, budget snapshot, fallback/degrade/block reason
```

## Runtime axes

RelayStack runtime should be modeled with at least two independent axes.

### VRAM / KV axis

```text
NORMAL_FULL
  FullKV path is still available.

RELAYKV_SHADOW
  FullKV is used for output while RelayKV decisions are evaluated in shadow.

RELAYKV_APPLY
  RelayKV is the active KV working-set path.

RELAYKV_SAFE_DEGRADE
  FullKV may no longer be available; RelayKV shrinks to a safer/smaller mode.

BLOCKED_NO_SAFE_KV_PATH
  There is no safe KV path within the current budget.

REQUEST_CONTEXT_REDUCTION
  The application or user must reduce context because the runtime cannot safely continue.
```

### Context / memory axis

```text
CONTEXT_NORMAL
  No special memory or context pressure handling is needed.

CTX_BUDGETED
  RelayCTX applies budgeted packing or compression before prefill.

MEM_RECALL
  RelayMEM adds selected external memory items before RelayCTX packing.

POST_STREAM_INDEX
  Memory candidate generation, summarization, indexing, and trace refinement happen after the live path.
```

These axes are not a single linear state machine. VRAM pressure can activate RelayKV even when RelayMEM and RelayCTX are not applied. Memory recall can happen without RelayKV if the active context and VRAM budgets remain safe.

## Implementation and evaluation order vs runtime activation

Implementation and quality evaluation should proceed top-down for clean attribution:

```text
1. RelayMEM only
2. RelayMEM + RelayCTX
3. RelayMEM + RelayCTX + RelayKV
```

Runtime pressure handling is different. Under VRAM pressure, RelayKV is the immediate runtime layer:

```text
1. NORMAL_FULL
2. RELAYKV_SHADOW when VRAM pressure is near
3. RELAYKV_APPLY when pressure is high and shadow/apply gates pass
4. RELAYKV_SAFE_DEGRADE if quality or budget risk increases after apply
5. BLOCKED_NO_SAFE_KV_PATH or REQUEST_CONTEXT_REDUCTION if no safe degraded path exists
```

This means development/evaluation order and runtime activation order are intentionally different:

```text
Development/evaluation:
  MEM → CTX → KV

VRAM-pressure runtime:
  KV first, then CTX/MEM for future turns or context planning
```

## Fallback, degrade, block, and context reduction

RelayStack should not treat fallback as a single universal operation.

```text
Fallback:
  Return to the safest still-available path.

RelayKV Degrade:
  FullKV is not available, so RelayKV shrinks to a safer or smaller KV mode.

Block:
  No safe KV path exists within the budget, so the runtime should not continue.

Request Context Reduction:
  The application or user must reduce active context because neither FullKV nor a safe RelayKV mode is available.
```

The critical RelayKV rule is:

```text
RelayKV before apply or in shadow:
  FullKV fallback may be possible.

RelayKV after apply:
  FullKV fallback is generally not available.
  Use degrade, block, or context reduction instead.
```

RelayKV apply-after-pressure must therefore avoid states that imply an automatic return to FullKV.

## Phase implications

This architecture changes the phase plan by adding a contract-consolidation step before runtime adapter restart.

```text
Phase 11:
  RelayKV fixed-budget working-set dry-run policy.
  Continue dry-run/schema/CLI/report work without runtime adapter, materialization,
  attention connection, KV-pool mutation, or scheduler changes.

Phase 11.5:
  RelayStack design contract consolidation.
  Lock the core boundary, data contract, lineage/attribution contract,
  runtime mode contract, fallback-vs-degrade terminology, and adapter contracts.

Phase 12:
  RelayStack adapter contract and runtime target selection.
  Choose SGLang, vLLM, HF, or another adapter target only after the contracts are explicit.

Phase 13:
  Safe materialization / shadow attention compare.
  FullKV is still available, so fallback-to-FullKV is valid in this phase.

Phase 14:
  Gated apply / safe degrade / block / context-reduction integration.
  RelayKV may become the active required path. FullKV fallback is not assumed after apply.

Phase 15:
  RelayCTX budgeted context integration.
  Add context packing, token-budget fitting, compression plan metadata,
  source attribution, and token-span mapping. Do not add tool execution.

Phase 16:
  RelayMEM + RelayCTX + RelayKV attribution evaluation.
  Evaluate MEM-only, MEM+CTX, and MEM+CTX+KV separately to attribute regressions.
```

## Minimal near-term design documents

The design should be split into focused documents rather than one large specification:

```text
docs/relaystack_architecture.md
  core boundary, layer responsibilities, externalization, runtime axes

docs/relaystack_data_contract.md
  RelayMEM / RelayCTX / RelayKV / Trace schemas

docs/relaystack_adapter_contracts.md
  MemoryBackend / Tokenizer / Engine / Observability adapter requirements

docs/relaystack_runtime_modes.md
  activation triggers, fallback/degrade/block semantics, pressure modes

docs/relaystack_eval_plan.md
  MEM-only, MEM+CTX, MEM+CTX+KV evaluation and attribution plan
```

This document is the architecture-boundary starting point for that split.
