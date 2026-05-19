# RelayStack Architecture Boundary

Design status: 2026-05-19 JST.

This document records the design-level boundary for RelayStack after the RelayMEM / RelayCTX / RelayKV split and the RelayLM naming consolidation. It is a design document only. It does not claim that the described runtime integration is implemented.

## RelayLM product framing

RelayLM is the product-facing name for the Relay Lineage Manager.

```text
RelayLM = Relay Lineage Manager
```

RelayLM is not a language model. It is a memory-context-token-runtime-cache lineage manager for local LLM applications, agents, and local inference runtimes.

```text
memory
  ↓
context
  ↓
token span
  ↓
runtime cache / KV cache
  ↓
budget and lifecycle decision
  ↓
trace
```

RelayStack remains the internal architecture name. RelayStack Core is a responsibility boundary, not a separate runtime component name.

```text
RelayLM
├─ RelayStack Core
│  ├─ RelayMEM
│  ├─ RelayCTX
│  ├─ RelayKV
│  ├─ RelayPLC
│  └─ RelayTRC
│
└─ Relay Adapter
   ├─ MemoryADP
   ├─ TokenizerADP
   ├─ EngineADP
   ├─ RuntimeADP
   ├─ APIADP
   ├─ AgentADP
   ├─ SessionADP
   ├─ ToolResultADP
   └─ ObservabilityADP
```

## Core definition

RelayStack Core is the memory, context, token-span, runtime-cache/KV, budget, lifecycle, and trace control boundary.

```text
RelayStack Core
├─ RelayMEM
├─ RelayCTX
├─ RelayKV
├─ RelayPLC
└─ RelayTRC
```

The core should decide and describe. External adapters, services, workers, applications, or inference engines should execute, store, connect, and translate.

```text
RelayStack Core:
  decision, metadata, budget, lifecycle, lineage, trace

Relay Adapter:
  connection, translation, capability probing, runtime/backend/API bridge

External systems:
  execution, storage, UI, approval, model-specific runtime behavior
```

## Layer responsibilities

### RelayMEM

RelayMEM decides what memories, records, summaries, retrieved facts, or structured state should be proposed for active context.

RelayMEM is a memory-lineage layer, not a direct physical allocator. In the short term, it covers semantic memory selection and context item proposal. In the longer term, it may also emit memory lifecycle, reuse, residency preference, or placement-intent hints that help RelayPLC, RelayKV, and adapters choose an efficient runtime path.

RelayMEM owns:

- memory record schemas
- retrieval result schemas
- memory ranking and budgeting policy
- context item proposal
- memory lifecycle state metadata
- future placement-intent or residency-preference metadata

RelayMEM does not own:

- direct physical memory allocation
- direct KV cache routing
- tokenizer internals
- inference-engine runtime state
- tool execution
- product-specific user interface or approval flows
- concrete heavy RAG, embedding, reranking, summarization, or graph extraction implementations

A meaningful RelayMEM runtime path depends on adapters. MemoryADP supplies memory backend records, while EngineADP / RuntimeADP supplies engine capability, runtime context limits, cache/runtime metadata availability, and placement capability. RelayMEM can emit memory lineage and placement intent, but engine-specific physical placement remains behind EngineADP / RuntimeADP and the target runtime such as SGLang, vLLM, HF, or llama.cpp.

### RelayCTX

RelayCTX transforms selected context items into model-input context and traceable token spans.

RelayCTX owns:

- context packing
- token budget fitting
- prompt layout at the context-item level
- token compression plan metadata
- source attribution
- token span mapping
- context rebuild planning when the active context approaches the model/runtime context limit
- reference-only context entries

RelayCTX does not own:

- tool execution
- workflow orchestration
- product-specific persona or UI prompt templates
- inference-engine KV routing
- long-term memory backend implementation
- concrete summarization model execution

A useful internal split is:

```text
RelayCTX
├─ ContextPacker
├─ ContextRebuilder
├─ TokenBudgetPlanner
├─ TokenCompressor
├─ SourceAttributor
└─ TokenSpanMapper
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

RelayKV is not always required for RelayLM to provide value. MEM_CTX_ONLY operation can improve memory and context quality without active KV working-set control. RelayKV becomes the critical layer when RelayLM must make max-context operation VRAM-aware.

### RelayPLC

RelayPLC is the policy, lifecycle, and control layer. It replaces the older RelayPolicy naming.

RelayPLC owns:

- global budget planning
- declared context window vs effective context budget separation
- memory / context / KV budget allocation
- runtime mode selection
- context rebuild triggers
- RelayKV OFF / SHADOW / APPLY mode decisions
- safe degrade / request context reduction / block decisions
- adapter capability based policy selection

RelayPLC does not own concrete engine mutation, tool execution, memory storage, UI approval, or observability storage.

### RelayTRC

RelayTRC is the trace, transition, and record schema layer. It replaces the older Trace Schema naming.

RelayTRC owns:

- decision trace schema
- budget trace schema
- lifecycle and state transition trace schema
- context rebuild trace schema
- memory-context-token-runtime-cache lineage records
- fallback / degrade / block reason records
- evaluation artifact schemas

```text
RelayPLC decides.
RelayTRC records.
```

RelayTRC is a schema/event layer. The emitted artifact may be local JSONL, OpenTelemetry, Langfuse, dashboard storage, or another ObservabilityADP target.

## Externalized responsibilities

The following responsibilities should stay outside RelayStack Core:

```text
App / Agent Layer:
  Tool execution
  Web search
  Approval
  UI
  user interaction
  product-specific prompts
  product workflows
  TTS / ASR
  VTube Studio or avatar integration

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

Relay Adapter:
  MemoryADP
  TokenizerADP
  EngineADP
  RuntimeADP
  APIADP
  AgentADP
  SessionADP
  ToolResultADP
  ObservabilityADP
```

RelayStack may consume records produced by tools or external workers, but RelayStack Core should not execute tools. Tool execution, web search, approval, side effects, rollback, retry, and authentication scope belong to the application or agent orchestration layer unless a later narrow adapter contract explicitly allows a read-only capability.

## Adapter boundary

RelayStack Core should remain engine-agnostic and product-agnostic. External dependencies enter through Relay Adapter contracts.

```text
RelayMEM
  ↓ MemoryADP + EngineADP / RuntimeADP capability facts
  SQLite / GBrain / Qdrant / GraphRAG / external memory service
  plus runtime context/cache/placement capability metadata

RelayCTX
  ↓ TokenizerADP
  model tokenizer / HF tokenizer / SGLang tokenizer / vLLM tokenizer

RelayKV
  ↓ EngineADP / RuntimeADP
  HF prototype / SGLang / vLLM / llama.cpp / runtime capability probes

RelayLM API boundary
  ↓ APIADP
  OpenAI-compatible endpoint / backend proxy

Agent or app runtime
  ↓ AgentADP / SessionADP / ToolResultADP
  Hermes-Agent / OpenClaw / LangGraph-like agent runtimes / tool or skill outputs

RelayTRC
  ↓ ObservabilityADP
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
runtime_cache_span / logical_block_id
  ↓
engine_block_ref or runtime_cache_ref
  ↓
budget / lifecycle / KV decision
```

Every transformed context should preserve lineage back to its source item or explicitly record why lineage was intentionally severed.

## Agent / assistant runtime integration

Hermes-Agent, OpenClaw, and similar systems should be treated as App / Agent layer systems, not as RelayStack Core components.

```text
Hermes-Agent / OpenClaw
  ↓ OpenAI-compatible API or AgentADP
RelayLM
  ↓ EngineADP
local LLM runtime
```

From the agent runtime perspective, RelayMEM makes context more memory-aware, RelayCTX may rebuild or transform the model-visible context, and RelayKV may make long-context operation VRAM-aware. Because agent-visible messages and model-visible context can diverge, RelayLM should support transparency modes:

```text
opaque:
  do not return detailed RelayLM metadata

summary:
  return memory/context/KV summary flags

trace:
  return a RelayTRC trace_id for later inspection

debug:
  expose rebuilt context and drop/compression reasons for development
```

RelayLM should not silently take over tool execution, approval, security policy, or app workflow orchestration from the agent runtime.

## Runtime operation modes

RelayLM should support modes that do not require RelayKV to be active from the start.

```text
PASS_THROUGH:
  behave close to a normal OpenAI-compatible proxy

MEM_CTX_ONLY:
  RelayMEM + RelayCTX + RelayPLC + RelayTRC are active
  RelayKV is OFF

KV_SHADOW:
  RelayKV decisions are evaluated in shadow while the normal/full path produces output

KV_APPLY:
  RelayKV working-set control is active

SAFE_DEGRADE:
  shrink, rebuild, reduce, block, or request context reduction when no safer path exists
```

MEM_CTX_ONLY can be the default for models whose KV cache is already relatively small or when the runtime has sufficient VRAM. KV_SHADOW and KV_APPLY become important when long context, low VRAM, KV-heavy attention layouts, or max-context operation require active KV working-set control.

Even in MEM_CTX_ONLY mode, RelayMEM and RelayCTX need EngineADP / RuntimeADP capability facts for safe operation: model/runtime context limit, tokenizer compatibility, cache/runtime constraints, available placement capabilities, and whether context rebuild or prefill reduction can be safely applied.

## Attention and runtime-cache abstraction

RelayLM should not permanently assume that every model is GQA with a conventional KV-cache layout. RelayKV remains the first concrete runtime-cache control implementation, but the RelayLM-level lineage should allow broader cache/state artifacts.

```text
attention_family:
  mha
  mqa
  gqa
  mla
  sliding_window
  hybrid_attention
  ssm_state

cache_artifact_type:
  kv_cache
  latent_cache
  recurrent_state
  compressed_state
  mixed_cache

budget_granularity:
  token_block
  layer
  head
  kv_head_group
  latent_block
  state_slot
```

For GQA, the lineage may specialize to token span → logical KV block → kv_head_group. For MHA, it may use token/layer/head/block granularity. For MQA, RelayKV savings may be smaller, but RelayMEM, RelayCTX, RelayPLC, and RelayTRC remain valuable. For MLA or hybrid state models, EngineADP should report the runtime cache/state capability before RelayKV-like control is attempted.

## Data contract priority

The next schema work should prioritize these contracts:

```text
RelayMEMContextItem
  source item, memory type, priority, evidence role, confidence, privacy/scope metadata,
  lifecycle metadata, future placement intent

RelayCTXContextSpan
  source item, compression status, token span, prompt layout role, attribution metadata

RelayKVBlockMeta
  logical block, token span, KV class, budget role, score hints, source lineage

RelayTRCEvent
  request/session, layer, decision state, input/output refs, budget snapshot, fallback/degrade/block reason
```

## Runtime axes

RelayStack runtime should be modeled with independent axes rather than a single linear state machine.

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

CTX_REBUILD_PLANNED
  RelayCTX plans context rebuild near the model/runtime context limit.

CTX_REBUILDING
  RelayCTX emits a rebuilt context plan before re-prefill.

CTX_REBUILT
  The rebuilt context plan has replaced the previous active context.

CTX_REBUILD_FAILED
  RelayCTX could not produce a safe rebuilt context plan.

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

## V0.1 boundary

V0.1 should prove the RelayLM / RelayStack Core contracts and the minimum product-facing OpenAI-compatible boundary before broadening runtime or application scope.

```text
Pre-V0.1 / V0.1 focus:
  RelayLM product framing
  RelayStack Core boundary
  RelayMEM / RelayCTX / RelayKV data and lineage contracts
  RelayPLC / RelayTRC naming and responsibility split
  Relay Adapter contracts
  runtime mode and fallback/degrade/block contracts
  HF-first validation path
  minimal OpenAI-compatible API boundary
  JSON-safe trace and evaluation artifacts

Post-V0.1 validation targets:
  SGLang OpenAI-compatible backend path
  Hermes-Agent / OpenClaw / Open-LLM-VTuber style application surfaces
  local AI friend / avatar-style practical conversation tests
  vLLM adapter evaluation
  optional read-only web search or other tool policy experiments
```

V0.1 should not absorb UI, TTS/ASR, VTube Studio, Open-LLM-VTuber-specific behavior, Hermes-Agent/OpenClaw-specific behavior, product workflows, tool execution, web search, or vLLM adapter work into RelayStack Core. Those remain App / Agent or adapter-layer concerns until a later contract explicitly narrows and promotes them.

## Phase implications

This architecture changes the phase plan by adding a contract-consolidation step before runtime adapter restart and by making RelayLM product naming explicit.

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
  RelayStack V0.1 runtime boundary and HF-first validation.
  Keep SGLang as the next practical OpenAI-compatible runtime target after V0.1.
  Keep vLLM as a post-V0.1 adapter target.

Phase 12.5:
  RelayLM naming and product-boundary consolidation.
  Treat relay-lm as the product / architecture repository and relay-kv as the
  RelayKV runtime/cache research repository until implementation migration is justified.

Phase 13:
  Safe materialization / shadow attention compare.
  FullKV is still available, so fallback-to-FullKV is valid in this phase.

Phase 14:
  Gated apply / safe degrade / block / context-reduction integration.
  RelayKV may become the active required path. FullKV fallback is not assumed after apply.

Phase 15:
  RelayCTX budgeted context integration.
  Add context packing, token-budget fitting, compression plan metadata,
  source attribution, token-span mapping, and context rebuild planning. Do not add tool execution.

Phase 16:
  RelayMEM + RelayCTX + RelayKV attribution evaluation.
  Evaluate MEM-only, MEM+CTX, and MEM+CTX+KV separately to attribute regressions.
```

## Minimal near-term design documents

The design should be split into focused documents rather than one large specification:

```text
docs/relaystack_architecture.md
  RelayLM framing, core boundary, layer responsibilities, externalization, runtime axes

docs/relaystack_data_contract.md
  RelayMEM / RelayCTX / RelayKV / RelayTRC schemas

docs/relaystack_adapter_contracts.md
  MemoryADP / TokenizerADP / EngineADP / RuntimeADP / APIADP / AgentADP / SessionADP / ToolResultADP / ObservabilityADP requirements

docs/relaystack_runtime_modes.md
  activation triggers, fallback/degrade/block semantics, pressure modes

docs/relaystack_eval_plan.md
  MEM-only, MEM+CTX, MEM+CTX+KV evaluation and attribution plan
```

This document is the architecture-boundary starting point for that split.

See [relaystack_adapter_contracts.md](relaystack_adapter_contracts.md) for the adapter-boundary requirements that Phase 11.5-B fixes before runtime target selection.
See [relaystack_runtime_modes.md](relaystack_runtime_modes.md) for the Phase 11.5-C runtime state and fallback-vs-degrade contract.
See [relaystack_eval_plan.md](relaystack_eval_plan.md) for the Phase 11.5-D evaluation attribution split across MEM-only, MEM+CTX, and MEM+CTX+KV stages.
