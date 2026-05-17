# RelayMEM

## Definition

**RelayMEM** is the design-level max-context-external memory layer for RelayStack.

Its job is to decide what memories, retrieved facts, summaries, and structured state should enter the model's active context before or during generation. RelayMEM is not the same thing as RelayKV.

RelayMEM should be treated as a **hierarchical memory and context assembly layer**, not as a single RAG adapter. RAG-like retrieval is one possible backend capability, but RelayMEM owns the higher-level decision about which memory items should be assembled into active context under a token and latency budget.

## Boundary with RelayCTX and RelayKV

The boundary is intentionally simple:

- **RelayMEM** decides what memories should be proposed for active context.
- **RelayCTX** decides how selected context items are packed, compressed, attributed, and token-span mapped before prefill.
- **RelayKV** decides how active-context KV blocks are routed under a residual VRAM budget after prefill.

RelayMEM operates outside the model's currently active context window. RelayCTX is the context-transform boundary between memory items and token spans. RelayKV operates inside the active context that is already admitted to the model.

The intended long-term data boundary is:

```text
RelayMEMContextItem
  ↓
RelayCTXContextSpan
  ↓
token_span
  ↓
logical_block_id
  ↓
RelayKVBlockMeta
```

RelayMEM may provide hints such as memory source, evidence role, or semantic anchor priority. RelayCTX should preserve these as source-attribution metadata where possible. RelayKV should treat those hints as policy inputs rather than as mandatory routing decisions.

See [relaystack_architecture.md](relaystack_architecture.md) for the broader RelayStack core boundary, adapter boundary, runtime modes, and fallback/degrade terminology.

## Components

RelayMEM is expected to include the following component types:

- **Profile Memory**: stable user, character, assistant, or task profile information
- **Episode Memory**: notable past interactions, events, or session fragments
- **Summary Memory**: compressed summaries of prior context or prior sessions
- **RAG Memory**: retrieved external knowledge and document-grounded evidence
- **Structured Memory**: key-value facts, tool state, schedules, settings, and explicit records
- **Context Assembly**: the layer that proposes active-context input from retrieved and summarized memory pieces
- **KV Checkpoint Metadata**: metadata about reusable prefixes, cached spans, or checkpointed context segments that may help future runtime decisions

These memory classes are not all equivalent. For example, Profile Memory and selected evidence summaries may become semantic anchor hints, while detailed episode fragments or RAG chunks may become retrieved context candidates.

## RelayStack responsibility split

RelayStack is not a RAG replacement and is not an agent/tool runtime. The responsibility split is:

```text
Retrieval backend
  finds candidate records, documents, chunks, or evidence chains

RelayMEM
  ranks, budgets, and proposes selected items for active context

RelayCTX
  packs, compresses, attributes, and token-span maps selected context items

Tokenizer / model prefill
  converts selected context into token spans and KV blocks

RelayKV
  manages the decode-time KV working set under a fixed residual VRAM budget
```

This split keeps RelayMEM focused on context-before-token decisions, RelayCTX focused on context-before-KV transformation, and RelayKV focused on post-prefill KV working-set control.

Tool execution, approval, UI, product-specific workflow orchestration, shell/GitHub/Codex/browser actions, and rollback/retry semantics should live outside RelayStack Core. RelayMEM may consume records produced by tools, but RelayMEM should not execute tools itself.

## Pluggable RAG Backends

RelayMEM owns the memory lifecycle and context assembly boundary, but the concrete retrieval implementation should remain pluggable.

That means RelayMEM is responsible for questions such as:

- what gets indexed
- what gets summarized
- what gets promoted to profile, episode, or structured memory
- what gets proposed for active context

But the specific RAG backend used to retrieve evidence can be replaced depending on latency, quality, and deployment needs.

Backend candidates include:

- **Fast Recall / keyword or token-overlap retrieval**
- **Fast Vector RAG**
- **BM25 / keyword search**
- **Hybrid Search**
- **Graph RAG**
- **Evidence-chain retrieval**
- **External RAG service**

## Backend Resource Contract

RelayMEM backends should be designed as CPU-side memory and control-plane components by default. RelayStack treats GPU memory as the critical resource for model weights, active KV blocks, RelayKV working-set control, and attention execution. Long-term memory storage and retrieval should therefore avoid consuming VRAM unless an optional accelerator is explicitly enabled.

The default backend contract is:

- **CPU-first**: the backend must be able to run on CPU without CUDA, GPU kernels, or model-runtime dependencies.
- **VRAM-zero by default**: the backend must not reserve GPU memory in the normal RelayMEM path.
- **Disk-backed or remote-backed**: persistent memory should live in files, SQLite, Postgres, GBrain, a vector/search service, or another CPU/IO-oriented store.
- **Engine-agnostic**: the backend must not depend on SGLang, vLLM, Hugging Face runtime internals, KV pools, attention backends, or scheduler state.
- **Latency-tiered**: low-latency recall and deeper recall should be separable so the live path can use a cheap backend while deeper indexing or evidence-chain retrieval runs outside the critical path.

Optional GPU use is allowed only as an accelerator for clearly bounded tasks such as embedding generation, reranking, summarization, or compression. Such use should remain disableable and must not be a hard requirement for the backend. Under VRAM pressure, RelayStack should be able to use a CPU-only memory path rather than competing with RelayKV's residual VRAM budget.

Backend capability metadata should eventually make these properties explicit, for example:

```text
RelayMEMBackendCapabilities
  runs_on_cpu: true
  requires_gpu: false
  uses_vram: false
  disk_backed: true | false
  local_first: true | false
  remote_allowed: true | false
  supports_hybrid_search: true | false
  supports_graph: true | false
  supports_timeline: true | false
  supports_compaction: true | false
```

GBrain fits this model as a possible long-term memory backend candidate when used behind the RelayMEM backend boundary. It should not replace RelayMEM itself. RelayMEM should continue to own memory policy, context assembly, conflict handling, and the boundary to RelayCTX and RelayKV.

### Runtime use

- **`LIVE_LOW_LATENCY`**: fast retrieval only
- **`MEMORY_RECALL_MODE`**: deeper retrieval, possibly application-gated
- **`POST_STREAM_INDEX_MODE`**: offline indexing and refinement after the live interaction path

Under this framing, a NeocorRAG-style evidence-chain retrieval flow is one possible **Deep Recall** backend candidate. It should not be treated as the default every-turn RAG path for RelayMEM.

The near-term implementation target is a stdlib-only **Fast Recall** backend that returns `RelayMEMRetrievalResult` objects without embeddings, vector databases, model loading, or external services.

## Use Cases

RelayMEM is being framed to support several persistent-assistant workloads:

- **AI Vtuber / AI character**: maintain character profile, remembered interactions, and retrieval of older episodes
- **Personal assistant**: maintain profile, preferences, plans, and structured state over long periods
- **Long-form writing**: preserve world facts, story summaries, recurring entities, and continuity constraints across longer projects
- **Project continuity assistant**: retrieve prior devlogs, decisions, phase plans, and implementation boundaries
- **Document-grounded work assistant**: assemble evidence, summaries, and structured facts before model generation

Profile Memory should be understood as a general memory category. Viewer-specific profiles are only one possible specialization in the AI Vtuber case.

## Implementation Status

RelayMEM is currently a **schema/log and smoke-test target**, not a claimed production memory runtime.

The present repository mainly implements RelayKV prototype components and supporting routing/budget schemas. RelayMEM is documented here so the architectural boundary stays clear while the RelayKV prototype remains the primary implemented path.

As a lightweight bridge for logs and experiments, `relaykv/relaymem.py` provides log-only schema objects for retrieval results and context assembly plans. It does not implement any concrete RAG backend.

For memory-side records, `relaykv/relaymem_records.py` defines log-only records for Profile Memory, Episode Memory, Summary Memory, Structured Memory, and KV Checkpoint Metadata. It does not implement storage, retrieval, embedding, or concrete RAG backends.

The next intended implementation step is a minimal Fast Recall backend and prompt-preview / CLI memory assistant smoke path before any heavier retrieval or runtime integration work.