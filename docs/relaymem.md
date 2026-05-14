# RelayMEM

## Definition

**RelayMEM** is the design-level max-context-external memory layer for RelayStack.

Its job is to decide what memories, retrieved facts, summaries, and structured state should enter the model's active context before or during generation. RelayMEM is not the same thing as RelayKV.

## Boundary with RelayKV

The boundary is intentionally simple:

- **RelayMEM** decides what memories enter active context
- **RelayKV** decides how active-context KV blocks are routed under a VRAM budget

RelayMEM operates outside the model's currently active context window. RelayKV operates inside the active context that is already admitted to the model.

## Components

RelayMEM is expected to include the following component types:

- **Profile Memory**: stable user, character, assistant, or task profile information
- **Episode Memory**: notable past interactions, events, or session fragments
- **Summary Memory**: compressed summaries of prior context or prior sessions
- **RAG Memory**: retrieved external knowledge and document-grounded evidence
- **Structured Memory**: key-value facts, tool state, schedules, settings, and explicit records
- **Context Assembly**: the layer that assembles active-context input from retrieved and summarized memory pieces
- **KV Checkpoint Metadata**: metadata about reusable prefixes, cached spans, or checkpointed context segments that may help future runtime decisions

## Pluggable RAG Backends

RelayMEM owns the memory lifecycle and context assembly boundary, but the concrete retrieval implementation should remain pluggable.

That means RelayMEM is responsible for questions such as:

- what gets indexed
- what gets summarized
- what gets promoted to profile, episode, or structured memory
- what gets assembled back into active context

But the specific RAG backend used to retrieve evidence can be replaced depending on latency, quality, and deployment needs.

Backend candidates include:

- **Fast Vector RAG**
- **BM25 / keyword search**
- **Hybrid Search**
- **Graph RAG**
- **Evidence-chain retrieval**
- **External RAG service**

### Runtime use

- **`LIVE_LOW_LATENCY`**: fast retrieval only
- **`MEMORY_RECALL_MODE`**: deeper retrieval, possibly user-gated
- **`POST_STREAM_INDEX_MODE`**: offline indexing and refinement after the live interaction path

Under this framing, a NeocorRAG-style evidence-chain retrieval flow is one possible **Deep Recall** backend candidate. It should not be treated as the default every-turn RAG path for RelayMEM.

## Use Cases

RelayMEM is being framed to support several persistent-assistant workloads:

- **AI Vtuber / AI character**: maintain character profile, remembered interactions, and retrieval of older episodes
- **Personal assistant**: maintain profile, preferences, plans, and structured state over long periods
- **Long-form writing**: preserve world facts, story summaries, recurring entities, and continuity constraints across longer projects

Profile Memory should be understood as a general memory category. Viewer-specific profiles are only one possible specialization in the AI Vtuber case.

## Implementation Status

RelayMEM is currently a **design target**, not a claimed implementation in this repository.

The present repository mainly implements RelayKV prototype components and supporting routing/budget schemas. RelayMEM is documented here so the architectural boundary stays clear while the RelayKV prototype remains the primary implemented path.

As a lightweight bridge for logs and experiments, `relaykv/relaymem.py` provides log-only schema objects for retrieval results and context assembly plans. It does not implement any concrete RAG backend.
