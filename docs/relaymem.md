# RelayMEM

## Definition

**RelayMEM** is the design-level max-context-external memory layer for RelayStack.

Its job is to decide what memories, retrieved facts, summaries, and structured state should enter the model's active context before or during generation. RelayMEM is not the same thing as RelayKV.

RelayMEM should be treated as a **hierarchical memory and context assembly layer**, not as a single RAG adapter. RAG-like retrieval is one possible backend capability, but RelayMEM owns the higher-level decision about which memory items should be assembled into active context under a token and latency budget.

## Boundary with RelayKV

The boundary is intentionally simple:

- **RelayMEM** decides what memories enter active context
- **RelayKV** decides how active-context KV blocks are routed under a VRAM budget

RelayMEM operates outside the model's currently active context window. RelayKV operates inside the active context that is already admitted to the model.

The intended long-term data boundary is:

```text
RelayMEMContextItem
  ↓
token_span
  ↓
logical_block_id
  ↓
RelayKVBlockMeta
```

RelayMEM may provide hints such as memory source, evidence role, or semantic anchor priority. RelayKV should treat those as policy inputs rather than as mandatory routing decisions.

## Components

RelayMEM is expected to include the following component types:

- **Profile Memory**: stable user, character, assistant, or task profile information
- **Episode Memory**: notable past interactions, events, or session fragments
- **Summary Memory**: compressed summaries of prior context or prior sessions
- **RAG Memory**: retrieved external knowledge and document-grounded evidence
- **Structured Memory**: key-value facts, tool state, schedules, settings, and explicit records
- **Context Assembly**: the layer that assembles active-context input from retrieved and summarized memory pieces
- **KV Checkpoint Metadata**: metadata about reusable prefixes, cached spans, or checkpointed context segments that may help future runtime decisions

These memory classes are not all equivalent. For example, Profile Memory and selected evidence summaries may become semantic anchor hints, while detailed episode fragments or RAG chunks may become retrieved context candidates.

## RelayStack responsibility split

RelayStack is not a RAG replacement. The responsibility split is:

```text
Retrieval backend
  finds candidate records, documents, chunks, or evidence chains

RelayMEM
  ranks, budgets, and assembles selected items into active context

Tokenizer / model prefill
  converts selected context into token spans and KV blocks

RelayKV
  manages the decode-time KV working set under a fixed VRAM budget
```

This split keeps RelayMEM focused on context-before-KV decisions and RelayKV focused on post-prefill KV working-set control.

## Pluggable RAG Backends

RelayMEM owns the memory lifecycle and context assembly boundary, but the concrete retrieval implementation should remain pluggable.

That means RelayMEM is responsible for questions such as:

- what gets indexed
- what gets summarized
- what gets promoted to profile, episode, or structured memory
- what gets assembled back into active context

But the specific RAG backend used to retrieve evidence can be replaced depending on latency, quality, and deployment needs.

Backend candidates include:

- **Fast Recall / keyword or token-overlap retrieval**
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
