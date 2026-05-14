# Max-Context-External Memory

## Core distinction

RelayKV manages **active-context KV inside the model's supported maximum context**.

If the system needs memory that conceptually exceeds the model's supported or practical active context window, that memory must be handled by **RelayMEM** rather than by RelayKV alone.

This distinction matters because RelayKV can improve usable active-context behavior under fixed VRAM, but it does not by itself create a larger trained context window.

## Memory mechanisms outside active context

Max-context-external behavior can include:

- **Sliding Window**: keep only the most recent active context directly in-window
- **Summary Memory**: compress older context into summaries
- **RAG / Vector Memory**: retrieve relevant older facts or documents on demand
- **Structured Memory**: maintain explicit records, profiles, tool outputs, and facts
- **Episode Memory**: retain notable prior interactions or event chunks
- **Profile Memory**: preserve stable traits, preferences, role definitions, and long-lived identity facts
- **KV Checkpoint / Prefix Cache**: reuse previously built or reusable prefix segments when applicable
- **Hybrid Memory Router**: choose among summary, retrieval, profile, episode, and cached-prefix sources

RAG in this layer should be treated as **pluggable** rather than tied to one retrieval implementation. NeocorRAG-style methods are better described as optional deep-recall backends, not the default every-turn retrieval path.

## Boundary of responsibility

The intended responsibility split is:

- **RelayMEM** retrieves, selects, and assembles the active context
- **RelayKV** routes KV inside that active context under the live VRAM budget

That means RelayMEM answers:

- what should be brought back from older memory
- what should be summarized instead of fully replayed
- what profile or structured facts must remain active

RelayKV answers:

- which active-context KV blocks stay on GPU
- which active-context KV blocks can move to RAM or colder tiers
- which active-context KV blocks should be retrieved into the current working set

## Why this split exists

Without this split, it is easy to overclaim that tiered KV management alone solves long-term memory or effectively infinite context. It does not.

RelayKV is a working-set and routing layer for active context. RelayMEM is the layer that makes max-context-external memory usable in the first place.
