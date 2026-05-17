# Devlog: RelayStack contract consolidation

Date basis: JST 2026-05-17.

## Summary

Phase 11.5 consolidated the RelayStack design contracts before restarting runtime adapter work.

This phase intentionally stayed in docs / contract / evaluation-planning scope. It did not add runtime adapter behavior, KV materialization, attention backend connection, scheduler changes, or model-loading paths.

The main outcome is that RelayStack now has explicit design documents for:

- data contracts
- adapter contracts
- runtime modes and fallback-vs-degrade semantics
- evaluation attribution

This closes the contract-consolidation step between the Phase 11 fixed-budget RelayKV dry-run chain and the later Phase 12 adapter/runtime target selection work.

## Completed PRs

### Phase 11.5-A: RelayStack data contracts

PR: #72, `Document RelayStack data contracts`

Added `docs/relaystack_data_contract.md` to define the core cross-layer data contracts:

- `RelayMEMContextItem`
- `RelayCTXContextSpan`
- `RelayKVBlockMeta`
- `RelayStackTraceEvent`

The key invariant is lineage preservation:

```text
RelayMEM item_id
  -> RelayCTX source_item_id
  -> token_span
  -> RelayKV logical_block_id
  -> engine_block_ref at adapter boundary
  -> KV decision / trace event
```

The data contract also keeps the following concepts separate:

```text
kv_class         = role in the working set
precision_level  = representation or compression level
residency_level  = location / storage tier
```

### Phase 11.5-B: RelayStack adapter contracts

PR: #73, `Document RelayStack adapter contracts`

Added `docs/relaystack_adapter_contracts.md` to define the boundary between RelayStack Core and external systems:

- `MemoryBackendAdapter`
- `TokenizerAdapter`
- `EngineAdapter`
- `ObservabilityAdapter`
- App / Agent boundary

The important boundary is:

```text
RelayStack Core:
  decision, metadata, budget, lineage, trace

External systems:
  execution, storage, UI, approval, model-specific runtime behavior
```

Tool execution, approval, UI, product workflow orchestration, heavy RAG / embedding / reranking / summarization execution, and engine-specific runtime internals remain outside RelayStack Core.

### Phase 11.5-C: RelayStack runtime modes

PR: #74, `Document RelayStack runtime modes`

Added `docs/relaystack_runtime_modes.md` to define runtime mode semantics before adapter implementation.

The VRAM / KV axis is:

```text
NORMAL_FULL
RELAYKV_SHADOW
RELAYKV_APPLY
RELAYKV_SAFE_DEGRADE
BLOCKED_NO_SAFE_KV_PATH
REQUEST_CONTEXT_REDUCTION
```

The context / memory axis is:

```text
CONTEXT_NORMAL
CTX_BUDGETED
MEM_RECALL
POST_STREAM_INDEX
```

These axes are independent. Development and quality evaluation proceed top-down:

```text
MEM -> CTX -> KV
```

Runtime activation under VRAM pressure can be KV-first:

```text
NORMAL_FULL
-> RELAYKV_SHADOW
-> RELAYKV_APPLY
-> RELAYKV_SAFE_DEGRADE
-> BLOCKED_NO_SAFE_KV_PATH or REQUEST_CONTEXT_REDUCTION
```

The most important safety rule is:

```text
Before RelayKV apply or in shadow:
  FullKV fallback may be available.

After RelayKV apply under VRAM pressure:
  FullKV fallback must not be assumed.
  Use safe degrade, block, or request context reduction instead.
```

### Phase 11.5-D: RelayStack evaluation attribution plan

PR: #75, `Document RelayStack evaluation attribution plan`

Added `docs/relaystack_eval_plan.md` to prevent mixed-layer quality regressions from becoming untraceable.

The evaluation stages are:

```text
MEM-only
MEM+CTX
MEM+CTX+KV
```

Attribution rules:

- If MEM-only fails, do not blame RelayCTX or RelayKV.
- If MEM-only passes but MEM+CTX fails, suspect context packing, compression, token-span, or attribution issues.
- If MEM+CTX passes but MEM+CTX+KV fails, suspect KV working-set budget, block selection, or RelayKV policy issues.
- If lineage is missing, classify it as a data-contract / trace issue before quality attribution.
- If an engine lacks a required capability, classify it as an adapter capability issue rather than a Core policy-quality failure.
- If RelayKV output differs from FullKV but the task answer remains correct, classify it as behavioral divergence, not necessarily failure.

The evaluation plan also distinguishes:

```text
Reference quality:
  FullKV when available, especially during shadow compare.

Practical low-VRAM baselines:
  sliding window, truncation, recent+static anchor, summary compression,
  RAG-style retrieval/context stuffing, or engine default context/KV policy.
```

## Relationship to Phase 11

Phase 11 produced the fixed-budget RelayKV dry-run artifact chain:

```text
pipeline/scoring artifact
  -> relaykv_candidates.json
  -> relaykv_fixed_budget_block_selection.json
  -> chain_summary.json
```

Phase 11.5 does not replace that chain. It clarifies how that chain fits into RelayStack contracts:

- candidate and block metadata belong to RelayKV working-set decisions
- lineage must eventually connect RelayMEM items through RelayCTX spans into RelayKV logical blocks
- runtime modes define when shadow, apply, degrade, block, or context reduction are valid
- evaluation attribution defines when a failure belongs to MEM, CTX, KV, adapter capability, or trace contract

## Safety boundary

Phase 11.5 remains design / docs / contract only.

Not changed:

- no model loading
- no runtime adapter
- no actual KV materialization
- no production attention backend connection
- no KV pool mutation
- no scheduler changes
- no tool execution or approval runtime
- no Open-LLM-VTuber runtime integration

## Validation recorded in PRs

The Phase 11.5 PRs used lightweight validation only, including:

```bash
python -m compileall relaykv tests scripts
python -m pytest -q tests/test_relaykv_fixed_budget_selection_chain.py
python -m pytest -q tests/test_relaykv_pipeline_candidate_export.py
python -m pytest -q tests/test_relaykv_fixed_budget_block_selection.py
python -m pytest -q tests/test_relaykv_fixed_budget_working_set.py
python -m pytest -q tests/test_relaystack_dry_run.py
```

Some PRs also continued to run report-layer tests such as:

```bash
python -m pytest -q tests/test_relaykv_pressure_shadow_quality_report.py
```

## Current status after Phase 11.5

RelayStack contract consolidation is now ready enough to support Phase 12 planning.

The next phase should not immediately jump into scheduler, attention, or KV-pool mutation. Instead, Phase 12 should choose the adapter/runtime target and define the first minimal adapter boundary.

Recommended Phase 12 direction:

```text
Phase 12-A:
  runtime target selection and adapter capability matrix

Phase 12-B:
  minimal EngineAdapter skeleton / schema mapping plan

Phase 12-C:
  adapter smoke artifact that emits capability flags and logical metadata only
```

Potential runtime targets remain:

- HF prototype adapter
- SGLang adapter
- vLLM adapter
- llama.cpp or other local runtime adapter

The immediate goal should be adapter-boundary clarity, not RelayKV apply.

## Notes

The contract split now gives RelayStack a cleaner product path:

```text
RelayMEM:
  what memory should enter active context

RelayCTX:
  how selected context becomes token spans with attribution

RelayKV:
  how decode-time KV working set stays within fixed VRAM budget

RelayStack Core:
  decision, metadata, budget, lineage, trace

External adapters/app layers:
  execution, storage, UI, approval, engine/runtime behavior
```

This phase is a design checkpoint before implementation risk increases again.
