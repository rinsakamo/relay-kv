# RelayKV Current Status

## Summary

RelayKV currently has a working research prototype for active-context KV approximation and comparison under a tiered memory path.

The implemented prototype flow is still:

```text
KV split
→ CPU cold offload
→ blockify
→ metadata
→ scoring
→ retrieval
→ candidate KV
→ working KV
→ attention comparison
```

This remains the main comparison path for the repository.

## Main executable path

```bash
python scripts/run_relaykv_pipeline.py
```

This script runs the representative RelayKV path and writes:

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

## Current design direction

The current project direction is broader than the initial small KV approximation framing.

- **RelayKV** is the VRAM-aware active-context KV routing layer.
- **RelayMEM** is the max-context-external memory layer that decides what should enter active context.
- **RelayCTX** is the context transform layer between RelayMEM and RelayKV. It handles context packing, token-budget fitting, token compression plan metadata, prompt layout at the context-item level, source attribution, and token/span mapping.
- **RelayStack** coordinates RelayMEM, RelayCTX, RelayKV, budget policy, runtime policy, and trace/evaluation schemas.
- **Open-LLM-VTuber** is the current practical product/demo target for low-VRAM Japanese AI character runtime integration.

RelayStack Core is intentionally not a tool-execution or agent-runtime layer. Tool execution, approval, UI, product-specific workflow orchestration, heavy RAG implementations, embedding/reranking/summarization model execution, engine-specific runtime internals, and observability product integrations should live outside RelayStack Core behind adapters or application/orchestration layers.

See [relaystack_architecture.md](relaystack_architecture.md) for the current RelayStack core-boundary, runtime-mode, adapter-boundary, and fallback/degrade design.

RelayKV extends usable active context under fixed VRAM budgets, but it does not by itself extend the model's trained or supported maximum context window.

RelayKV should now be treated as a **fixed-VRAM-budget decode-time KV working-set controller**. The working-set budget is conceptually split across role-specific KV classes:

```text
B_total_working_kv
  = B_recent
  + B_anchor
  + B_retrieved
  + B_transient
```

The design keeps three concepts separate:

```text
kv_class         = role in the working set
precision_level  = representation or compression level
residency_level  = location such as GPU HBM, CPU RAM, SSD, or remote tier
```

This separation keeps future compressed or checkpointed KV representations from being confused with RelayKV's selection role.

RelayKV fallback semantics should be treated carefully. Before apply, or while running in shadow, FullKV fallback may be possible because the FullKV path is still available. After RelayKV apply under VRAM pressure, FullKV fallback should generally be treated as unavailable. In that case RelayKV should use safe degrade, block, or request context reduction semantics rather than implying an automatic return to FullKV.

## Implemented today

The repository currently implements or prototypes the following pieces:

- PyTorch prototype execution path through `scripts/run_relaykv_pipeline.py`
- hot/cold KV split and CPU cold offload
- cold blockification and block metadata construction
- scoring and retrieval of candidate cold blocks
- candidate KV and working KV assembly
- attention comparison against full KV attention
- working-budget dry-run decisions
- VRAM-budget dry-run decisions
- demotion dry-run decisions
- memory-block schemas and routing-decision schemas for future runtime integration
- RelayMEM schema objects for retrieval results and context assembly plans
- RelayMEM Fast Recall backend v1 for stdlib-only keyword-based memory recall
- RelayMEM prompt preview / user-gated fallback planning schemas and smoke path
- RelayMEM record schemas for profile, episode, summary, structured, and KV checkpoint metadata records
- User-Gated Fallback schema fields
- VRAM reservation schema and smoke path
- RelayStack no-model/no-GPU dry-run JSON combining RelayMEM retrieval results, context assembly, prompt preview planning, final routing decisions, RelayKV, VRAM reservation, runtime policy, and fallback fields
- RelayStack HF smoke report layer that joins a synthetic-or-real HF context-length smoke artifact with a RelayStack dry-run artifact

## Design-only or not yet integrated

The following items are part of the current direction, but should not be described as implemented runtime features yet:

- RelayCTX runtime implementation beyond design-level context-transform responsibilities
- concrete RelayMEM retrieval backend beyond the current lightweight Fast Recall path
- concrete RelayStack data-contract and adapter-contract implementations
- Open-LLM-VTuber integration
- User-Gated Fallback runtime UX
- real KV materialization in an inference engine
- real routed execution in a production backend
- full end-to-end runtime policy integration across GPU, RAM, and SSD tiers
- compressed KV implementation
- disk-backed or RAM-backed full KV checkpoint execution
- SGLang or vLLM runtime adapter changes beyond prior exploratory work
- RelayKV safe-degrade / block / request-context-reduction runtime handling after apply

## Revised phase direction

The next work should keep model/GPU/runtime risk low until RelayMEM and RelayStack planning are useful as standalone artifacts. The phase order now distinguishes implementation/evaluation order from runtime activation order.

```text
Phase 6:
  Completed: RelayMEM Fast Recall backend v1

Phase 6.5:
  Completed: RelayMEM prompt preview / user-gated fallback planning
  Scope: schema and smoke only. No model, GPU, runtime, or KV path is called.

Phase 7:
  Completed: RelayStack dry-run artifact includes RelayMEM prompt_preview_plan
  Scope: dry-run planning only. No model, GPU, runtime, attention, KV, or scheduler path is called.

Phase 8:
  Completed: RelayStack final routing decision dry-run layer
  Scope: schema and dry-run planning only. No model, GPU, runtime, attention, KV, or scheduler path is called.

Phase 9:
  Current: HF smoke artifact + RelayStack dry-run report layer
  Scope: report/join only with synthetic-artifact tests. No model, GPU, runtime, attention, KV, or scheduler path is changed.

Phase 10:
  RelayKV pressure-triggered shadow policy quality test
  Phase 10-A: completed planning/report-only recommendation fields for when a shadow quality test
  should be prioritized from existing pressure and smoke signals
  Phase 10-B: completed pressure-triggered shadow quality join/report layer combining
  RelayStack pressure recommendation with existing RelayKV pipeline quality metrics
  Phase 10-C: completed synthetic/no-model artifact-chain wrapper for reproducing the report inputs
  and final pressure shadow quality report in a fixed order
  Phase 10-D: completed existing-pipeline-artifact validation guide and threshold calibration scaffold
  for joining real RelayKV pipeline summaries into the fixed chain without changing runtime behavior
  Phase 10-E: completed JST-dated real-artifact smoke/devlog pass covering
  synthetic-chain confirmation, fresh real-pipeline generation attempt logging, and existing real-artifact join results
  Scope: the default synthetic artifact chain remains no-model/no-GPU/report-only.
  The fresh real-artifact smoke may invoke the existing PyTorch RelayKV pipeline,
  including model loading and attention comparison, to generate a real pipeline summary.
  Even in Phase 10-E, there is still no runtime adapter, no RelayKV apply,
  no production attention backend connection, no KV pool mutation, and no scheduler change.

Phase 11:
  RelayKV fixed-budget working-set dry-run policy
  Phase 11-A: completed fixed-budget working-set dry-run policy schema and CLI
  for allocating RECENT / ANCHOR / RETRIEVED token budgets without runtime integration
  Phase 11-B: completed fixed-budget block candidate selection dry-run
  joining fixed-budget decisions to synthetic/candidate block metadata and emitting selected/rejected/overflow block ids
  Phase 11-C: completed pipeline/scoring candidate exporter
  for converting pipeline-style block scoring artifacts into Phase 11-B candidates-json
  and feeding fixed-budget block selection from exported metadata
  Phase 11-D: completed fixed-budget selection artifact chain
  chaining pipeline candidate export into fixed-budget block selection and emitting
  candidates, selection, and chain_summary artifacts in one no-model/no-GPU dry-run path
  Phase 11-E: completed JST-dated devlog/status consolidation for the fixed-budget dry-run chain
  with the current RelayMEM / RelayCTX / RelayKV / RelayStack Core boundary clarified
  Scope: dry-run/schema/CLI/report only. No materialization, attention connection,
  runtime adapter, or scheduler path is changed. The fixed-budget chain artifacts are:
    - relaykv_candidates.json
    - relaykv_fixed_budget_block_selection.json
    - chain_summary.json

Phase 11.5:
  Current: RelayStack design contract consolidation
  Phase 11.5-A: completed RelayStack data contract documentation in
  docs/relaystack_data_contract.md for RelayMEMContextItem, RelayCTXContextSpan,
  RelayKVBlockMeta, and RelayStackTraceEvent
  Phase 11.5-B: completed RelayStack adapter contract documentation in
  docs/relaystack_adapter_contracts.md for MemoryBackendAdapter, TokenizerAdapter,
  EngineAdapter, ObservabilityAdapter, and the App / Agent boundary
  Phase 11.5-C: completed RelayStack runtime mode documentation in
  docs/relaystack_runtime_modes.md for VRAM/KV runtime states and
  fallback-vs-degrade-vs-block-vs-context-reduction semantics
  Scope: docs/schema planning only before runtime adapter restart.
  No runtime behavior is changed.
  Required contracts:
    - RelayStack Core Boundary
    - RelayMEM → RelayCTX → RelayKV data contract
    - Lineage / Attribution contract
    - Runtime Mode contract
    - Fallback vs RelayKV Degrade / Block / Context Reduction contract
    - Adapter contract for MemoryBackend / Tokenizer / Engine / Observability
  Next:
    - Phase 11.5-D evaluation attribution plan

Phase 12:
  RelayStack adapter contract and runtime target selection
  Choose SGLang, vLLM, HF, or another adapter target after the adapter contracts are clear.
  Scope: adapter-boundary planning and target selection first; no scheduler, attention, or KV-pool mutation by default.

Phase 13:
  Safe materialization / shadow attention compare
  Scope: RelayKV decisions are evaluated while FullKV is still available.
  FullKV fallback is valid in this phase because RelayKV is not yet the active required path.

Phase 14:
  Gated apply / safe degrade / block / context-reduction integration
  Scope: RelayKV becomes the active path only behind gates.
  After RelayKV apply under VRAM pressure, FullKV fallback is not assumed to be available.
  The required safety paths are safe degrade, block-no-safe-path, and request-context-reduction.

Phase 15:
  RelayCTX budgeted context integration
  Scope: context packing, token-budget fitting, token compression plan metadata, source attribution,
  and token-span mapping before prefill. RelayCTX should not execute tools.

Phase 16:
  RelayMEM + RelayCTX + RelayKV attribution evaluation
  Scope: evaluate MEM-only, MEM+CTX, and MEM+CTX+KV paths separately so quality regressions can be attributed
  to memory selection, context transformation, or KV working-set decisions.
```

## Runtime activation vs evaluation order

RelayStack implementation and quality evaluation should proceed top-down for clean attribution:

```text
1. RelayMEM only
2. RelayMEM + RelayCTX
3. RelayMEM + RelayCTX + RelayKV
```

Runtime activation under pressure is different. When VRAM pressure is the trigger, RelayKV is the immediate runtime layer:

```text
NORMAL_FULL
→ RELAYKV_SHADOW
→ RELAYKV_APPLY
→ RELAYKV_SAFE_DEGRADE
→ BLOCKED_NO_SAFE_KV_PATH or REQUEST_CONTEXT_REDUCTION
```

RelayMEM and RelayCTX remain context-planning layers. They are especially useful for future turns, token/context pressure, long-term memory recall, and prefill reduction, while RelayKV handles decode-time KV pressure after active context has already entered the model.

## Current empirical picture

The prototype already supports direct approximation-quality measurement. The current results suggest that:

- candidate coverage is a strong predictor of error
- larger hot windows improve stability
- deeper layers are harder than shallow layers
- longer contexts remain harder, but still follow the same coverage-driven trend

See [experimental_findings.md](experimental_findings.md) for the current measured findings, [phase10_pressure_shadow_quality.md](phase10_pressure_shadow_quality.md) for the current Phase 10 chain and real-artifact validation workflow, and [evaluation_targets.md](evaluation_targets.md) for the next target directions.
