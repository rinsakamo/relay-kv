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

RelayLM is now the product-facing name for the Relay Lineage Manager. RelayLM is not a language model; it is a memory-context-token-runtime-cache lineage manager for local LLM applications, agents, and local inference runtimes.

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

- **RelayLM** is the product-facing lineage-manager framing.
- **RelayStack Core** is the internal responsibility boundary. It is not a separate runtime component name.
- **RelayKV** is the VRAM-aware active-context KV routing layer.
- **RelayMEM** is the max-context-external memory layer that decides what should enter active context.
- **RelayCTX** is the context transform layer between RelayMEM and RelayKV. It handles context packing, token-budget fitting, context rebuild planning, token compression plan metadata, prompt layout at the context-item level, source attribution, and token/span mapping.
- **RelayPLC** is the policy, lifecycle, and control layer. It replaces the older RelayPolicy naming.
- **RelayTRC** is the trace, transition, and record schema layer. It replaces the older Trace Schema naming.
- **Relay Adapter** is the external connection and translation layer. Concrete adapter modules use `*ADP` naming.
- **OpenAI-compatible API boundaries** are the preferred product-facing boundary for practical validation. Application surfaces such as Hermes-Agent, OpenClaw, and Open-LLM-VTuber are post-V0.1 validation targets, not pre-V0.1 Core implementation scope.

RelayStack Core is intentionally not a tool-execution or agent-runtime layer. Tool execution, web search, approval, UI, product-specific workflow orchestration, TTS/ASR, VTube Studio or avatar integration, Hermes-Agent/OpenClaw-specific behavior, heavy RAG implementations, embedding/reranking/summarization model execution, engine-specific runtime internals, and observability product integrations should live outside RelayStack Core behind adapters or application/orchestration layers.

See [relaystack_architecture.md](relaystack_architecture.md) for the current RelayLM / RelayStack core-boundary, runtime-mode, adapter-boundary, and fallback/degrade design.

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

## RelayLM modes

RelayLM does not require RelayKV to be active from the start.

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

For models whose KV cache is already relatively small, or when the runtime has sufficient VRAM, MEM_CTX_ONLY can still provide value through memory selection, context packing, context rebuild planning, and traceability. RelayKV becomes the critical layer when RelayLM must make model-max-context operation VRAM-aware.

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
- HF adapter capability smoke artifact generator for `relaystack_adapter_capabilities.json`
- HF tokenizer span probe artifact generator for `relaystack_tokenizer_span_probe.json`
- HF engine/model metadata probe artifact generator for `relaystack_engine_metadata_probe.json`

## Design-only or not yet integrated

The following items are part of the current direction, but should not be described as implemented runtime features yet:

- RelayLM product-facing runtime or server implementation
- Relay Adapter concrete implementation beyond current metadata/artifact planning
- RelayCTX runtime implementation beyond design-level context-transform responsibilities
- concrete context rebuild execution path
- concrete RelayMEM retrieval backend beyond the current lightweight Fast Recall path
- concrete RelayStack data-contract and adapter-contract implementations beyond metadata/artifact planning
- OpenAI-compatible product-facing RelayStack / RelayLM server
- Hermes-Agent, OpenClaw, Open-LLM-VTuber, or other agent/avatar-app integration
- tool execution or web search integration
- User-Gated Fallback runtime UX
- real KV materialization in an inference engine
- real routed execution in a production backend
- full end-to-end runtime policy integration across GPU, RAM, and SSD tiers
- compressed KV implementation
- disk-backed or RAM-backed full KV checkpoint execution
- SGLang runtime adapter changes beyond prior exploratory work
- vLLM runtime adapter support
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
  Completed: RelayStack design contract consolidation
  Phase 11.5-A: completed RelayStack data contract documentation in
  docs/relaystack_data_contract.md for RelayMEMContextItem, RelayCTXContextSpan,
  RelayKVBlockMeta, and RelayStackTraceEvent
  Phase 11.5-B: completed RelayStack adapter contract documentation in
  docs/relaystack_adapter_contracts.md for MemoryBackendAdapter, TokenizerAdapter,
  EngineAdapter, ObservabilityAdapter, and the App / Agent boundary
  Phase 11.5-C: completed RelayStack runtime mode documentation in
  docs/relaystack_runtime_modes.md for VRAM/KV runtime states and
  fallback-vs-degrade-vs-block-vs-context-reduction semantics
  Phase 11.5-D: completed RelayStack evaluation attribution plan documentation in
  docs/relaystack_eval_plan.md for MEM-only, MEM+CTX, and MEM+CTX+KV evaluation separation
  and regression attribution rules
  Phase 11.5-E: completed JST-dated consolidation devlog in
  notes/devlog_2026-05-17_relaystack_contract_consolidation_ja.md
  Scope: docs/schema/evaluation planning only before runtime adapter restart.
  No runtime behavior is changed.

Phase 12:
  Current: RelayStack V0.1 runtime boundary and HF-first validation
  Phase 12-A: completed runtime target selection and adapter capability matrix in
  docs/phase12_runtime_target_selection.md
  Phase 12-B: completed HF adapter capability schema and skeleton plan in
  docs/phase12_hf_adapter_capability_schema.md
  Phase 12-C: completed HF adapter capability smoke artifact in
  scripts/run_hf_adapter_capability_smoke.py
  Phase 12-D: completed JST-dated status/devlog checkpoint for Phase 12-A through Phase 12-C in
  notes/devlog_2026-05-17_phase12_hf_adapter_boundary_ja.md
  Phase 12-E: completed HF tokenizer span probe artifact in
  scripts/run_hf_tokenizer_span_probe.py
  Phase 12-F: completed HF engine/model metadata probe artifact in
  scripts/run_hf_engine_metadata_probe.py
  Phase 12-G: completed JST-dated status/devlog checkpoint for Phase 12-E through Phase 12-F in
  notes/devlog_2026-05-17_phase12_hf_metadata_probes_ja.md
  Phase 12-H: completed HF adapter readiness report in
  scripts/run_hf_adapter_readiness_report.py
  Phase 12-I: completed HF tokenizer/config-backed metadata probe in
  scripts/run_hf_tokenizer_config_probe.py
  Artifacts:
    - relaystack_adapter_capabilities.json
    - relaystack_tokenizer_span_probe.json
    - relaystack_engine_metadata_probe.json
    - relaystack_hf_adapter_readiness_report.json
    - relaystack_hf_tokenizer_config_probe.json
  Use HF as the first concrete runtime path to validate the Core contracts and minimal OpenAI-compatible boundary without adding scheduler, attention, or KV-pool mutation by default. SGLang remains the next practical OpenAI-compatible runtime target after V0.1. vLLM remains a post-V0.1 adapter target and should not broaden the V0.1 implementation scope.
  Scope: tokenizer/config metadata probe only through Phase 12-I. No runtime adapter,
  model loading of weights, GPU inspection, KV materialization, attention connection, scheduler path, or KV-pool mutation is changed.
  Phase 12-I may load tokenizer/config metadata only, behind the existing readiness gate.
  Next likely implementation step:
    - tokenizer/config-backed checkpoint devlog or next HF metadata consolidation step

Phase 12.5:
  Current: RelayLM naming and product-boundary consolidation
  Scope: docs-only clarification that relay-lm is the product / architecture repository while relay-kv remains the RelayKV runtime/cache research repository until implementation migration is justified.

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
  token-span mapping, and context rebuild planning before prefill. RelayCTX should not execute tools.

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

## V0.1 boundary and post-V0.1 validation

V0.1 should validate the RelayLM / RelayStack Core contracts and the minimum product-facing OpenAI-compatible API boundary before expanding runtime or application scope.

```text
V0.1 scope:
  RelayLM product framing
  RelayMEM / RelayCTX / RelayKV contract validation
  RelayPLC / RelayTRC contract validation
  Relay Adapter contract validation
  HF-first runtime validation
  minimal OpenAI-compatible API boundary
  traceable budget, lineage, and runtime-mode artifacts

Post-V0.1 validation:
  SGLang OpenAI-compatible backend path
  Hermes-Agent / OpenClaw / Open-LLM-VTuber or similar application surfaces
  local AI friend / avatar-style practical conversation tests
  vLLM adapter evaluation
  optional read-only web search or other tool policies
```

Tool execution, web search, UI, TTS/ASR, VTube Studio, Hermes-Agent/OpenClaw/Open-LLM-VTuber-specific behavior, and product workflows remain outside RelayStack Core. They should be treated as post-V0.1 validation surfaces or App / Agent responsibilities unless a later contract explicitly promotes a narrow capability behind an adapter boundary.

## Current empirical picture

The prototype already supports direct approximation-quality measurement. The current results suggest that:

- candidate coverage is a strong predictor of error
- larger hot windows improve stability
- deeper layers are harder than shallow layers
- longer contexts remain harder, but still follow the same coverage-driven trend

See [experimental_findings.md](experimental_findings.md) for the current measured findings, [phase10_pressure_shadow_quality.md](phase10_pressure_shadow_quality.md) for the current Phase 10 chain and real-artifact validation workflow, and [evaluation_targets.md](evaluation_targets.md) for the next target directions.
