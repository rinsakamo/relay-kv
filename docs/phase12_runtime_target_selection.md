# Phase 12 Runtime Target Selection

Design status: 2026-05-17 JST.

This document records the Phase 12-A runtime target selection and adapter capability matrix for RelayStack V0.1. It is a docs/planning artifact only. It does not implement a runtime adapter, KV materialization, attention connection, scheduler change, or model-loading path.

## Goal

Phase 12 should restart implementation risk only at the adapter boundary. The immediate goal is not RelayKV apply. The immediate goal is to choose the first concrete runtime target and define the minimum adapter capability surface needed to validate the Phase 11.5 contracts.

The selected V0.1 direction is:

```text
Primary V0.1 target:
  HF prototype adapter

Next practical target:
  SGLang adapter

Post-V0.1 target:
  vLLM adapter

Other future target:
  llama.cpp / local runtime adapter
```

## Selection rationale

### HF prototype adapter first

HF should be the first runtime target because it is the lowest-risk way to validate RelayStack Core contracts against real tokenizer/model artifacts without entering production scheduler or KV-pool internals.

HF-first lets Phase 12 validate:

- TokenizerAdapter behavior
- prompt / token-span mapping
- RelayStackTraceEvent-compatible artifact emission
- EngineAdapter capability flags
- FullKV reference availability for future shadow comparison
- minimal OpenAI-compatible boundary planning
- local no-server / script-first smoke loops

HF-first should not be treated as the final serving stack. It is a validation adapter.

### SGLang next

SGLang remains the next practical OpenAI-compatible runtime target because previous exploratory work showed where RelayKV observation metadata can attach, and SGLang is closer to the intended serving/runtime integration path.

SGLang should wait until the HF-first adapter boundary proves the Core contracts and capability reporting shape.

### vLLM post-V0.1

vLLM should remain a post-V0.1 adapter target. It is important for engine-agnostic product direction, but adding it before the first HF/SGLang adapter boundary would broaden scope too early.

### llama.cpp / local runtime later

llama.cpp or other local runtimes may be useful for product reach, but they should not drive Phase 12-A. They can be evaluated after the adapter capability matrix and HF/SGLang mapping are stable.

## Capability matrix

| Capability | HF prototype adapter | SGLang adapter | vLLM adapter | llama.cpp / local adapter |
| --- | --- | --- | --- | --- |
| TokenizerAdapter validation | Strong first target | Likely available through engine/tokenizer path | Likely available through engine/tokenizer path | Runtime-specific |
| OpenAI-compatible serving relevance | Indirect / wrapper needed | Strong | Strong | Varies |
| FullKV reference path | Strong for prototype | Possible in shadow phases | Possible but adapter-specific | Runtime-specific |
| Scheduler/KV-pool risk | Low | Medium/high | Medium/high | Runtime-specific |
| Logical block metadata observation | Prototype-owned | Needs adapter mapping | Needs adapter mapping | Runtime-specific |
| Safe materialization path | Prototype only | Future phase | Future phase | Future phase |
| RelayKV apply path | Out of V0.1 scope | Future phase | Future phase | Future phase |
| Best Phase 12 role | First validation adapter | Next practical serving target | Post-V0.1 engine-agnostic target | Later product/runtime exploration |

## Phase 12-A decision

Phase 12-A chooses HF-first validation for RelayStack V0.1.

The acceptance criteria for Phase 12-A are docs/planning only:

- runtime target order is explicit
- capability matrix is explicit
- HF-first scope is limited to adapter-boundary validation
- SGLang and vLLM are not removed from the roadmap
- no runtime apply, scheduler, attention, or KV-pool mutation is introduced

## Minimal Phase 12-B adapter boundary

Phase 12-B should create a minimal adapter-boundary plan or skeleton around capability reporting and artifact emission.

Recommended Phase 12-B output:

```text
HF adapter capability artifact
  -> tokenizer_ref
  -> model_ref
  -> context_window_hint
  -> supports_fullkv_reference
  -> supports_shadow_compare
  -> supports_materialization = false for V0.1
  -> supports_apply = false for V0.1
  -> supports_safe_degrade = false or planned
  -> supports_context_reduction_request = planned/core-only
```

The initial HF adapter should be allowed to emit JSON-safe capability and metadata artifacts without changing model execution semantics.

## Minimal Phase 12-C smoke path

Phase 12-C can add a no-apply smoke path that emits adapter capability and logical metadata only.

Suggested output artifacts:

```text
relaystack_adapter_capabilities.json
relaystack_tokenizer_span_probe.json
relaystack_engine_metadata_probe.json
```

The smoke path should not:

- materialize KV
- connect to attention backend
- change scheduler behavior
- mutate engine KV pool
- require Open-LLM-VTuber integration
- imply RelayKV apply

## Relationship to Phase 13 and Phase 14

Phase 13 can use the first adapter boundary for safe materialization / shadow attention compare only after capability reporting and metadata probes are stable.

Phase 14 remains gated apply / safe degrade / block / context-reduction integration. FullKV fallback must not be assumed after RelayKV apply under VRAM pressure.

## Non-goals

Phase 12-A does not implement:

- runtime adapter code
- actual KV materialization
- production attention backend connection
- scheduler or KV-pool changes
- Open-LLM-VTuber integration
- tool execution or approval UX
- SGLang or vLLM adapter code

## Recommended next PRs

```text
Phase 12-A:
  runtime target selection and adapter capability matrix docs

Phase 12-B:
  HF adapter capability schema / skeleton plan

Phase 12-C:
  HF adapter metadata smoke artifact

Phase 12-D:
  Phase 12 status/devlog checkpoint before any materialization work
```
