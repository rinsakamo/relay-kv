# Devlog: Phase 12 HF adapter boundary checkpoint

Date basis: JST 2026-05-17.

## Summary

Phase 12 restarted RelayStack runtime-boundary work after the Phase 11.5 contract consolidation. The phase deliberately began with HF-first validation and adapter capability artifacts rather than scheduler, attention, or KV-pool integration.

The main outcome so far is a safe V0.1 runtime target decision and the first no-model/no-GPU HF adapter capability smoke artifact.

Completed scope:

- Phase 12-A: runtime target selection and adapter capability matrix
- Phase 12-B: HF adapter capability schema / skeleton plan
- Phase 12-C: HF adapter capability smoke artifact generator

The work remains metadata-only and does not perform runtime adapter integration or RelayKV apply.

## Completed PRs

### Phase 12-A: runtime target selection

PR: #77, `Document Phase 12 runtime target selection`

Phase 12-A selected the runtime target order for RelayStack V0.1:

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

Rationale:

- HF is the lowest-risk first validation path for tokenizer/model metadata and contract validation.
- SGLang remains the next practical OpenAI-compatible runtime target.
- vLLM remains important for engine-agnostic direction but is kept post-V0.1 to avoid scope expansion.
- llama.cpp / local runtimes are useful later product/runtime exploration targets.

Artifact:

```text
docs/phase12_runtime_target_selection.md
```

### Phase 12-B: HF adapter capability schema

PR: #78, `Document HF adapter capability schema`

Phase 12-B defined the expected JSON-safe HF adapter capability artifact shape:

```text
relaystack_adapter_capabilities.json
```

Key capability fields:

```text
supports_tokenizer_span_probe
supports_engine_metadata_probe
supports_fullkv_reference
supports_shadow_compare
supports_materialization
supports_apply
supports_safe_degrade
supports_context_reduction_request
```

Important Phase 12-B defaults:

```text
supports_tokenizer_span_probe: true
supports_engine_metadata_probe: true
supports_fullkv_reference: true
supports_shadow_compare: false
supports_materialization: false
supports_apply: false
supports_safe_degrade: false
supports_context_reduction_request: true
```

Artifact:

```text
docs/phase12_hf_adapter_capability_schema.md
```

### Phase 12-C: HF adapter capability smoke artifact

PR: #79, `Add HF adapter capability smoke artifact`

Phase 12-C added a no-model/no-GPU smoke CLI that emits the HF adapter capability artifact.

Added:

```text
scripts/run_hf_adapter_capability_smoke.py
tests/test_hf_adapter_capability_smoke.py
```

Output artifact:

```text
relaystack_adapter_capabilities.json
```

The artifact records:

- schema version
- phase
- adapter kind/name
- runtime target
- model reference
- tokenizer reference
- context window hint
- capability flags
- safety scope
- compact summary

The smoke path intentionally avoids:

- model loading
- GPU inspection
- KV materialization
- attention backend connection
- runtime adapter behavior
- scheduler changes
- KV-pool mutation
- RelayKV apply

## Validation recorded in Phase 12-C

PR #79 recorded the following validation:

```bash
python -m compileall relaykv tests scripts
python -m pytest -q tests/test_hf_adapter_capability_smoke.py
python scripts/run_hf_adapter_capability_smoke.py --output /tmp/relaystack_adapter_capabilities.json
python -m json.tool /tmp/relaystack_adapter_capabilities.json >/dev/null
python -m pytest -q tests/test_relaystack_dry_run.py
python -m pytest -q tests/test_relaystack_hf_smoke_report.py
```

## Current safety boundary

Phase 12-A through Phase 12-C are adapter-boundary preparation, not engine integration.

Still not changed:

- no model loading
- no GPU inspection required
- no runtime adapter execution path
- no actual KV materialization
- no production attention backend connection
- no scheduler change
- no KV-pool mutation
- no RelayKV apply
- no Open-LLM-VTuber integration
- no tool execution or approval UX

## Relationship to earlier contracts

Phase 12-C is the first concrete metadata artifact after Phase 11.5's contract consolidation.

It maps onto the existing contracts as follows:

```text
RelayStack data contract:
  artifact is JSON-safe and replayable

RelayStack adapter contract:
  adapter declares capabilities before execution

RelayStack runtime modes:
  supports_apply remains false, so this is not RELAYKV_APPLY

RelayStack eval plan:
  artifact supports future attribution by recording model/tokenizer/runtime capability metadata
```

## Recommended next work

The next step should continue to be metadata-only or shadow-safe.

Recommended order:

```text
Phase 12-D:
  status/devlog checkpoint for Phase 12-A through Phase 12-C

Phase 12-E:
  tokenizer span probe artifact

Phase 12-F:
  engine/model metadata probe artifact

Phase 13-A:
  safe materialization / shadow attention compare planning or minimal shadow artifact
```

The most natural next implementation step is tokenizer span probing, because it validates the RelayCTX token-span contract before any KV working-set or attention work.

## Notes

Phase 12 is intentionally conservative. The project is now moving from design contracts back toward implementation, but the first concrete artifact only declares what the HF prototype adapter can safely provide.

This keeps RelayStack V0.1 aligned with the project boundary:

```text
RelayStack Core:
  decision, metadata, budget, lineage, trace

HF prototype adapter:
  first validation target for model/tokenizer/capability metadata

Later adapters:
  SGLang next, vLLM post-V0.1
```
