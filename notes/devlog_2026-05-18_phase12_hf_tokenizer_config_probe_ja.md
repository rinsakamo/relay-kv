# Devlog: Phase 12 HF tokenizer/config metadata probe checkpoint

Date basis: JST 2026-05-18.

## Summary

Phase 12-I completed the HF tokenizer/config-backed metadata probe checkpoint.

This phase is the first Phase 12 step that allows a guarded HF metadata load. The allowed scope is intentionally narrow: tokenizer/config metadata may be loaded behind the existing readiness report gate, but model weights, GPU state, KV materialization, attention, scheduler, runtime adapter paths, and RelayKV apply remain untouched.

The new probe artifact is:

```text
relaystack_hf_tokenizer_config_probe.json
```

The current Phase 12 artifact chain is now:

```text
relaystack_adapter_capabilities.json
relaystack_tokenizer_span_probe.json
relaystack_engine_metadata_probe.json
relaystack_hf_adapter_readiness_report.json
relaystack_hf_tokenizer_config_probe.json
```

## Completed PR

PR:

```text
#86 Add HF tokenizer config metadata probe
```

Merge commit:

```text
9e521c65a55f0777a2914cbe4f76e32aecf4b956
```

Merged at:

```text
2026-05-18T12:25:44Z
```

## Files added or updated

Primary implementation:

```text
scripts/run_hf_tokenizer_config_probe.py
```

Tests:

```text
tests/test_hf_tokenizer_config_probe.py
```

Docs/status:

```text
docs/current_status.md
docs/phase12_hf_adapter_capability_schema.md
```

Related upstream gate:

```text
scripts/run_hf_adapter_readiness_report.py
tests/test_hf_adapter_readiness_report.py
```

## What changed

Phase 12-I adds a readiness-gated HF tokenizer/config metadata probe.

The probe supports:

- no-load skip mode
- local-files-only tokenizer/config metadata loading
- tokenizer/config-backed metadata extraction when the readiness gate permits it
- graceful artifact output for unavailable dependency, unavailable local files, or blocked readiness state
- comparison against readiness-scoped model and tokenizer references

The emitted artifact records:

- model_ref
- tokenizer_ref
- readiness_ref
- tokenizer_probe
- config_probe
- consistency
- safety_scope
- summary
- notes

This keeps the output useful as a contract artifact rather than a hidden runtime side effect.

## Readiness-gate hardening

The Codex review cycle tightened the reference scope so the tokenizer/config probe cannot silently drift away from the readiness-validated adapter scope.

The final behavior is:

- model_id mismatch is blocking
- tokenizer_name_or_path mismatch is blocking
- local_path is a readiness-scoped field and is strict
- tokenizer_revision is a readiness-scoped field and is strict
- when readiness did not record local_path, caller-supplied local_path overrides are rejected
- when readiness did not record tokenizer_revision, caller-supplied tokenizer_revision overrides are rejected
- mismatch failures apply even when tokenizer/config loading is skipped
- mismatch failures apply even when tokenizer/config loading itself succeeds

This matters because Phase 12-I is the first checkpoint that may consult real HF tokenizer/config metadata. The readiness report remains the authority for whether a caller is still operating inside the validated artifact chain.

## Safety boundary

Phase 12-I allows tokenizer/config metadata probing only.

Still not included:

```text
model weight loading
GPU inspection
actual KV materialization
attention backend connection
runtime adapter/server path
scheduler changes
KV-pool mutation
RelayKV apply
OpenAI-compatible product server
Open-LLM-VTuber integration
tool execution or approval UX
```

Config metadata loading is not treated as model loading. The probe may inspect configuration fields such as attention head counts, key-value head counts, context-window hints, dtype strings, or quantization config metadata, but it must not instantiate or execute model weights.

## Why this checkpoint matters

Phase 12-H made the earlier HF metadata probes usable as a readiness-gated artifact chain.

Phase 12-I takes the next step: it validates that RelayStack can safely cross from purely synthetic/no-load metadata into tokenizer/config-backed metadata without broadening into runtime execution.

This gives RelayStack a safer bridge toward HF-first V0.1 validation:

```text
synthetic capability metadata
→ estimated tokenizer span metadata
→ no-load engine/model metadata hints
→ readiness report gate
→ tokenizer/config-backed metadata probe
```

The important design point is that the first real HF metadata read is still controlled by an artifact-chain gate, not by an ad-hoc script path.

## Validation recorded for this checkpoint

Recommended validation for this docs checkpoint and adjacent Phase 12-I implementation is:

```bash
python -m compileall relaykv tests scripts
python -m pytest -q tests/test_hf_tokenizer_config_probe.py
python -m pytest -q tests/test_hf_adapter_readiness_report.py
python -m pytest -q tests/test_hf_adapter_capability_smoke.py
python -m pytest -q tests/test_hf_tokenizer_span_probe.py
python -m pytest -q tests/test_hf_engine_metadata_probe.py
```

Manual artifact chain smoke path:

```bash
python scripts/run_hf_adapter_capability_smoke.py \
  --output /tmp/relaystack_adapter_capabilities.json
python scripts/run_hf_tokenizer_span_probe.py \
  --output /tmp/relaystack_tokenizer_span_probe.json
python scripts/run_hf_engine_metadata_probe.py \
  --output /tmp/relaystack_engine_metadata_probe.json
python scripts/run_hf_adapter_readiness_report.py \
  --adapter-capabilities /tmp/relaystack_adapter_capabilities.json \
  --tokenizer-span-probe /tmp/relaystack_tokenizer_span_probe.json \
  --engine-metadata-probe /tmp/relaystack_engine_metadata_probe.json \
  --output /tmp/relaystack_hf_adapter_readiness_report.json
python scripts/run_hf_tokenizer_config_probe.py \
  --readiness-report /tmp/relaystack_hf_adapter_readiness_report.json \
  --output /tmp/relaystack_hf_tokenizer_config_probe.json \
  --local-files-only
python -m json.tool /tmp/relaystack_hf_tokenizer_config_probe.json >/dev/null
```

For no-load safety confirmation, use the probe's skip-load path and verify that mismatch checks still fail when caller-supplied refs are outside readiness scope.

## Recommended next work

The next natural phase is:

```text
Phase 12-J:
  5-artifact chain report / acceptance report
```

Phase 12-J should not add runtime behavior. It should consolidate the five Phase 12 artifacts into one acceptance report that answers:

- whether the artifact chain is complete
- whether every artifact is schema-compatible
- whether the chain stays inside the HF-first V0.1 boundary
- whether tokenizer/config metadata is available or explicitly skipped
- whether the project is ready for the next safe HF metadata consolidation step

Still defer:

```text
model loading
GPU inspection
KV materialization
attention connection
runtime adapter/server path
scheduler changes
KV-pool mutation
RelayKV apply
```