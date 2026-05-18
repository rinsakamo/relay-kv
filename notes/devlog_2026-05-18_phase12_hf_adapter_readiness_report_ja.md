# Devlog: Phase 12 HF adapter readiness report checkpoint

Date basis: JST 2026-05-18.

## Summary

Phase 12-H completed the HF adapter readiness report checkpoint.

This phase adds a no-model / no-GPU / no-tokenizer-load readiness validator that joins the three earlier HF metadata artifacts and decides whether the HF adapter boundary is ready for the next metadata step.

The new report artifact is:

```text
relaystack_hf_adapter_readiness_report.json
```

It joins these existing artifacts:

```text
relaystack_adapter_capabilities.json
relaystack_tokenizer_span_probe.json
relaystack_engine_metadata_probe.json
```

## Completed PR

PR:

```text
#84 Add HF adapter readiness report
```

Merge commit:

```text
570810ff507003da86226871dbd6ea59adb93932
```

Merged at:

```text
2026-05-18T10:35:31Z
```

## Files added or updated

Primary implementation:

```text
scripts/run_hf_adapter_readiness_report.py
```

Tests:

```text
tests/test_hf_adapter_readiness_report.py
```

Docs/status:

```text
docs/current_status.md
docs/phase12_hf_adapter_capability_schema.md
```

## What the readiness report validates

The readiness report checks the artifact chain before any real tokenizer/config/model-backed work begins.

It validates:

- schema versions and artifact kinds
- adapter kind consistency
- adapter name consistency when present
- runtime target consistency
- model id / model revision / local path consistency
- tokenizer name / tokenizer revision / tokenizer config hash / tokenizer family consistency
- per-span embedded tokenizer_ref consistency
- input summary.ok is present and exactly true
- required probe support flags are present and true
- safety scope remains dry-run / no-load / no-apply
- materialization and apply capabilities remain disabled
- tokenizer spans are estimated, tokenizer-scoped, and not bound to engine block refs
- engine metadata remains no-model-loaded / no-tokenizer-loaded / no-GPU-inspected
- attention metadata is internally coherent for MHA/GQA/unknown cases

## Robustness added during review

The review cycle was useful because the readiness report became a real artifact-chain validator rather than only a happy-path joiner.

The final implementation now handles malformed inputs as structured failures instead of crashing.

It covers:

- invalid or missing JSON inputs
- valid JSON that is not a top-level object
- malformed nested mappings such as model_ref, tokenizer_ref, capabilities, safety_scope, summary, or engine metadata fields
- malformed span entries
- missing or malformed span tokenizer_ref / lineage fields
- failed upstream artifacts with summary.ok false
- missing, false, or non-bool required capability flags
- mismatched tokenizer revision / config hash
- mismatched model revision
- tokenizer probe model_revision being null when adapter and engine revisions match
- mismatched local model paths when both adapter and engine provide them
- mismatched adapter names when both adapter and engine provide them

## Safety boundary

Phase 12-H remains metadata-only.

Still not included:

```text
model loading
tokenizer loading
GPU inspection
runtime adapter execution
actual KV materialization
attention backend connection
scheduler changes
KV-pool mutation
RelayKV apply
OpenAI-compatible server implementation
Open-LLM-VTuber integration
tool execution or approval UX
```

This preserves the Phase 12 boundary: HF-first validation is being built as a sequence of artifact contracts before runtime integration.

## Validation recorded in PR

PR #84 recorded the following validation set:

```bash
python -m compileall relaykv tests scripts
python -m pytest -q tests/test_hf_adapter_readiness_report.py
python -m pytest -q tests/test_hf_adapter_capability_smoke.py
python -m pytest -q tests/test_hf_tokenizer_span_probe.py
python -m pytest -q tests/test_hf_engine_metadata_probe.py
python -m pytest -q tests/test_relaystack_dry_run.py
```

Manual smoke path:

```bash
python scripts/run_hf_adapter_capability_smoke.py --output /tmp/relaystack_adapter_capabilities.json
python scripts/run_hf_tokenizer_span_probe.py --output /tmp/relaystack_tokenizer_span_probe.json
python scripts/run_hf_engine_metadata_probe.py --output /tmp/relaystack_engine_metadata_probe.json
python scripts/run_hf_adapter_readiness_report.py \
  --adapter-capabilities /tmp/relaystack_adapter_capabilities.json \
  --tokenizer-span-probe /tmp/relaystack_tokenizer_span_probe.json \
  --engine-metadata-probe /tmp/relaystack_engine_metadata_probe.json \
  --output /tmp/relaystack_hf_adapter_readiness_report.json
python -m json.tool /tmp/relaystack_hf_adapter_readiness_report.json >/dev/null
```

## Why this checkpoint matters

Before this phase, RelayStack had three separate HF metadata probes:

```text
capability declaration
estimated tokenizer-span metadata
engine/model-shape metadata hints
```

Phase 12-H makes those probes usable as an artifact chain.

The important shift is that the next step no longer has to trust each artifact independently. It can first ask whether the artifacts describe the same adapter/model/tokenizer/runtime scope and whether the chain is safe to advance.

This is the correct boundary before moving from synthetic/no-load metadata to real tokenizer/config-backed metadata.

## Recommended next work

The next implementation step should be the first real tokenizer/config-backed probe, guarded by the readiness report.

Candidate next phase:

```text
Phase 12-I or Phase 12-J:
  HF tokenizer/config-backed metadata probe
```

Recommended constraints:

```text
Allowed:
  - tokenizer/config metadata probe only
  - local-files-only option
  - graceful unavailable/dependency failure artifact
  - comparison against prior estimated tokenizer span and engine metadata artifacts

Still disallowed:
  - model loading
  - GPU inspection
  - KV materialization
  - attention connection
  - scheduler changes
  - KV-pool mutation
  - RelayKV apply
```

The next script should likely produce a new artifact such as:

```text
relaystack_hf_tokenizer_config_probe.json
```

The readiness report should remain the gate before running or accepting that artifact chain.
