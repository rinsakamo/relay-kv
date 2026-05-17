# Devlog: Phase 12 HF metadata probes checkpoint

Date basis: JST 2026-05-17.

## Summary

Phase 12-E and Phase 12-F extended the HF-first adapter boundary work from a capability declaration into concrete metadata probe artifacts.

The important point is that the work still does not cross into runtime adapter execution. It remains no-model / no-tokenizer-load / no-GPU metadata smoke validation.

Completed scope:

- Phase 12-E: HF tokenizer span probe artifact
- Phase 12-F: HF engine/model metadata probe artifact

The resulting Phase 12 metadata artifact set is now:

```text
relaystack_adapter_capabilities.json
relaystack_tokenizer_span_probe.json
relaystack_engine_metadata_probe.json
```

## Phase 12-E: HF tokenizer span probe artifact

PR: #81, `Add HF tokenizer span probe artifact`

Added:

```text
scripts/run_hf_tokenizer_span_probe.py
tests/test_hf_tokenizer_span_probe.py
```

Output artifact:

```text
relaystack_tokenizer_span_probe.json
```

Purpose:

- Validate the RelayCTX token-span artifact shape at the HF adapter boundary.
- Preserve tokenizer-scoped span metadata before any real KV, attention, or engine integration.
- Keep RelayMEM -> RelayCTX -> RelayKV lineage placeholders explicit.

Important design choice:

```text
Phase 12-E does not load a tokenizer.
```

Token spans are intentionally marked as estimated and tokenizer-scoped. This keeps the artifact honest and prevents it from being mistaken for a real tokenization result.

The artifact records:

- schema version
- phase
- adapter kind
- runtime target
- model reference
- tokenizer reference
- input source item
- estimated token span
- tokenizer-scoped flag
- lineage placeholders
- safety scope
- compact summary

Codex review found one valid issue:

```text
Reject non-positive --estimated-token-count overrides.
```

Resolution:

- `--estimated-token-count <= 0` is rejected before writing the artifact.
- Generated spans keep `token_end > token_start`.
- Regression coverage was added for invalid overrides.

## Phase 12-F: HF engine/model metadata probe artifact

PR: #82, `Add HF engine metadata probe artifact`

Added:

```text
scripts/run_hf_engine_metadata_probe.py
tests/test_hf_engine_metadata_probe.py
```

Output artifact:

```text
relaystack_engine_metadata_probe.json
```

Purpose:

- Preserve model/tokenizer/context-window metadata at the HF adapter boundary.
- Preserve a capability snapshot aligned with Phase 12-B/C.
- Record engine-shape hints without loading model config, tokenizer, or GPU state.

Important design choice:

```text
Phase 12-F does not load model config, model weights, tokenizer, torch, or transformers.
```

The artifact records:

- schema version
- phase
- adapter kind/name
- runtime target
- model reference
- tokenizer reference
- context window hint
- engine metadata hints
- attention type hint
- KV head grouping metadata
- capability snapshot
- safety scope
- compact summary

Codex review found one valid issue:

```text
Compute KV group count from the attention-head to KV-head ratio.
```

Resolution:

- MHA reports `kv_head_group_count = 1`.
- Valid GQA reports `kv_head_group_count = num_attention_heads // num_key_value_heads`.
- Non-divisible GQA-style inputs are rejected before writing the artifact.
- `num_key_value_heads > num_attention_heads` is rejected.
- Regression coverage was added for GQA, MHA, and invalid head-count cases.

## Validation recorded by PRs

Phase 12-E recorded:

```bash
python -m compileall relaykv tests scripts
python -m pytest -q tests/test_hf_tokenizer_span_probe.py
python scripts/run_hf_tokenizer_span_probe.py --output /tmp/relaystack_tokenizer_span_probe.json
python -m json.tool /tmp/relaystack_tokenizer_span_probe.json >/dev/null
python -m pytest -q tests/test_hf_adapter_capability_smoke.py
python -m pytest -q tests/test_relaystack_dry_run.py
```

Phase 12-F recorded:

```bash
python -m compileall relaykv tests scripts
python -m pytest -q tests/test_hf_engine_metadata_probe.py
python scripts/run_hf_engine_metadata_probe.py --output /tmp/relaystack_engine_metadata_probe.json
python -m json.tool /tmp/relaystack_engine_metadata_probe.json >/dev/null
python -m pytest -q tests/test_hf_adapter_capability_smoke.py
python -m pytest -q tests/test_hf_tokenizer_span_probe.py
python -m pytest -q tests/test_relaystack_dry_run.py
```

## Current safety boundary

Still not changed:

- no model loading
- no tokenizer loading
- no GPU inspection
- no torch import requirement
- no transformers import requirement
- no runtime adapter execution path
- no actual KV materialization
- no production attention backend connection
- no scheduler change
- no KV-pool mutation
- no RelayKV apply
- no OpenAI-compatible server implementation
- no Open-LLM-VTuber integration
- no tool execution or approval UX

## Why this checkpoint matters

Phase 12-A through Phase 12-F now provide a clean metadata boundary for the HF prototype adapter:

```text
capabilities
  -> what the adapter claims it can safely provide

tokenizer span probe
  -> how RelayCTX span/lineage metadata should look before real tokenization

engine metadata probe
  -> how model/runtime shape hints should be represented before real config/model loading
```

This gives RelayStack a safer path toward later runtime work:

```text
metadata artifacts first
→ readiness report
→ optional real tokenizer/config probe
→ shadow-safe materialization planning
→ shadow compare
→ gated apply only much later
```

## Recommended next work

The next step should combine the three Phase 12 metadata artifacts into a readiness report before moving toward real tokenizer/config loading.

Recommended next PR:

```text
Phase 12-H:
  HF adapter readiness report

Inputs:
  relaystack_adapter_capabilities.json
  relaystack_tokenizer_span_probe.json
  relaystack_engine_metadata_probe.json

Output:
  relaystack_hf_adapter_readiness_report.json
```

Readiness report should check:

- schema versions
- adapter kind consistency
- runtime target consistency
- model/tokenizer reference consistency
- safety scope remains dry-run/no-load/no-apply
- tokenizer span artifact is estimated, not real-tokenized
- engine metadata says model/tokenizer/GPU were not loaded
- capability snapshot does not claim materialization/apply support
- attention metadata is internally valid when head counts are supplied

Do not proceed directly to KV materialization or attention work yet.
