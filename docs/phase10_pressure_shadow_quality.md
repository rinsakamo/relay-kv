# Phase 10 Pressure Shadow Quality Guide

## Scope

This guide fixes the reproducible artifact chain for Phase 10 pressure-triggered RelayKV shadow quality reporting.

The current scope is still:

- report-only
- no model loading in the default path
- no actual shadow attention execution
- no RelayKV apply
- no runtime adapter integration

The goal is to make it straightforward to validate a pressure-triggered quality report with either:

- the synthetic default chain
- an existing RelayKV pipeline summary JSON produced elsewhere

## Synthetic Default Chain

Use the synthetic/no-model path first.

```bash
python scripts/run_relaykv_pressure_shadow_quality_chain.py \
  --output-dir /tmp/relaykv_phase10_chain
```

Expected artifacts:

- `/tmp/relaykv_phase10_chain/relaystack_dry_run.json`
- `/tmp/relaykv_phase10_chain/hf_context_smoke_synthetic.json`
- `/tmp/relaykv_phase10_chain/relaystack_hf_smoke_report.json`
- `/tmp/relaykv_phase10_chain/synthetic_relaykv_pipeline_summary.json`
- `/tmp/relaykv_phase10_chain/relaykv_pressure_shadow_quality_report.json`
- `/tmp/relaykv_phase10_chain/chain_summary.json`

Basic validity checks:

```bash
python -m json.tool /tmp/relaykv_phase10_chain/chain_summary.json >/dev/null
python -m json.tool /tmp/relaykv_phase10_chain/relaykv_pressure_shadow_quality_report.json >/dev/null
jq '.quality_status, .notes' /tmp/relaykv_phase10_chain/relaykv_pressure_shadow_quality_report.json
```

## Existing Pipeline Artifact Path

To validate the Phase 10 report against a real or manually generated RelayKV pipeline summary JSON, pass that artifact explicitly.

```bash
python scripts/run_relaykv_pressure_shadow_quality_chain.py \
  --output-dir /tmp/relaykv_phase10_chain_real \
  --relaykv-pipeline-json results/raw/prototype_checks/relaykv_pipeline_summary.json
```

Basic checks:

```bash
python -m json.tool /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json >/dev/null
jq '.quality_status, .notes' /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json
jq '.artifacts' /tmp/relaykv_phase10_chain_real/chain_summary.json
```

The wrapper does not run `scripts/run_relaykv_pipeline.py` for you in this mode. It only joins the existing artifact into the fixed chain.

If a fresh pipeline artifact is needed, generate it separately with the repository's main comparison path:

```bash
python scripts/run_relaykv_pipeline.py
```

Expected output path:

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

This command is intentionally not part of default validation because it may depend on local model, torch, and GPU setup.

## Quality Status Meanings

- `not_recommended`: pressure signals did not recommend a shadow quality test for this case.
- `recommended_quality_unknown`: pressure signals recommended a quality test, but no usable RelayKV attention-compare metrics were available.
- `recommended_quality_observed`: pressure signals recommended a quality test and some quality artifact exists, but threshold-ready diff metrics are incomplete.
- `recommended_quality_within_threshold`: recommended case with diff metrics present and both thresholds satisfied.
- `recommended_quality_exceeds_threshold`: recommended case with diff metrics present and one or both thresholds exceeded.
- `recommended_quality_context_mismatch`: recommended case, but the pressure-target context and the RelayKV pipeline `seq_len` do not match.
- `recommended_quality_model_mismatch`: recommended case, but the HF/RelayStack model and the RelayKV pipeline model do not match.

## Mismatch Interpretation

### Model Mismatch

`recommended_quality_model_mismatch` means the pressure recommendation and the quality artifact describe different models.

Typical cause:

- HF smoke report was produced from one model
- RelayKV pipeline summary was produced from another model

In this case, do not treat the reported quality metrics as the pressure-triggered quality result for that recommendation.

### Context Mismatch

`recommended_quality_context_mismatch` means the pressure-triggered target context length and the RelayKV pipeline `seq_len` differ.

Typical cause:

- pressure recommendation was triggered around `first_failed_context_tokens`
- pipeline quality summary was measured at a different `seq_len_actual`

In this case, do not treat the reported quality metrics as the pressure-triggered quality result for that recommendation.

## Threshold Calibration Scaffold

The current default thresholds are provisional triage settings:

- `mean_abs_diff_threshold = 0.01`
- `max_abs_diff_threshold = 0.10`

These thresholds should currently be treated as a routing and triage signal, not as a strong quality claim.

Recommended calibration workflow:

1. Collect multiple real RelayKV pipeline summary artifacts for the same model and comparable context conditions.
2. Join each artifact through the fixed Phase 10 chain with `--relaykv-pipeline-json`.
3. Compare `quality_status`, `notes`, `mean_abs_diff`, and `max_abs_diff`.
4. Adjust thresholds only after the observed metric distribution is clear across several artifacts.

Useful inspection commands:

```bash
jq '.quality_status' /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json
jq '.quality_summary.mean_abs_diff, .quality_summary.max_abs_diff' /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json
jq '.notes' /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json
```

## Current Boundaries

Phase 10-D does not add:

- actual shadow attention execution
- runtime integration
- RelayKV apply
- KV pool changes
- scheduler changes
