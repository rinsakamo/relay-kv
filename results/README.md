# Results

This directory contains experiment outputs for RelayKV.

## Structure

- `raw/prototype_checks/` — direct outputs from prototype verification scripts
- `raw/sweeps/` — direct outputs from sweep experiments
- `processed/` — cleaned tables and analysis-ready CSV files
- `figures/` — generated plots and visual summaries

## Main files

### Prototype pipeline artifact

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

This is the main end-to-end RelayKV prototype output produced by:

```bash
python scripts/run_relaykv_pipeline.py
```

It contains:

- sequence configuration
- hot/cold ranges
- candidate and working KV lengths
- coverage and working ratios
- selected top-scoring blocks
- attention comparison against full KV

### Sweep artifacts

```text
results/raw/sweeps/attention_sweep.csv
results/raw/sweeps/attention_sweep_large.csv
```

These contain parameter sweeps over sequence length, hot window, block size, top-k, and layer index.

### Processed and visual outputs

```text
results/processed/relaykv_coverage_vs_error.csv
results/figures/relaykv_coverage_vs_error.png
```

These files summarize the current coverage-versus-error trend used in the documentation.

## Notes

Files in `raw/` are usually direct script outputs and may be overwritten.
Files in `processed/` and `figures/` are intended for documentation and reporting.
