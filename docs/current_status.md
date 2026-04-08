# RelayKV Current Status

## Summary

RelayKV currently has a working end-to-end prototype for:

- tier splitting
- CPU cold offload
- cold blockification
- block metadata construction
- block scoring
- block retrieval
- candidate KV assembly
- working KV assembly
- attention output comparison

## Main executable path

The prototype now has a single main executable path through:

```bash
python scripts/run_relaykv_pipeline.py
```

This script runs the representative RelayKV path and writes:

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

## Current empirical picture

The prototype already supports direct approximation-quality measurement. The current results suggest that:

- candidate coverage is a strong predictor of error
- larger hot windows improve stability
- deeper layers are harder than shallow layers
- longer contexts remain harder, but still follow the same coverage-driven trend
