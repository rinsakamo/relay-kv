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

## What is already verified

### Verified on real model KV tensors

The prototype successfully operates on actual KV tensors extracted from a model cache object rather than on synthetic toy tensors only.

### DynamicCache inspection

The current model backend exposes KV layers through `DynamicCache.layers`, where each layer provides:

- `keys`
- `values`

with tensor shape:

```text
[batch, heads, seq_len, head_dim]
```

### Real split path

The current split path is:

```text
full KV
→ split into hot / cold
→ move cold to CPU
→ blockify cold KV
→ build metadata
→ score and retrieve
→ build candidate KV
→ merge with hot KV
→ compare to full attention output
```

## Current empirical picture

The prototype already supports direct approximation-quality measurement. The current results suggest that:

- candidate coverage is a strong predictor of error
- larger hot windows improve stability
- deeper layers are harder than shallow layers
- longer contexts remain harder, but still follow the same coverage-driven trend

## Figure reference

See `docs/figures/relaykv_coverage_vs_error.png` for the current coverage-vs-error view, and `docs/data/relaykv_coverage_vs_error.csv` for the aggregated plotting data.
