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

- **RelayKV** is the VRAM-aware active-context KV routing layer.
- **RelayMEM** is the max-context-external memory layer that decides what should enter active context.
- **Open-LLM-VTuber** is the current practical product/demo target for low-VRAM Japanese AI character runtime integration.

RelayKV extends usable active context under fixed VRAM budgets, but it does not by itself extend the model's trained or supported maximum context window.

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

## Design-only or not yet integrated

The following items are part of the current direction, but should not be described as implemented runtime features yet:

- RelayMEM integration
- Open-LLM-VTuber integration
- User-Gated Fallback runtime UX
- real KV materialization in an inference engine
- real routed execution in a production backend
- full end-to-end runtime policy integration across GPU, RAM, and SSD tiers

## Current empirical picture

The prototype already supports direct approximation-quality measurement. The current results suggest that:

- candidate coverage is a strong predictor of error
- larger hot windows improve stability
- deeper layers are harder than shallow layers
- longer contexts remain harder, but still follow the same coverage-driven trend

See [experimental_findings.md](experimental_findings.md) for the current measured findings and [evaluation_targets.md](evaluation_targets.md) for the next target directions.
