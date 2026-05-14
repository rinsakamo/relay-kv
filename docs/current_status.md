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
- **RelayStack** coordinates RelayMEM, RelayKV, VRAM reservation, runtime policy, and User-Gated Fallback.
- **Open-LLM-VTuber** is the current practical product/demo target for low-VRAM Japanese AI character runtime integration.

RelayKV extends usable active context under fixed VRAM budgets, but it does not by itself extend the model's trained or supported maximum context window.

RelayKV should now be treated as a **fixed-VRAM-budget decode-time KV working-set controller**. The working-set budget is conceptually split across role-specific KV classes:

```text
B_total_working_kv
  = B_recent
  + B_anchor
  + B_retrieved
  + B_transient
```

The design keeps three concepts separate:

```text
kv_class         = role in the working set
precision_level  = representation or compression level
residency_level  = location such as GPU HBM, CPU RAM, SSD, or remote tier
```

This separation keeps future compressed or checkpointed KV representations from being confused with RelayKV's selection role.

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
- RelayMEM schema objects for retrieval results and context assembly plans
- RelayMEM record schemas for profile, episode, summary, structured, and KV checkpoint metadata records
- User-Gated Fallback schema fields
- VRAM reservation schema and smoke path
- RelayStack no-model/no-GPU dry-run JSON combining RelayMEM, RelayKV, VRAM reservation, runtime policy, and fallback fields

## Design-only or not yet integrated

The following items are part of the current direction, but should not be described as implemented runtime features yet:

- concrete RelayMEM retrieval backend
- Open-LLM-VTuber integration
- User-Gated Fallback runtime UX
- real KV materialization in an inference engine
- real routed execution in a production backend
- full end-to-end runtime policy integration across GPU, RAM, and SSD tiers
- compressed KV implementation
- disk-backed or RAM-backed full KV checkpoint execution
- SGLang or vLLM runtime adapter changes beyond prior exploratory work

## Revised phase direction

The next work should keep model/GPU/runtime risk low until RelayMEM and RelayStack planning are useful as standalone artifacts.

```text
Phase 6:
  RelayMEM Fast Recall backend

Phase 7:
  RelayMEM prompt preview / CLI memory assistant smoke

Phase 8:
  RelayStack runtime planning artifact practical configuration

Phase 9:
  HF max-context / FullKV baseline quality smoke

Phase 10:
  RelayKV pressure-triggered shadow policy quality test

Phase 11:
  RelayKV fixed-budget working-set dry-run policy

Phase 12:
  Runtime adapter selection and restart
  Choose SGLang, vLLM, HF, or another adapter target for the next integration pass.

Phase 13:
  Safe materialization / shadow attention compare

Phase 14:
  Gated apply / fallback integration
```

## Current empirical picture

The prototype already supports direct approximation-quality measurement. The current results suggest that:

- candidate coverage is a strong predictor of error
- larger hot windows improve stability
- deeper layers are harder than shallow layers
- longer contexts remain harder, but still follow the same coverage-driven trend

See [experimental_findings.md](experimental_findings.md) for the current measured findings and [evaluation_targets.md](evaluation_targets.md) for the next target directions.
