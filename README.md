# RelayKV

![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-success)

**RelayKV** is a research prototype for a VRAM-aware active-context KV routing layer and tiered KV memory manager for local LLM inference.

RelayKV should be understood as a **fixed-VRAM-budget decode-time KV working-set controller**, not as a simple KV-cache reduction algorithm. Its goal is to keep the active decode-time KV working set inside the residual VRAM budget left after model weights and other local runtime components are accounted for.

It is intended to extend the amount of *usable active context under fixed VRAM* by routing and managing KV blocks across GPU, RAM, and colder tiers. It does **not** extend a model's trained or supported maximum context length by itself.

## Current Status

RelayKV remains a **working research prototype**.

The current prototype supports this end-to-end path:

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

The main executable path is:

```bash
python scripts/run_relaykv_pipeline.py
```

Detailed status is tracked in [docs/current_status.md](docs/current_status.md).

## RelayStack Direction

The current project direction is **RelayStack**:

- **RelayKV**: active-context KV block routing, GPU/RAM/SSD tiered KV management, and VRAM-aware working-set control inside the model's supported context window
- **RelayMEM**: max-context-external memory layer for retrieval, summaries, profiles, episodes, structured memory, context assembly, and KV checkpoint metadata
- **Runtime policy**: request-time policy that decides when to stay in low-latency mode, when to retrieve older memory, and when to require explicit user approval for fallback behavior

RelayStack is the broader system direction. The repository today is still centered on the RelayKV prototype path and supporting policy/design work.

RelayStack is **not** a RAG replacement. It is a memory/context/KV-budget orchestration layer that can combine RAG-like retrieval, hierarchical memory, active-context assembly, fixed-budget KV routing, VRAM reservation, and fallback policy.

The intended responsibility split is:

```text
RAG or retrieval backend
  retrieves external evidence or memory candidates

RelayMEM
  selects and assembles memories into active context

RelayKV
  controls the decode-time KV working set under a fixed VRAM budget

RelayStack
  coordinates RelayMEM, RelayKV, VRAM reservation, and fallback policy
```

## Evaluation Targets

Current research and demo targets are:

1. **Code / structured retrieval**
   Model candidate: `Qwen2.5-Coder-7B-Instruct-AWQ`
2. **Ultra-long context**
   Model candidate: `Qwen2.5-7B-Instruct-1M`
3. **Japanese long-form / character consistency**
   Model candidate: `LLM-jp-4-8B`
4. **Low-VRAM AI character demo**
   Target stack: `Open-LLM-VTuber + LLM-jp-4 8B 4bit + RelayMEM + RelayKV`

See [docs/evaluation_targets.md](docs/evaluation_targets.md) for the full rationale and metrics.

## Product / Demo Target

The current product-facing demo target is **Open-LLM-VTuber** for a low-VRAM Japanese AI character / AI Vtuber setup.

The intended comparison is cautious and practical:

- ordinary context-limited local runtime
- RelayMEM + RelayKV assisted runtime under the same hardware constraints

This is a target direction, not a claim of product readiness. See [docs/open_llm_vtuber_target.md](docs/open_llm_vtuber_target.md).

## Docs

- [docs/README.md](docs/README.md): docs index
- [docs/current_status.md](docs/current_status.md): implemented prototype path vs design-only direction
- [docs/evaluation_targets.md](docs/evaluation_targets.md): research evaluation pillars and demo target
- [docs/relaymem.md](docs/relaymem.md): RelayMEM design boundary
- [docs/max_context_external_memory.md](docs/max_context_external_memory.md): active-context vs max-context-external memory split
- [docs/user_gated_fallback.md](docs/user_gated_fallback.md): runtime fallback approval policy
- [docs/open_llm_vtuber_target.md](docs/open_llm_vtuber_target.md): Open-LLM-VTuber integration target

## Main Entry Points

### Baseline measurement

```bash
python scripts/run_baseline.py
```

### RelayKV prototype pipeline

```bash
python scripts/run_relaykv_pipeline.py \
  --seq-len 1024 \
  --hot-window 128 \
  --block-size 128 \
  --top-k 2 \
  --layer-idx 27
```

Default output:

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

### Sweep experiments

```bash
python scripts/run_attention_sweep.py
```

For the current empirical findings, see [docs/experimental_findings.md](docs/experimental_findings.md).
