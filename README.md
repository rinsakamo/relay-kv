# RelayKV

![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-success)
![Focus](https://img.shields.io/badge/focus-tiered%20KV%20cache-informational)
![Target](https://img.shields.io/badge/target-long--context%20local%20inference-purple)

**RelayKV** is a recall-aware tiered KV cache engine for long-context local LLM inference.

It re-lays cold KV cache across GPU and CPU memory tiers, recalls relevant candidates on demand, and helps local models run with longer context on limited hardware.

## Overview

Long-context inference is often constrained not only by compute, but by KV cache growth.

As context length increases, KV cache becomes a dominant memory cost. This is especially problematic for local inference on commodity GPUs, where limited VRAM makes long-context workloads difficult to sustain.

RelayKV is built around a simple idea:

> not all KV entries need to remain on GPU at all times, but useful entries must remain reachable when needed.

RelayKV treats KV cache as a managed memory system rather than a fixed GPU-only artifact. It keeps hot KV close to the GPU, offloads cold KV to CPU memory, and restores only the most relevant candidates for the current query.

## Motivation

Existing local inference backends can run quantized models efficiently, but KV cache itself remains a major bottleneck for long-context workloads.

RelayKV is designed to address this by:

- reducing GPU KV residency
- organizing cold KV for efficient retrieval
- restoring only useful candidates on demand
- extending practical context length on limited hardware
- supporting local-first inference workflows where privacy and cost matter

## What RelayKV Is

RelayKV is a **KV engine**, not a full inference runtime.

Its role is to manage how KV cache is:

- placed across memory tiers
- grouped into retrievable units
- recalled for attention
- optionally compressed in colder tiers

RelayKV is intended to sit beneath or alongside an existing inference backend.

## What RelayKV Does

RelayKV is designed to:

- maintain a **hot tier** on GPU for recent or important KV entries
- maintain a **cold tier** on CPU for older or less active KV entries
- organize cold KV into blocks for efficient selection and transfer
- score cold blocks using lightweight metadata
- recall only relevant blocks for the current query
- reduce memory pressure while preserving useful candidate access
- enable longer-context local inference on commodity hardware

## Key Ideas

### 1. Tiered KV Cache

RelayKV splits KV cache across multiple memory tiers.

- **GPU tier** stores hot KV that should remain immediately accessible
- **CPU tier** stores cold KV that would otherwise exceed VRAM limits

This allows inference to continue beyond the point where a GPU-only KV cache becomes impractical.

### 2. Recall-Aware Retrieval

Instead of restoring all offloaded KV entries, RelayKV recalls only the blocks that appear relevant to the current query.

This turns cold KV from passive storage into an active memory tier that can be searched and reused.

### 3. Block-Wise Candidate Selection

Cold KV is grouped into fixed-size blocks.

RelayKV scores these blocks using lightweight metadata, selects a small subset, and restores only those candidates for higher-precision attention.

This reduces transfer overhead and keeps retrieval practical.

### 4. Practical Long-Context Inference

RelayKV is designed for local environments with constrained VRAM.

The goal is not just to increase theoretical context length, but to make long-context inference more usable in real workflows.

## Architecture

RelayKV can be used beneath or alongside an existing inference backend.

```text
Application / Agent / API Server
        ↓
   Inference Runtime
        ↓
      RelayKV
   ├─ GPU hot tier
   ├─ CPU cold tier
   ├─ block metadata
   └─ candidate recall path
        ↓
   Attention Execution
```

RelayKV does not aim to replace the entire runtime stack.

Instead, it focuses on KV placement, movement, selective restoration, and memory-aware attention support.

## Suggested Repository Structure

```text
relay-kv/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ default.yaml
│  ├─ profiling.yaml
│  └─ experiments.yaml
├─ relaykv/
│  ├─ __init__.py
│  ├─ engine/
│  │  ├─ tier_manager.py
│  │  ├─ hot_cache.py
│  │  ├─ cold_cache.py
│  │  ├─ block_index.py
│  │  └─ recall_scheduler.py
│  ├─ attention/
│  │  ├─ scorer.py
│  │  ├─ candidate_selector.py
│  │  └─ block_attention.py
│  ├─ quant/
│  │  ├─ int8.py
│  │  ├─ int4.py
│  │  └─ adapters.py
│  ├─ backends/
│  │  ├─ base.py
│  │  ├─ llama_cpp.py
│  │  └─ transformers.py
│  ├─ profiling/
│  │  ├─ memory.py
│  │  ├─ latency.py
│  │  └─ transfer.py
│  └─ utils/
│     ├─ config.py
│     └─ logging.py
├─ scripts/
│  ├─ run_baseline.py
│  ├─ run_tiered_kv.py
│  ├─ run_block_recall.py
│  └─ benchmark_long_context.py
├─ experiments/
│  ├─ prompts/
│  ├─ notebooks/
│  └─ results/
└─ tests/
   ├─ test_tiering.py
   ├─ test_block_selection.py
   └─ test_recall_path.py
```

## Design Goals

RelayKV is built with the following goals:

- **practicality**: useful on real local hardware
- **modularity**: works with existing inference backends
- **efficiency**: reduce unnecessary KV residency and transfer
- **scalability**: support longer context windows through tiered storage
- **quality retention**: preserve attention quality through useful candidate recall

## Non-Goals

RelayKV is not intended to be:

- a complete LLM serving framework
- a model training system
- a replacement for all runtime optimizations
- a guarantee of exact full-attention equivalence in every setting

Its focus is narrower: efficient KV management for long-context local inference.

## Why “RelayKV”?

RelayKV reflects the idea that KV cache should not simply be discarded when it leaves fast memory.

Instead, cold KV can be re-laid across memory tiers, relayed when needed, and brought back into computation selectively.

In that sense, RelayKV treats KV cache as something to preserve, route, and reactivate rather than merely evict.

## Planned Features

- GPU/CPU tiered KV cache
- block-wise cold KV layout
- lightweight block scoring
- on-demand candidate recall
- configurable hot-window retention
- optional cold-tier quantization
- profiling for memory, transfer, and latency
- integration path for local inference backends

## Early Target Use Cases

RelayKV is especially aimed at:

- long chat histories on local models
- note-centric workflows such as Obsidian-assisted reasoning
- agent systems that maintain extended task context
- local-first inference where privacy and cost matter
- commodity hardware environments with constrained VRAM

## Project Status

RelayKV is currently a research prototype and system design effort focused on tiered KV caching for long-context local inference.

The initial implementation targets:

- simple GPU/CPU KV tiering
- block-wise cold KV organization
- selective candidate recall
- evaluation on consumer hardware

## Roadmap

### Phase 1
- baseline long-context inference measurements
- GPU-only KV cache evaluation
- CPU offload prototype

### Phase 2
- block-based cold KV layout
- candidate block scoring
- selective recall path

### Phase 3
- quantized cold-tier storage
- recall / latency / memory evaluation
- integration with local inference workflows

### Phase 4
- backend integration
- asynchronous prefetching
- multi-request and serving-oriented extensions

## Research Direction

RelayKV explores the hypothesis that, for practical long-context local inference, quality is governed more by useful candidate recall than by keeping the entire KV cache resident on GPU at all times.

This motivates a system in which:

- hot KV remains immediately accessible
- cold KV remains compressible and relocatable
- relevant context remains recoverable on demand

## Intended Integrations

RelayKV is designed to complement, not replace, existing local inference systems.

Potential integration targets include:

- local inference backends
- OpenAI-compatible local serving layers
- agent systems that rely on long conversation state
- note-centric local knowledge workflows

## Repository Scope

This repository focuses on:

- KV tiering experiments
- cold-tier layout and retrieval
- recall-aware attention support
- profiling and evaluation tools
- prototype integration paths

It does **not** currently aim to provide:

- a production-ready chat server
- a fully general model runtime
- a complete frontend or UI layer

## Contributing

The project is still early, but discussion and experimental contributions are welcome.

Areas of interest include:

- KV block scoring strategies
- hot/cold tier policies
- candidate recall evaluation
- cold-tier quantization
- backend integration
- profiling and benchmark tooling

## License

MIT
