# RelayKV

![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-success)

**RelayKV** is a recall-aware tiered KV cache engine for long-context local LLM inference.

It re-lays cold KV cache across GPU and CPU memory tiers, recalls relevant candidates on demand, and helps local models run with longer context on limited hardware.

## Current Status

RelayKV is now a **working research prototype**.

The current prototype can:

- split KV cache into **hot** and **cold** ranges
- offload cold KV tensors to **CPU**
- split cold KV into **blocks**
- build lightweight **block metadata**
- score blocks against a query
- retrieve selected cold blocks
- build a **candidate KV**
- merge candidate KV with hot KV into a **working KV**
- compare the resulting attention output against full KV attention

This means RelayKV already supports the following end-to-end prototype path:

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

## Main Entry Points

The current repository has three main entry points.

### 1. Baseline measurement

```bash
python scripts/run_baseline.py
```

Use this to collect baseline memory and latency measurements without RelayKV candidate reconstruction.

### 2. RelayKV prototype pipeline

```bash
python scripts/run_relaykv_pipeline.py   --seq-len 1024   --hot-window 128   --block-size 128   --top-k 2   --layer-idx 27
```

Use this to run the full RelayKV prototype path in one execution:

```text
forward
→ hot/cold split
→ cold offload
→ blockify
→ metadata
→ scoring
→ retrieval
→ candidate KV
→ working KV
→ attention comparison
```

The default output is:

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

### 3. Sweep experiment

```bash
python scripts/run_attention_sweep.py
```

Use this to run a parameter sweep over sequence length, hot window, block size, top-k, and layer index.

## Motivation

Long-context inference is often constrained not only by compute, but by KV cache growth.

As context length increases, KV cache becomes a dominant memory cost. This is especially problematic for local inference on commodity GPUs, where limited VRAM makes long-context workloads difficult to sustain.

RelayKV is built around a simple idea:

> not all KV entries need to remain on GPU at all times, but useful entries must remain reachable when needed.

RelayKV treats KV cache as a managed memory system rather than a fixed GPU-only artifact. It keeps hot KV close to the GPU, offloads cold KV to CPU memory, and restores only the most relevant candidates for the current query.

## What RelayKV Is

RelayKV is a **KV engine**, not a full inference runtime.

Its role is to manage how KV cache is:

- placed across memory tiers
- grouped into retrievable units
- recalled for attention
- optionally compressed in colder tiers

RelayKV is intended to sit beneath or alongside an existing inference backend.

## Prototype Architecture

```text
Application / Agent / API Server
        ↓
   Inference Runtime
        ↓
      RelayKV
   ├─ Tier split (hot / cold)
   ├─ CPU cold cache
   ├─ Cold block layout
   ├─ Block metadata
   ├─ Block scoring
   ├─ Block retrieval
   ├─ Candidate KV assembly
   └─ Working KV assembly
        ↓
   Attention Comparison / Future Re-Attention
```

## Example Pipeline Output

A representative pipeline run with:

- `seq_len = 1024`
- `hot_window = 128`
- `block_size = 128`
- `top_k = 2`
- `layer_idx = 27`

produced:

- `cold_k_len = 896`
- `candidate_k_len = 256`
- `coverage_ratio = 0.2857`
- `working_k_len = 384`
- `working_ratio = 0.375`
- `mean_abs_diff = 0.0182`
- `max_abs_diff = 0.0931`

The corresponding JSON artifact is stored at:

```text
results/raw/prototype_checks/relaykv_pipeline_summary.json
```

## Sweep Findings

A larger sweep on `seq_len=1024`, `2048`, and `4096` at `layer_idx=27` showed a stable trend:

- approximation error decreases as **candidate coverage** increases
- larger hot windows improve stability
- different `(block_size, top_k)` pairs often produce similar error when they yield similar effective coverage
- the same qualitative trend persists even at `seq_len=4096`

### Coverage vs. Error (3 sequence lengths)

![RelayKV coverage vs error](docs/figures/relaykv_coverage_vs_error_3seq.png)

**Figure 1.** Mean absolute attention-output difference as a function of candidate coverage ratio for `layer_idx=27`, shown for `seq_len=1024`, `2048`, and `4096`. In all three cases, approximation error decreases as coverage increases.

### Working Ratio vs. Error (3 sequence lengths)

![RelayKV working ratio vs error](docs/figures/relaykv_working_ratio_vs_error_3seq.png)

**Figure 2.** Mean absolute attention-output difference as a function of working ratio for `layer_idx=27`, shown for `seq_len=1024`, `2048`, and `4096`. This complementary view highlights the role of both selected cold candidates and preserved hot KV in the reconstructed working set.

### Prompt-Type Comparison at 4096 Tokens

To check whether the same qualitative behavior persists across different input styles, RelayKV was also evaluated at `seq_len=4096`, `layer_idx=27` with three prompt types:

- `repetitive`
- `prose`
- `structured`

![RelayKV prompt-type comparison](docs/figures/relaykv_prompt_types_4096.png)

**Figure 3.** Mean absolute attention-output difference as a function of candidate coverage ratio for `seq_len=4096` and `layer_idx=27`, shown for three prompt styles. Although the absolute error level varies by prompt type, the same qualitative trend remains: approximation error decreases as effective candidate coverage increases.

### Prompt-Type Interpretation

The current prompt-type comparison suggests that:

- the coverage-driven trend is preserved across multiple prompt styles
- absolute error varies somewhat by prompt type
- the structured prompt currently appears easiest under this prototype setup
- RelayKV behavior is therefore not limited to a single repetitive prompt pattern

For more detailed experimental observations, see `docs/experimental_findings.md`.

### Current Interpretation

The current prototype evidence supports the following empirical view:

- higher **coverage_ratio** generally reduces approximation error
- for matched coverage, different block sizes may behave similarly
- longer sequence lengths remain harder, but follow the same trend
- larger hot windows improve stability by preserving more recent KV directly

For compact tables, see the corresponding files in `docs/` or the project notes.

## Repository Structure

```text
relay-kv/
├─ README.md
├─ LICENSE
├─ docs/
│  ├─ README.md
│  ├─ current_status.md
│  ├─ experimental_findings_2026-04-07.md
│  ├─ figures/
│  │  └─ relaykv_coverage_vs_error.png
│  └─ data/
│     └─ relaykv_coverage_vs_error.csv
├─ results/
│  ├─ README.md
│  ├─ raw/
│  │  ├─ prototype_checks/
│  │  │  └─ relaykv_pipeline_summary.json
│  │  └─ sweeps/
│  ├─ processed/
│  └─ figures/
├─ relaykv/
├─ scripts/
└─ tests/
```

## Design Goals

RelayKV is built with the following goals:

- **practicality**: useful on real local hardware
- **modularity**: works with existing inference backends
- **efficiency**: reduce unnecessary KV residency and transfer
- **scalability**: support longer context windows through tiered storage
- **quality retention**: preserve attention quality through useful candidate recall

## Planned Next Steps

Near-term:

- expand sweeps over longer sequence lengths
- compare more layers systematically
- summarize results in compact tables and plots
- test additional scoring variants

Mid-term:

- build a cleaner single-pipeline script
- test more practical prompt sets
- scale to Qwen2.5-3B and Qwen2.5-7B

Long-term:

- quantized cold tier
- async prefetch
- backend integration
- vLLM-aware design path

## Documentation Note

Core project documents are written in English.  
Informal development notes may be written in Japanese.

## License

MIT
