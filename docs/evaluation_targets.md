# Evaluation Targets

## Purpose

RelayKV is moving toward a broader RelayStack direction, but the evaluation plan still needs to stay concrete and separable.

The current targets are organized around three research pillars plus one practical demo target:

1. code / structured retrieval
2. ultra-long context behavior
3. Japanese long-form / character consistency
4. low-VRAM AI character runtime demo

## 1. Code / structured retrieval

- **Model candidate**: `Qwen2.5-Coder-7B-Instruct-AWQ`
- **Purpose**: evaluate whether active-context KV routing preserves exact or near-exact retrieval behavior for code and structured prompts where the relevant span is sparse but important
- **Primary metrics**: code exact match, structured retrieval accuracy, pass/fail task completion, approximation error under fixed working-set budget, latency and VRAM budget stability
- **Why it matters for RelayKV**: code and structured tasks are a strong stress test for selective recall because small misses can break correctness even when overall language quality still appears acceptable

## 2. Ultra-long context

- **Model candidate**: `Qwen2.5-7B-Instruct-1M`
- **Purpose**: evaluate RelayKV behavior when the model itself supports very large context and the bottleneck becomes practical working-set control under finite VRAM
- **Primary metrics**: retrieval quality at long distance, approximation error vs coverage, usable working-set ratio, latency growth, and stability across longer active contexts
- **Why it matters for RelayKV**: this target isolates the value of VRAM-aware active-context routing even when the underlying model already supports a very large context window

## 3. Japanese long-form / character consistency

- **Model candidate**: `LLM-jp-4-8B`
- **Purpose**: evaluate whether RelayKV preserves long-form Japanese coherence, recurring details, and character consistency under constrained local hardware
- **Primary metrics**: long-form consistency checks, retained detail accuracy, character/profile consistency, summary-to-generation continuity, and approximation error under working-set constraints
- **Why it matters for RelayKV**: RelayKV is intended to be useful for persistent low-VRAM local applications, and Japanese long-form character behavior is an important target workload rather than only a synthetic benchmark

## 4. Low-VRAM AI character demo

- **Target stack**: `Open-LLM-VTuber + LLM-jp-4 8B 4bit + RelayMEM + RelayKV`
- **Purpose**: demonstrate a cautious end-to-end target where long-running AI character interaction benefits from both max-context-external memory assembly and VRAM-aware active-context KV control
- **Primary metrics**: runtime stability, latency consistency, fallback frequency, memory recall usefulness, subjective character continuity, and sustained operation within the hardware budget
- **Why it matters for RelayKV**: this is the practical integration target where RelayKV is not evaluated alone, but as part of a full local stack with memory assembly, runtime policy, and low-VRAM constraints

## Interpretation

These targets deliberately separate two questions:

- whether RelayKV preserves useful behavior inside active context under fixed VRAM
- whether RelayMEM plus runtime policy can decide *what* belongs in that active context in the first place

That separation is important because RelayKV does not extend maximum context by itself. Max-context-external behavior must come from RelayMEM and context assembly policy.
