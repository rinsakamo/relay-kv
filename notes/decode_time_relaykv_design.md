# Decode-Time RelayKV Design Notes

## Goal

Extend RelayKV from post-hoc attention approximation analysis to an actual decode-time path.

The immediate goal is not full production integration, but a prototype path that uses RelayKV working sets during incremental generation and makes it possible to benchmark:

- output quality
- latency
- throughput
- KV working-set size

---

## Current status

RelayKV currently supports the following prototype flow:

- KV split
- CPU cold offload
- cold blockification
- block metadata construction
- block scoring
- block retrieval
- candidate KV assembly
- working KV assembly
- attention output comparison

This confirms that the retrieval-and-reconstruction path works on real KV tensors.

However, the current benchmark path still uses:

- full generation for output speed measurement
- post-hoc RelayKV attention comparison for approximation quality

So RelayKV is not yet being used as the actual KV path during decode.

---

## Immediate implementation target

Build a prototype decode-time RelayKV path that does the following:

1. Run normal prefill on the prompt
2. Split prefill KV into hot and cold regions
3. Build cold blocks and metadata
4. During each decode step:
   - obtain the current query
   - score cold blocks
   - retrieve a small candidate set
   - build a working KV
   - use that working KV for the attention computation
5. Continue generation using this approximate path

The first version can prioritize correctness and observability over speed.

---

## Scope for the first decode-time prototype

### Keep fixed for the first version
- model: current Qwen test model
- seq_len target: 4096
- block_size: 256
- hot_window: 256
- scoring_variant: mean_plus_norm
- layer-wise allocation: current best tested deep-biased setting

### Keep simple
- start with greedy decoding
- start with a small `max_new_tokens`
- prefer CPU-side reconstruction if needed
- accept extra overhead in the first implementation

### Defer for later
- flash-attention-compatible fast path
- multi-request batching
- optimized memory movement
- quantized cold storage
- production-grade caching policy

---

## Core design questions

## 1. Where to intercept generation

The most important design choice is where RelayKV enters the generation path.

### Candidate approach
Use standard model prefill first, then intercept after `past_key_values` is produced.

That gives:

- prompt tokens processed normally
- initial KV cache available
- a clean split point before incremental decoding begins

This is the simplest first integration point.

### Reason
RelayKV already works on real KV tensors extracted from the cache.
So the first decode-time path should reuse exactly that representation.

---

## 2. What happens at each decode step

At each new token step:

1. Take the current last-token query for each target layer
2. Score cold blocks against that query
3. Retrieve blocks according to the layer-wise budget
4. Build candidate KV
5. Concatenate candidate KV with the hot KV to form working KV
6. Run approximate attention using working KV instead of full KV

This is the conceptual decode loop.

---

## 3. Layer coverage policy

The current experiments suggest that uniform allocation is not optimal.

Current best tested allocation in the inspected setup:
- layer 0: low budget
- layer 14: low budget
- layer 27: high budget

For the first decode-time prototype, use a fixed layer-wise budget policy rather than trying to adapt it online.

### Recommendation
Start with the current best tested policy as a fixed map.

Example:
- layer 0: 1
- layer 14: 1
- layer 27: 7

If the decode-time path is later expanded beyond the representative layers, this should generalize into a per-layer allocation table.

---

## 4. Representative layers vs all layers

There are two ways to build the first prototype.

### Option A: representative-layer RelayKV
Use RelayKV only on representative layers and leave the rest unchanged.

Pros:
- easier to debug
- smaller implementation change
- aligns with current experimental setup

Cons:
- not a full decode-time RelayKV path
- final speed and quality implications may be hard to interpret

### Option B: all-layer RelayKV with fixed budget table
Use RelayKV across all layers, with a per-layer retrieval budget.

Pros:
- closer to the eventual real system
- more meaningful end-to-end evaluation

Cons:
- more moving parts
- harder to debug initially

### Recommendation
Start with Option A for implementation validation, then move to Option B.

---

## 5. Decode-Time Block Boundary Policy

A practical issue appears during decode-time shadow retrieval: newly generated tokens extend the cache one step at a time, and the current prototype allows the boundary-adjacent cold block to become effectively variable-length.

This is observable in the current shadow prototype logs, where spans such as `3840–3841`, `3840–3842`, and `3840–3843` appear near the hot/cold boundary. This is acceptable for an early prototype, but it should not remain ambiguous in the actual decode-time design.

### Design question

When decode-time generation extends the cache, how should block boundaries be handled?

### Candidate policies

#### Option A: fixed cold blocks from prefill, decode additions stay hot-only

- Freeze cold block boundaries at the end of prefill
- Treat all newly generated tokens as part of the hot region
- Do not allow decode-added tokens to create or resize cold blocks during the decode loop

Pros:
- simplest and easiest to reason about
- avoids variable-length cold blocks
- keeps cold block metadata stable across decode steps
- makes retrieval logs easier to compare across steps

Cons:
- hot region grows unless explicitly trimmed or refreshed
- may require an additional policy for hot-window rollover

#### Option B: rolling repartition during decode

- Recompute hot/cold split during decode
- Allow newly generated tokens to push older hot tokens into cold storage
- Rebuild block boundaries as the decode loop advances

Pros:
- closer to a true long-running streaming policy
- keeps the hot region bounded more naturally

Cons:
- much more complex
- metadata and block ids can shift across steps
- harder to debug and benchmark fairly in the first implementation

#### Option C: hybrid rollover policy

- Freeze cold blocks for a fixed interval
- Keep new decode tokens hot
- After every N steps, refresh the split and rebuild cold blocks

Pros:
- bounded hot growth
- easier than full rolling repartition
- may offer a practical speed/stability compromise

Cons:
- adds another scheduling hyperparameter
- still more complex than a fixed-prefill policy

### Current recommendation

For the first actual decode-time RelayKV path, use **Option A**:

- freeze cold block boundaries at prefill time
- keep decode-added tokens in the hot region only
- do not let boundary-adjacent cold blocks become variable-length during the first assisted-decode implementation

This is the cleanest choice for the first implementation because it separates:

1. retrieval behavior
2. layer-wise budget allocation
3. decode-time cache growth policy

Only after the assisted decode path is working should a rolling or hybrid boundary update policy be tested.

### Follow-up implementation note

If Option A is used, the next implementation question becomes:

- how to handle hot-window rollover once decode length exceeds the original hot budget

That should be treated as a later design step, not as part of the first assisted-decode prototype.

---

## 6. Current Observations from Shadow Decode Prototype

A shadow-mode decode-time RelayKV prototype was run with representative layers `0`, `14`, and `27`, using the current deep-biased budget setting.

The main observations so far are:

### 1. Decode-time retrieval behavior is layer-dependent

- `layer 0` is highly stable and strongly recent-heavy.
- `layer 14` is also mostly recent-heavy, but shows some boundary-sensitive switching behavior.
- `layer 27` is the most dynamic and retrieval-demanding layer during decode.

This is consistent with the earlier inspected-layer difficulty picture.

### 2. `layer 27` often requires non-contiguous retrieval during decode

In the shadow decode prototype, `layer 27` frequently selected non-contiguous block sets rather than a single compact recent span.

This suggests that the hard deep layer may require a working set that combines:

- a recent-core region
- additional mid-range support blocks

This point is important for the later assisted-decode path, because candidate reconstruction cost may depend not only on candidate size but also on candidate contiguity.

### 3. Prompt style appears to affect decode-time retrieval structure

The current shadow runs suggest a qualitative difference between prompt styles:

- `structured` tends to produce a more recent-local and often more contiguous selection pattern
- `prose` tends to show more non-contiguous and more widely distributed retrieval at `layer 27`

This should still be treated as an early prototype observation, but it suggests that decode-time retrieval structure may depend on prompt style.

### 4. Boundary-adjacent tiny spans appear in the current prototype

The current shadow decode prototype allows boundary-adjacent spans such as `3840–3841`, `3840–3842`, and similar short recent spans to appear during decode.

This is acceptable for the current observation phase, but it reinforces the need for an explicit decode-time block boundary policy before implementing the first actual assisted-decode path.

### 5. Current implementation implication

The shadow prototype supports the following near-term implementation stance:

- keep the first assisted-decode prototype simple
- freeze cold block boundaries at prefill time
- keep decode-added tokens in the hot region
- treat non-contiguous candidate reconstruction, especially at `layer 27`, as a first-class implementation concern

---

## 7. Retrieval cadence

A practical design question is whether retrieval should happen at every decode step.

### Option A: retrieve every step
Pros:
- most accurate
- simplest conceptually

Cons:
- may be slow

### Option B: retrieve every N steps
Pros:
- cheaper
- easier to benchmark

Cons:
- may reduce approximation quality

### Recommendation
Start with retrieve-every-step for correctness.
Later test every-2 or every-4-step refresh as a speed-quality tradeoff.

---

## 8. What to benchmark first

The first decode-time benchmark should stay small.

### Compare
- full generation
- RelayKV-assisted generation (representative-layer version)
- later: RelayKV-assisted generation (all-layer version)

### Measure
- generated continuation
- elapsed time
- tokens/sec
- approximate KV sizes
- layer-wise attention error summary if available

### Initial objective
Not “beat full generation immediately,” but:
- produce valid output
- preserve reasonable output behavior
- expose the main overheads clearly

---

## 9. Minimum observable outputs

The first decode-time prototype should save:

- prompt_type
- max_new_tokens
- generated continuation
- elapsed time
- tokens/sec
- per-step or aggregated retrieval info
- candidate / working KV lengths
- selected block ids per step or sampled steps
- layer-wise budget map used
- any approximation error summary that can still be computed

This is more important than making the code elegant in the first version.

---

## 10. Risks and expected bottlenecks

### Likely bottlenecks
- repeated cold block scoring each step
- repeated candidate KV reconstruction
- CPU/GPU movement overhead
- mismatch between prototype path and model’s optimized attention path

### Likely early outcome
The first decode-time RelayKV path may be slower than full generation.

That is acceptable if it demonstrates:
- correct integration
- controllable approximation
- a path toward later optimization

---

## 11. Recommended implementation order

### Phase 1
Build a representative-layer decode-time prototype
- normal prefill
- RelayKV path on a small set of layers
- save outputs and timings

### Phase 2
Generalize to all-layer budget tables
- fixed per-layer retrieval allocation
- same decode loop

### Phase 3
Add cadence / optimization experiments
- retrieve every N steps
- lower-overhead reconstruction
- later hardware-aware optimization

---

## 12. Immediate next coding task

The next coding task should be:

Create a decode-time prototype script that:
- runs standard prefill
- enters a manual decode loop
- extracts current queries
- runs RelayKV retrieval on representative layers
- logs selected blocks, working-set sizes, and generated continuation

Suggested filename:
- `scripts/run_relaykv_decode_prototype.py`

This script should prioritize transparency and debuggability over speed.