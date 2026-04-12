# RelayKV Quick Benchmark — 3 Prompt Styles Comparison Memo

## Scope

This memo summarizes the current quick-benchmark status across three prompt styles:

- repetitive
- prose
- structured

The current comparison focuses on:

- full generation speed (when available)
- RelayKV approximation quality under:
  - uniform `(3,3,3)`
  - very-heavy `(1,1,7)`

---

## 1. Overall takeaway

Across all three prompt styles examined so far, the same qualitative pattern holds for the inspected layer-wise budget setup:

- uniform < balanced < hard-layer-heavy < very-heavy

In other words, allocating more retrieval budget to the harder deep layer continues to improve approximation quality under the same total budget.

Among the tested allocations, `very-heavy (1,1,7)` is currently the strongest candidate.

---

## 2. Prompt-style summary

### repetitive

Current status:
- layer-wise budget comparison has been completed
- quick benchmark markdown with full-generation speed is not yet included in this memo

Observed pattern:
- `very-heavy` performed best among the tested plans
- the main gain again came from a large reduction in layer 27 error
- layer 14 became worse, but the overall average and worst-case error still improved

Representative layer-wise budget table result:
- uniform avg: `0.010283850`
- balanced avg: `0.009389106`
- hard-layer-heavy avg: `0.008408868`
- very-heavy avg: `0.007548443`

Layer 27:
- uniform: `0.030168409`
- very-heavy: `0.020364562`

Interpretation:
- even in the repetitive setting, shifting budget toward the hard deep layer is beneficial

---

### prose

Full generation:
- prompt tokens: `4694`
- generated tokens: `64`
- elapsed: `2.900450 sec`
- throughput: `22.065544 tok/s`

RelayKV comparison:
- uniform avg: `0.008483595`
- very-heavy avg: `0.005448804`

Layer 27:
- uniform: `0.019356238`
- very-heavy: `0.009709065`

Interpretation:
- very-heavy substantially improves the representative hard-layer error
- the improvement is large enough to reduce both average and worst-case error

Notes:
- uniform selected block ids:
  - layer 0: `[14, 9, 11]`
  - layer 14: `[14, 13, 12]`
  - layer 27: `[14, 12, 11]`
- very-heavy selected block ids:
  - layer 0: `[14]`
  - layer 14: `[14]`
  - layer 27: `[14, 12, 11, 13, 8, 10, 9]`

---

### structured

Full generation:
- prompt tokens: `5050`
- generated tokens: `64`
- elapsed: `3.008345 sec`
- throughput: `21.274153 tok/s`

RelayKV comparison:
- uniform avg: `0.007499031`
- very-heavy avg: `0.004137125`

Layer 27:
- uniform: `0.021780726`
- very-heavy: `0.011168486`

Interpretation:
- very-heavy again gives the best inspected result
- the same qualitative trend seen in prose also appears in structured prompts

Notes:
- uniform selected block ids:
  - layer 0: `[14, 7, 12]`
  - layer 14: `[14, 13, 12]`
  - layer 27: `[14, 13, 12]`
- very-heavy selected block ids:
  - layer 0: `[14]`
  - layer 14: `[14]`
  - layer 27: `[14, 13, 12, 11, 10, 9, 8]`

---

## 3. Cross-style interpretation

The current evidence suggests:

1. The benefit of layer-wise budget allocation is not limited to a single prompt style.
2. A stronger deep-layer bias consistently improves approximation quality in the inspected setup.
3. `very-heavy (1,1,7)` is currently the best-tested allocation among the inspected plans.
4. Lightweight scoring changes were much less influential than allocation changes.
5. The current prototype appears more sensitive to block construction and budget allocation than to small scoring modifications.

---

## 4. Practical current recommendation

For the current inspected setup, the strongest working recommendation is:

- `block_size=256`
- `hot_window=256`
- `scoring_variant=mean_plus_norm`
- layer-wise allocation:
  - layer 0: `1`
  - layer 14: `1`
  - layer 27: `7`

This should still be treated as a prototype-stage best-tested configuration, not yet as a final globally optimal design.

---

## 5. What is still missing

To complete the quick-benchmark picture more cleanly, the next useful additions are:

1. repetitive quick benchmark markdown with full-generation speed
2. a direct side-by-side summary across all three prompt styles in one table
3. a later benchmark that measures actual RelayKV-assisted generation latency, not just full generation plus approximation summaries

---

## 6. One-sentence summary

Across repetitive, prose, and structured prompts, the current RelayKV prototype consistently benefits from allocating more retrieval budget to the hardest deep layer, and the best tested plan so far is `very-heavy (1,1,7)`.
