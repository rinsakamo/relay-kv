# Dynamic Gate Findings for Layer14 Next-Step Apply

## Background

This note summarizes findings from layer14 next-step apply experiments in the RelayKV assisted decode prototype.

We introduced two gating dimensions:

- **spatial confidence**: `min_score_margin`
- **temporal confidence**: `min_gate_step`

The goal was to understand whether divergence quality depends not only on *which block* is selected, but also on *when* replacement is allowed.

## Key Finding

The main result is:

> RelayKV quality is governed by both spatial routing and temporal routing.

In particular, for structured prompts, instability was driven less by block choice itself and more by **early application timing**.

## Score Margin Gate

For `2048 / prose`, increasing `min_score_margin` reduced gate passes monotonically:

| min_score_margin | gate pass count |
|---|---:|
| 10 | 10 |
| 30 | 5 |
| 50 | 1 |

This shows that `score_margin` is a useful confidence proxy for block dominance.

## Early-Step Suppress

### Structured / 2048 / margin50

Without temporal suppression:

- gate pass steps: `[0, 1, 2, 6]`
- first divergence: `step 1`
- divergence type: candidate shift
- top5 overlap: `1`

With `min_gate_step=2`:

- gate pass steps: `[2, 12]`
- first divergence: `step 10`
- step 9: apply/baseline fully identical
- step 10: same top5 set, top1 reordered only
- divergence type: delayed rank flip
- top5 overlap: `5`

This indicates that **early-step suppress strongly stabilizes structured prompts**.

## Prose Cross-Check

For `2048 / prose / margin30`, adding `min_gate_step=2` caused almost no observable change:

- gate pass steps remained `[2, 3, 5, 6, 8]`
- first divergence remained `step 3`
- divergence stayed rank-flip-like with strong top5 overlap

This suggests that temporal suppression has **low side effect** on prose while providing strong benefit for structured prompts.

## Interpretation

These results suggest:

1. `min_score_margin` controls whether the selected block is spatially dominant.
2. `min_gate_step` controls whether replacement is temporally safe.
3. Structured instability is highly sensitive to early injection.
4. Temporal gating can weaken divergence from **candidate shift** into **rank flip**.

## Design Implication

RelayKV should not be viewed only as block selection.

A better framing is:

- **spatial routing**: which block to use
- **temporal routing**: when to use it

## Next Steps

- Extend comparison summary with:
  - `min_score_margin`
  - `min_gate_step`
  - `gate_pass_steps`
  - `first_divergence_step`
  - `top5_overlap_count`
  - `same_top5_set`
  - `change_type`
- Test whether pre-divergence signals such as:
  - rising `mean_abs_diff`
  - low `score_margin`
  can predict future divergence.
- Connect temporal gating back to retrieval policy design.

## Focused Single-Style Results

### Prose / 2048 / `min_score_margin=30` / `min_gate_step=2`

- gate pass steps: `[2, 3, 5, 6, 8]`
- first divergence: `step 3`
- divergence lag: `0`
- pre-divergence step: `2`
- pre-divergence mean abs diff: `8.87e-4`
- pre-divergence score margin: `30.17`
- divergence-step mean abs diff: `1.75e-5`
- divergence-step score margin: `33.73`
- divergence type: `rank_flip_partial_overlap`
- top-5 overlap: `4/5`
- top-5 Jaccard: `0.667`
- change subtype: `mutual_retention`

This condition remains stable in the sense that divergence is limited to a high-overlap rank-flip-like change. Adding `min_gate_step=2` does not materially change the prose behavior, which suggests low side effect on already stable prompt styles.

### Structured / 2048 / `min_score_margin=50` / `min_gate_step=2`

- gate pass steps: `[2, 12]`
- first divergence: `step 10`
- divergence lag: `7`
- pre-divergence step: `9`
- pre-divergence mean abs diff: `2.13e-4`
- pre-divergence score margin: `3.87`
- divergence-step mean abs diff: `1.17e-5`
- divergence-step score margin: `4.79`
- divergence type: `rank_flip_same_set`
- top-5 overlap: `5/5`
- top-5 Jaccard: `1.0`
- change subtype: `mutual_retention`

This is a major improvement over the earlier structured setting, where divergence happened at step 1 with candidate-shift behavior. With temporal suppression, structured divergence becomes delayed and non-destructive.

### Practical Interpretation

These focused results support the following view:

- `min_score_margin` provides **spatial confidence**
- `min_gate_step` provides **temporal safety**
- both are required for stable behavior

Most importantly, structured instability appears to be caused primarily by **early unsafe application timing**, not merely by prompt style itself.

### Implication for RelayKV

RelayKV should not be framed purely as block selection. A better abstraction is:

- **spatial routing**: which KV blocks to use
- **temporal routing**: when to apply them

The structured result shows that temporal gating can transform divergence from an early destructive candidate shift into a delayed same-set rank flip.

### Repetitive / 2048 / `min_score_margin=30` / `min_gate_step=2`

- gate pass steps: `[12]`
- first divergence: `step 13`
- divergence lag: `1`
- divergence type: `rank_flip_partial_overlap`
- top-5 overlap: `2/5`

At this setting, repetitive prompts become strongly conservative: only one gated step is allowed, and divergence appears one step later. The resulting change is weaker than candidate shift but less stable than the prose case, remaining in a partial-overlap regime.

### Cross-Style Summary

Under the tested gated settings:

- **prose** remains largely unchanged and already stable
- **repetitive** becomes strongly conservative, with only a single gated step
- **structured** benefits the most from temporal suppression, shifting from early candidate-shift behavior to delayed same-set rank-flip behavior

This supports the view that `min_gate_step` acts as a practical temporal safety bias:
it is highly effective for early-sensitive conditions, while introducing little or moderate side effect elsewhere.

## Dry-Run Divergence Predictor

A simple dry-run predictor was tested using:

- `score_margin < 10`
- `mean_abs_diff > 1e-4`

In the best structured setting
(`seq_len=2048`, `min_score_margin=50`, `min_gate_step=2`):

- predictor danger steps: `[9, 11]`
- first divergence: `step 10`

This means the predictor successfully flagged the pre-divergence step (`step 9`) while avoiding earlier low-margin but stable steps (`steps 7 and 8`).

This suggests that the predictor is better interpreted as a **post-injection hazard detector** than as a replacement for the gating rule itself.

---

## Current phase closure

We close the current phase at the point where the core behavior of the dynamic gate is sufficiently understood. At 2048, a provisional cross-style policy candidate is already visible, while successful runs indicate that the main success factor is temporal routing rather than predictor-based control. In contrast, 4096 is not yet in a gate-policy tuning phase; it first requires sanity checks on the baseline/apply comparison setup. 7

## Main conclusions from the dynamic-gate phase

### Provisional common policy at 2048

For `medium_2048`, the current provisional common policy candidate is:

- `min_score_margin = 20`
- `min_gate_step = 7`

### Failure is dominated by timing, not block identity

The best current explanation is that instability is governed more by when a block is applied than by which block is selected. In condensed form:

- block 7 is not unsafe
- early block 7 is unsafe
- late block 7 is safe

### Predictor is currently auxiliary

The predictor now supports separate observation of requested vs. effective blocking. However, in successful runs it does not act as the primary control mechanism. At this phase, it is better interpreted as an auxiliary hazard-observation component rather than the main driver of stability. 8

## Known constraints at the end of this phase

The known constraints at this point are:

- early steps are risky
- temporal routing is dominant
- the predictor remains auxiliary
- 4096 requires comparison sanity checks before policy tuning
- retrieval metrics, fixed GPU live KV budget, and a recent/anchor/retrieval three-tier design are not yet introduced. 9

## Treatment of 4096 results

In this phase, 4096 should not be treated as a target for further dynamic-gate tuning. It should instead be treated as a sanity-check target for the comparison setup. The required checks are:

- step-0 token/top5 agreement
- generated token count
- `step_logs` length
- baseline file pairing correctness. 10

## Transition to the fast-track plan

The next phase shifts the experimental subject away from dynamic-gate micro-tuning and toward memory/retrieval architecture. The next primary design axes are:

- fixed GPU live KV budget
- recent/anchor/retrieval three-tier memory design
- retrieval quality metrics
- coarse-to-fine retrieval

This phase transition should begin once the 2048 representative success case is fixed, the divergence summary is stable in its current form, 4096 is explicitly treated as a sanity-check target, and the present phase conclusion is documented as temporal-routing-dominant. 11
