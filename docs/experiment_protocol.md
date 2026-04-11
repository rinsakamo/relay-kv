# RelayKV Experiment Protocol

## Purpose

This document defines the standard comparison protocol for RelayKV experiments.

The goal is to make scoring-variant comparisons reproducible and directly comparable across runs.
In particular, pipeline-level comparisons should use the same prompt generation logic, sequence setup, and logging format.

---

## Standard comparison settings

### Sequence lengths
- 1024
- 2048
- 4096

### Prompt types
- repetitive
- prose
- structured

### Representative layers
- 0
- 14
- 27

### Primary evaluation script
- `scripts/run_relaykv_pipeline.py`

### Standard comparison output
- `results/raw/prototype_checks/relaykv_pipeline_summary.json`

---

## Required controls for fair comparison

The following settings must be matched when comparing scoring variants:

- model
- sequence length
- prompt type
- layer index
- hot window
- block size
- top-k or equivalent retrieval budget
- prompt generation logic
- seed, when applicable

Pipeline and sweep comparisons must use the same prompt generation logic.
`run_relaykv_pipeline.py` should be treated as the reference path for scoring-variant comparisons.

---

## Standard metrics

### Primary metric
- `mean_abs_diff` from attention output comparison

### Supporting metrics
- `coverage_ratio`
- `working_ratio`
- `num_selected_blocks`
- `num_layer_blocks`
- `candidate_k_len`
- `working_k_len`
- `full_k_len`

---

## Standard selection log

The canonical selection log is `top_scores` in `relaykv_pipeline_summary.json`.

Each selected entry should include:
- `layer_idx`
- `block_id`
- `start`
- `end`
- `score`

This is the standard source for:
- selected block identities
- selected block spans
- selected block scores
- ranking order

If an explicit rank field is later added, it should remain consistent with the order of `top_scores`.

---

## Comparison rules for scoring variants

When comparing scoring variants such as:
- `mean_only`
- `mean_plus_norm`
- `mean_plus_vnorm`
- `headwise_max_mean`
- future variants such as `mean_plus_max` or `query_to_block_max`

the following procedure should be used:

1. Fix the standard comparison settings.
2. Run `scripts/run_relaykv_pipeline.py`.
3. Save the summary to the standard output format.
4. Compare:
   - `mean_abs_diff`
   - `coverage_ratio`
   - `working_ratio`
   - `top_scores`
5. Inspect whether block selection changed:
   - top-k overlap
   - ranking changes
   - span/location changes

A scoring variant should not be judged only by output error.
Selection behavior should also be checked explicitly.

---

## Recommended evaluation order

### Stage 1: representative hard case
Use this setting first when testing a new scoring variant:

- `seq_len=4096`
- `prompt_type=prose`
- `layer_idx=27`

This is the preferred initial condition because it is a harder inspected case.

### Stage 2: representative layer slice
Then expand to:

- `layer_idx=0`
- `layer_idx=14`
- `layer_idx=27`

### Stage 3: broader condition check
Then expand across:

- `seq_len=1024, 2048, 4096`
- `prompt_type=repetitive, prose, structured`

---

## Interpretation guidelines

### If error improves and selection changes
This is evidence that the new scoring variant is altering retrieval behavior in a useful way.

### If selection changes but error does not
This suggests that retrieval behavior is sensitive, but the new selection is not yet better.

### If selection does not meaningfully change
This suggests that the current retrieval ranking is stable under that scoring modification.

This case is still a meaningful result.

---

## Notes

- `relaykv_pipeline_summary.json` is the primary comparison artifact.
- `top_scores` is the canonical selection log.
- Additional prototype summaries may still be useful for debugging, but scoring comparisons should be based on the pipeline summary.
