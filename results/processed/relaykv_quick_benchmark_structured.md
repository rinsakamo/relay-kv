# RelayKV Quick Benchmark Summary

## Setup

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- seq_len_target: `4096`
- seq_len_actual: `4096`
- prompt_type: `structured`
- hot_window: `256`
- block_size: `256`
- scoring_variant: `mean_plus_norm`
- max_new_tokens: `64`

## Full generation

- prompt_tokens: `5050`
- generated_tokens: `64`
- elapsed_sec: `3.053305`
- tokens_per_sec: `20.960894`

## RelayKV comparison

| plan | layer 0 mean_abs_diff | layer 14 mean_abs_diff | layer 27 mean_abs_diff | avg | max |
|---|---:|---:|---:|---:|---:|
| uniform | 0.000001254 | 0.000715114 | 0.021780726 | 0.007499031 | 0.021780726 |
| very-heavy | 0.000001207 | 0.001241682 | 0.011168486 | 0.004137125 | 0.011168486 |

## Layer details

### uniform

| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 7, 12]` |
| 14 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 13, 12]` |
| 27 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 13, 12]` |

### very-heavy

| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 256 | 512 | 0.066667 | 0.125000 | `[14]` |
| 14 | 1 | 256 | 512 | 0.066667 | 0.125000 | `[14]` |
| 27 | 7 | 1792 | 2048 | 0.466667 | 0.500000 | `[14, 13, 12, 11, 10, 9, 8]` |

## Generated continuation

```text
 full attention - factors: coverage ratio, block size, hot window, layer index - observation: harder layers may require larger retrieval budgets - note: scoring changes and block granularity should be evaluated separately Experiment summary: - system: RelayKV - goal: compare approximate attention against full attention - factors: coverage ratio, block size,
```

