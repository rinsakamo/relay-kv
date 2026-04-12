# RelayKV Quick Benchmark Summary

## Setup

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- seq_len_target: `4096`
- seq_len_actual: `4096`
- prompt_type: `repetitive`
- hot_window: `256`
- block_size: `256`
- scoring_variant: `mean_plus_norm`
- max_new_tokens: `64`

## Full generation

- prompt_tokens: `5462`
- generated_tokens: `64`
- elapsed_sec: `2.791314`
- tokens_per_sec: `22.928271`

## RelayKV comparison

| plan | layer 0 mean_abs_diff | layer 14 mean_abs_diff | layer 27 mean_abs_diff | avg | max |
|---|---:|---:|---:|---:|---:|
| uniform | 0.000000013 | 0.000683127 | 0.030168409 | 0.010283850 | 0.030168409 |
| very-heavy | 0.000000036 | 0.002280731 | 0.020364562 | 0.007548443 | 0.020364562 |

## Layer details

### uniform

| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 11, 12]` |
| 14 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 13, 12]` |
| 27 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 13, 12]` |

### very-heavy

| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 256 | 512 | 0.066667 | 0.125000 | `[14]` |
| 14 | 1 | 256 | 512 | 0.066667 | 0.125000 | `[14]` |
| 27 | 7 | 1792 | 2048 | 0.466667 | 0.500000 | `[14, 13, 12, 11, 10, 9, 2]` |

## Generated continuation

```text
 retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context retrieval behavior. RelayKV checks recent context
```

