# RelayKV Quick Benchmark Summary

## Setup

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- seq_len_target: `4096`
- seq_len_actual: `4096`
- prompt_type: `prose`
- hot_window: `256`
- block_size: `256`
- scoring_variant: `mean_plus_norm`
- max_new_tokens: `64`

## Full generation

- prompt_tokens: `4694`
- generated_tokens: `64`
- elapsed_sec: `2.957480`
- tokens_per_sec: `21.640045`

## RelayKV comparison

| plan | layer 0 mean_abs_diff | layer 14 mean_abs_diff | layer 27 mean_abs_diff | avg | max |
|---|---:|---:|---:|---:|---:|
| uniform | 0.005545032 | 0.000549515 | 0.019356238 | 0.008483595 | 0.019356238 |
| very-heavy | 0.005552961 | 0.001084385 | 0.009709065 | 0.005448804 | 0.009709065 |

## Layer details

### uniform

| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 9, 11]` |
| 14 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 13, 12]` |
| 27 | 3 | 768 | 1024 | 0.200000 | 0.250000 | `[14, 12, 11]` |

### very-heavy

| layer | top_k | candidate_k_len | working_k_len | coverage_ratio | working_ratio | selected_block_ids |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 256 | 512 | 0.066667 | 0.125000 | `[14]` |
| 14 | 1 | 256 | 512 | 0.066667 | 0.125000 | `[14]` |
| 27 | 7 | 1792 | 2048 | 0.466667 | 0.500000 | `[14, 12, 11, 13, 8, 10, 9]` |

## Generated continuation

```text
 working set, and comparing approximate attention outputs against full attention. The current experiments examine how coverage ratio, block granularity, and layer difficulty affect approximation quality across different sequence lengths and prompt styles. RelayKV is a prototype for splitting KV cache into hot and cold regions, retrieving a smaller working set, and comparing approximate attention outputs against
```

