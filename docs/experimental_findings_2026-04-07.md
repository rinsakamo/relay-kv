# RelayKV Experimental Findings (Prototype Snapshot)

## Scope

This note summarizes the current prototype findings for RelayKV.

The current snapshot is still small-scale, but the main trend is already visible.

## Main findings

### 1. Candidate coverage strongly affects approximation quality

In the current sweeps, approximation error decreased as candidate coverage increased.

This trend was visible at both:

- `seq_len=1024`
- `seq_len=2048`

and remained especially clear for the harder case `layer_idx=27`.

### 2. Matched coverage often leads to similar error

Different `(block_size, top_k)` combinations often produced nearly identical errors when they yielded similar effective coverage.

This suggests that approximation quality may be explained better by **effective candidate coverage** than by block granularity alone.

### 3. Larger hot windows improve stability

Increasing the hot window consistently made the approximation more stable.

This is expected, since more recent KV remains directly available and does not need to be reconstructed from cold storage.

### 4. Longer contexts are harder, but follow the same trend

`seq_len=2048` generally shows higher residual error than `seq_len=1024`, but the same monotonic coverage-driven improvement pattern still appears.

## Figure

![RelayKV coverage vs error](figures/relaykv_coverage_vs_error.png)

**Figure 1.** Mean absolute attention-output difference versus coverage ratio for `layer_idx=27`. Both `seq_len=1024` and `seq_len=2048` show decreasing error as coverage increases. The longer context remains harder overall, but preserves the same qualitative trend.

## Example slice

| seq_len | hot_window | block_size | top_k | coverage_ratio | mean_abs_diff |
|---:|---:|---:|---:|---:|---:|
| 1024 | 128 | 64  | 1 | 0.0714 | 0.072750889 |
| 1024 | 128 | 128 | 3 | 0.4286 | 0.038244553 |
| 1024 | 256 | 256 | 3 | 1.0000 | 0.000000000 |
| 2048 | 128 | 64  | 1 | 0.0333 | 0.051156793 |
| 2048 | 128 | 256 | 3 | 0.3333 | 0.036251646 |
| 2048 | 256 | 256 | 3 | 0.4286 | 0.033087544 |

## Cautious interpretation

The current evidence supports the following empirical statement:

> approximation error appears to correlate more strongly with effective candidate coverage than with execution granularity alone.

This should still be treated as a prototype-stage observation rather than a final general theorem.

## Recommended next analyses

- extend to more sequence lengths
- compare more layers systematically
- add multiple prompts
- generate compact result tables for the paper
- add a second plot using `working_ratio`
