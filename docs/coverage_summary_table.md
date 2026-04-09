# RelayKV Coverage Summary Table

Representative rows showing how increasing effective coverage reduces error.

| seq_len | representative coverage_ratio | representative setting | mean_abs_diff |
|---:|---:|---|---:|
| 1024 | 0.0714 | hot=128, block=64, top_k=1 | 0.072750889 |
| 1024 | 0.4286 | hot=128, block=128, top_k=3 / 256,2 | 0.038244553 |
| 1024 | 1.0000 | hot=256, block=256, top_k=3 | 0.000000000 |
| 2048 | 0.0333 | hot=128, block=64, top_k=1 | 0.051156793 |
| 2048 | 0.2000 | hot=128, block=128, top_k=3 / 256,2 | 0.043847952 |
| 2048 | 0.4286 | hot=256, block=256, top_k=3 | 0.033087544 |
| 4096 | 0.0323 | hot=128, block=128, top_k=1 / 256,1 | 0.041207977 |
| 4096 | 0.0968 | hot=128, block=128, top_k=3 / 256,2 | 0.036567513 |
| 4096 | 0.2000 | hot=256, block=256, top_k=3 | 0.032424182 |

## Short interpretation

Across `1024`, `2048`, and `4096`, approximation error decreases as effective candidate coverage increases. Different `(block_size, top_k)` combinations often yield very similar error when they produce similar effective coverage. The same qualitative trend persists at `4096`.
