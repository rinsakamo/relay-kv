| model_name | prompt_type | layer_idx | seq_len | top_k | num_selected_blocks | selected_block_ids | working_ratio | coverage_ratio | candidate_k_len | working_k_len | cold_k_len | mean_abs_diff | max_abs_diff |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | 1024 | 3 | 3 | [5,4,3] | 0.625000000 | 0.500000000 | 384 | 640 | 768 | 0.369262010 | 2.575494528 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | 1024 | 6 | 6 | [5,4,3,2,1,0] | 1.000000000 | 1.000000000 | 768 | 1024 | 768 | 0.000000000 | 0.000000000 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | 1024 | 8 | 6 | [5,4,3,2,1,0] | 1.000000000 | 1.000000000 | 768 | 1024 | 768 | 0.000000000 | 0.000000000 |
