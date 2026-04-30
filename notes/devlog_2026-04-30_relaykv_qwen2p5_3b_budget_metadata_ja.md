# Devlog: Qwen2.5-3B budget metadata comparison

## 日付

2026-04-30

## 対象repo

- repo: `~/work/relay-kv`
- branch: `budget-planner-metadata`

## 目的

RelayKVのbudget-first評価で、Qwen2.5-1.5Bより少し現実的なKV負荷を出すため、`Qwen/Qwen2.5-3B-Instruct` を metadata-only budget cases に追加した。

この段階では、まだ `retrieval_top_k_effective` を実retrievalへ反映しない。

```text
actual runtime:
  --hot-window
  --top-k

budget metadata:
  --recent-window-tokens
  --retrieval-top-k
  retrieval_top_k_effective
```

## 実施内容

`scripts/run_budget_metadata_cases.py` に `--model-name` を追加し、`scripts/run_relaykv_pipeline.py --model` へ pass-through できるようにした。

また、非default modelの場合は出力名に suffix を付け、既存1.5B結果を破壊しないようにした。

## 変更ファイル

```text
scripts/run_budget_metadata_cases.py

results/raw/budget_metadata_cases/budget_tokens_seq1024_structured_layer14_qwen2p5_3b/
results/processed/budget_metadata_cases_budget_tokens_seq1024_structured_layer14_qwen2p5_3b.md
results/processed/budget_metadata_cases_budget_tokens_seq1024_structured_layer14_qwen2p5_3b.json

results/raw/budget_metadata_cases/residual_mib_seq1024_structured_layer14_qwen2p5_3b/
results/processed/budget_metadata_cases_residual_mib_seq1024_structured_layer14_qwen2p5_3b.md
results/processed/budget_metadata_cases_residual_mib_seq1024_structured_layer14_qwen2p5_3b.json
```

## 実行条件

```text
model_name: Qwen/Qwen2.5-3B-Instruct
case_set: budget_tokens / residual_mib
seq_len: 1024
prompt_type: structured
layer_idx: 14
top_k: 3
hot_window: 256
block_size: 128
```

offline env:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache
```

3Bはローカルcacheに存在し、network downloadなしで実行できた。

## model KV profile

### Qwen2.5-1.5B

```text
model_name: Qwen/Qwen2.5-1.5B-Instruct
num_layers: 28
num_key_value_heads: 2
head_dim: 128
kv_dtype_bytes: 2
kv_bytes_per_token: 28672
```

計算:

```text
28 layers × 2(K,V) × 2 kv_heads × 128 head_dim × 2 bytes
= 28672 bytes/token
```

### Qwen2.5-3B

```text
model_name: Qwen/Qwen2.5-3B-Instruct
num_layers: 36
num_key_value_heads: 2
head_dim: 128
kv_dtype_bytes: 2
kv_bytes_per_token: 36864
```

計算:

```text
36 layers × 2(K,V) × 2 kv_heads × 128 head_dim × 2 bytes
= 36864 bytes/token
```

### 比率

```text
3B / 1.5B = 36864 / 28672 = 1.285714...
```

3Bは1.5BよりKV bytes/tokenが約1.29倍。

同じMiB budgetで使えるworking KV token数は以下。

```text
1 / 1.285714... ≒ 0.7778
```

つまり、3Bでは同じ残VRAM予算で使えるworking KV token数が約22%減る。

## budget_tokens結果

```markdown
| plan | model_name | num_layers | num_key_value_heads | head_dim | kv_dtype_bytes | kv_bytes_per_token | kv_working_budget_tokens | recent_window_tokens | budget_block_size | anchor_blocks | anchor_budget_tokens | retrieval_budget_tokens | retrieval_block_budget | retrieval_top_k_requested | retrieval_top_k_effective | budget_overflow | budget_policy_reason | top_k | num_selected_blocks | working_ratio | mean_abs_diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| tokens_1024 | Qwen/Qwen2.5-3B-Instruct | 36 | 2 | 128 | 2 | 36864 | 1024 | 768 | 128 | 4 | 256 | 0 | 0 | 8 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000031693 |
| tokens_2048 | Qwen/Qwen2.5-3B-Instruct | 36 | 2 | 128 | 2 | 36864 | 2048 | 768 | 128 | 4 | 512 | 768 | 6 | 8 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000031693 |
| tokens_4096 | Qwen/Qwen2.5-3B-Instruct | 36 | 2 | 128 | 2 | 36864 | 4096 | 768 | 128 | 4 | 512 | 2816 | 22 | 8 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000031693 |
```

### 解釈

`tokens_1024/2048/4096` は explicit token budget なので、1.5Bと3Bで `kv_working_budget_tokens` 自体は同じ。

ただし、同じ `top_k=3`, `working_ratio=0.625` 条件で、3Bの方が attention差分が大きい。

```text
1.5B mean_abs_diff: 0.000008604
3B   mean_abs_diff: 0.000031693
```

このため、3Bは次のquality sweep対象として1.5Bより有効と判断できる。

## residual_mib結果

```markdown
| plan | model_name | num_layers | num_key_value_heads | head_dim | kv_dtype_bytes | kv_bytes_per_token | kv_working_budget_tokens | recent_window_tokens | budget_block_size | anchor_blocks | anchor_budget_tokens | retrieval_budget_tokens | retrieval_block_budget | retrieval_top_k_requested | retrieval_top_k_effective | budget_overflow | budget_policy_reason | top_k | num_selected_blocks | working_ratio | mean_abs_diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| mib_128 | Qwen/Qwen2.5-3B-Instruct | 36 | 2 | 128 | 2 | 36864 | 3640 | 768 | 128 | 4 | 512 | 2360 | 18 | 8 | 8 | False | estimated_from_available_kv_budget_mib | 3 | 3 | 0.625000000 | 0.000031693 |
| mib_256 | Qwen/Qwen2.5-3B-Instruct | 36 | 2 | 128 | 2 | 36864 | 7281 | 768 | 128 | 4 | 512 | 6001 | 46 | 8 | 8 | False | estimated_from_available_kv_budget_mib | 3 | 3 | 0.625000000 | 0.000031693 |
| mib_512 | Qwen/Qwen2.5-3B-Instruct | 36 | 2 | 128 | 2 | 36864 | 14563 | 768 | 128 | 4 | 512 | 13283 | 103 | 8 | 8 | False | estimated_from_available_kv_budget_mib | 3 | 3 | 0.625000000 | 0.000031693 |
```

### 1.5Bとの比較

```text
128MiB:
  1.5B: 4681 tokens
  3B:   3640 tokens

256MiB:
  1.5B: 9362 tokens
  3B:   7281 tokens

512MiB:
  1.5B: 18724 tokens
  3B:   14563 tokens
```

同じMiB budgetで3Bのworking token数は約22%減る。

## 注意点

今回の3B residual MiB caseでも、128MiB時点でまだ budget 的には余裕がある。

```text
mib_128:
  recent = 768
  anchor = 512
  retrieval_budget = 2360
  retrieval_block_budget = 18
  retrieval_top_k_effective = 8
```

そのため、MiB sweepだけでは制約が弱い。

制約感を出すには、引き続き token budget 指定の方が有効。

```text
tokens_1024:
  retrieval_top_k_effective = 0

tokens_2048:
  retrieval_top_k_effective = 6

tokens_4096:
  retrieval_top_k_effective = 8
```

## 既存挙動への影響

変更していないもの:

```text
scoring
attention comparison
retrieval block selection
--top-k behavior
KV tensor construction
retrieval_top_k_effective の実反映
```

`retrieval_top_k_effective` は引き続き metadata-only。

## 確認コマンド

```bash
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python -m compileall relaykv scripts
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python scripts/run_budget_metadata_cases.py   --model-name Qwen/Qwen2.5-3B-Instruct   --case-set budget_tokens   --seq-len 1024   --prompt-type structured   --layer-idx 14   --top-k 3   --hot-window 256   --block-size 128
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python scripts/run_budget_metadata_cases.py   --model-name Qwen/Qwen2.5-3B-Instruct   --case-set residual_mib   --seq-len 1024   --prompt-type structured   --layer-idx 14   --top-k 3   --hot-window 256   --block-size 128
```

```bash
git diff --name-status | grep '.github/workflows' || true
git diff --check
```

結果:

```text
compileall: pass
3B budget_tokens: pass
3B residual_mib: pass
.github/workflows差分: なし
git diff --check: pass
network download: なし
```

## 結論

Qwen2.5-3Bは、RelayKV budget metadata quality sweep の次の標準モデル候補として妥当。

理由:

```text
1. 3Bはローカルcacheにあり、offline実行できる
2. 1.5BよりKV bytes/tokenが約1.29倍大きい
3. 同じMiB予算でworking KV token数が約22%減る
4. 同じtop_k/working_ratio条件でmean_abs_diffが大きい
5. ただし128MiB residual budgetでもまだ緩い
6. 次は token budget中心の quality sweep がよい
```

## 次の推奨ステップ

3Bで prompt_type と layer を広げる。

```text
model: Qwen/Qwen2.5-3B-Instruct
case_set: budget_tokens
seq_len: 1024
prompt_type: repetitive / prose / structured
layer_idx: 0 / 14 / 27
top_k: 3
hot_window: 256
block_size: 128
```

まだ `retrieval_top_k_effective` は実retrievalに反映しない。

このsweepで、3Bにおける prompt/layer別の `mean_abs_diff` と budget metadata を比較する。
