# Devlog: Qwen2.5-3B budget metadata quality sweep

## 日付

2026-04-30

## 対象repo

- repo: `~/work/relay-kv`
- branch: `budget-planner-metadata`

## 目的

Qwen2.5-3Bを使って、RelayKVの `budget_tokens` case-set を `prompt_type × layer_idx` に広げ、metadata-onlyのbudget評価表を作成した。

この段階では、まだ `retrieval_top_k_effective` を実retrievalへ反映していない。

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

既存の `scripts/run_budget_metadata_cases.py` はそのまま使用できたため、実行ロジックは変更していない。

追加したのは、複数caseの結果を集約するためのsummary scriptのみ。

```text
scripts/make_budget_metadata_sweep_summary.py
```

## 変更 / 追加ファイル

```text
scripts/make_budget_metadata_sweep_summary.py

results/raw/budget_metadata_cases/budget_tokens_seq1024_repetitive_layer0_qwen2p5_3b/
results/raw/budget_metadata_cases/budget_tokens_seq1024_repetitive_layer14_qwen2p5_3b/
results/raw/budget_metadata_cases/budget_tokens_seq1024_repetitive_layer27_qwen2p5_3b/

results/raw/budget_metadata_cases/budget_tokens_seq1024_prose_layer0_qwen2p5_3b/
results/raw/budget_metadata_cases/budget_tokens_seq1024_prose_layer14_qwen2p5_3b/
results/raw/budget_metadata_cases/budget_tokens_seq1024_prose_layer27_qwen2p5_3b/

results/raw/budget_metadata_cases/budget_tokens_seq1024_structured_layer0_qwen2p5_3b/
results/raw/budget_metadata_cases/budget_tokens_seq1024_structured_layer14_qwen2p5_3b/
results/raw/budget_metadata_cases/budget_tokens_seq1024_structured_layer27_qwen2p5_3b/

results/processed/budget_metadata_cases_budget_tokens_seq1024_{prompt}_layer{layer}_qwen2p5_3b.md
results/processed/budget_metadata_cases_budget_tokens_seq1024_{prompt}_layer{layer}_qwen2p5_3b.json

results/processed/budget_metadata_3b_sweep_seq1024_summary.md
results/processed/budget_metadata_3b_sweep_seq1024_summary.json
```

## 実行条件

```text
model_name: Qwen/Qwen2.5-3B-Instruct
case_set: budget_tokens
seq_len: 1024
prompt_type: repetitive / prose / structured
layer_idx: 0 / 14 / 27
top_k: 3
hot_window: 256
block_size: 128
```

offline env:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache
```

## 実行した prompt_type × layer_idx

```text
repetitive: 0, 14, 27
prose:      0, 14, 27
structured: 0, 14, 27
```

合計9条件。

各条件で `tokens_1024`, `tokens_2048`, `tokens_4096` の3 planを出力した。

## summary table

```markdown
| model_name | prompt_type | layer_idx | plan | kv_bytes_per_token | kv_working_budget_tokens | retrieval_top_k_effective | budget_overflow | budget_policy_reason | top_k | num_selected_blocks | working_ratio | mean_abs_diff |
|---|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| Qwen/Qwen2.5-3B-Instruct | repetitive | 0 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000407403 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 0 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000407403 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 0 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000407403 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 14 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000012910 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 14 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000012910 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 14 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000012910 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.369262010 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.369262010 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.369262010 |
| Qwen/Qwen2.5-3B-Instruct | prose | 0 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000069693 |
| Qwen/Qwen2.5-3B-Instruct | prose | 0 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000069693 |
| Qwen/Qwen2.5-3B-Instruct | prose | 0 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000069693 |
| Qwen/Qwen2.5-3B-Instruct | prose | 14 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000483775 |
| Qwen/Qwen2.5-3B-Instruct | prose | 14 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000483775 |
| Qwen/Qwen2.5-3B-Instruct | prose | 14 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000483775 |
| Qwen/Qwen2.5-3B-Instruct | prose | 27 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.033340055 |
| Qwen/Qwen2.5-3B-Instruct | prose | 27 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.033340055 |
| Qwen/Qwen2.5-3B-Instruct | prose | 27 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.033340055 |
| Qwen/Qwen2.5-3B-Instruct | structured | 0 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000376783 |
| Qwen/Qwen2.5-3B-Instruct | structured | 0 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000376783 |
| Qwen/Qwen2.5-3B-Instruct | structured | 0 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000376783 |
| Qwen/Qwen2.5-3B-Instruct | structured | 14 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000031693 |
| Qwen/Qwen2.5-3B-Instruct | structured | 14 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000031693 |
| Qwen/Qwen2.5-3B-Instruct | structured | 14 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000031693 |
| Qwen/Qwen2.5-3B-Instruct | structured | 27 | tokens_1024 | 36864 | 1024 | 0 | True | anchor_budget_clipped_after_recent_window | 3 | 3 | 0.625000000 | 0.000554743 |
| Qwen/Qwen2.5-3B-Instruct | structured | 27 | tokens_2048 | 36864 | 2048 | 6 | True | retrieval_top_k_clipped_to_remaining_budget | 3 | 3 | 0.625000000 | 0.000554743 |
| Qwen/Qwen2.5-3B-Instruct | structured | 27 | tokens_4096 | 36864 | 4096 | 8 | False | explicit_working_budget_tokens | 3 | 3 | 0.625000000 | 0.000554743 |
```

## metadata-onlyとしての確認

今回も `retrieval_top_k_effective` はsummary表示のみで、実retrievalには反映していない。

全rowで実挙動は固定されている。

```text
top_k = 3
num_selected_blocks = 3
working_ratio = 0.625
```

そのため、同じ `prompt_type × layer_idx` の中では、`tokens_1024`, `tokens_2048`, `tokens_4096` の `mean_abs_diff` が同じになる。

これは意図通り。

budget metadataは以下のように変化している。

```text
tokens_1024:
  retrieval_top_k_effective = 0
  budget_overflow = True
  budget_policy_reason = anchor_budget_clipped_after_recent_window

tokens_2048:
  retrieval_top_k_effective = 6
  budget_overflow = True
  budget_policy_reason = retrieval_top_k_clipped_to_remaining_budget

tokens_4096:
  retrieval_top_k_effective = 8
  budget_overflow = False
  budget_policy_reason = explicit_working_budget_tokens
```

## 主な観察

### 1. layer差が非常に大きい

特に layer 27 で差が大きくなるcaseがある。

```text
repetitive layer 27:
  mean_abs_diff = 0.369262010

prose layer 27:
  mean_abs_diff = 0.033340055

structured layer 27:
  mean_abs_diff = 0.000554743
```

特に `repetitive × layer27` は極端に大きい。

### 2. layer14だけでは安全側に見えすぎる

layer14の結果は全体的に小さい。

```text
repetitive layer14:
  mean_abs_diff = 0.000012910

prose layer14:
  mean_abs_diff = 0.000483775

structured layer14:
  mean_abs_diff = 0.000031693
```

これだけを見ると、3BでもRelayKV差分はかなり小さく見える。

しかし layer27 では大きな差分が出るため、今後の評価では layer27 を外すべきではない。

### 3. prompt_type依存も強い

同じ layer27 でもprompt_typeで差が大きい。

```text
repetitive layer27:
  0.369262010

prose layer27:
  0.033340055

structured layer27:
  0.000554743
```

このため、structuredだけで評価すると過度に楽観的になる可能性がある。

## 現時点の解釈

3Bでは、1.5Bよりも実験対象としての感度が上がった。

特に次の観点で有用。

```text
1. Qwen2.5-3Bはoffline cacheで実行できる
2. kv_bytes_per_token = 36864
3. 1.5BよりKV負荷が約1.29倍
4. layer27で大きなattention差分が出る
5. prompt_typeによる差が明確
6. layer14だけでは危険側を見落とす可能性がある
```

## 変更していないもの

```text
scoring
attention comparison
retrieval block selection
--top-k behavior
KV tensor construction
retrieval_top_k_effective の実反映
```

## 確認コマンド

```bash
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python -m compileall relaykv scripts
```

3Bの全9条件を以下のような形で実行。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python scripts/run_budget_metadata_cases.py   --model-name Qwen/Qwen2.5-3B-Instruct   --case-set budget_tokens   --seq-len 1024   --prompt-type structured   --layer-idx 14   --top-k 3   --hot-window 256   --block-size 128
```

summary確認:

```bash
head -n 30 results/processed/budget_metadata_3b_sweep_seq1024_summary.md
```

差分確認:

```bash
git diff --name-status | grep '.github/workflows' || true
git diff --check
```

結果:

```text
compileall: pass
3B 全9条件: pass
summary table: 作成済み
.github/workflows差分: なし
git diff --check: pass
network download: なし
```

## 結論

3Bのmetadata-only budget sweepは成功。

次の判断:

```text
Qwen2.5-3Bは、RelayKVの次の標準評価モデルとして妥当。
```

理由:

```text
1. ローカルcacheにあり、offline実行可能
2. 1.5BよりKV負荷が高い
3. 同じtop_k/working_ratio条件で差分が出やすい
4. prompt/layer差が明確に見える
5. 特に layer27 が危険側の検出に有効
```

## 次の推奨ステップ

次はすぐに `retrieval_top_k_effective` を実反映するより、まず **layer27の高diff原因確認** を行う。

候補:

```text
1. repetitive layer27 の raw JSON を詳しく見る
2. selected block / score / cold block配置を確認する
3. top_k=3 固定が厳しすぎるのか確認する
4. top_k=6 / 8 で同じ layer27 を追加実行する
5. その後、retrieval_top_k_effective を実反映する optional flag を設計する
```

推奨する次実験:

```text
model: Qwen/Qwen2.5-3B-Instruct
prompt_type: repetitive
layer_idx: 27
seq_len: 1024
hot_window: 256
block_size: 128
top_k: 3 / 6 / 8
case_set: budget_tokens
```

目的:

```text
layer27の大diffが、top_k不足によるものか、prompt構造によるものか、後段layer固有の敏感さによるものかを切り分ける。
