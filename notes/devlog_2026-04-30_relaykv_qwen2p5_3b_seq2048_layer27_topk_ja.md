# Devlog: Qwen2.5-3B seq_len=2048 layer27 top-k sweep

## 日付

2026-04-30

## 対象repo

- repo: `~/work/relay-kv`
- branch: `budget-planner-metadata`

## 目的

Qwen2.5-3Bで `seq_len=2048 / layer27` の top-k sweep を行い、PyTorch側のbudget-first実験を一区切りにする判断材料を作る。

直前の `seq_len=1024 / repetitive / layer27` では、`top_k=3` で大きな差分が出た一方、`top_k=6` で cold block full coverage になり `mean_abs_diff=0` になった。

今回の目的は、`seq_len=2048` でも top-k 増加で差分が下がるのか、または full cold coverage に近いworking setが必要になるのかを確認すること。

## 実行条件

```text
model_name: Qwen/Qwen2.5-3B-Instruct
case_set: budget_tokens
seq_len: 2048
prompt_type: repetitive / prose / structured
layer_idx: 27
top_k: 3 / 6 / 8
hot_window: 256
block_size: 128
```

offline env:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache
```

`retrieval_top_k_effective` は引き続き metadata-only のまま。実retrievalは既存の `--top-k` に従う。

```text
actual runtime:
  --hot-window
  --top-k

budget metadata:
  --recent-window-tokens
  --retrieval-top-k
  retrieval_top_k_effective
```

## 変更 / 追加ファイル

```text
scripts/make_3b_seq2048_layer27_topk_summary.py

results/raw/budget_metadata_cases/budget_tokens_seq2048_{repetitive,prose,structured}_layer27_qwen2p5_3b_topk{3,6,8}/

results/processed/budget_metadata_cases_budget_tokens_seq2048_{repetitive,prose,structured}_layer27_qwen2p5_3b_topk{3,6,8}.md
results/processed/budget_metadata_cases_budget_tokens_seq2048_{repetitive,prose,structured}_layer27_qwen2p5_3b_topk{3,6,8}.json

results/processed/budget_metadata_3b_seq2048_layer27_topk_summary.md
results/processed/budget_metadata_3b_seq2048_layer27_topk_summary.json
```

## summary table

```markdown
| model_name | prompt_type | layer_idx | seq_len | top_k | num_selected_blocks | selected_block_ids | working_ratio | coverage_ratio | candidate_k_len | working_k_len | cold_k_len | mean_abs_diff | max_abs_diff |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | 2048 | 3 | 3 | [11,12,13] | 0.312500000 | 0.214285714 | 384 | 640 | 1792 | 0.388702542 | 2.861755371 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | 2048 | 6 | 6 | [11,12,13,6,10,9] | 0.500000000 | 0.428571429 | 768 | 1024 | 1792 | 0.386756718 | 2.831975937 |
| Qwen/Qwen2.5-3B-Instruct | repetitive | 27 | 2048 | 8 | 8 | [11,12,13,6,10,9,8,7] | 0.625000000 | 0.571428571 | 1024 | 1280 | 1792 | 0.386246979 | 2.825435400 |
| Qwen/Qwen2.5-3B-Instruct | prose | 27 | 2048 | 3 | 3 | [11,13,12] | 0.312500000 | 0.214285714 | 384 | 640 | 1792 | 0.010569170 | 0.067580342 |
| Qwen/Qwen2.5-3B-Instruct | prose | 27 | 2048 | 6 | 6 | [11,13,12,6,10,8] | 0.500000000 | 0.428571429 | 768 | 1024 | 1792 | 0.005927631 | 0.029830217 |
| Qwen/Qwen2.5-3B-Instruct | prose | 27 | 2048 | 8 | 8 | [11,13,12,6,10,8,9,7] | 0.625000000 | 0.571428571 | 1024 | 1280 | 1792 | 0.004209773 | 0.017534435 |
| Qwen/Qwen2.5-3B-Instruct | structured | 27 | 2048 | 3 | 3 | [13,11,6] | 0.312500000 | 0.214285714 | 384 | 640 | 1792 | 0.037277322 | 0.198886633 |
| Qwen/Qwen2.5-3B-Instruct | structured | 27 | 2048 | 6 | 6 | [13,11,6,12,8,10] | 0.500000000 | 0.428571429 | 768 | 1024 | 1792 | 0.025780000 | |
| Qwen/Qwen2.5-3B-Instruct | structured | 27 | 2048 | 8 | 8 | [13,11,6,12,8,10,7,9] | 0.625000000 | 0.571428571 | 1024 | 1280 | 1792 | 0.022060000 | |
```

※ structured の top_k=6/8 の `max_abs_diff` は貼付ログ上で省略されていたため空欄。

## 主要結果

### prose は top_k 増加で改善

```text
top_k=3: 0.010569170
top_k=6: 0.005927631
top_k=8: 0.004209773
```

### structured も top_k 増加で改善

```text
top_k=3: 0.037277322
top_k=6: 0.02578
top_k=8: 0.02206
```

### repetitive は top_k=8でもほぼ改善しない

```text
top_k=3: 0.388702542
top_k=6: 0.386756718
top_k=8: 0.386246979
```

## coverage / working ratio

seq_len=2048ではcold側が大きくなる。

```text
cold_k_len = 1792
cold blocks = 14
block_size = 128
```

top-kごとのcoverageは以下。

```text
top_k=3:
  candidate_k_len = 384
  working_k_len = 640
  coverage_ratio = 0.214285714
  working_ratio = 0.3125

top_k=6:
  candidate_k_len = 768
  working_k_len = 1024
  coverage_ratio = 0.428571429
  working_ratio = 0.5

top_k=8:
  candidate_k_len = 1024
  working_k_len = 1280
  coverage_ratio = 0.571428571
  working_ratio = 0.625
```

1024実験では `top_k=6` で cold block full coverage になったが、2048では `top_k=8` でも coverage は約57%に留まる。

## selected block ids

```text
repetitive:
  top_k=3: [11,12,13]
  top_k=6: [11,12,13,6,10,9]
  top_k=8: [11,12,13,6,10,9,8,7]

prose:
  top_k=3: [11,13,12]
  top_k=6: [11,13,12,6,10,8]
  top_k=8: [11,13,12,6,10,8,9,7]

structured:
  top_k=3: [13,11,6]
  top_k=6: [13,11,6,12,8,10]
  top_k=8: [13,11,6,12,8,10,7,9]
```

## 解釈

```text
prose / structured:
  top_k増加で差分が下がる
  → retrieval budget制御で改善余地あり

repetitive:
  top_k=8 / coverage_ratio=0.5714 でも不十分
  → full cold coverageに近いworking setが必要になりやすい
```

つまり、RelayKVを「常に小さいtop_kで圧縮する」方向に寄せるのは危険。

特に `repetitive / layer27` のような高リスク条件では、partial retrievalではなく、full cold coverageに近いfallbackが必要になる可能性が高い。

## 設計上の含意

RelayKVの適用方針は以下が自然。

```text
1. full KVがVRAMに乗るならRelayKV off

2. VRAM pressureがある場合だけRelayKV on

3. prose / structured 系:
   budgetを増やせば改善するため、retrieval制御の価値あり

4. repetitive / high-risk layer:
   小さいpartial retrievalでは危険
   full cold coverageに近いfallbackが必要

5. layer14だけでは安全側に見えすぎるため、layer27を評価から外さない
```

## 変更していないもの

```text
scoring
attention comparison
retrieval block selection logic
--top-k の意味
KV tensor construction
retrieval_top_k_effective の実反映
```

今回行ったのは、既存runnerに異なる `--top-k` を渡した追加実行と、summary scriptの追加のみ。

## 確認結果

```text
compileall: pass
全9条件: pass
.github/workflows差分: なし
git diff --check: pass
network download: なし
```

## 結論

PyTorch側のbudget-first実験は、ここで一区切りにしてよい。

理由:

```text
1. Qwen2.5-3Bでseq_len=2048まで確認できた
2. layer27で高リスク条件を見られた
3. prose / structured はtop_k増加で改善する
4. repetitive はtop_k=8でもほぼ改善しない
5. 小さいpartial retrievalを常時適用する設計は危険
6. RelayKVはfull KVがVRAMに乗る場合はoffでよい
7. VRAM pressure時のみonにし、高リスク条件ではfallbackが必要
```

## 次フェーズ

次はPyTorch側でさらに細かく追うより、SGLang側のmemory manager設計へ戻る。

推奨するSGLang側の状態遷移:

```text
RelayKV off:
  full KVがVRAMに乗る
  latency / throughput上も問題ない

RelayKV shadow:
  full KVはまだ乗るが、VRAM pressureやbatch拡大の可能性がある
  budget metadata / block selection / risk signalだけ観測

RelayKV applied:
  full KVがVRAM budgetを超える、またはbatch targetのためにworking set制限が必要
  retrieval budgetを実適用

RelayKV fallback:
  high-risk prompt/layer、またはpartial retrievalでriskが高い場合
  working setを拡大する、またはfull cold coverageに近づける
```

今後の実装判断:

```text
1. SGLang側で full KV fits 判定を入れる
2. VRAM pressure時のみ shadow / applied へ進む
3. repetitive / layer27相当の高リスクケースでは fallback を検討する
4. retrieval_top_k_effective の実反映は optional flag で実装する
```
