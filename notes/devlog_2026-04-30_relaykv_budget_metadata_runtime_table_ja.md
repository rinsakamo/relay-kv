# Devlog: RelayKV budget metadata runtime smoke / table support

## 日付

2026-04-30

## 対象repo

- repo: `~/work/relay-kv`
- branch: `budget-planner-metadata`

## 目的

RelayKV PyTorch実験repo側で、budget-first metadata が実 pipeline JSON に入り、さらに table 出力で確認できるところまで固定した。

今回の目的は、`retrieval_top_k_effective` を実挙動に反映することではなく、まず **metadata-onlyで安全に可視化できる状態** を作ること。

## 到達点

```text
budget planner pure function:
  完了

run_relaykv_pipeline.py の JSON metadata:
  完了

budget sweep script:
  完了

runtime smoke JSONで budget fields 確認:
  完了

single JSON table化:
  完了

既存 --top-k retrieval挙動:
  変更なし
```

## 変更ファイル

今回の最終差分:

```text
scripts/make_layer_budget_table.py
```

追加済みの関連ファイル / 既存push済み内容:

```text
relaykv/budget_planner.py
relaykv/__init__.py
scripts/run_relaykv_pipeline.py
scripts/relaykv_budget_sweep.py
README.md
results/processed/relaykv_budget_sweep_2026-04-30.md
notes/devlog_2026-04-30_relaykv_pytorch_budget_planner_integration_ja.md
```

## 追加内容

### 1. requested budget columns の追加

既存 layer budget table に、budget-first評価用の列を追加した。

主な列:

```text
kv_working_budget_tokens
recent_window_tokens
budget_block_size
anchor_blocks
anchor_budget_tokens
retrieval_budget_tokens
retrieval_block_budget
retrieval_top_k_requested
retrieval_top_k_effective
budget_overflow
budget_policy_reason
top_k
num_selected_blocks
working_ratio
mean_abs_diff
```

### 2. --single-json の追加

`/tmp/relaykv_pipeline_budget_smoke.json` のような単一 pipeline JSON を table 化できるようにした。

これにより、layer0/layer14/layer27 の既存比較モードに加えて、単発pipeline smokeのmetadata確認ができる。

### 3. 既存比較モードの維持

既存の layer0/layer14/layer27 比較モードは維持した。

## runtime smoke確認

offline modeで最小pipelineを実行し、budget fields が output JSON に入ることを確認した。

実行条件:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
seq_len = 32
hot_window = 8
block_size = 8
top_k = 1
layer_idx = 0
kv_working_budget_tokens = 16
recent_window = 8
budget_block_size = 8
anchor_blocks = 1
retrieval_top_k = 2
```

確認された output JSON の budget fields:

```json
{
  "top_k": 1,
  "num_selected_blocks": 1,
  "retrieval_top_k": 2,
  "kv_working_budget_tokens": 16,
  "recent_window_tokens": 8,
  "budget_block_size": 8,
  "anchor_blocks": 1,
  "anchor_budget_tokens": 8,
  "retrieval_budget_tokens": 0,
  "retrieval_block_budget": 0,
  "retrieval_top_k_requested": 2,
  "retrieval_top_k_effective": 0,
  "budget_overflow": true,
  "budget_policy_reason": "no_retrieval_room_after_recent_and_anchor"
}
```

## 解釈

この結果は期待通り。

```text
実retrieval:
  --top-k 1
  num_selected_blocks = 1

budget metadata:
  retrieval_top_k_requested = 2
  retrieval_top_k_effective = 0
  retrieval_budget_tokens = 0
```

つまり、budget planner は「recent + anchor でworking budgetを使い切るため retrieval枠なし」と判断している。

一方で、既存pipelineの実retrievalはまだ `--top-k 1` のまま動く。

これは意図通りで、現段階では `retrieval_top_k_effective` は metadata として出すだけで、既存挙動には反映していない。

## table smoke確認

確認コマンド:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache \
.venv/bin/python scripts/make_layer_budget_table.py \
  --single-json /tmp/relaykv_pipeline_budget_smoke.json \
  --output-md /tmp/relaykv_budget_table_smoke.md \
  --output-json /tmp/relaykv_budget_table_smoke.json
```

出力table例:

```markdown
| plan | kv_working_budget_tokens | recent_window_tokens | budget_block_size | anchor_blocks | anchor_budget_tokens | retrieval_budget_tokens | retrieval_block_budget | retrieval_top_k_requested | retrieval_top_k_effective | budget_overflow | budget_policy_reason | top_k | num_selected_blocks | working_ratio | mean_abs_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| relaykv_pipeline_budget_smoke | 16 | 8 | 8 | 1 | 8 | 0 | 0 | 2 | 0 | True | no_retrieval_room_after_recent_and_anchor | 1 | 1 | 0.500000000 | 0.000000013 |
```

## 確認コマンド

```bash
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python -m compileall relaykv scripts
```

```bash
git diff --name-status | grep '.github/workflows' || true
```

```bash
git diff --check
```

結果:

```text
compileall: pass
.github/workflows 差分: なし
git diff --check: pass
network download: なし
```

## 変更していないもの

```text
scoring
attention差し替え
retrieval top-k の実選択
KV tensor materialization
CPU copy
GPU free
host backup dry-copy
network-dependent model download
```

## 現在の評価

```text
metadata-only budget evaluation:
  成功

single JSON smoke table:
  成功

既存実験との互換性:
  維持

次に retrieval_top_k_effective を実挙動に反映する準備:
  まだ未実施
```

## commit候補

```bash
cd ~/work/relay-kv

git status --short
git diff --name-status
git diff --name-status | grep '.github/workflows' || true
git diff --check

git add scripts/make_layer_budget_table.py

git commit -m "Add single JSON budget table support"

git push
```

devlogもrepoへ入れる場合:

```bash
cd ~/work/relay-kv

cp /mnt/data/devlog_2026-04-30_relaykv_budget_metadata_runtime_table_ja.md \
  notes/devlog_2026-04-30_relaykv_budget_metadata_runtime_table_ja.md

git add notes/devlog_2026-04-30_relaykv_budget_metadata_runtime_table_ja.md

git commit -m "Add RelayKV budget metadata table devlog"

git push
```

## 次の候補

次の実装候補は、まだ `retrieval_top_k_effective` の実反映ではなく、以下が安全。

```text
1. 実験結果テーブルに budget columns を継続的に含める
2. budget制約ごとの quality metrics を比較する実験設計を固める
3. その後に optional flag で retrieval_top_k_effective を実retrievalに反映する
```

推奨は、次に以下の比較を作ること。

```text
metadata-only:
  --top-k fixed

budget-applied optional:
  --top-k := retrieval_top_k_effective
```

ただし、これは既存比較軸を壊すため、必ず optional flag として導入する。
