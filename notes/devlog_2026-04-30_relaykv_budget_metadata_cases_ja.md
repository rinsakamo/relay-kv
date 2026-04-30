# Devlog: RelayKV budget metadata cases

## 日付

2026-04-30

## 対象repo

- repo: `~/work/relay-kv`
- branch: `budget-planner-metadata`

## 目的

RelayKV PyTorch実験repo側で、budget-first評価を **metadata-only** で複数case実行できるようにした。

この段階の目的は、実際の retrieval / attention / scoring 挙動を変えずに、以下を同じ表で確認できるようにすること。

```text
actual runtime parameter:
  --hot-window
  --top-k

budget evaluation metadata:
  --recent-window-tokens
  --retrieval-top-k
  retrieval_top_k_effective
```

これにより、実挙動を固定したまま、budget設計だけを比較できる。

## 到達点

```text
budget planner pure function:
  完了

pipeline JSON metadata:
  完了

--recent-window-tokens による metadata-only recent 分離:
  完了

budget sweep script:
  完了

single JSON budget table:
  完了

metadata-only budget case runner:
  完了

results/raw/budget_metadata_cases/*.json:
  作成済み

results/processed/budget_metadata_cases_table.md/json:
  作成済み
```

## 変更ファイル

```text
scripts/run_budget_metadata_cases.py
scripts/run_relaykv_pipeline.py
results/raw/budget_metadata_cases/*.json
results/processed/budget_metadata_cases_table.md
results/processed/budget_metadata_cases_table.json
```

## 実行case

```text
tiny_16
small_32
medium_64
mib_512
```

## table出力

```markdown
| plan | kv_working_budget_tokens | recent_window_tokens | budget_block_size | anchor_blocks | anchor_budget_tokens | retrieval_budget_tokens | retrieval_block_budget | retrieval_top_k_requested | retrieval_top_k_effective | budget_overflow | budget_policy_reason | top_k | num_selected_blocks | working_ratio | mean_abs_diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| tiny_16 | 16 | 8 | 8 | 1 | 8 | 0 | 0 | 2 | 0 | True | no_retrieval_room_after_recent_and_anchor | 1 | 1 | 0.500000000 | 0.000006579 |
| small_32 | 32 | 8 | 8 | 1 | 8 | 16 | 2 | 2 | 2 | False | explicit_working_budget_tokens | 1 | 1 | 0.500000000 | 0.000006579 |
| medium_64 | 64 | 16 | 8 | 1 | 8 | 40 | 5 | 4 | 4 | False | explicit_working_budget_tokens | 1 | 1 | 0.500000000 | 0.000006579 |
| mib_512 | 18724 | 768 | 128 | 4 | 512 | 17444 | 136 | 8 | 8 | False | estimated_from_available_kv_budget_mib | 1 | 1 | 0.500000000 | 0.000006579 |
```

## 解釈

### tiny_16

```text
working = 16
recent = 8
anchor = 8
retrieval = 0
retrieval_top_k_effective = 0
budget_overflow = True
```

recent + anchor で budget を使い切るため、budget planner上は retrieval room がない。

一方、実pipelineでは `--top-k 1` のため `num_selected_blocks=1` のまま。

これは metadata-only段階では意図通り。

### small_32

```text
working = 32
recent = 8
anchor = 8
retrieval = 16
retrieval_block_budget = 2
retrieval_top_k_effective = 2
budget_overflow = False
```

budget planner上は requested retrieval top-k 2 が入る。

実pipelineはまだ `--top-k 1` 固定。

### medium_64

```text
working = 64
recent = 16
anchor = 8
retrieval = 40
retrieval_block_budget = 5
retrieval_top_k_effective = 4
budget_overflow = False
```

budget planner上は requested retrieval top-k 4 が入る。

実pipelineはまだ `--top-k 1` 固定。

### mib_512

```text
working = 18724
recent = 768
anchor = 512
retrieval = 17444
retrieval_block_budget = 136
retrieval_top_k_effective = 8
budget_overflow = False
```

512MiB budgetをQwen2.5-1.5B想定の `kv_bytes_per_token=28672` で token換算したcase。

この短い smoke pipelineでは実runtimeは `seq_len=32`, `hot_window=8` のまま維持している。

## --recent-window-tokens を追加した理由

`scripts/run_relaykv_pipeline.py` に `--recent-window-tokens` を追加した。

理由は、既存の `--recent-window` が実pipelineの `--hot-window` alias になっていたため。

今回の `mib_512` case では、以下を同時に表現したかった。

```text
actual runtime:
  seq_len = 32
  hot_window = 8

budget metadata:
  recent_window_tokens = 768
  available_kv_budget_mib = 512
```

もし既存の `--recent-window` を使うと、実pipelineの hot/cold split, retrieval, attention比較に影響してしまう。

そのため、budget planner metadata専用の `--recent-window-tokens` を追加し、actual runtime と budget metadata を分離した。

## 既存挙動への影響

Codex確認結果:

```text
scoring:
  影響なし

attention comparison:
  影響なし

retrieval block selection:
  影響なし

--top-k behavior:
  影響なし

KV tensor construction:
  影響なし
```

実際のretrieval選択は引き続き以下。

```text
top_scores = top_k_blocks(scores, k=min(top_k, len(scores)))
```

`retrieval_top_k_effective` はまだ実retrievalには反映していない。

## 確認コマンド

```bash
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python -m compileall relaykv scripts
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python scripts/run_budget_metadata_cases.py
```

```bash
head -n 20 results/processed/budget_metadata_cases_table.md
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
budget metadata cases: pass
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

今回の作業で、RelayKV PyTorch側に以下の評価軸が整った。

```text
1. budget planner単体のsweep
2. 実pipeline JSONへのbudget metadata埋め込み
3. 単一JSONのbudget table化
4. 複数budget caseの一括実行
5. actual runtimeとbudget metadataの分離
```

ここまでで、metadata-only budget評価の基礎は完成。

## 次の候補

次は、まだ `retrieval_top_k_effective` を実挙動に反映せず、以下に進むのが安全。

```text
1. 同じcase runnerを seq_len=1024/2048/4096 に拡張する
2. prompt_type別に budget metadata cases を出す
3. mean_abs_diff / working_ratio / first_divergence_step を budget tableに接続する
4. その後、optional flag で retrieval_top_k_effective を実retrievalへ反映する
```

推奨する次ステップ:

```text
budget metadata quality sweep:
  seq_len: 1024, 2048, 4096
  prompt_type: repetitive, prose, structured
  layer_idx: 0, 14, 27
  actual top_k: fixed
  budget metadata: variable
```

この段階でも、`retrieval_top_k_effective` は実挙動へ反映しない。
