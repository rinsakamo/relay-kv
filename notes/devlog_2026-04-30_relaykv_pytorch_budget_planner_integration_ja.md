# RelayKV PyTorch Budget Planner Integration Devlog

Date: 2026-04-30

## 目的

SGLang 側で整理した RelayKV budget-first 方針を、PyTorch 側 `relay-kv`
評価スクリプトにも反映した。評価軸を単純な `seq_len` / `coverage_ratio`
中心から、残 VRAM 制約下で `recent + anchor + retrieved cold blocks` を
どう配分するかが追える metadata 中心へ拡張した。

## 変更ファイル

- `relaykv/budget_planner.py`
- `relaykv/__init__.py`
- `scripts/run_relaykv_pipeline.py`
- `scripts/make_layer_budget_table.py`
- `scripts/relaykv_budget_sweep.py`
- `README.md`
- `results/processed/relaykv_budget_sweep_2026-04-30.md`

## Budget Planner 式

`kv_working_budget_tokens` が明示されている場合はそれを優先する。
未指定の場合は `available_kv_budget_mib` と `kv_bytes_per_token` から推定する。

```text
kv_working_budget_tokens =
  explicit_tokens
  or floor(available_kv_budget_mib * 1024 * 1024 / kv_bytes_per_token)

recent_window_tokens = min(requested_recent_window_tokens, kv_working_budget_tokens)
anchor_budget_tokens = min(anchor_blocks * budget_block_size,
                           max(0, kv_working_budget_tokens - recent_window_tokens))
retrieval_budget_tokens = max(0,
                              kv_working_budget_tokens
                              - recent_window_tokens
                              - anchor_budget_tokens)
retrieval_block_budget = floor(retrieval_budget_tokens / budget_block_size)
retrieval_top_k_effective = min(retrieval_top_k_requested, retrieval_block_budget)
```

## 1024 Token Budget Case

確認条件:

```text
kv_working_budget_tokens = 1024
recent_window_tokens = 768
anchor_blocks = 4
budget_block_size = 128
retrieval_top_k_requested = 8
```

確認結果:

```text
anchor_budget_tokens = 256
retrieval_budget_tokens = 0
retrieval_block_budget = 0
retrieval_top_k_effective = 0
budget_overflow = true
budget_policy_reason = anchor_budget_clipped_after_recent_window
```

## 512 MiB Budget Case

確認条件:

```text
available_kv_budget_mib = 512
kv_bytes_per_token = 28672
recent_window_tokens = 768
anchor_blocks = 4
budget_block_size = 128
retrieval_top_k_requested = 8
```

確認結果:

```text
kv_working_budget_tokens = 18724
kv_working_budget_source = estimated_from_available_kv_budget_mib
anchor_budget_tokens = 512
retrieval_budget_tokens = 17444
retrieval_block_budget = 136
retrieval_top_k_effective = 8
budget_overflow = false
budget_policy_reason = estimated_from_available_kv_budget_mib
```

## 既存 Retrieval 挙動

既存の `--top-k` による retrieval 選択は変更していない。
`--retrieval-top-k` は budget metadata 用の requested value として扱い、
現時点では scoring / retrieval / attention 差し替え挙動には反映していない。

## 実行した確認コマンド

この環境では `python` が PATH に存在しなかったため、同じ repo の
`.venv/bin/python` を使用した。また既存 `__pycache__` が read-only のため、
`PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache` を指定した。

```bash
git branch --show-current
git status --short
PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python -m compileall relaykv scripts
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python scripts/relaykv_budget_sweep.py | head -n 40
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/relaykv_pycache .venv/bin/python scripts/relaykv_budget_sweep.py > results/processed/relaykv_budget_sweep_2026-04-30.md
git diff --name-status | grep '.github/workflows' || true
git diff --check
```

## 未実施

- network model smoke は未実施。
- Hugging Face model download は未実施。
- attention 変更は未実施。
- CPU copy 実装変更は未実施。
- GPU free 実装変更は未実施。
