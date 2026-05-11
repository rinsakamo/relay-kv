# RelayKV Devlog: VRAM Budget Schema v1

- Date basis: JST 2026-05-11
- Repository: `rinsakamo/relay-kv`
- Merged PR: #24 `Add RelayKV VRAM budget dry-run schema`
- Branch: `relaykv-vram-budget-schema-v1`
- Merge commit: `be1b463611cd0fdd074ffaa5bc377bb8f34ccc65`

## 1. Summary

RelayKV に `global / per-request VRAM budget dry-run schema v1` を追加した。

これにより RelayKV は、単なる selection 型 KV block 評価ではなく、モデル重みロード後の残余 VRAM を前提に decode-time working KV を budget 内へ収める **VRAM-aware KV residency controller** としての metadata layer を一段進めた。

今回の PR は metadata-only であり、runtime KV pool mutation、attention backend 接続、scheduler 変更、runtime writeback は行っていない。

## 2. 実装内容

追加・変更:

- `relaykv/vram_budget.py`
  - `RelayKVVramBudgetDecision`
  - `build_vram_budget_decision()`
- `relaykv/__init__.py`
  - VRAM budget decision builder を package root から export
- `scripts/run_relaykv_pipeline.py`
  - VRAM budget dry-run CLI を追加
  - JSON output に `vram_budget_decision` を追加
- `tests/test_vram_budget.py`
  - VRAM budget schema の unit tests を追加

CLI 追加:

- `--vram-budget-mode {off,dry_run}`
- `--global-residual-vram-mib`
- `--global-working-kv-budget-mib`
- `--target-concurrent-requests`
- `--allocation-policy {equal_share}`
- `--kv-dtype-bytes`
- `--num-layers`
- `--num-kv-heads`
- `--head-dim`

## 3. 設計

今回の schema は以下を分離して扱う。

- global budget
  - global residual VRAM
  - global working KV budget
- per-request budget
  - target concurrent requests に基づく request working KV budget
- KV shape estimate
  - dtype bytes
  - layer 数
  - KV head 数
  - head dim
  - block size
- derived budget
  - KV bytes per token
  - KV bytes per block
  - derived target keep blocks

計算式:

```text
kv_bytes_per_token = 2 * kv_dtype_bytes * num_layers * num_kv_heads * head_dim
kv_bytes_per_block = kv_bytes_per_token * block_size
request_working_kv_budget_mib = global_working_kv_budget_mib / target_concurrent_requests
derived_target_keep_blocks = floor(request_budget_bytes / kv_bytes_per_block)
```

`2` は K/V の 2 本分。

allocation policy は v1 では `equal_share` のみ。

## 4. JSON metadata

`run_relaykv_pipeline.py` の output に `vram_budget_decision` を追加した。

off mode:

```json
{
  "vram_budget_decision": null
}
```

dry-run mode の observed example:

```json
{
  "global_residual_vram_mib": null,
  "global_working_kv_budget_mib": 128.0,
  "target_concurrent_requests": 2,
  "request_working_kv_budget_mib": 64.0,
  "allocation_policy": "equal_share",
  "kv_dtype_bytes": 2,
  "num_layers": 28,
  "num_kv_heads": 2,
  "head_dim": 128,
  "block_size": 128,
  "kv_bytes_per_token": 28672,
  "kv_bytes_per_block": 3670016,
  "derived_target_keep_blocks": 18,
  "budget_ok": true,
  "fallback_reason": null,
  "dry_run_only": true
}
```

## 5. Validation

実行結果:

```text
python -m pytest tests/test_vram_budget.py -q
12 passed in 0.94s

python -m pytest -q
14 passed in 3.40s
```

Smoke:

```text
python scripts/run_relaykv_pipeline.py   --seq-len 512   --hot-window 128   --block-size 128   --top-k 1   --layer-idx 0   --prompt-type structured   --vram-budget-mode off   --output relaykv_vram_budget_off_smoke.json
```

結果:

```json
{
  "has_vram_budget_decision": true,
  "vram_budget_decision": null
}
```

Smoke:

```text
python scripts/run_relaykv_pipeline.py   --seq-len 512   --hot-window 128   --block-size 128   --top-k 1   --layer-idx 0   --prompt-type structured   --vram-budget-mode dry_run   --global-working-kv-budget-mib 128   --target-concurrent-requests 2   --kv-dtype-bytes 2   --output relaykv_vram_budget_dry_run_smoke.json
```

結果:

```text
request_working_kv_budget_mib = 64.0
kv_bytes_per_token = 28672
kv_bytes_per_block = 3670016
derived_target_keep_blocks = 18
budget_ok = true
dry_run_only = true
```

`derived_target_keep_blocks = 18` は以下と整合する。

```text
64 MiB / 3,670,016 bytes per block = 約18.28 blocks
floor = 18
```

## 6. 意義

これで RelayKV の policy chain は以下の形に近づいた。

```text
global residual VRAM budget
  -> global working KV budget
  -> per-request working KV budget
  -> derived target keep blocks
  -> activation policy
  -> demotion policy
  -> future materialization / retrieval policy
```

現時点では `derived_target_keep_blocks` は metadata として算出するだけで、demotion policy へは未接続。

この分離により、今後は以下のような budget source を明示できる。

- CLI explicit target keep blocks
- VRAM-derived target keep blocks
- future dynamic pressure-derived target keep blocks
- per-request concurrency-adjusted target keep blocks

## 7. 制約

今回の PR で行っていないこと:

- runtime KV pool mutation
- attention backend connection
- scheduler decision change
- runtime writeback
- SGLang / vLLM adapter change
- demotion policy への自動適用
- results/raw または results/processed の commit

## 8. 次の候補

次 PR 候補:

```text
relaykv-vram-budget-to-demotion-dry-run-v1
```

目的:

- `vram_budget_decision.derived_target_keep_blocks` を `demotion_policy` の `target_keep_blocks` に dry-run 接続する
- 明示 CLI の `--target-keep-blocks` がある場合は CLI を優先する
- VRAM budget 由来の場合は JSON に source を残す
  - 例: `target_keep_blocks_source = "vram_budget"`
- runtime demotion はまだ行わない
- `demotion_applied=false`
- `dry_run_only=true`

候補 metadata:

```text
target_keep_blocks_source:
  - explicit_cli
  - vram_budget
  - unset

effective_target_keep_blocks:
  - int | null

vram_budget_to_demotion_connected:
  - bool
```

## 9. Commit reference

Merged PR:

```text
#24 Add RelayKV VRAM budget dry-run schema
```

Merge commit:

```text
be1b463611cd0fdd074ffaa5bc377bb8f34ccc65
```
