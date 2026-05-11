# RelayKV Devlog: VRAM Budget Schema → Demotion Dry-Run Connection v1

- Date basis: JST 2026-05-11
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #24: `Add RelayKV VRAM budget dry-run schema`
  - PR #25: `Add devlog for RelayKV VRAM budget schema`
  - PR #26: `Connect VRAM budget target to demotion dry run`

## 1. Summary

RelayKV の設計転換である **VRAM-aware KV residency controller** の metadata path が一段進んだ。

今回の節目では、まず global / per-request VRAM budget から request 単位の working KV budget と `derived_target_keep_blocks` を算出する schema を追加し、その後、その `derived_target_keep_blocks` を demotion dry-run の `target_keep_blocks` に接続した。

これにより、以下の metadata chain が成立した。

```text
global working KV budget
  -> per-request working KV budget
  -> KV bytes per token / block
  -> derived_target_keep_blocks
  -> demotion_target_resolution
  -> demotion_policy_decision.target_keep_blocks
```

ただし、今回も runtime 適用は行っていない。

- runtime KV pool mutation なし
- attention backend connection なし
- scheduler change なし
- runtime writeback なし
- demotion_applied=false
- dry_run_only=true

## 2. PR #24: VRAM Budget Dry-Run Schema

PR #24 では `relaykv/vram_budget.py` を追加し、VRAM budget から per-request KV budget と keep block 数を導出する metadata-only schema を実装した。

追加内容:

- `RelayKVVramBudgetDecision`
- `build_vram_budget_decision()`
- `run_relaykv_pipeline.py` の `vram_budget_decision` JSON 出力
- `tests/test_vram_budget.py`

主要 CLI:

```text
--vram-budget-mode {off,dry_run}
--global-residual-vram-mib
--global-working-kv-budget-mib
--target-concurrent-requests
--allocation-policy {equal_share}
--kv-dtype-bytes
--num-layers
--num-kv-heads
--head-dim
```

計算式:

```text
kv_bytes_per_token = 2 * kv_dtype_bytes * num_layers * num_kv_heads * head_dim
kv_bytes_per_block = kv_bytes_per_token * block_size
request_working_kv_budget_mib = global_working_kv_budget_mib / target_concurrent_requests
derived_target_keep_blocks = floor(request_budget_bytes / kv_bytes_per_block)
```

Observed:

```text
request_working_kv_budget_mib = 64.0
kv_bytes_per_token = 28672
kv_bytes_per_block = 3670016
derived_target_keep_blocks = 18
budget_ok = true
dry_run_only = true
```

Validation:

```text
python -m pytest tests/test_vram_budget.py -q
12 passed

python -m pytest -q
14 passed
```

## 3. PR #25: VRAM Budget Schema Devlog

PR #25 では、PR #24 の内容を `notes/devlog_2026-05-11_relaykv_vram_budget_schema_v1.ja.md` として repo 側にも記録した。

記録した主な内容:

- RelayKV を VRAM-aware KV residency controller として扱う設計方針
- `vram_budget_decision` schema
- `equal_share` allocation
- KV bytes/token と KV bytes/block の推定
- `derived_target_keep_blocks=18` の smoke 結果
- 次段階として demotion dry-run への接続を予定

## 4. PR #26: VRAM Budget Target → Demotion Dry-Run

PR #26 では、PR #24 で算出した `vram_budget_decision.derived_target_keep_blocks` を demotion dry-run の `target_keep_blocks` に接続した。

追加された metadata:

```json
{
  "demotion_target_resolution": {
    "effective_target_keep_blocks": 18,
    "target_keep_blocks_source": "vram_budget",
    "fallback_reason": null,
    "vram_budget_to_demotion_connected": true
  }
}
```

基本ルール:

```text
1. explicit --target-keep-blocks があれば最優先
   -> target_keep_blocks_source = "explicit_cli"

2. explicit target がなく、
   --vram-budget-mode=dry_run かつ budget_ok=true なら
   vram_budget_decision.derived_target_keep_blocks を使用
   -> target_keep_blocks_source = "vram_budget"

3. demotion_policy_mode=off の場合は demotion に接続しない
   -> target_keep_blocks_source = "demotion_policy_off"
   -> vram_budget_to_demotion_connected = false
   -> demotion_policy_decision = null

4. demotion dry-run で target が解決できない場合は失敗
   -> build_demotion_decision(target_keep_blocks=None) に進ませない
```

## 5. Review Fixes in PR #26

PR #26 では Codex review により、metadata semantics / validation に関する重要な修正を行った。

### 5.1 target 未解決の demotion dry-run を成功扱いしない

問題:

```text
--demotion-policy-mode=dry_run
--target-keep-blocks なし
vram budget usable target なし
```

この状態で `build_demotion_decision(target_keep_blocks=None)` に進むと、全 block keep / `budget_ok=true` の no-op 成功に見えてしまう。

修正:

- `effective_target_keep_blocks is None` の場合は `build_demotion_decision()` に進ませない
- runtime guard で `ValueError` を出す

### 5.2 no-target / no-VRAM-budget は model load 前に拒否

問題:

runtime guard だけだと、明らかに失敗する CLI でも model load / forward 後に失敗する。

修正:

```text
--demotion-policy-mode=dry_run
--target-keep-blocks なし
--vram-budget-mode off
```

このケースは argparse 段階で失敗させる。

Observed:

```text
exited nonzero with code 2
message:
demotion dry-run requires --target-keep-blocks or --vram-budget-mode=dry_run
device=cpu / device=cuda は出力されない
```

### 5.3 demotion off の VRAM-budget-only run を connected=true にしない

問題:

```text
--vram-budget-mode=dry_run
--demotion-policy-mode=off
```

この VRAM-budget-only 実験で `vram_budget_to_demotion_connected=true` が出ると、demotion decision がないのに接続済みと誤読できる。

修正:

demotion off の場合:

```json
{
  "demotion_target_resolution": {
    "effective_target_keep_blocks": null,
    "target_keep_blocks_source": "demotion_policy_off",
    "fallback_reason": null,
    "vram_budget_to_demotion_connected": false
  },
  "demotion_policy_decision": null
}
```

## 6. Validation Summary

PR #26 final validation:

```text
python -m pytest -q
24 passed in 4.34s
```

Positive demotion dry-run smoke:

```text
python scripts/run_relaykv_pipeline.py \
  --seq-len 256 \
  --hot-window 128 \
  --block-size 128 \
  --top-k 1 \
  --layer-idx 0 \
  --prompt-type structured \
  --demotion-policy-mode dry_run \
  --vram-budget-mode dry_run \
  --global-working-kv-budget-mib 128 \
  --target-concurrent-requests 2 \
  --kv-dtype-bytes 2 \
  --output relaykv_vram_budget_to_demotion_smoke_seq256.json
```

Observed:

```text
derived_target_keep_blocks = 18
demotion_target_resolution.target_keep_blocks_source = "vram_budget"
demotion_policy_decision.target_keep_blocks = 18
demotion_policy_decision.dry_run_only = true
demotion_policy_decision.demotion_applied = false
```

Early negative CLI check:

```text
python scripts/run_relaykv_pipeline.py \
  --seq-len 256 \
  --hot-window 128 \
  --block-size 128 \
  --top-k 1 \
  --layer-idx 0 \
  --prompt-type structured \
  --demotion-policy-mode dry_run \
  --output relaykv_demotion_target_unset_should_fail.json
```

Observed:

```text
exited nonzero with code 2
argparse usage printed
demotion dry-run requires --target-keep-blocks or --vram-budget-mode=dry_run
device=cpu / device=cuda was not printed
```

VRAM-budget-only smoke with demotion off:

```text
python scripts/run_relaykv_pipeline.py \
  --seq-len 256 \
  --hot-window 128 \
  --block-size 128 \
  --top-k 1 \
  --layer-idx 0 \
  --prompt-type structured \
  --vram-budget-mode dry_run \
  --global-working-kv-budget-mib 128 \
  --target-concurrent-requests 2 \
  --kv-dtype-bytes 2 \
  --output relaykv_vram_budget_only_no_demotion_smoke_seq256.json
```

Observed:

```text
vram_budget_decision.derived_target_keep_blocks = 18
demotion_target_resolution.target_keep_blocks_source = "demotion_policy_off"
demotion_target_resolution.vram_budget_to_demotion_connected = false
demotion_policy_decision = null
```

## 7. Current Design State

現時点の RelayKV dry-run metadata path:

```text
activation_policy_decision
  - diagnostic / practical activation gating

vram_budget_decision
  - global / per-request working KV budget
  - derived_target_keep_blocks

demotion_target_resolution
  - explicit_cli / vram_budget / demotion_policy_off
  - effective_target_keep_blocks
  - vram_budget_to_demotion_connected

demotion_policy_decision
  - RECENT_PROTECTED
  - BOUNDARY_NEAR_RECENT
  - PREFIX_PROTECTED only when explicit
  - EVICTION_CANDIDATE
  - DEMOTE_OLDEST
  - demotion_applied=false
```

これで、VRAM budget 由来の working KV target を demotion dry-run に安全に渡せるようになった。

## 8. Remaining Constraints

今回も以下は行っていない。

- runtime KV pool mutation
- attention backend connection
- scheduler decision change
- runtime writeback
- SGLang / vLLM adapter change
- actual demotion materialization
- results/raw または results/processed の commit

## 9. Next Candidate

次の候補は、まだ runtime に進まず、metadata / sweep 側を広げるのが自然。

候補 branch:

```text
relaykv-vram-budget-demotion-sweep-v1
```

目的:

- global working KV budget を複数値で sweep
- target_concurrent_requests を複数値で sweep
- derived_target_keep_blocks の変化を見る
- demotion_policy_decision の keep/drop block 変化を見る
- attention diff を比較する
- practical baseline として:
  - explicit target
  - VRAM-derived target
  - demotion off
  - short-context disabled
  を比較可能にする

または、より小さく進めるなら:

```text
relaykv-demotion-target-summary-v1
```

目的:

- `demotion_target_resolution` / `vram_budget_decision` / `demotion_policy_decision` を読みやすく summary する script を追加
- results を commit せず、diagnostic output の jq 依存を減らす

## 10. Commit / PR References

PR #24:

```text
Add RelayKV VRAM budget dry-run schema
merge_commit_sha = be1b463611cd0fdd074ffaa5bc377bb8f34ccc65
```

PR #25:

```text
Add devlog for RelayKV VRAM budget schema
merge_commit_sha = d51daa51264d75b72948c2ecebcc5a94b5d98280
```

PR #26:

```text
Connect VRAM budget target to demotion dry run
merge_commit_sha = 495429dd8b7c801971626701cb6deb1e6d37dfd4
```
