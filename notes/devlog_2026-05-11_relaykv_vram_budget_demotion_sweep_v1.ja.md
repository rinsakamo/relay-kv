# RelayKV Devlog: VRAM Budget Demotion Sweep v1

- Date basis: JST 2026-05-11
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #28: `Add VRAM budget demotion sweep`
- Branch:
  - `relaykv-vram-budget-demotion-sweep-v1`
- Merge commit:
  - `d6600f49bf14bf3700a5aac816871a1c4ceccfb6`

## 1. Summary

RelayKV の VRAM-aware KV residency controller 化に向けて、`vram_budget_decision` と `demotion_policy_decision` を複数条件で横断評価する diagnostic-only sweep を追加した。

これにより、単発 pipeline ではなく、以下の関係を一覧で確認できるようになった。

```text
global_working_kv_budget_mib
target_concurrent_requests
  -> request_working_kv_budget_mib
  -> derived_target_keep_blocks
  -> demotion_target_resolution
  -> demotion_policy_decision.keep/drop
  -> attention diff
```

今回も runtime 適用は行っていない。

- runtime KV pool mutation なし
- attention backend connection なし
- scheduler change なし
- runtime writeback なし
- demotion_applied=false
- diagnostic-only

## 2. Added Files

追加ファイル:

```text
scripts/run_vram_budget_demotion_sweep.py
tests/test_vram_budget_demotion_sweep.py
```

`run_vram_budget_demotion_sweep.py` は、model load / forward を1回だけ行い、その取得済み KV に対して複数の VRAM budget / concurrency case を評価する。

主な reuse:

- `load_model`
- `make_prompt_for_target_tokens`
- `build_vram_budget_decision`
- `resolve_demotion_target_resolution`
- `require_usable_demotion_target_for_dry_run`
- `build_demotion_decision`
- `ColdSegment -> to_blocks`
- `retrieve_blocks_by_ids`
- `build_candidate_kv`
- `compare_attention_outputs`

## 3. CLI

主要 CLI:

```text
--model
--seq-len
--block-size
--layer-idx
--prompt-type
--global-working-kv-budget-mib-list
--target-concurrent-requests-list
--global-residual-vram-mib
--kv-dtype-bytes
--demotion-recent-blocks
--protect-boundary-blocks
--protect-prefix-blocks
--output-json
--output-md
```

list 型は comma-separated で指定する。

例:

```text
--global-working-kv-budget-mib-list 32,64,128
--target-concurrent-requests-list 1,2
```

## 4. Output

出力先:

```text
results/processed/vram_budget_demotion_sweep*.json
results/processed/vram_budget_demotion_sweep*.md
```

ただし、これらは validation output であり commit 対象ではない。

case summary には主に以下を記録する。

```text
case_name
global_working_kv_budget_mib
target_concurrent_requests
request_working_kv_budget_mib
derived_target_keep_blocks
keep_block_ids
drop_block_ids
working_ratio
mean_abs_diff
max_abs_diff
fallback_reason
error
vram_budget_decision
demotion_target_resolution
demotion_policy_decision
candidate_kv
attention_compare
```

## 5. Implementation Notes

今回の重要点は、`run_drop_policy_sweep.py` と同じ思想で **model-load-once / forward-once** にしたこと。

単発 pipeline を budget case ごとに複数回起動すると、HF model load / CUDA / WSL2 / VRAM が競合しやすい。今回の sweep script は、1回の forward で得た FullKV を使い、case ごとの demotion dry-run と attention comparison を行う。

case ごとの処理:

```text
1. build_vram_budget_decision()
2. resolve_demotion_target_resolution(demotion_policy_mode="dry_run")
3. require_usable_demotion_target_for_dry_run()
4. build_demotion_decision()
5. keep_block_ids で candidate KV を構築
6. full attention と approx attention を compare
7. JSON / Markdown に記録
```

case が失敗した場合も、そこで sweep 全体を止めずに error として summary に記録し、最後に nonzero exit する設計。

## 6. Validation

PR #28 final validation:

```text
python -m py_compile scripts/run_vram_budget_demotion_sweep.py
passed

python -m pytest -q
27 passed in 4.62s
```

## 7. Smoke 1: seq_len=256

Command:

```text
python scripts/run_vram_budget_demotion_sweep.py \
  --seq-len 256 \
  --block-size 128 \
  --layer-idx 0 \
  --prompt-type structured \
  --global-working-kv-budget-mib-list 32,64,128 \
  --target-concurrent-requests-list 1,2 \
  --kv-dtype-bytes 2 \
  --output-json results/processed/vram_budget_demotion_sweep_seq256.json \
  --output-md results/processed/vram_budget_demotion_sweep_seq256.md
```

Observed:

```text
total_blocks = 2
derived_target_keep_blocks varied by budget/concurrency
all cases fallback_reason = fullkv_within_budget
keep_block_ids = [0, 1]
drop_block_ids = []
working_ratio = 1.0
mean_abs_diff = 0.0
max_abs_diff = 0.0
```

Interpretation:

`seq_len=256` / `block_size=128` では total block が2つしかないため、今回の budget ではすべて FullKV within budget になった。これは short-context safety / no-op path の確認として有用。

## 8. Smoke 2: seq_len=1024 tight budget

Command:

```text
python scripts/run_vram_budget_demotion_sweep.py \
  --seq-len 1024 \
  --block-size 128 \
  --layer-idx 0 \
  --prompt-type structured \
  --global-working-kv-budget-mib-list 1,2,4 \
  --target-concurrent-requests-list 1,2 \
  --kv-dtype-bytes 2 \
  --output-json results/processed/vram_budget_demotion_sweep_seq1024_tight.json \
  --output-md results/processed/vram_budget_demotion_sweep_seq1024_tight.md
```

Observed:

```text
device = cuda
seq_len_actual = 1024
total_blocks = 8
num_layers = 28
num_kv_heads = 2
head_dim = 128
kv_bytes_per_token = 28672
kv_bytes_per_block = 3670016
```

budget が小さすぎる case:

```text
budget_1_concurrency_1
budget_1_concurrency_2
budget_2_concurrency_1
budget_2_concurrency_2
budget_4_concurrency_2
```

これらは request budget が1 block 未満、または concurrency により per-request budget が小さくなり、`budget_not_ok` になった。

代表例:

```text
global_working_kv_budget_mib = 1.0
target_concurrent_requests = 1
request_working_kv_budget_mib = 1.0
derived_target_keep_blocks = 0
budget_ok = false
fallback_reason = request_budget_smaller_than_one_block
demotion_target_resolution.vram_budget_to_demotion_connected = false
demotion_policy_decision = null
```

usable target が出た case:

```text
case_name = budget_4_concurrency_1
request_working_kv_budget_mib = 4.0
derived_target_keep_blocks = 1
keep_block_ids = [7]
drop_block_ids = [0, 1, 2, 3, 4, 5, 6]
working_k_len = 128
working_ratio = 0.125
mean_abs_diff = 2.741216121648904e-06
max_abs_diff = 2.5153160095214844e-05
demotion_policy_decision.dry_run_only = true
demotion_policy_decision.demotion_applied = false
```

Interpretation:

`budget_4_concurrency_1` は、VRAM budget 由来の `derived_target_keep_blocks=1` が demotion dry-run に入り、最新 block 7 のみ keep、block 0-6 を drop candidate とする metadata を生成した。attention diff は非常に小さく、少なくともこの synthetic structured / layer0 / last-token attention comparison では、強い demotion でも差分が小さい case が存在することを確認できた。

ただし、これは layer0 / synthetic prompt / single-step attention diff であり、品質保証ではない。今後は layer / prompt / seq_len / budget を広げて見る必要がある。

## 9. Current Position

PR #24〜#28 で、RelayKV の VRAM-aware metadata path は以下まで進んだ。

```text
#24
VRAM budget dry-run schema

#26
derived_target_keep_blocks -> demotion dry-run target connection

#28
VRAM budget / concurrency sweep with attention diff
```

現在の到達点:

```text
global/per-request VRAM budget
  -> derived target keep blocks
  -> demotion dry-run keep/drop
  -> candidate KV
  -> attention comparison
```

まだ runtime に対する実適用はしていないが、residual VRAM budget から working KV target を導出し、それが keep/drop と attention diff にどう出るかを sweep できるようになった。

## 10. Remaining Constraints

今回も以下は行っていない。

- runtime KV pool mutation
- actual KV eviction / demotion
- attention backend connection
- scheduler decision change
- runtime writeback
- SGLang / vLLM adapter change
- results/processed の commit
- quality benchmark / generation-level evaluation

## 11. Next Candidate

次の候補は2方向。

### Candidate A: Sweep Summary / Analysis Helper

branch:

```text
relaykv-vram-budget-sweep-summary-v1
```

目的:

- `results/processed/vram_budget_demotion_sweep*.json` を読み込む
- error / budget_not_ok / fullkv_within_budget / applied dry-run を分類
- derived_target_keep_blocks と working_ratio / mean_abs_diff の表を作る
- jq 依存を減らす
- Markdown / CSV summary を出す

これは小さく安全で、review 負荷が低い。

### Candidate B: Larger Sweep Protocol

branch:

```text
relaykv-vram-budget-demotion-sweep-protocol-v1
```

目的:

- seq_len: 512 / 1024 / 2048
- layer_idx: 0 / 14 / 27
- prompt_type: structured / prose / repetitive
- global_working_kv_budget_mib と concurrency の grid
- budget_not_ok / fullkv_within_budget / actual demotion の割合
- attention diff trend

こちらは実験寄りで、runtime 実装の前に policy の妥当性を見るための材料になる。

## 12. Commit / PR Reference

PR #28:

```text
Add VRAM budget demotion sweep
merge_commit_sha = d6600f49bf14bf3700a5aac816871a1c4ceccfb6
```
