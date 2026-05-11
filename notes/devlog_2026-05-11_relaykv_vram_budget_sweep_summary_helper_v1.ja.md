# RelayKV Devlog: VRAM Budget Sweep Summary Helper v1

- Date basis: JST 2026-05-11
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #30: `Add VRAM budget sweep summary helper`
- Branch:
  - `relaykv-vram-budget-sweep-summary-v1`
- Merge commit:
  - `603dc68cac0e812407da1aaee1a67e20ef542e25`

## 1. Summary

RelayKV の VRAM budget demotion sweep 結果を、model load なしで offline 集計する summary helper を追加した。

これにより、`results/processed/vram_budget_demotion_sweep*.json` を jq や ad hoc parser で読む代わりに、repo 内の標準 helper で Markdown / CSV summary を出せるようになった。

今回の変更は diagnostic / analysis-only であり、runtime 挙動には影響しない。

- model loading なし
- runtime KV pool mutation なし
- attention backend connection なし
- scheduler change なし
- runtime writeback なし
- results/processed outputs は validation 用で commit 対象外

## 2. Added Files

追加ファイル:

```text
scripts/summarize_vram_budget_demotion_sweep.py
tests/test_vram_budget_sweep_summary.py
```

## 3. Purpose

直前の PR #28 で、VRAM budget / concurrency を sweep して、case ごとの以下を記録できるようになった。

```text
global_working_kv_budget_mib
target_concurrent_requests
request_working_kv_budget_mib
derived_target_keep_blocks
demotion_target_resolution
demotion_policy_decision
keep_block_ids
drop_block_ids
working_ratio
attention_compare
```

しかし、結果確認にはまだ jq や目視が必要だった。

今回の summary helper は、1つの processed sweep JSON を読み込み、case を分類し、aggregate counts と per-case rows を Markdown / CSV に整形する。

## 4. CLI

基本形:

```text
python scripts/summarize_vram_budget_demotion_sweep.py \
  --input-json results/processed/vram_budget_demotion_sweep_seq1024_tight.json
```

明示出力:

```text
python scripts/summarize_vram_budget_demotion_sweep.py \
  --input-json results/processed/vram_budget_demotion_sweep_seq1024_tight.json \
  --output-md results/processed/vram_budget_demotion_sweep_seq1024_tight_summary.md \
  --output-csv results/processed/vram_budget_demotion_sweep_seq1024_tight_summary.csv
```

`--output-md` / `--output-csv` を省略した場合は、input stem に `_summary` を付けた path を使う。

## 5. Case Classification

case classification は以下。

```text
budget_not_ok
fullkv_within_budget
actual_demotion
demotion_failed
case_error
unknown
```

最終的な classification rule:

```text
1. budget_not_ok
   - error is not None and fallback_reason == "vram_budget_not_ok"
   - or demotion_target_resolution.fallback_reason == "vram_budget_not_ok"

2. case_error
   - error is not None
   - and not budget_not_ok

3. fullkv_within_budget
   - fallback_reason == "fullkv_within_budget"

4. demotion_failed
   - error is None
   - fallback_reason is not None
   - fallback_reason != "fullkv_within_budget"

5. actual_demotion
   - error is None
   - fallback_reason is None
   - len(drop_block_ids) > 0

6. unknown
   - otherwise
```

重要な点は、`drop_block_ids` の有無だけで成功 / within budget を判断しないこと。

## 6. Review Fix: No-drop fallback classification

初期実装では、non-error かつ `drop_block_ids=[]` の case を広く `fullkv_within_budget` に分類していた。

これは不正確だった。

例えば、保護設定が強すぎる場合:

```text
--demotion-recent-blocks high
--protect-prefix-blocks high
--protect-boundary-blocks high
```

`build_demotion_decision()` が以下のような case を返すことがある。

```text
fallback_reason = "insufficient_eviction_candidates"
drop_block_ids = []
error = None
```

これは fullkv が budget 内に収まったのではなく、demotion target を満たせなかった失敗系である。

修正後:

```text
fallback_reason="fullkv_within_budget"
  -> fullkv_within_budget

fallback_reason="insufficient_eviction_candidates"
  -> demotion_failed
```

## 7. Review Fix: Partial insufficient-candidate classification

さらに、`insufficient_eviction_candidates` は `drop_block_ids=[]` の場合だけではない。

一部 block は drop できたが、target まで到達できなかった場合:

```text
fallback_reason = "insufficient_eviction_candidates"
drop_block_ids = [0, 1]
error = None
```

この場合も、demotion target 未達なので `actual_demotion` ではなく `demotion_failed` に分類すべき。

そのため、最終 rule では `drop_block_ids` の判定より先に non-fullkv fallback を見るようにした。

```text
fallback_reason != None and fallback_reason != "fullkv_within_budget"
  -> demotion_failed

fallback_reason == None and len(drop_block_ids) > 0
  -> actual_demotion
```

これにより、`actual_demotion` は明確に以下の case に限定された。

```text
error = None
fallback_reason = None
drop_block_ids is non-empty
```

## 8. Aggregate Counts

summary helper は以下の aggregate counts を出す。

```text
total_cases
budget_not_ok_cases
fullkv_within_budget_cases
actual_demotion_cases
demotion_failed_cases
case_error_cases
unknown_cases
```

`demotion_failed_cases` を明示的に分けたことで、保護設定により target 未達となった case を successful demotion と混ぜずに確認できる。

## 9. Output Rows

CSV / Markdown の per-case rows には主に以下を含める。

```text
case_name
case_class
global_working_kv_budget_mib
target_concurrent_requests
request_working_kv_budget_mib
derived_target_keep_blocks
kept_blocks_count
dropped_blocks_count
working_ratio
mean_abs_diff
max_abs_diff
fallback_reason
error
```

## 10. Validation

Final validation:

```text
python -m py_compile scripts/summarize_vram_budget_demotion_sweep.py
passed

python -m pytest -q
36 passed in 4.33s
```

Optional local summary smoke:

```text
python scripts/summarize_vram_budget_demotion_sweep.py \
  --input-json results/processed/vram_budget_demotion_sweep_seq1024_tight.json
```

Observed aggregate counts:

```json
{
  "total_cases": 6,
  "budget_not_ok_cases": 5,
  "fullkv_within_budget_cases": 0,
  "actual_demotion_cases": 1,
  "demotion_failed_cases": 0,
  "case_error_cases": 0,
  "unknown_cases": 0
}
```

The smoke wrote:

```text
results/processed/vram_budget_demotion_sweep_seq1024_tight_summary.md
results/processed/vram_budget_demotion_sweep_seq1024_tight_summary.csv
```

These outputs were not staged and should not be committed.

## 11. Observed Real Demotion Case

From the existing `seq1024_tight` processed sweep:

```text
case_name = budget_4_concurrency_1
case_class = actual_demotion
kept_blocks_count = 1
dropped_blocks_count = 7
mean_abs_diff = 0.000002741
max_abs_diff = 0.000025153
```

This remains `actual_demotion` because:

```text
error = None
fallback_reason = None
drop_block_ids is non-empty
```

## 12. Current Position

PR #24〜#30 で、RelayKV の VRAM-aware demotion metadata / diagnostics path は以下まで進んだ。

```text
#24
VRAM budget dry-run schema

#26
derived_target_keep_blocks -> demotion dry-run target connection

#28
VRAM budget / concurrency sweep with attention diff

#30
Offline sweep summary helper with strict classification
```

現在の到達点:

```text
global/per-request VRAM budget
  -> derived target keep blocks
  -> demotion dry-run keep/drop
  -> candidate KV
  -> attention comparison
  -> offline Markdown/CSV summary
```

## 13. Remaining Constraints

まだ以下は行っていない。

- runtime KV pool mutation
- actual KV eviction / demotion
- attention backend connection
- scheduler decision change
- runtime writeback
- SGLang / vLLM adapter integration
- generation-level quality evaluation
- large grid protocol sweep
- results/processed の commit

## 14. Next Candidate

次は大きく2択。

### Candidate A: Larger Sweep Protocol v1

branch:

```text
relaykv-vram-budget-demotion-sweep-protocol-v1
```

目的:

```text
seq_len: 512 / 1024 / 2048
layer_idx: 0 / 14 / 27
prompt_type: structured / prose / repetitive
budget/concurrency grid
```

出力:

```text
sweep JSON
summary Markdown/CSV
case_class counts
attention diff trend
```

これは PR #28 + #30 を組み合わせた実験 protocol。

### Candidate B: Demotion failure scenario fixtures

branch:

```text
relaykv-demotion-failure-fixtures-v1
```

目的:

```text
insufficient_eviction_candidates
budget_not_ok
fullkv_within_budget
actual_demotion
```

の synthetic fixture JSON を用意し、summary helper の分類を実データに近い形で固定する。

review churn を減らすなら Candidate B、次の実験に進むなら Candidate A がよい。
