# RelayKV Devlog - 2026-05-10 JST
## Drop Policy Sweep v1 / FullKV Demotion 評価への転換

日付確認:
- 本 devlog は **JST 2026-05-10** 基準で作成。
- 今回は `relaykv-drop-policy-sweep-v1` の実装・push 後の区切りとして記録する。

---

## 1. 今日の結論

RelayKV の評価軸を、従来の **selection 型** から **FullKV demotion 型**へ進めた。

従来の問い:

```text
限られた working KV budget に、どの block を選ぶか？
```

今回から重視する問い:

```text
FullKV から始めて、VRAM budget に合わせてどの block を drop / demote してもよいか？
```

この整理により、RelayKV は単なる KV block selection ではなく、以下のように再定義される。

```text
RelayKV is a VRAM-aware KV residency controller.

It starts from FullKV when VRAM allows, protects blocks that should not be evicted,
demotes low-value GPU-resident KV blocks under memory pressure,
and retrieves/swap cold blocks only when their value exceeds the blocks being demoted.
```

日本語では:

```text
RelayKV は、モデル重みロード後の残余 VRAM を管理し、
decode-time working KV を指定予算内に収める KV residency controller。

FullKV が載る間は FullKV を維持し、
VRAM pressure が出たら低価値 block を demote し、
cold 側により価値の高い block がある場合だけ retrieve / swap する。
```

---

## 2. 実装したもの

### Branch

```text
relaykv-drop-policy-sweep-v1
```

### 追加ファイル

```text
scripts/run_drop_policy_sweep.py
```

### 変更方針

既存の以下は変更しない方針で進めた。

```text
scripts/run_relaykv_pipeline.py
scripts/run_budget_policy_sweep.py
relaykv/budget_policy.py
```

今回の追加 script は、既存の selection 型 budget policy とは独立した dry-run 診断用 script として実装した。

---

## 3. 実装内容

`run_drop_policy_sweep.py` は、FullKV block set を基準に、case ごとに明示的な `keep_block_ids` / `drop_block_ids` を作り、working KV を構成して FullKV attention output との差分を測る。

### Built-in cases

```text
full
drop_oldest_1
drop_oldest_2
drop_middle_2
drop_boundary_before_recent_1
keep_recent4
keep_recent4_boundary1
keep_recent4_anchor1_boundary1
```

### 出力

JSON / Markdown summary に以下を出力する。

```text
case_name
total_blocks
keep_block_ids
drop_block_ids
working_k_len
working_ratio
mean_abs_diff
max_abs_diff
fallback_reason / error
decision_summary
```

`decision_summary` には以下を含める。

```text
eviction_excluded_block_ids
eviction_candidate_block_ids
demoted_block_ids
reason_labels_by_block
```

### reason label 例

```text
FULLKV_REFERENCE
RECENT_PROTECTED
BOUNDARY_NEAR_RECENT
PREFIX_CANDIDATE
DROP_OLDEST
DROP_MIDDLE
KEEP_BY_CASE
```

---

## 4. 検証コマンド

### py_compile

```bash
python -m py_compile scripts/run_drop_policy_sweep.py
```

### 1024 structured / layer14 sweep

```bash
python scripts/run_drop_policy_sweep.py \
  --seq-len 1024 \
  --block-size 128 \
  --layer-idx 14 \
  --prompt-type structured \
  --output-json results/processed/drop_policy_sweep_structured_1024_l14.json \
  --output-md results/processed/drop_policy_sweep_structured_1024_l14.md
```

### summary 確認

```bash
cat results/processed/drop_policy_sweep_structured_1024_l14.md
```

```bash
jq '.cases[] | {
  case_name,
  keep_block_ids,
  drop_block_ids,
  working_ratio,
  mean_abs_diff,
  max_abs_diff,
  decision_summary
}' results/processed/drop_policy_sweep_structured_1024_l14.json
```

---

## 5. 観測結果

`seq_len=1024`, `block_size=128`, `layer_idx=14`, `prompt_type=structured` の結果。

| case | keep_blocks | drop_blocks | working_ratio | mean_abs_diff | max_abs_diff | interpretation |
|---|---|---|---:|---:|---:|---|
| full | `[0,1,2,3,4,5,6,7]` | `[]` | 1.000 | 0.000000000 | 0.000000000 | FullKV reference |
| drop_oldest_1 | `[1,2,3,4,5,6,7]` | `[0]` | 0.875 | 約2e-8〜4e-8 | 約2e-7〜5e-7 | block0 はほぼ落とせる |
| drop_oldest_2 | `[2,3,4,5,6,7]` | `[0,1]` | 0.750 | 約1.4e-6〜1.5e-6 | 約1.2e-5 | block0/1 もかなり安全 |
| drop_middle_2 | candidates 内の中間2 blocks | 中間2 blocks | 0.750 | 要確認 | 要確認 | eviction scope と矛盾しない形に修正済み |
| drop_boundary_before_recent_1 | `[0,1,2,4,5,6,7]` | `[3]` | 0.875 | 約5.7e-5〜6.1e-5 | 約6e-4 | boundary block 3 は落とすと悪化 |
| keep_recent4 | `[4,5,6,7]` | `[0,1,2,3]` | 0.500 | 約6.4e-5〜6.8e-5 | 約6.8e-4〜7.3e-4 | recent-only baseline |
| keep_recent4_boundary1 | `[3,4,5,6,7]` | `[0,1,2]` | 0.625 | 約7.8e-6〜8.1e-6 | 約7e-5 | boundary block 追加で大きく改善 |
| keep_recent4_anchor1_boundary1 | `[0,3,4,5,6,7]` | `[1,2]` | 0.750 | 約7.8e-6〜8.1e-6 | 約7e-5 | anchor block0 の上積みは小さい |

注:
- 実行タイミングにより `mean_abs_diff` の極小値はわずかに揺れた。
- 重要なのは絶対値よりも相対傾向。
- `keep_recent4_boundary1` が `keep_recent4` より大幅に良い。
- `block0` anchor の寄与はこの条件ではほぼ見えない。
- `recent` 直前 boundary block `[3]` は重要。

---

## 6. 今日の主要な発見

### 6.1 FullKV demotion 型の評価が必要

従来の budget sweep は、主に以下を比較していた。

```text
recent4
vs
recent1 + retrieval3
vs
anchor1 + recent1 + retrieval2
```

これは「どの block を選ぶか」という selection 型評価である。

しかし実運用では、短〜中文脈では FullKV が基本であり、RelayKV は VRAM pressure が出てから動く。

したがって、実用上の中心問題は以下である。

```text
FullKV からどの block を安全に落とせるか？
```

今回の `run_drop_policy_sweep.py` によって、この方向の評価が可能になった。

---

### 6.2 固定 anchor block0 はこの条件では弱い

以前の additive budget 比較では以下が観測された。

```text
recent4:
  working = [4,5,6,7]
  mean_abs_diff ≈ 6.8e-5

recent4 + anchor1:
  working = [0,4,5,6,7]
  mean_abs_diff ≈ 6.8e-5

recent4 + retrieval1 / boundary1:
  working = [3,4,5,6,7]
  mean_abs_diff ≈ 8e-6
```

今回の drop-policy sweep でも同様に、`block0` を追加しても `boundary block 3` の追加ほど効果がない。

結論:

```text
ANCHOR は固定 block class ではなく、
PREFIX_PROTECTED / instruction / shared prefix などの reason label として扱う方がよい。
```

---

### 6.3 boundary block が重要

今回の条件では、`recent` window の直前 block が非常に重要だった。

```text
recent4:
  keep [4,5,6,7]

keep_recent4_boundary1:
  keep [3,4,5,6,7]
```

この差だけで attention diff が大きく改善した。

仮説:

```text
FullKV demotion 初期ルールでは、
RECENT_PROTECTED に加えて BOUNDARY_NEAR_RECENT を保護するのが有効。
```

---

### 6.4 block class ではなく policy scope / decision / reason が必要

今後の schema は以下に寄せるべき。

```text
residency:
  GPU_WORKING
  CPU_COLD
  COMPRESSED_COLD
  REMOTE_COLD

eviction_scope:
  EVICTION_EXCLUDED
  EVICTION_CANDIDATE

retrieve_scope:
  NOT_RETRIEVABLE
  COLD_CANDIDATE
  RETRIEVE_CANDIDATE

decision:
  KEEP
  DEMOTE
  RETRIEVE
  SWAP
  FALLBACK

reason:
  RECENT_PROTECTED
  BOUNDARY_NEAR_RECENT
  PREFIX_PROTECTED
  LOW_DROP_SCORE
  HIGH_RETRIEVAL_SCORE
  TRANSFER_COST_TOO_HIGH
```

`RECENT / ANCHOR / RETRIEVED` は block の固定分類ではなく、policy decision の reason label / transition label として扱う。

---

## 7. VRAM budget 設計の追加方針

RelayKV は単一 request の長文脈化だけでなく、global / per-request budget の制御層としても扱う。

### Budget levels

```text
global_residual_vram_bytes
global_working_kv_budget_bytes
target_concurrent_requests
request_working_kv_budget_bytes
```

### 目的

```text
1. 低VRAM環境での OOM 回避
2. per-request / per-thread の動作保証
3. memory limit による副次的 sparse working KV 化
4. 結果として attention 対象が減り、速度改善する可能性
```

基本方針:

```text
memory guarantee first,
speedup as a side effect.
```

---

## 8. 現在の位置づけ

RelayKV の定義を以下に更新する。

```text
RelayKV は KV cache 削減アルゴリズムではなく、
FullKV から安全に落としていく
VRAM-aware KV residency controller。
```

より実装寄りには:

```text
RelayKV は、残余 VRAM budget を global / per-request に配分し、
各 request の decode-time working KV set を予算内に保つ。
recent / boundary / prefix など保護すべき block は eviction から除外し、
低価値 block を demote し、
cold 側により価値がある場合だけ retrieve / swap する。
```

---

## 9. 次にやること

### 最優先

```text
1. PR の review / merge
2. drop-policy sweep の結果を複数条件で確認
3. demotion policy の初期ルールを決める
```

### 次の実験候補

```bash
python scripts/run_drop_policy_sweep.py \
  --seq-len 1024 \
  --block-size 128 \
  --layer-idx 14 \
  --prompt-type prose \
  --output-json results/processed/drop_policy_sweep_prose_1024_l14.json \
  --output-md results/processed/drop_policy_sweep_prose_1024_l14.md
```

```bash
python scripts/run_drop_policy_sweep.py \
  --seq-len 2048 \
  --block-size 128 \
  --layer-idx 14 \
  --prompt-type structured \
  --output-json results/processed/drop_policy_sweep_structured_2048_l14.json \
  --output-md results/processed/drop_policy_sweep_structured_2048_l14.md
```

### 次PR候補

```text
relaykv-activation-policy-v1
```

内容:

```text
- short context では RelayKV を実用適用しない
- diagnostic mode と practical mode を分ける
- activation_state を導入する
  - DISABLED_SHORT_CONTEXT
  - FULLKV_ACTIVE
  - SHADOW_ONLY
  - DEMOTION_CANDIDATE
  - APPLY
  - FALLBACK
```

または:

```text
relaykv-demotion-policy-v1
```

内容:

```text
- RECENT_PROTECTED + BOUNDARY_NEAR_RECENT を優先保護
- oldest / low-value candidate から demote
- drop-policy sweep の結果を policy 化する
```

---

## 10. Commit / push

今回の作業は push 済み。

参考コマンド:

```bash
git status --short
git branch -vv
git log --oneline --decorate -n 5
```

commit / push 例:

```bash
git add scripts/run_drop_policy_sweep.py
git commit -m "Add RelayKV drop policy sweep"
git push -u origin relaykv-drop-policy-sweep-v1
```

---

## 11. PR body draft

```text
Summary
- Add a dry-run RelayKV drop-policy sweep.
- Evaluate FullKV demotion patterns with explicit keep/drop block sets.
- Report keep_block_ids, drop_block_ids, working ratio, attention diff metrics, and decision summaries.
- Keep existing budget policy and pipeline behavior unchanged.

Validation
- Compiled the new script.
- Ran a 1024-token structured layer14 drop-policy sweep.
- Confirmed full/reference, drop-oldest, drop-middle, boundary-drop, recent-only, and boundary-retention cases emit JSON and Markdown summaries.

Command
    python -m py_compile scripts/run_drop_policy_sweep.py

    python scripts/run_drop_policy_sweep.py \
      --seq-len 1024 \
      --block-size 128 \
      --layer-idx 14 \
      --prompt-type structured \
      --output-json results/processed/drop_policy_sweep_structured_1024_l14.json \
      --output-md results/processed/drop_policy_sweep_structured_1024_l14.md

    jq '.cases[] | {
      case_name,
      keep_block_ids,
      drop_block_ids,
      working_ratio,
      mean_abs_diff,
      max_abs_diff,
      decision_summary
    }' results/processed/drop_policy_sweep_structured_1024_l14.json

Observed
- full produced zero attention diff.
- drop_oldest_1 and drop_oldest_2 produced very small attention diffs.
- dropping the boundary block before the recent window produced much larger diff.
- keep_recent4_boundary1 improved substantially over keep_recent4.
- fixed block0 anchor added little beyond boundary retention in this 1024 structured layer14 case.

Notes
- This is a diagnostic-only PyTorch prototype script.
- This does not modify attention kernels, runtime KV pools, scheduler behavior, SGLang/vLLM integration, or existing budget policy behavior.
- results/raw/ and results/processed/ are not committed.
```

---

## 12. 次セッションへの引き継ぎ

次セッションでは以下から開始する。

```text
RelayKV は selection 型評価から FullKV demotion 型評価へ移行した。
`relaykv-drop-policy-sweep-v1` では `scripts/run_drop_policy_sweep.py` を追加し、
FullKV block set から keep/drop blocks を明示して attention diff を測れるようにした。

重要な観測:
- block0 は落としてもほぼ影響が小さい。
- block0 固定 anchor の上積みは小さい。
- recent window 直前の boundary block は重要。
- keep_recent4_boundary1 は keep_recent4 より大幅に FullKV に近い。
- RelayKV の分類は RECENT/ANCHOR/RETRIEVED 固定classではなく、
  eviction_scope / retrieve_scope / decision / reason に分解する方針。

次にやること:
1. PR review / merge
2. prose 1024 / structured 2048 でも drop-policy sweep を確認
3. activation-policy-v1 か demotion-policy-v1 に進む
4. global / per-request VRAM budget を dry-run schema に入れる
```
