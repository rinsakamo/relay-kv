# RelayKV Devlog: Routing Policy Skeleton v1

- Date basis: JST 2026-05-12
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #34: `Add RelayKV routing policy skeleton`
- Branch:
  - `relaykv-routing-policy-skeleton-v1`
- Merge commit:
  - `e62384416a99168c9151e10a8cde656d32fddc08`

## 1. Summary

RelayKV の既存 demotion dry-run 出力を `RelayKVDecision` に接続する、log-only / schema-only の routing policy skeleton を追加した。

今回の PR により、これまでの `keep_block_ids` / `drop_block_ids` / `fallback_reason` を、RelayKV Core の routing decision schema として扱えるようになった。

主な追加:

```text
relaykv/routing_policy.py
tests/test_routing_policy.py
```

主な helper:

```text
build_routing_decision_from_demotion()
```

この変更により、VRAM budget -> demotion dry-run -> RelayKVDecision までがつながり、次の memory pressure state machine / tiered budget accounting / shadow router に進むための routing output contract ができた。

## 2. Context

直前の PR #32 では、RelayKV を **KV block router / tiered KV memory manager** として扱うために、以下の schema を追加した。

```text
RelayKVMemoryBlock
RelayKVDecision
KVClass
ResidencyLevel
PrecisionLevel
RopeStatus
CachePositionPolicy
ExecutionMode
```

しかし、schema は存在していても、既存の demotion dry-run と接続されていなかった。

今回の PR #34 は、既存の demotion policy decision を `RelayKVDecision` に変換する薄い wrapper を追加し、RelayKV routing policy の最初の利用実績を作った。

## 3. Added / Changed Files

追加:

```text
relaykv/routing_policy.py
tests/test_routing_policy.py
```

更新:

```text
relaykv/__init__.py
relaykv/routing_decision.py
```

## 4. build_routing_decision_from_demotion()

追加した主関数:

```text
build_routing_decision_from_demotion()
```

目的:

```text
existing demotion dry-run output
  -> RelayKVDecision
```

主な入力:

```text
total_blocks
target_keep_blocks
keep_block_ids
drop_block_ids
fallback_reason
dry_run_only
eviction_excluded_block_ids
eviction_candidate_block_ids
estimated_working_kv_bytes
estimated_ram_swap_bytes
estimated_ssd_read_bytes
```

主な出力 mapping:

```text
keep_block_ids
  -> selected_active_block_ids

drop_block_ids
  -> demoted_block_ids

eviction_excluded_block_ids
  -> protected_block_ids

eviction_candidate_block_ids
  -> demotion_candidate_block_ids

fallback_reason
  -> fallback_reason

fallback_reason / dry_run_only
  -> apply_blocked_reason

dry_run_only / fallback state / routed state
  -> execution_mode
```

## 5. Execution Mode Mapping

今回の PR で、routing wrapper は execution mode を明示的に返すようになった。

最終的な mapping:

```text
fallback_reason == "fullkv_within_budget"
  -> ExecutionMode.FULLKV_GPU

fallback_reason is not None
  -> ExecutionMode.SHADOW_ONLY

fallback_reason is None
and drop_block_ids is non-empty
and dry_run_only is False
  -> ExecutionMode.RELAYKV_ROUTED

otherwise
  -> ExecutionMode.SHADOW_ONLY
```

重要な点:

```text
successful dry-run demotion
  -> SHADOW_ONLY + apply_blocked_reason=dry_run_only

successful non-dry-run routed demotion
  -> RELAYKV_ROUTED + apply_blocked_reason=None

fullkv_within_budget
  -> FULLKV_GPU + apply_blocked_reason=None

insufficient_eviction_candidates
  -> SHADOW_ONLY + apply_blocked_reason=insufficient_eviction_candidates
```

当初は `APPLY_VRAM_WORKING` への写像案も検討されたが、既存 enum には存在しないため、新 enum は追加せず、既存の `ExecutionMode.RELAYKV_ROUTED` を使用した。

## 6. Protection Metadata Preservation

レビュー指摘により、demotion policy が持っている protected / candidate metadata を routing summary でも保持するよう修正した。

対応 mapping:

```text
eviction_excluded_block_ids
  -> protected_block_ids

eviction_candidate_block_ids
  -> demotion_candidate_block_ids
```

また、防御的に protected block が demotion candidate に混入しないようにした。

これにより、recent / boundary / prefix protection が active なケースでも、routing summary が demotion decision と矛盾しなくなった。

## 7. Fallback Blocker Preservation

レビュー指摘により、`apply_blocked_reason` の優先順位を修正した。

最終的な優先順位:

```text
1. fallback_reason があり、fallback_reason != "fullkv_within_budget"
   -> apply_blocked_reason = fallback_reason

2. fallback_reason == "fullkv_within_budget"
   -> apply_blocked_reason = None

3. dry_run_only
   -> apply_blocked_reason = "dry_run_only"

4. otherwise
   -> apply_blocked_reason = None
```

これにより、以下のような diagnostic value が失われなくなった。

```text
insufficient_eviction_candidates + dry_run_only
  -> apply_blocked_reason = insufficient_eviction_candidates
```

`dry_run_only` は execution status であり、`insufficient_eviction_candidates` は routing / demotion target を満たせなかった policy blocker なので、後者を優先する設計にした。

## 8. Review Fixes

PR #34 では、以下の重要なレビュー修正を取り込んだ。

### 8.1 Preserve demotion protection metadata

問題:

```text
protected_block_ids = []
demotion_candidate_block_ids = list(range(total_blocks))
```

としていたため、protected recent / boundary / prefix blocks が demotion candidate に見えていた。

修正:

```text
eviction_excluded_block_ids -> protected_block_ids
eviction_candidate_block_ids -> demotion_candidate_block_ids
```

### 8.2 Preserve fallback blockers in dry-run routing

問題:

```text
fallback_reason = insufficient_eviction_candidates
dry_run_only = True
apply_blocked_reason = dry_run_only
```

となり、actionable fallback reason が隠れていた。

修正:

```text
non-fullkv fallback_reason を dry_run_only より優先
```

### 8.3 Set applied demotions to routed mode

問題:

```text
fallback_reason is None
drop_block_ids non-empty
dry_run_only is False
execution_mode = SHADOW_ONLY
```

となり、実際に APPLY 可能な routed demotion が shadow-only として扱われていた。

修正:

```text
successful non-dry-run routed demotion
  -> ExecutionMode.RELAYKV_ROUTED
```

### 8.4 Use defined execution mode

問題:

```text
ExecutionMode.APPLY_VRAM_WORKING
```

を使う修正案があったが、現行 enum には存在しなかった。

修正:

```text
ExecutionMode.RELAYKV_ROUTED
```

を使い、新 enum は追加しない方針にした。

## 9. Tests

追加された test coverage:

```text
successful demotion dry-run maps keep/drop into RelayKVDecision
protected blocks are preserved
protected blocks are excluded from demotion candidates
fallback-specific blockers take precedence over dry_run_only
fullkv_within_budget maps to FULLKV_GPU
successful non-dry-run routed demotion maps to RELAYKV_ROUTED
summary() remains JSON-serializable
integration test wrapping build_demotion_decision() output
```

## 10. Validation

Targeted tests:

```text
python -m pytest tests/test_routing_policy.py -q
8 passed in 2.11s
```

Full suite:

```text
python -m pytest -q
51 passed in 4.62s
```

## 11. Constraints Preserved

今回の変更では以下を行っていない。

```text
model loading
runtime KV pool mutation
attention backend connection
scheduler change
runtime writeback
actual KV materialization
actual KV eviction
SGLang/vLLM adapter integration
```

この PR は schema/log-only の routing wrapper であり、runtime behavior は変更していない。

## 12. Current Position

PR #24〜#34 で、RelayKV の dry-run control plane は以下まで進んだ。

```text
#24
VRAM budget dry-run schema

#26
VRAM budget target -> demotion dry-run connection

#28
VRAM budget demotion sweep

#30
Offline sweep summary helper

#32
Core routing schema

#34
Routing policy skeleton
```

現在の接続:

```text
global/per-request VRAM budget
  -> derived target keep blocks
  -> demotion dry-run keep/drop/fallback
  -> RelayKVDecision
  -> routing summary-ready metadata
```

これにより、RelayKV Core は単なる keep/drop list ではなく、以下を含む routing decision を扱えるようになった。

```text
selected active blocks
demoted blocks
protected blocks
demotion candidates
fallback reason
apply blocked reason
execution mode
estimated byte fields
```

## 13. Next Candidate

次は以下が自然。

```text
relaykv-memory-pressure-state-machine-v1
```

目的:

```text
seq_len
projected_fullkv_bytes
residual_vram_budget_bytes
labels_ready
host_backup_available
shadow_compare_passed
selection_stability_ratio
estimated_net_benefit_ms
```

などから、log-only に execution readiness / blocked reason / fallback state を判断する。

候補 state:

```text
DISABLED_SHORT_CONTEXT
SHADOW_ONLY_WARMUP
BUDGET_PRESSURE
APPLY_READY
FALLBACK_REQUIRED
```

ただし、既存の `ExecutionMode` には `FULL_ATTENTION`, `FULLKV_GPU`, `RELAYKV_ROUTED`, `SHADOW_COMPARE`, `SHADOW_ONLY` があるため、次 PR では enum の追加よりも、まず state machine 側の internal decision/state schema と `RelayKVDecision.execution_mode` への conservative mapping を分けるのが安全。

## 14. Recommendation

次の実装は `relaykv-memory-pressure-state-machine-v1` がよい。

理由:

```text
RelayKVDecision が使えるようになった
routing output に execution_mode / fallback_reason / apply_blocked_reason が揃った
次は「いつ SHADOW_ONLY から RELAYKV_ROUTED に進めるか」を log-only に判定する段階
tiered budget accounting より前に readiness / blocked_reason の整理が必要
runtime にはまだ触らず、安全に control plane を伸ばせる
```
