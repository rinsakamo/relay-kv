# RelayKV Devlog: Memory Pressure State Machine v1

- Date basis: JST 2026-05-12
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #36: `Add RelayKV memory pressure state machine`
- Branch:
  - `relaykv-memory-pressure-state-machine-v1`
- Merge commit:
  - `0cc6d2d927ac5b11a4e90bd39af4085b0e21b556`

## 1. Summary

RelayKV の control-plane に、log-only / schema-only の memory pressure state machine を追加した。

今回の PR により、RelayKV が以下のどの状態にあるかを、runtime APPLY / materialization なしで判定できるようになった。

```text
short context なので無効
FullKV が GPU 予算内に収まる
shadow warmup 中
fallback が必要
RelayKV routed ready
```

主な追加:

```text
relaykv/memory_pressure.py
tests/test_memory_pressure.py
```

主な型と関数:

```text
RelayKVMemoryPressureState
RelayKVMemoryPressureDecision
decide_memory_pressure_state()
```

これにより、VRAM budget / demotion dry-run / routing decision の上位に、RelayKV をいつ shadow-only に留め、いつ `RELAYKV_ROUTED` に進めるかを判定する readiness layer ができた。

## 2. Context

直前までに、RelayKV の dry-run control plane は以下まで進んでいた。

```text
VRAM budget dry-run schema
  -> VRAM budget target
  -> demotion dry-run
  -> RelayKVDecision
```

PR #34 では `build_routing_decision_from_demotion()` により、demotion dry-run の keep/drop/fallback を `RelayKVDecision` に接続した。

ただし、以下のような条件を統合して「実際に routed-ready と見なしてよいか」を判断する state machine はまだなかった。

```text
seq_len
projected_fullkv_bytes
residual_vram_budget_bytes
labels_ready
host_backup_available
shadow_compare_passed
selection_stability_ratio
estimated_net_benefit_ms
fallback_reason
```

今回の PR #36 は、この readiness 判定を log-only に formalize した。

## 3. Added / Changed Files

追加:

```text
relaykv/memory_pressure.py
tests/test_memory_pressure.py
```

更新:

```text
relaykv/__init__.py
```

## 4. RelayKVMemoryPressureState

追加した state enum:

```text
DISABLED_SHORT_CONTEXT
FULLKV_WITHIN_BUDGET
SHADOW_ONLY_WARMUP
BUDGET_PRESSURE
RELAYKV_ROUTED_READY
FALLBACK_REQUIRED
```

意味:

```text
DISABLED_SHORT_CONTEXT
  seq_len が RelayKV 適用閾値未満。Full attention / normal path でよい。

FULLKV_WITHIN_BUDGET
  FullKV が residual VRAM budget 内に収まる。RelayKV demotion は不要。

SHADOW_ONLY_WARMUP
  RelayKV の dry-run / shadow observation は可能だが、まだ apply-ready ではない。

BUDGET_PRESSURE
  予算圧はあるが、まだ routed-ready / fallback の最終状態にはしていない中間概念。

RELAYKV_ROUTED_READY
  log-only 判定上、RelayKV routed path に進める準備が整っている。

FALLBACK_REQUIRED
  RelayKV routed path に進むべきではなく、fallback / shadow-only path に留める必要がある。
```

## 5. RelayKVMemoryPressureDecision

追加した decision schema:

```text
RelayKVMemoryPressureDecision
```

主な fields:

```text
state
execution_mode
apply_blocked_reason
fallback_reason
budget_pressure
short_context
labels_ready
host_backup_available
shadow_compare_passed
selection_stability_ratio
estimated_net_benefit_ms
projected_fullkv_bytes
residual_vram_budget_bytes
min_seq_len_for_relaykv
seq_len
```

`summary()` は JSON-serializable dict を返す。

validation:

```text
seq_len >= 0
projected_fullkv_bytes >= 0
residual_vram_budget_bytes >= 0
min_seq_len_for_relaykv >= 0
0.0 <= selection_stability_ratio <= 1.0
```

## 6. decide_memory_pressure_state()

追加した main helper:

```text
decide_memory_pressure_state()
```

目的:

```text
runtime-independent input signals
  -> RelayKVMemoryPressureDecision
```

主な入力:

```text
seq_len
min_seq_len_for_relaykv
projected_fullkv_bytes
residual_vram_budget_bytes
labels_ready
host_backup_available
shadow_compare_passed
selection_stability_ratio
min_selection_stability_ratio
estimated_net_benefit_ms
min_estimated_net_benefit_ms
fallback_reason
```

## 7. Decision Logic

最終的な優先順位:

```text
1. seq_len < min_seq_len_for_relaykv
   -> DISABLED_SHORT_CONTEXT
   -> ExecutionMode.FULL_ATTENTION
   -> apply_blocked_reason = short_context

2. fallback_reason == "fullkv_within_budget"
   -> FULLKV_WITHIN_BUDGET
   -> ExecutionMode.FULLKV_GPU
   -> apply_blocked_reason = None

3. fallback_reason is not None
   -> FALLBACK_REQUIRED
   -> ExecutionMode.SHADOW_ONLY
   -> apply_blocked_reason = fallback_reason

4. projected_fullkv_bytes <= residual_vram_budget_bytes
   -> FULLKV_WITHIN_BUDGET
   -> ExecutionMode.FULLKV_GPU
   -> apply_blocked_reason = None

5. labels_ready is False
   -> SHADOW_ONLY_WARMUP
   -> ExecutionMode.SHADOW_ONLY
   -> apply_blocked_reason = labels_not_ready

6. host_backup_available is False
   -> SHADOW_ONLY_WARMUP
   -> ExecutionMode.SHADOW_ONLY
   -> apply_blocked_reason = host_backup_unavailable

7. shadow_compare_passed is False
   -> FALLBACK_REQUIRED
   -> ExecutionMode.SHADOW_ONLY
   -> fallback_reason = shadow_compare_failed
   -> apply_blocked_reason = shadow_compare_failed

8. shadow_compare_passed is None
   -> SHADOW_ONLY_WARMUP
   -> ExecutionMode.SHADOW_ONLY
   -> apply_blocked_reason = shadow_compare_not_ready

9. selection_stability_ratio < min_selection_stability_ratio
   -> SHADOW_ONLY_WARMUP
   -> ExecutionMode.SHADOW_ONLY
   -> apply_blocked_reason = selection_unstable

10. estimated_net_benefit_ms < min_estimated_net_benefit_ms
    -> SHADOW_ONLY_WARMUP
    -> ExecutionMode.SHADOW_ONLY
    -> apply_blocked_reason = net_benefit_too_low

11. otherwise
    -> RELAYKV_ROUTED_READY
    -> ExecutionMode.RELAYKV_ROUTED
    -> apply_blocked_reason = None
```

## 8. Important Review Fixes

PR #36 では、重要なレビュー修正を2件取り込んだ。

### 8.1 Preserve `fullkv_within_budget` as FullKV

問題:

```text
fallback_reason = fullkv_within_budget
```

は、既存 demotion / routing 側では「FullKV が予算内に収まる正常系」を意味する。

しかし当初の generic fallback branch では、これを以下に誤分類していた。

```text
FALLBACK_REQUIRED
ExecutionMode.SHADOW_ONLY
```

修正:

```text
fallback_reason == "fullkv_within_budget"
  -> FULLKV_WITHIN_BUDGET
  -> ExecutionMode.FULLKV_GPU
  -> apply_blocked_reason = None
```

これにより、`build_demotion_decision()` / `build_routing_decision_from_demotion()` と意味が揃った。

### 8.2 Require explicit shadow pass before routed readiness

問題:

```text
shadow_compare_passed = None
```

は「未計測」を意味するが、当初は explicit `False` だけを blocker にしていた。

そのため、以下の条件が揃うと shadow compare 未実施でも routed-ready になり得た。

```text
labels_ready = True
host_backup_available = True
shadow_compare_passed = None
```

修正:

```text
shadow_compare_passed is None
  -> SHADOW_ONLY_WARMUP
  -> apply_blocked_reason = shadow_compare_not_ready

shadow_compare_passed is True
  -> routed-ready 判定に進める
```

これにより、RelayKV の shadow-first gate 方針と整合した。

## 9. ExecutionMode Mapping

memory pressure state から既存 `ExecutionMode` への conservative mapping:

```text
DISABLED_SHORT_CONTEXT
  -> FULL_ATTENTION

FULLKV_WITHIN_BUDGET
  -> FULLKV_GPU

SHADOW_ONLY_WARMUP
  -> SHADOW_ONLY

FALLBACK_REQUIRED
  -> SHADOW_ONLY

RELAYKV_ROUTED_READY
  -> RELAYKV_ROUTED
```

新しい execution mode enum は追加していない。

## 10. Test Coverage

追加された test coverage:

```text
short context -> DISABLED_SHORT_CONTEXT + FULL_ATTENTION
full-KV byte budget fit -> FULLKV_WITHIN_BUDGET + FULLKV_GPU
fallback_reason=fullkv_within_budget -> FULLKV_WITHIN_BUDGET + FULLKV_GPU
non-fullkv fallback reason -> FALLBACK_REQUIRED + SHADOW_ONLY
labels_not_ready -> SHADOW_ONLY_WARMUP
host_backup_unavailable -> SHADOW_ONLY_WARMUP
shadow_compare_passed=False -> FALLBACK_REQUIRED + shadow_compare_failed
shadow_compare_passed=None -> SHADOW_ONLY_WARMUP + shadow_compare_not_ready
selection_unstable -> SHADOW_ONLY_WARMUP
net_benefit_too_low -> SHADOW_ONLY_WARMUP
readiness pass with shadow_compare_passed=True -> RELAYKV_ROUTED_READY + RELAYKV_ROUTED
missing budget bytes do not imply FULLKV_WITHIN_BUDGET
summary() is JSON-serializable
invalid selection_stability_ratio raises ValueError
```

## 11. Validation

Codex run result before PR:

```text
python -m pytest tests/test_memory_pressure.py -q
13 passed in 2.07s

python -m pytest -q
65 passed in 4.57s
```

PR body still had placeholder validation text, but local Codex run reported the above results before review fixes. The merged PR body records the final behavior including:

```text
fallback_reason=fullkv_within_budget maps to FULLKV_WITHIN_BUDGET + FULLKV_GPU
shadow_compare_passed=None remains SHADOW_ONLY_WARMUP
RELAYKV_ROUTED_READY requires shadow_compare_passed=True
```

## 12. Constraints Preserved

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

この PR は log-only / schema-only の state machine であり、runtime behavior は変更していない。

## 13. Current Position

PR #24〜#36 で、RelayKV の dry-run / control-plane は以下まで進んだ。

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

#36
Memory pressure state machine
```

現在の logical pipeline:

```text
global/per-request VRAM budget
  -> projected FullKV / residual budget
  -> memory pressure state
  -> target keep blocks
  -> demotion dry-run keep/drop/fallback
  -> RelayKVDecision
  -> routing summary-ready metadata
```

ただし、現時点ではまだ log-only であり、実際の KV 移動・materialization・attention 接続はない。

## 14. Next Candidate

次は以下が自然。

```text
relaykv-memory-pressure-summary-helper-v1
```

目的:

```text
RelayKVMemoryPressureDecision.summary()
  -> offline summary aggregation
```

集計候補:

```text
state_counts
execution_mode_counts
apply_blocked_reason_counts
fallback_reason_counts
short_context_count
budget_pressure_count
routed_ready_count
shadow_not_ready_count
fullkv_within_budget_count
```

これにより、sweep / smoke / future JSON logs から、RelayKV がなぜ apply-ready にならないかを定量的に見られるようになる。

## 15. Recommendation

次の実装は `relaykv-memory-pressure-summary-helper-v1` がよい。

理由:

```text
state machine は入ったが、複数ケースの集計 helper がまだない
routing summary helper と同じ流れで追加しやすい
runtime に触らず control-plane 観測性を上げられる
次の tiered budget accounting / shadow router へ進む前に blocker 分布を見られる
```
