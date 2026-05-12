# RelayKV Devlog: Memory Pressure Summary Helper v1

- Date basis: JST 2026-05-12
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #38: `Add RelayKV memory pressure summary helper`
- Branch:
  - `relaykv-memory-pressure-summary-helper-v1`
- Merge commit:
  - `12d8f210e979cec7c47c9487b1e5464c15cb8dfe`

## 1. Summary

RelayKV の memory pressure decision を offline で集計する helper を追加した。

今回の PR により、`RelayKVMemoryPressureDecision` またはその `summary()` dict を複数件まとめて、state / execution mode / blocker / fallback reason の分布を JSON-serializable な summary として取得できるようになった。

主な追加:

```text
relaykv/memory_pressure_summary.py
tests/test_memory_pressure_summary.py
```

主な helper:

```text
summarize_memory_pressure_decisions()
```

この変更により、PR #36 で追加した memory pressure state machine の出力を、sweep / smoke / future JSON logs で集計できる観測 layer ができた。

## 2. Context

PR #36 では、以下を追加した。

```text
RelayKVMemoryPressureState
RelayKVMemoryPressureDecision
decide_memory_pressure_state()
```

これにより、RelayKV の readiness / blocked reason / fallback state を log-only に判定できるようになった。

ただし、複数 request / 複数 sweep case / 複数 synthetic scenario をまとめたときに、以下を数える helper はまだなかった。

```text
どの state が多いか
どの execution_mode が多いか
なぜ SHADOW_ONLY_WARMUP に留まったか
fallback_required がどれくらいあるか
routed_ready がどれくらい出たか
fullkv_within_budget がどれくらいあるか
```

今回の PR #38 は、この集計 helper を追加した。

## 3. Added / Changed Files

追加:

```text
relaykv/memory_pressure_summary.py
tests/test_memory_pressure_summary.py
```

更新:

```text
relaykv/__init__.py
```

## 4. summarize_memory_pressure_decisions()

追加した main helper:

```text
summarize_memory_pressure_decisions()
```

入力:

```text
Iterable[RelayKVMemoryPressureDecision | dict]
```

対応する入力形式:

```text
RelayKVMemoryPressureDecision object
RelayKVMemoryPressureDecision.summary() が返す dict
object / dict の mixed input
unknown state / reason を含む dict
empty input
```

出力:

```text
JSON-serializable dict
```

## 5. Aggregated Fields

主な出力 fields:

```text
total_decisions
state_counts
execution_mode_counts
apply_blocked_reason_counts
fallback_reason_counts
short_context_count
budget_pressure_count
routed_ready_count
shadow_warmup_count
fallback_required_count
fullkv_within_budget_count
shadow_compare_not_ready_count
shadow_compare_failed_count
labels_not_ready_count
host_backup_unavailable_count
selection_unstable_count
net_benefit_too_low_count
```

意図:

```text
state_counts
  RelayKVMemoryPressureState の分布を見る。

execution_mode_counts
  FULL_ATTENTION / FULLKV_GPU / SHADOW_ONLY / RELAYKV_ROUTED などの出現分布を見る。

apply_blocked_reason_counts
  apply-ready にならなかった理由を数える。

fallback_reason_counts
  fallback-required になった policy / safety reason を数える。

routed_ready_count
  RELAYKV_ROUTED_READY に到達した件数を見る。

shadow_warmup_count
  SHADOW_ONLY_WARMUP に留まった件数を見る。

fullkv_within_budget_count
  RelayKV 不要で FullKV が予算内に収まる件数を見る。
```

## 6. Counting Behavior

設計上の方針:

```text
empty input は total_decisions=0 と空 counter を返す
None reason は reason counter では skip する
enum values は string に normalize する
unknown state / reason は reject せず count する
missing optional fields で crash しない
object と dict の mixed input を許容する
```

これにより、future JSON logs や sweep output の schema が少し増減しても、summary helper が壊れにくい。

## 7. Blocker-Specific Counts

PR #36 の state machine で導入した主な blocker を直接 count できるようにした。

```text
shadow_compare_not_ready_count
shadow_compare_failed_count
labels_not_ready_count
host_backup_unavailable_count
selection_unstable_count
net_benefit_too_low_count
```

このため、RelayKV が `RELAYKV_ROUTED_READY` に進まない原因を、sweep 後にすぐ比較できる。

例:

```text
shadow_compare_not_ready が多い
  -> shadow compare warmup / measurement path が足りない

labels_not_ready が多い
  -> metadata / labels preparation が遅れている

host_backup_unavailable が多い
  -> host backup copy / cold candidate source が足りない

selection_unstable が多い
  -> N-step reuse / hysteresis / stability policy が必要

net_benefit_too_low が多い
  -> materialization cost estimate / transfer cost / benefit threshold の調整が必要
```

## 8. Relationship to Existing Pipeline

現在の logical pipeline:

```text
global/per-request VRAM budget
  -> projected FullKV / residual budget
  -> memory pressure state
  -> memory pressure summary
  -> target keep blocks
  -> demotion dry-run keep/drop/fallback
  -> RelayKVDecision
  -> routing summary-ready metadata
```

今回の helper は、runtime path ではなく offline observation / sweep analysis 用。

## 9. Test Coverage

追加された test coverage:

```text
empty input returns total_decisions=0
RelayKVMemoryPressureDecision objects are accepted
summary() dicts are accepted
mixed object/dict input is accepted
state_counts are aggregated
execution_mode_counts are aggregated
apply_blocked_reason_counts skip None and count known blockers
fallback_reason_counts skip None and count fallback reasons
routed_ready_count is aggregated
shadow_warmup_count is aggregated
fallback_required_count is aggregated
fullkv_within_budget_count is aggregated
short_context_count is aggregated
budget_pressure_count is aggregated
blocker-specific counts are aggregated
unknown state / reason does not crash and is counted
summary output is JSON-serializable
```

## 10. Validation

PR body records the intended validation commands:

```text
python -m pytest tests/test_memory_pressure_summary.py -q
python -m pytest -q
```

The PR body did not include filled numeric results, but the change was merged after focused tests were added.

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

この PR は offline summary helper のみであり、runtime behavior は変更していない。

## 12. Current Position

PR #24〜#38 で、RelayKV の dry-run / control-plane は以下まで進んだ。

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

#38
Memory pressure summary helper
```

現在、RelayKV Core は以下を log-only に表現・集計できる。

```text
budget pressure
short context disable
fullkv within budget
shadow warmup blocker
fallback-required reason
routed-ready count
execution mode distribution
apply blocked reason distribution
```

## 13. Next Candidate

次は以下が自然。

```text
relaykv-memory-pressure-sweep-integration-v1
```

目的:

```text
existing sweep / smoke outputs
  -> decide_memory_pressure_state()
  -> summarize_memory_pressure_decisions()
```

ただし、まだ runtime には触らない。

候補作業:

```text
既存の vram budget / demotion sweep helper に memory pressure summary を optional に追加
あるいは新しい small script で synthetic memory pressure cases を JSON 出力
state_counts / blocker_counts / routed_ready_count を result JSON に含める
```

## 14. Recommendation

次の実装は `relaykv-memory-pressure-sweep-integration-v1` がよい。

理由:

```text
state machine と summary helper は揃った
次は実際の sweep JSON に載せて観測できる状態にするのが自然
runtime に触らず、control-plane の可視性を高められる
将来の tiered budget accounting / shadow router の blocker 分布を先に見られる
```
