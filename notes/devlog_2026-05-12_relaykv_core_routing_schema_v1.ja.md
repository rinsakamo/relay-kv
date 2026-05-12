# RelayKV Devlog: Core Routing Schema v1

- Date basis: JST 2026-05-12
- Repository: `rinsakamo/relay-kv`
- Scope:
  - PR #32: `Add RelayKV core routing schema`
- Branch:
  - `relaykv-core-routing-schema-v1`
- Merge commit:
  - `5d7e2da44f87292ba7589358cc70fb2dadd07769`

## 1. Summary

RelayKV を **KV block router / tiered KV memory manager** として進めるための core schema を追加した。

今回の変更は schema-only であり、runtime 実行経路には接続していない。

追加された主な型は以下。

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

これにより、直近の VRAM budget / demotion dry-run / sweep / summary helper の上位概念として、RelayKV Core が扱う block metadata と routing decision の package-level contract ができた。

## 2. Context

RelayKV の定義を、単純な KV-cache reduction algorithm ではなく、以下のように更新している。

```text
RelayKV =
  KV block router
  + decode-time working-set controller
  + GPU/RAM/SSD tiered KV memory manager
  + cost-aware materialization and fallback layer
```

設計上の重要点は、モデル最大文脈長までの logical KV state を GPU/RAM/SSD 階層に保持し、decode 時には bounded active KV working set だけを GPU attention に routing すること。

今回の schema は、この設計を実装上の型として固定する最初の PR である。

## 3. Added Files

追加:

```text
relaykv/memory_block.py
relaykv/routing_decision.py
tests/test_memory_block.py
tests/test_routing_decision.py
```

更新:

```text
relaykv/__init__.py
```

## 4. RelayKVMemoryBlock

`RelayKVMemoryBlock` は、RelayKV Core が扱う logical KV block の最小 schema。

主な field:

```text
logical_block_id
logical_token_span
kv_class
residency_level
precision_level
rope_status
cache_position_policy
protected
protection_reason
protection_ttl
demotion_priority
retrieval_priority
reuse_eligible
score_skip_reason
materialization_cost_estimate
last_retrieved_step
retrieval_reuse_count
```

### 4.1 KVClass

```text
RECENT
ANCHOR
RETRIEVED
COLD_CANDIDATE
LOW_VALUE_COLD
```

これは block の役割を表す。

重要なのは、role / location / representation を分離すること。

```text
kv_class != residency_level
kv_class != precision_level
```

例えば、`ANCHOR` は常に GPU resident とは限らないし、`COLD_CANDIDATE` は常に SSD resident とも限らない。

### 4.2 ResidencyLevel

```text
GPU
CPU_RAM
SSD
```

これは block の現在の所在を表す。

### 4.3 PrecisionLevel

```text
FP16
BF16
INT8
INT4
COMPRESSED
```

これは block の表現形式を表す。

今後の `COMPRESSED_COLD` / quantized KV / SSD cold KV の拡張に備える。

### 4.4 RopeStatus / CachePositionPolicy

```text
RopeStatus:
  APPLIED_ORIGINAL_POSITION
  RAW_NEEDS_ROPE
  UNKNOWN

CachePositionPolicy:
  ORIGINAL_POSITION_ONLY
  COMPACTED_POSITION_FORBIDDEN
```

RelayKV の短期方針は、RoPE-applied KV を値変更なしで移動すること。

```text
K tensor values must not be recomputed.
K tensor values must not be re-RoPE'd.
V tensor values must not be changed.
original logical_token_span must be preserved.
```

そのため、RoPE / cache_position の状態を block schema に入れた。

## 5. RelayKVMemoryBlock Validation

`RelayKVMemoryBlock` には軽量 validation を追加した。

```text
logical_block_id >= 0
logical_token_span start/end validity
retrieval_reuse_count >= 0
protection_ttl is None or >= 0
```

これは runtime safety というより、今後の routing policy / adapter / materialization が前提にする basic invariant を固定するためのもの。

## 6. RelayKVDecision

`RelayKVDecision` は、routing policy の出力 schema。

主な field:

```text
execution_mode
selected_active_block_ids
protected_block_ids
demotion_candidate_block_ids
demoted_block_ids
retrieved_block_ids
prefetched_block_ids
reused_block_ids
newly_retrieved_block_ids

estimated_working_kv_bytes
estimated_ram_swap_bytes
estimated_ssd_read_bytes
estimated_materialization_latency_ms
estimated_policy_compute_ms
estimated_attention_tokens_saved
estimated_net_benefit_ms

fallback_reason
apply_blocked_reason
shadow_compare_passed
selection_stability_ratio
```

これにより、単なる `keep_block_ids` / `drop_block_ids` ではなく、routing decision 全体を表現できるようになった。

## 7. ExecutionMode

追加された execution mode:

```text
FULLKV_GPU
SHADOW_ONLY
APPLY_VRAM_WORKING
APPLY_RAM_BACKED
FALLBACK_FULLKV_RAM
FALLBACK_FULLKV_TIERED
FALLBACK_RECENT_ANCHOR
FALLBACK_TRUNCATION
```

この enum によって、RelayKV の状態遷移と fallback path を明示できる。

## 8. Summary Methods

以下の `summary()` を追加した。

```text
RelayKVMemoryBlock.summary()
RelayKVDecision.summary()
```

目的:

```text
JSON-serializable dict を返す
sweep / smoke / devlog / metadata log にそのまま出せる
```

## 9. Package Exports

`relaykv/__init__.py` から新しい schema types を export した。

これにより、今後の script / tests / policy code から以下のように使える。

```text
from relaykv import RelayKVMemoryBlock, RelayKVDecision
from relaykv import KVClass, ResidencyLevel, PrecisionLevel
from relaykv import RopeStatus, CachePositionPolicy, ExecutionMode
```

## 10. Validation

Targeted tests:

```text
python -m pytest tests/test_memory_block.py tests/test_routing_decision.py -q
7 passed in 2.08s
```

Full suite:

```text
python -m pytest -q
43 passed in 4.57s
```

## 11. Constraints Preserved

今回の変更では以下を行っていない。

```text
model loading
runtime KV pool mutation
attention backend connection
scheduler change
runtime writeback
actual KV eviction
actual KV materialization
SGLang/vLLM adapter integration
```

schema-only のため、既存の実験 pipeline や sweep の runtime behavior は変わらない。

## 12. Current Position

PR #24〜#32 で、RelayKV は以下の段階まで来た。

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
```

設計上のつながり:

```text
global/per-request VRAM budget
  -> derived target keep blocks
  -> demotion dry-run keep/drop
  -> candidate KV
  -> attention comparison
  -> offline Markdown/CSV summary
  -> RelayKVMemoryBlock / RelayKVDecision schema
```

これにより、今後の routing policy / memory pressure state machine / tiered budget accounting の土台ができた。

## 13. Next Candidate

次の実装候補は以下。

### Candidate A: Routing Policy Skeleton v1

branch:

```text
relaykv-routing-policy-skeleton-v1
```

目的:

```text
既存の demotion dry-run を RelayKVDecision に接続し、
selected_active_block_ids / demoted_block_ids / fallback_reason / execution_mode
として返す skeleton を作る。
```

### Candidate B: Memory Pressure State Machine v1

branch:

```text
relaykv-memory-pressure-state-machine-v1
```

目的:

```text
DISABLED_SHORT_CONTEXT
SHADOW_ONLY_WARMUP
BUDGET_PRESSURE
APPLY_VRAM_WORKING_READY
APPLY_RAM_BACKED_READY
FALLBACK_REQUIRED
```

のような execution mode / readiness state を log-only で判断する。

### Candidate C: Tiered Budget Accounting v1

branch:

```text
relaykv-tiered-budget-accounting-v1
```

目的:

```text
GPU working KV budget
CPU RAM backup KV budget
SSD cold KV budget
RAM/SSD/materialization I/O budget
```

を capacity と I/O に分けて accounting する。

## 14. Recommendation

次は Candidate A がよい。

```text
relaykv-routing-policy-skeleton-v1
```

理由:

```text
#32 の RelayKVDecision をすぐ使える
既存 demotion_policy と自然につながる
runtime に触らず schema の利用実績を作れる
memory pressure / tiered budget の前に routing output の形を固められる
```
