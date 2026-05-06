# RelayKV統合設計メモ: Working KV制御・Anchor予算・投機的プリフェッチ

作成日: 2026-05-06 JST  
日付確認: このメモはJST基準で 2026-05-06 に作成した設計引き継ぎ用メモです。

---

## 0. このメモの目的

このメモは、これまで検討してきたRelayKVの設計アイディアを、他の実装スレッドへ渡しやすい形に統合したものです。

RelayKVは単なるKV cache削減アルゴリズムではなく、長文コンテキスト・低VRAM環境・serving engine統合を前提にした、**decode-time KV working-set control layer** として扱う。

特に今後の設計では、以下を重視する。

- SGLang専用機能に閉じない
- vLLMにも同等レベルで刺せる設計にする
- RelayKV Coreをengine非依存にする
- SGLang/vLLM固有処理はAdapterに閉じ込める
- Prefix cache / residency manager / scheduler / attention backendと責務を分離する
- まずmetadata/log受け皿を作り、attentionやKV poolには触らない
- 将来の大設計を壊さないschemaを先に置く

---

## 1. RelayKVの上位定義

RelayKVは、SGLang専用のKV削減機能ではなく、SGLang / vLLM / 将来のGatewayやControl Planeに刺さる、engine-agnosticなdecode-time KV working-set control layerである。

```text
RelayKV = long-context decode時に、
残VRAM予算内でどのKVをworking setとしてGPU上・attention対象に戻すかを決める制御層
```

### 1.1 基本構造

```text
RelayKV Core:
  engine非依存のmetadata / budget / policy / scoring / decisionを扱う

SGLang Adapter:
  ForwardBatch, req_pool_idx, RadixAttention, RadixTree, HiCache,
  SGLang KV pool / attention backendとの対応を扱う

vLLM Adapter:
  KVCacheManager, KVCacheBlock, block table, PagedAttention,
  prefix cache hash/ref_cntとの対応を扱う

Gateway / Control Plane:
  将来のSMG / router / external policy / prefetch / routing / semantic indexを扱う
```

### 1.2 Coreに入れてはいけないもの

RelayKV Coreには、engine固有名を入れない。

避けるもの:

```text
ForwardBatch
req_pool_idx
RadixTree node
RadixAttention backend
HiCache内部参照
KVCacheBlock
PagedAttention page id
vLLM block table
```

Core側は以下の抽象schemaを使う。

```text
engine_request_id
logical_sequence_id
logical_block_id
token_span
layer_id
kv_head_group
kv_class
residency_level
precision_level
prefix_cache_role
anchor_score
retrieval_score
decision_state
layer_budget_policy
working_kv_budget
fallback_reason
```

---

## 2. 6本柱としての統合設計

これまでのアイディアは、以下の6本柱にまとめられる。

```text
RelayKV
├─ 1. Engine-agnostic KV Working-Set Control
├─ 2. Role-aware KV Classes
├─ 3. Budgeted Retrieval and Anchor Control
├─ 4. Retrieval-Critical Profiling and Query-Block Scoring
├─ 5. Speculative Block Prefetch and Temporal Reuse
└─ 6. Shadow-first Safety and Gated Apply
```

---

## 3. 1本目: Engine-agnostic KV Working-Set Control

### 3.1 目的

SGLang / vLLM / 将来のGatewayで同じRelayKV Coreを使えるようにする。

```text
RelayKV Core:
  logical metadataだけを扱う

Engine Adapter:
  logical_block_id -> engine_block_ref へ変換する

Materializer:
  engine_block_ref -> 実KV tensor / page / span へ変換する
```

### 3.2 logical_block_id と engine_block_ref の分離

最重要の設計境界。

```text
logical_block_id:
  RelayKV内部の抽象block ID

engine_block_ref:
  SGLang/vLLM固有の物理参照
  Adapter境界でのみ扱う

token_span:
  元prompt上の論理token範囲
```

### 3.3 Adapter責務

SGLang Adapter:

```text
ForwardBatch -> RelayKVRequestMeta
req_pool_idx -> engine_request_id / engine_block_ref
RadixTree / HiCache metadata -> prefix/cache/residency feature
SGLang materialization -> RelayKVMaterializedKV
```

vLLM Adapter:

```text
KVCacheManager / KVCacheBlock -> RelayKVBlockMeta
block table -> token_span / engine_block_ref
PagedAttention path -> RelayKVMaterializedKV
prefix cache hash/ref_cnt -> prefix-derived anchor feature
```

---

## 4. 2本目: Role-aware KV Classes

RelayKVでは、KVの分類を「保存場所」ではなく「役割」として扱う。

```text
KV class = 役割
precision_level = 表現
residency_level = 所在
```

### 4.1 KV class

短期の論理KV class:

```text
RECENT:
  直近文脈。品質安定の基礎。

ANCHOR:
  system prompt, schema, header, prefix-derived span, sink-like token,
  gather-like aggregation pointなど、常に残す価値が高いKV。

RETRIEVED:
  今のquery/decode stepでcold tierからworking setへ戻すKV。

COLD_CANDIDATE:
  host/cold tierに保持される候補KV。

TRANSIENT:
  speculative decoding, draft tree, branch verificationなどの一時KV。
  短期実装では物理classにせず、decode/log/metadata概念でもよい。
```

中期以降:

```text
HEAVY_HITTER:
  実測で重要性が高いKV。

COMPRESSED_COLD:
  圧縮表現で保持されるcold KV。

HIERARCHICAL_REMOTE:
  remote / L3 / distributed storage上のKV。

DROPPED:
  破棄済み、または復元不可のKV。
```

### 4.2 Prefix cacheとの関係

Prefix cacheはRelayKVそのものではない。  
Prefix cacheはengine側のprefix reuse layerであり、RelayKVではread-only metadata sourceとして扱う。

```text
Prefix cache:
  同一prefixのKVを再利用するengine側最適化

Anchor KV:
  decode時のworking setに残すべき品質保護class
```

RelayKVでは、Prefix cache spanをAnchor候補として使う。

```text
Prefix-derived Anchor
Shared Prefix Anchor
```

ただし、prefix由来だから無条件にAnchorにするのではなく、Anchor budget内で選ぶ。

```text
prefix_cache_bonusをanchor_scoreに加える
ただしB_anchor内で選ぶ
```

---

## 5. 3本目: Budgeted Retrieval and Anchor Control

RelayKVの基本budget model:

```text
B_total_working_kv =
  B_recent
+ B_anchor
+ B_transient
+ B_retrieval
```

将来的には、layer別・kv_head_group別に拡張する。

```text
B_retrieval[layer]
B_retrieval[layer][kv_head_group]
```

### 5.1 budgetの意味

```text
B_recent:
  直近文脈用の予算

B_anchor:
  品質保護用Anchor予算
  Prefix-derived Anchorもここに入る

B_transient:
  speculative / draft / branch / verify用の一時KV予算

B_retrieval:
  query-dependentにcold tierから戻すKV予算
```

### 5.2 budget単位

初期実装:

```text
working_block_count
working_token_count
```

中期以降:

```text
estimated_working_kv_bytes
estimated_working_kv_mib
GPU page count
layer-wise bytes
kv_head_group-wise bytes
```

設計原則:

```text
Policy budget:
  token / block単位で扱う

Runtime budget:
  bytes / MiB / page単位で検証する
```

### 5.3 Anchor過剰保護問題

Prefix-derived Anchorを入れると、Anchor候補が増えすぎる可能性がある。

問題:

```text
prefix spanが大きい
↓
全部Anchor候補になる
↓
B_anchorが膨らむ
↓
B_retrievalが残らない
```

対策:

```text
B_anchor_max
anchor_score_threshold
anchor_top_k_blocks
anchor_type_priority
```

Anchor優先順位の例:

```text
1. explicit structural anchor
   schema / header / system / format

2. sink-like anchor
   initial tokens / BOS

3. prefix-derived anchor
   shared prefix span由来

4. gather-like anchor
   delimiter / key-value aggregation point

5. learned/heavy-hitter anchor
   実測で重要なblock
```

---

## 6. 4本目: Retrieval-Critical Profiling and Query-Block Scoring

最新論文系のattention head分析、BlockRank、S3-Attention、Massive Q/K、Gather-and-Aggregate head分析は、この柱にまとめる。

目的:

```text
どのlayer / kv_head_group / blockが検索に効くかを測り、
working KV budgetとblock selectionに反映する
```

### 6.1 Retrieval-critical layer / kv_head_group profiling

Transformer-SSM hybridやretrieval-aware head analysisの考え方は、RelayKVではモデル変換ではなく、retrieval-critical profilingとして使う。

```text
論文側:
  retrievalに効く少数Attention headを特定する

RelayKV側:
  retrievalに効くlayer / kv_head_groupを特定し、
  B_retrieval[layer][kv_head_group]へ反映する
```

GQAモデルでは、query head単位ではなくkv_head_group単位へ集約する。

```text
head score
  ↓
kv_head_group_retrieval_score
```

### 6.2 Gather / Aggregate的な解釈

```text
Gather-like signal:
  header / schema / delimiter / key-value aggregation point
  → Anchor scoreへ反映

Aggregate-like signal:
  query-dependent relevant block
  → RETRIEVED scoreへ反映
```

### 6.3 query-to-block scoring

BlockRank / attention-aligned retrieval / S3-Attention系の考え方を、RelayKVではquery-block relevanceとして扱う。

候補:

```text
query_block_score
query_to_block_max
middle_layer_query_block_score
block_relevance_rank
```

これまでの軽量scoring変更で差が小さい場合、次に試す価値が高いのはquery-to-block系。

### 6.4 Massive Q/Kの扱い

Massive Q/K values系の知見は、削ると危険なblockを検出する特徴量として使う。

```text
massive_qk_score
query_key_outlier_score
rope_sensitive_anchor_score
```

---

## 7. 5本目: Speculative Block Prefetch and Temporal Reuse

これはRelayKVの実時間性能に効く中核テーマ。

```text
良いblockを選ぶ
  +
将来使いそうなblockを投機的に選ぶ
  +
使う前にprefetchする
  +
安定しているselected_block_idsを数step再利用する
```

設計名:

```text
Speculative Block Prefetch and Temporal Reuse
```

日本語:

```text
投機的ブロック選択・プリフェッチ・時間再利用ポリシー
```

### 7.1 なぜ重要か

毎decode stepで最適なblockを選び直すと、理論上は良くても実時間では遅くなる可能性がある。

```text
毎stepで違うblockを選ぶ
↓
CPU/GPU転送が不規則
↓
prefetchできない
↓
cache localityが悪化
↓
attention対象は減っても実時間は遅くなる
```

そのため、RelayKVは以下をpolicyに含める。

```text
future_reuse_probability
prefetchability_score
materialization_cost_estimate
selection_volatility_score
stable_selection_window
selection_hysteresis
prefetch_hint_block_ids
prefetch_deadline_step
reused_block_ids
newly_retrieved_block_ids
selection_refresh_reason
```

### 7.2 画像/動画生成DiT系からの流用

画像/動画生成DiT高速化研究からは、以下の発想を流用する。

```text
TokenCache / ToCa:
  token/blockごとにcache/reuse判断

Region-Adaptive Sampling:
  focus領域だけ更新し、他はreuse

Pyramid Attention Broadcast:
  attention/selectionを数step再利用

MixCache:
  状況に応じてcache粒度を変える

SDTM:
  stage-aware importance

FastCache:
  redundancy/motion-aware pruning

DiffSparse:
  fixed budget下でlayer-wise sparsityを最適化

ToPi:
  reference / target / role-aware token pruning
```

RelayKVへの変換:

```text
temporal_reuse_enabled
selection_hysteresis
working_set_stability_score
focus_block_score
selection_refresh_interval
policy_granularity
stage_aware_budget
reused_block_ids
newly_retrieved_block_ids
cache_reuse_reason_counts
```

### 7.3 近縁分野からの流用

MoE expert caching:

```text
次に必要なexpertを予測してcache/prefetch
↓
次に必要なKV blockを予測してprefetch
```

KV制約付きonline scheduling:

```text
working_kv_budget_bucket
batching_compatibility_score
expected_decode_kv_pressure
```

Disaggregated inference:

```text
materialization_cost_estimate
transfer_coalescing_group
prefetch_window_steps
```

RAG lookahead retrieval:

```text
lookahead_window_steps
next_query_block_prediction
prefetch_before_decode_step
```

Agent memory / neural long-term memory:

```text
revisitable_block
verified_anchor
provisional_anchor
memory_commit_state
COMPRESSED_COLD
HIERARCHICAL_REMOTE
```

### 7.4 実験案

中期のPyTorch/dry-run policy実験:

```text
1. 現stepのbest blockだけでなく、次N stepで使いそうなblockも候補に入れる
2. selected_block_idsをN step固定して再利用する
3. score_marginやrisk signalが悪化した時だけrefreshする
4. mean_abs_diff, task accuracy, refresh回数, materialization回数を比較する
```

重要仮説:

```text
毎step最適なblockを選ぶより、
少し妥協しても安定したblock setを使う方が速い可能性が高い
```

---

## 8. 6本目: Shadow-first Safety and Gated Apply

RelayKVは出力を変え得るため、最初から常時ONにしない。

実装順:

```text
OBSERVE
JOIN
DRY_RUN
MATERIALIZE_SHADOW
WORKING_KV_ASSEMBLY_DRY_RUN
SHADOW_COMPARE
GATED_APPLY
FALLBACK
```

### 8.1 decision_state

fallbackは例外ではなく、通常のdecision stateとして扱う。

```text
APPLY:
  RelayKV working KVを実際に適用

SHADOW_ONLY:
  RelayKV pathを裏で計算・比較するが出力には使わない

FALLBACK:
  full/normal pathを使う
```

### 8.2 fallback条件

最低限のfallback理由:

```text
metadata_mismatch
token_span_mismatch
shape_mismatch
dtype_mismatch
device_mismatch
layer_mismatch
kv_head_group_mismatch
position_mismatch
rope_mismatch
attention_mask_mismatch
budget_overflow
policy_uncertain
materialization_failed
attention_compare_too_large
```

### 8.3 quality / safety指標

```text
mean_abs_diff
max_abs_diff
l2_diff
top5_overlap
first_diff_index
same_first_code
same_output_ids
task_accuracy
anchor_coverage
score_margin
fallback_count
apply_count
shadow_count
```

### 8.4 MTP / DFlash / DDTreeとの関係

MTP / DFlash / DDTreeは、RelayKVの代替ではなくdecode高速化レイヤー。

```text
MTP / DFlash / DDTree:
  未来tokenをdraftし、decode step数を減らす

RelayKV:
  target verify / attention時のKV working setを制御する
```

接続点:

```text
speculative_mode
draft_token_count
verify_token_count
accepted_token_count
rejected_token_count
draft_tree_depth
draft_branch_count
transient_kv_budget_tokens
shared_prefix_anchor_block_ids
```

注意:

```text
speculative decodingのtarget verify pathをRelayKVで近似すると、
acceptance判定が変わり、lossless性を壊す可能性がある。
最初はfull/normal verifyを維持し、RelayKV verifyはshadow compareから始める。
```

---

## 9. Metadata schema案

### 9.1 RelayKVRequestMeta

```python
RelayKVRequestMeta:
    engine_name: str
    engine_request_id: str
    logical_sequence_id: str
    seq_len: int
    decode_step: int
    batch_index: int | None
    model_arch: str | None
    attention_type: str | None

    # budget / scheduling
    working_kv_budget_bucket: str | None
    expected_decode_kv_pressure: float | None
    lookahead_window_steps: int | None

    # speculative decoding future fields
    speculative_mode: str | None
    draft_token_count: int | None
    verify_token_count: int | None
    accepted_token_count: int | None
    rejected_token_count: int | None
```

### 9.2 RelayKVBlockMeta

```python
RelayKVBlockMeta:
    logical_block_id: int
    token_span: tuple[int, int]
    layer_id: int
    kv_head_group: int | None

    kv_class: str
    residency_level: str
    precision_level: str
    prefix_cache_role: str | None

    # Adapter境界のみ
    engine_block_ref: object | None

    # scoring
    anchor_score: float | None
    retrieval_score: float | None
    query_block_score: float | None
    middle_layer_query_block_score: float | None
    retrieval_criticality_rank: int | None
    gather_anchor_score: float | None
    aggregate_retrieval_score: float | None
    massive_qk_score: float | None

    # temporal reuse / prefetch
    future_reuse_probability: float | None
    prefetchability_score: float | None
    materialization_cost_estimate: float | None
    working_set_stability_score: float | None
    selection_volatility_score: float | None
    last_retrieved_step: int | None
    retrieval_reuse_count: int | None

    # memory role
    revisitable_block: bool
    verified_anchor: bool
    provisional_anchor: bool
    memory_commit_state: str | None

    # locality
    locality_penalty: float | None
    layout_group_id: str | None
```

### 9.3 RelayKVGroupMeta

```python
RelayKVGroupMeta:
    layer_id: int
    kv_head_group: int
    retrieval_head_score: float | None
    kv_head_group_retrieval_score: float | None
    query_dependent_group_score: float | None
    group_budget_bonus: int | None
```

### 9.4 RelayKVPolicyDecision

```python
RelayKVPolicyDecision:
    decision_state: str  # APPLY / SHADOW_ONLY / FALLBACK
    fallback_reason: str | None

    selected_block_ids: list[int]
    recent_token_range: tuple[int, int] | None
    anchor_block_ids: list[int]
    retrieved_block_ids: list[int]
    cold_candidate_block_ids: list[int]

    # budget
    layer_budget_policy: dict
    working_kv_budget_tokens: int | None
    estimated_working_kv_mib: float | None

    # anchor / retrieval accounting
    anchor_budget_tokens: int | None
    retrieval_budget_tokens: int | None
    transient_kv_budget_tokens: int | None

    # temporal reuse / prefetch
    temporal_reuse_enabled: bool
    stable_selection_window: int | None
    selection_refresh_interval: int | None
    selection_refresh_reason: str | None
    reused_block_ids: list[int]
    newly_retrieved_block_ids: list[int]
    selection_stability_ratio: float | None
    selection_reason_counts: dict[str, int]

    prefetch_hint_block_ids: list[int]
    prefetch_deadline_step: int | None
    prefetch_priority: float | None
    transfer_coalescing_group: str | None

    # speculative decoding future fields
    speculative_verify_enabled: bool
    draft_tree_depth: int | None
    draft_branch_count: int | None
    ancestor_mask_mode: str | None
    transient_token_count: int | None
    shared_prefix_anchor_block_ids: list[int]
```

---

## 10. 今すぐ実装スレッドへ渡す場合の範囲

現フェーズでは、全アイディアを実装しない。

現在フェーズ:

```text
SGLang metadata-only / dry-run準備
```

今回入れるもの:

```text
将来拡張のmetadata/log受け皿
```

入れないもの:

```text
attention変更
KV pool read/write
scheduler変更
prefix cache変更
speculative decoding本体
learned router
real prefetch
real materialization
RadixTree/RadixAttention変更
vLLM実装
SMG/Gateway実装
```

最小追加候補:

```text
Retrieval-critical profiling fields
Temporal reuse / prefetch fields
Decision state / fallback fields
Prefix-derived Anchor fields
B_transient / speculative fields
```

短縮した実装依頼文:

```text
今は将来のpolicy実装ではなく、RelayKV Core/metadata/logの受け皿を予約するだけ。
attention / KV pool / scheduler / prefix cacheには触らない。
SGLang固有名はAdapter/runtime observation側に閉じ込め、
Coreはlogical_block_id / token_span / kv_head_group / kv_class / budget / decision_stateのみを扱う。
```

---

## 11. 実装ロードマップ

### Phase 0: metadata observation

```text
ForwardBatch上で request_id / req_pool_idx / seq_len をread-only観測
出力変更なし
```

### Phase 1: candidate summary join

```text
runtime observation metadata と host backup copy candidate summary をjoin
KV poolは読まない
copyもしない
```

### Phase 2: dry-run policy

```text
RECENT / ANCHOR / RETRIEVED / COLD_CANDIDATE を選ぶだけ
attentionへ接続しない
```

### Phase 3: safe materialization

```text
selected candidateを安全にmaterializeできるか確認
出力には使わない
shape / dtype / device / layer / kv_head_group / token_span を確認
```

### Phase 4: working KV assembly dry-run

```text
RECENT + ANCHOR + RETRIEVED をworking KVとして組めるか確認
position / RoPE / mask / token orderを検証
```

### Phase 5: shadow attention compare

```text
normal SGLang/vLLM pathは維持
RelayKV pathを裏で比較
mean_abs_diff / top5_overlap / first_diff_index等を見る
```

### Phase 6: gated attention connection

```text
限定条件でRelayKV working KVを適用
fallback可能な状態で始める
```

### Phase 7: quality benchmark

```text
needle retrieval
table lookup
structured prompt
code lookup
long prose
format following
```

### Phase 8: residual VRAM budget mode

```text
残VRAM予算内でworking KVを制御
RTX 3060 12GBなど低VRAM環境を意識
```

### Phase 9: optimization

```text
query-to-block scoring
retrieval-critical profiling
temporal reuse
speculative block prefetch
selection hysteresis
```

### Phase 10以降: Control Plane / Rust / Gateway

```text
RelayKV metadata/policy/indexを外部Control Plane化
SMG/Gateway/router/prefetch hintへ接続
必要に応じてRust化
```

---

## 12. さらに短い要約

RelayKVの設計思想は、以下に圧縮できる。

```text
1. 何を残すか
   RECENT / ANCHOR / RETRIEVED / TRANSIENT

2. どれだけ残すか
   B_recent / B_anchor / B_transient / B_retrieval

3. いつ戻すか
   speculative selection / prefetch / temporal reuse

4. どこで動かすか
   Core / Adapter / Gateway

5. いつ適用するか
   shadow-first / gated apply / fallback
```

最重要の研究テーマ名:

```text
Retrieval-Critical Working-Set Selection
Speculative Block Prefetch and Temporal Reuse
```

一文でまとめると:

```text
RelayKVは、検索に効くKV blockを重要度・予算・将来再利用確率・転送コスト・選択安定性に基づいて選び、
必要になる前にprefetchし、安定している間は複数decode stepで再利用する、
engine-agnosticなdecode-time KV working-set control layerである。
```

---

## 13. commit用メモ

このメモをrepoへ入れる場合の候補パス:

```text
notes/relaykv_unified_design_working_set_prefetch_2026-05-06_ja.md
```

commit例:

```bash
git add notes/relaykv_unified_design_working_set_prefetch_2026-05-06_ja.md

git commit -m "Document RelayKV unified working-set and prefetch design"

git push -u origin "$(git branch --show-current)"
```

fork remoteを `mine` にしている場合:

```bash
git push -u mine "$(git branch --show-current)"
```
