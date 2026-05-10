# RelayKV Devlog - 2026-05-10 JST
## Activation Policy v1 / Demotion Policy v1 merge

日付確認:
- 本 devlog は **JST 2026-05-10** 基準で作成。
- `relaykv-activation-policy-v1` と `relaykv-demotion-policy-v1` の merge 完了後の区切りとして記録する。
- 前段の `relaykv-drop-policy-sweep-v1` に続く、RelayKV の設計転換フェーズの記録。

---

## 1. 今日の結論

RelayKV の実用設計を、以下の方向へさらに固定した。

    RelayKV is not primarily a KV block selection algorithm.
    RelayKV is a VRAM-aware KV residency controller.

今回の到達点:

    1. 短文脈では RelayKV を実用適用しない activation policy を追加。
    2. diagnostic mode と practical mode を分離。
    3. FullKV demotion 型の dry-run demotion policy を追加。
    4. RECENT + BOUNDARY_NEAR_RECENT を初期保護対象にした。
    5. fixed block0 anchor はデフォルト保護しない方針にした。
    6. selection-style budget policy と demotion policy を分離した。
    7. まだ runtime KV mutation / attention backend / scheduler には接続していない。

これにより、RelayKV の中心問題は以下に更新された。

    限られた working KV に何を選ぶか？

ではなく、

    FullKV から始めて、VRAM budget に合わせて何を安全に demote するか？
    必要な場合だけ、cold 側からより価値の高い block を retrieve / swap するか？

---

## 2. 完了したPR / branch

### 2.1 relaykv-activation-policy-v1

目的:

    RelayKV を短文脈でも強制実行する diagnostic mode と、
    実用上は FullKV を維持して必要時だけ動かす practical mode に分ける。

主な追加:

    relaykv/activation_policy.py

主な概念:

    ActivationState:
    - DISABLED_SHORT_CONTEXT
    - FULLKV_ACTIVE
    - SHADOW_ONLY
    - DEMOTION_CANDIDATE
    - APPLY
    - FALLBACK

JSON output に `activation_policy_decision` を追加。

重要な挙動:

    activation_mode=diagnostic:
      既存同様、短文脈でも診断のために RelayKV policy を強制実行できる。

    activation_mode=practical:
      seq_len が min_relaykv_seq_len 未満なら DISABLED_SHORT_CONTEXT。
      seq_len が working budget 内なら FULLKV_ACTIVE。
      それ以外で RelayKV metadata path を有効化。

Review follow-up:
- `build_activation_decision` / `RelayKVActivationDecision` の package root export を `relaykv/__init__.py` に追加。
- `scripts/run_relaykv_pipeline.py` / `scripts/run_budget_policy_sweep.py` の startup import smoke を確認。

---

### 2.2 relaykv-demotion-policy-v1

目的:

    drop-policy sweep で見えた初期ルールを、runtime非接続の dry-run demotion decision として実装する。

主な追加:

    relaykv/demotion_policy.py

主な出力:

    demotion_policy_decision

主なdecision fields:

    keep_block_ids
    drop_block_ids
    eviction_excluded_block_ids
    eviction_candidate_block_ids
    demoted_block_ids
    reason_labels_by_block
    fallback_reason
    budget_ok
    target_keep_blocks
    total_blocks
    dry_run_only
    demotion_applied

初期ルール:

    - recent protected:
      last recent_blocks は eviction excluded

    - boundary protected:
      recent window 直前の protect_boundary_blocks も eviction excluded

    - prefix protected:
      protect_prefix_blocks が明示指定された場合だけ prefix block を eviction excluded

    - default:
      fixed block0 anchor は強制保護しない

    - demotion:
      eviction candidates から oldest first で demote

    - safety:
      protected block は demoted に入れない

---

## 3. 背景になった観測

前段の `run_drop_policy_sweep.py` では、`seq_len=1024`, `block_size=128`, `layer_idx=14`, `prompt_type=structured` で以下が見えた。

    full:
      keep [0,1,2,3,4,5,6,7]
      diff = 0

    drop_oldest_1:
      drop [0]
      diff は極小

    drop_oldest_2:
      drop [0,1]
      diff はかなり小さい

    drop_boundary_before_recent_1:
      drop [3]
      diff が大きく悪化

    keep_recent4:
      keep [4,5,6,7]
      recent-only baseline

    keep_recent4_boundary1:
      keep [3,4,5,6,7]
      recent-only より大幅改善

    keep_recent4_anchor1_boundary1:
      keep [0,3,4,5,6,7]
      boundary1 からの上積みは小さい

この観測から、以下の初期仮説を採用した。

    重要:
    - RECENT_PROTECTED
    - BOUNDARY_NEAR_RECENT

    現時点で固定保護しない:
    - block0 fixed anchor

    落としやすい:
    - oldest non-protected candidates

---

## 4. Activation Policy の検証

### py_compile

    python -m py_compile \
      relaykv/__init__.py \
      relaykv/activation_policy.py \
      scripts/run_relaykv_pipeline.py \
      scripts/run_budget_policy_sweep.py \
      scripts/run_drop_policy_sweep.py

結果:

    exit=0

### diagnostic mode

    python scripts/run_relaykv_pipeline.py \
      --seq-len 1024 \
      --block-size 128 \
      --layer-idx 14 \
      --prompt-type structured \
      --working-budget-blocks 4 \
      --recent-budget-blocks 4 \
      --anchor-budget-blocks 0 \
      --retrieval-budget-blocks 0 \
      --activation-mode diagnostic \
      --output relaykv_activation_diagnostic_structured_1024_l14.json

確認:

    - import error なし
    - exit=0
    - activation_policy_decision が出力される
    - diagnostic mode では既存の forced diagnostic behavior を維持

### practical short context

    python scripts/run_relaykv_pipeline.py \
      --seq-len 1024 \
      --block-size 128 \
      --layer-idx 14 \
      --prompt-type structured \
      --working-budget-blocks 8 \
      --recent-budget-blocks 4 \
      --anchor-budget-blocks 0 \
      --retrieval-budget-blocks 0 \
      --activation-mode practical \
      --min-relaykv-seq-len 4096 \
      --disable-relaykv-below-budget \
      --output relaykv_activation_practical_short_1024_l14.json

観測:

    activation_state = DISABLED_SHORT_CONTEXT
    relaykv_enabled = false

意味:

    短文脈では実用上 RelayKV を動かさない。
    ただし diagnostic metadata としては観測可能。

---

## 5. Demotion Policy の検証

### py_compile

    python -m py_compile \
      relaykv/demotion_policy.py \
      relaykv/activation_policy.py \
      scripts/run_relaykv_pipeline.py \
      scripts/run_drop_policy_sweep.py \
      scripts/run_budget_policy_sweep.py

結果:

    exit=0

### unit / smoke

    python - <<'PY'
    from relaykv.demotion_policy import build_demotion_decision

    d = build_demotion_decision(
        total_blocks=8,
        target_keep_blocks=5,
        recent_blocks=4,
        protect_boundary_blocks=1,
    )

    print(d)

    assert d.keep_block_ids == [3, 4, 5, 6, 7]
    assert d.drop_block_ids == [0, 1, 2]
    assert d.eviction_excluded_block_ids == [3, 4, 5, 6, 7]
    assert d.demoted_block_ids == [0, 1, 2]
    PY

結果:

    pass

### pipeline dry-run

    python scripts/run_relaykv_pipeline.py \
      --seq-len 1024 \
      --block-size 128 \
      --layer-idx 14 \
      --prompt-type structured \
      --activation-mode diagnostic \
      --demotion-policy-mode dry_run \
      --target-keep-blocks 5 \
      --demotion-recent-blocks 4 \
      --protect-boundary-blocks 1 \
      --protect-prefix-blocks 0 \
      --output relaykv_demotion_policy_dry_run_structured_1024_l14.json

確認結果:

    keep_block_ids = [3,4,5,6,7]
    drop_block_ids = [0,1,2]
    eviction_excluded_block_ids = [3,4,5,6,7]
    block0 は保護されていない
    budget_policy_decision = null
    demotion_applied = false

意味:

    - recent + boundary を保護できている。
    - fixed block0 anchor は protect-prefix-blocks=0 では保護しない。
    - selection-style budget policy と demotion policy が混ざっていない。
    - demotion は metadata-only で、runtimeには適用していない。

### practical disabled + demotion dry-run metadata

    python scripts/run_relaykv_pipeline.py \
      --seq-len 1024 \
      --block-size 128 \
      --layer-idx 14 \
      --prompt-type structured \
      --activation-mode practical \
      --min-relaykv-seq-len 4096 \
      --disable-relaykv-below-budget \
      --demotion-policy-mode dry_run \
      --target-keep-blocks 8 \
      --demotion-recent-blocks 4 \
      --protect-boundary-blocks 1 \
      --output relaykv_demotion_policy_practical_short_disabled.json

確認結果:

    activation_state = DISABLED_SHORT_CONTEXT
    relaykv_enabled = false
    demotion_policy_decision は出る
    dry_run_only = true
    demotion_applied = false
    fallback_reason = fullkv_within_budget

意味:

    practical mode では RelayKV を適用しない条件でも、
    dry-run metadata として demotion decision を観測できる。
    ただし demotion_applied=false のため、実適用とは誤読されにくい。

---

## 6. 現在の設計状態

### 6.1 RelayKVの中心定義

    RelayKV is a VRAM-aware KV residency controller.

日本語:

    RelayKV は、モデル重みロード後の残余VRAMを管理し、
    decode-time working KV を指定予算内に収める KV residency controller。

### 6.2 実用動作イメージ

    1. FullKV が載る間は FullKV を維持
    2. practical activation threshold を超えたら SHADOW_ONLY / DEMOTION_CANDIDATE
    3. recent + boundary などを eviction excluded にする
    4. oldest / low-value eviction candidate を demote
    5. cold 側により価値のある block がある場合だけ retrieve / swap

### 6.3 まだやっていないこと

    - 実KV pool mutation
    - attention backend 接続
    - scheduler変更
    - runtime writeback
    - coldからの実retrieve/swap
    - latency/throughput評価
    - 16K以上の実用長文脈評価

---

## 7. Schema方針

今後、KV blockを固定classで扱うのではなく、以下を分ける。

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

`RECENT / ANCHOR / RETRIEVED` は固定block classではなく、reason label / transition label として扱う。

---

## 8. VRAM budget設計の方向性

今後は budget を global / per-request で分ける。

    global_residual_vram_bytes
    global_working_kv_budget_bytes
    target_concurrent_requests
    request_working_kv_budget_bytes

目的:

    1. OOM回避
    2. per-request / per-thread の動作保証
    3. memory limit による副次的 sparse working KV 化
    4. 副次的な速度改善の可能性

基本方針:

    memory guarantee first,
    speedup as a side effect.

---

## 9. 次にやること

### 第一候補: global / per-request VRAM budget dry-run schema

目的:

    指定VRAM budget から working KV block budget を計算し、
    global / request 単位でログに出す。

候補branch:

    relaykv-vram-budget-schema-v1

内容:

    - global_working_kv_budget_mib
    - request_working_kv_budget_mib
    - target_concurrent_requests
    - kv_bytes_per_token estimate
    - kv_bytes_per_block estimate
    - derived target_keep_blocks
    - allocation_policy = equal_share

### 第二候補: demotion policy sweep 拡張

目的:

    demotion policy の初期ルールが 1024 structured 以外でも成立するか確認。

候補:

    1024 prose
    2048 structured
    8192 structured
    layer 0 / 14 / 27

### 第三候補: retrieval-aware task

目的:

    古いblockが本当に必要なタスクで、demotion / retrieve の価値を見る。

---

## 10. Commit / branch整理

merge後のローカル整理:

    cd ~/work/relay-kv
    git switch main
    git pull origin main
    git status --short
    git branch -d relaykv-activation-policy-v1
    git branch -d relaykv-demotion-policy-v1
    git fetch --prune

---

## 11. 次セッションへの引き継ぎ

次セッションでは以下から開始する。

    RelayKV は selection 型評価から FullKV demotion / residency controller 型へ移行した。

    完了済み:
    - drop-policy-sweep-v1
      - FullKVからkeep/drop blocksを明示して attention diff を測る診断scriptを追加
    - activation-policy-v1
      - diagnostic / practical mode を分離
      - short contextでは practical modeで RelayKV disabled
    - demotion-policy-v1
      - recent + boundary を eviction excluded とする dry-run demotion decision を追加
      - fixed block0 anchor はデフォルト保護しない
      - demotion_applied=false の metadata-only 実装

    重要な観測:
    - block0 は落としても差分が小さい
    - fixed block0 anchor の上積みは小さい
    - recent window直前の boundary block は重要
    - keep_recent4_boundary1 は keep_recent4 より大幅改善

    次にやるなら:
    1. global / per-request VRAM budget dry-run schema
    2. demotion policy sweep を prose/2048/8192/layer違いで確認
    3. retrieval-aware task
