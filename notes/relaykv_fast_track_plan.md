# RelayKV Fast-Track Implementation Plan
## Three-Tier Memory + 5 Key Ideas
## (Phase / Commit / Metrics / Exit Criteria)

---

## 0. 目的

三層設計

- `recent`
- `anchor`
- `retrieval`

を土台にしつつ、次の5つのアイデアを最小実験で順に導入する。

1. 固定 GPU live KV budget
2. anchor bank
3. 二段階 retrieval / rerank
4. warmup 限定 adaptive budget
5. 非同期プリフェッチ

狙いは、

**RelayKV プロジェクトの実験を最小化しつつ、最速で実装可能な順番で前進すること**

である。

---

## 1. 全体戦略

### 実装の原則
- 一度に増やす自由度は1つだけ
- 各段階が単独で比較可能であること
- 比較ケースは最小限に固定すること
- retrieval 改善と runtime 改善を混ぜすぎないこと

### 固定する主要評価ケース
- `seq_len = 4096`
- `prompt_type = prose` を主ケース
- `structured` を補助ケース
- `layer = 14, 27`

### 最小評価指標
- retrieval recall@k
- mean_abs_diff
- first divergence step
- top-1 changed
- 簡易品質判定
  - 強要約一致
  - 重要点抽出一致
  - フォーマット維持

---

## 2. フェーズ表

| Phase | 主目的 | 導入内容 | ねらい |
|---|---|---|---|
| Phase 1 | メモリ設計の土台化 | 固定 GPU live KV budget + 三層設計 | 今後の比較軸を固定する |
| Phase 2 | retrieval の本命改善 | 二段階 retrieval / rerank | 同じ budget で重要 block の recall を上げる |
| Phase 3 | 危険区間だけ守る | warmup 限定 adaptive budget | 平均コストを抑えつつ初期不安定を減らす |
| Phase 4 | 難しい層にだけ厚くする | layer-specific policy | layer 27 問題を直接詰める |
| Phase 5 | offload 実運用へ寄せる | 非同期プリフェッチ | 転送待ちを隠す |

---

## 3. Commit 単位の実装計画

---

### Commit 1
## 固定 GPU live KV budget の明示化

### 実装内容
- `B_total` を導入
- `B_recent`
- `B_anchor`
- `B_retrieval`
- `B_transient`

の4分割を明示する

### 初期方針
- `B_total` は固定
- `B_recent` は固定
- `B_anchor` は 0 のままでもよい
- `B_retrieval = B_total - B_recent - B_transient`

### 評価指標
- OOM しないこと
- 実行パスが壊れないこと
- 既存 baseline と同条件比較が可能であること

### 完了条件
- GPU live KV 使用量を設計上説明できる
- 実行ログから budget 内訳が追える
- 既存結果が大きく崩れない

### 備考
これはアルゴリズム改善ではなく、**設計の主語を固定するコミット**である。

---

### Commit 2
## 三層設計の導入（recent / anchor / retrieval）

### 実装内容
- `anchor bank` を追加
- 最初は文頭 `N` token もしくは先頭数 block を固定保持
- `recent` と `anchor` を別枠管理

### 初期方針
- `recent` は現行の recent full
- `anchor` は小さな固定枠
- `retrieval` は現行 cold retrieval を維持

### 評価指標
- mean_abs_diff
- first divergence step
- フォーマット維持
- anchor あり / なし比較

### 完了条件
- anchor 追加でフォーマット崩れが悪化しない
- recent と prefix 制御が分離された状態で実行できる
- 予算上 `B_anchor` が明示管理される

### 備考
StreamingLLM / H2O 的な知見を最小実装で取り込む段階。

---

### Commit 3
## retrieval 指標の導入

### 実装内容
- full attention 側から block relevance を集約
- retrieval recall@k
- retrieval precision@k
- coverage 的指標

を出せるようにする

### 評価指標
- recall@k
- precision@k
- mean_abs_diff との相関

### 完了条件
- 「何を取れて何を落としているか」を block 単位で可視化できる
- layer 14 / 27 の retrieval 差が確認できる

### 備考
以降の rerank や adaptive budget の効果判定の土台になる。

---

### Commit 4
## 二段階 retrieval / rerank

### 実装内容
- Stage 1: `mean_plus_norm` など軽量 score で top-M 候補取得
- Stage 2: `query_to_block_max` または `mean_plus_max` で rerank
- 最終 top-k は従来通り固定

### 評価指標
- retrieval recall@k
- retrieval precision@k
- mean_abs_diff
- layer 14 / 27 比較

### 完了条件
- baseline より retrieval recall が改善する
- 少なくとも主ケースで mean_abs_diff が同等以上または改善
- rerank の overhead が測定できる

### 備考
Quest / ParisKV 的な知見の最も直接的な導入。

---

### Commit 5
## warmup 限定 adaptive budget

### 実装内容
- `WARMUP` 中のみ
  - `B_recent`
  - `retrieval_topk`
  - 必要なら `working_ratio`
を動的化
- `ACTIVE` では cheap 固定 policy に戻す

### 使う signal
- recent-window 安定度
- logits margin
- predictor danger
- top-1 changed

### 評価指標
- first divergence step
- average working budget
- decode 成功率
- mean_abs_diff

### 完了条件
- warmup 中だけ保守化することで early divergence が減る
- 平均予算は大きく増えない
- ACTIVE の実装は単純なまま保てる

### 備考
DynamicKV 的な動的制御を最小コストで入れる段階。

---

### Commit 6
## layer-specific policy

### 実装内容
- layer 14 と 27 で policy を分ける
- 例:
  - layer 14: Stage 1 のみ
  - layer 27: Stage 1 + Stage 2
- または layer 27 だけ `B_retrieval` を厚くする

### 評価指標
- layer別 mean_abs_diff
- layer別 retrieval recall
- 主ケースでの全体品質

### 完了条件
- layer 27 の難しさに対して明確な改善が出る
- 全層共通 policy より有利な証拠がある

### 備考
ChunkKV / DynamicKV 的知見を、reuse ではなく差分設定として先に使う。

---

### Commit 7
## ACTIVE 限定 1-step 非同期プリフェッチ

### 実装内容
- `ACTIVE` 状態のみ
- 次 step 候補 block を 1 step 先読みで staging
- `B_transient` 内で管理

### 評価指標
- decode latency
- transfer wait
- throughput
- 品質が悪化しないこと

### 完了条件
- 転送待ちの削減が見える
- 予測外れで大きく悪化しない
- 品質は baseline と同等

### 備考
ParisKV / AsyncTLS 系の runtime 改善知見を最後に入れる。

---

## 4. 各フェーズの完了条件

### Phase 1 完了条件
- `B_total` と三層設計が導入済み
- anchor を含む live KV 構成が明示できる
- retrieval 指標が取れる

### Phase 2 完了条件
- 二段階 retrieval が実装済み
- 主ケースで retrieval recall の改善が確認できる
- mean_abs_diff も悪化しないか改善する

### Phase 3 完了条件
- warmup 中だけ adaptive budget が機能する
- early divergence の抑制が確認できる
- ACTIVE は cheap policy に維持できる

### Phase 4 完了条件
- layer-specific policy が有効と確認できる
- layer 27 に対する明確な対策が示せる

### Phase 5 完了条件
- 非同期プリフェッチで runtime 改善が見える
- 転送待ちが抑制される
- 品質面の副作用が小さい

---

## 5. いま捨てるもの

最速実装を優先するため、次は後回しにする。

- KV quantization の本格導入
- semantic block 再設計
- 再構成型メモリ
- drift-robust scorer の本格導入
- layer-wise index reuse の本格導入

理由は、評価軸が増えすぎて原因切り分けが難しくなるため。

---

## 6. 最小比較セット

最小比較セットは次の5条件で十分。

1. Baseline
2. + anchor
3. + rerank
4. + warmup adaptive budget
5. + prefetch

評価ケースは次に固定する。

- `prose / 4096 / layer 14`
- `prose / 4096 / layer 27`
- `structured / 4096 / layer 27` （補助）

---

## 7. 一番大事なまとめ

最速で進めるなら、順番は次でよい。

1. **固定 GPU live KV budget を主語にする**
2. **recent / anchor / retrieval の三層設計を入れる**
3. **二段階 retrieval を最初の本命改善にする**
4. **adaptive budget は warmup 限定で薄く入れる**
5. **難しい層だけ layer-specific に厚くする**
6. **最後に非同期プリフェッチで runtime を詰める**

この順番なら、実験数を増やしすぎず、各段階の効果を切り分けながら最速で前進できる。
