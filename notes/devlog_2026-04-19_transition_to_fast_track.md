# devlog_2026-04-19 追記案：現行実験の切り上げと fast track plan への移行

## 1. 現時点の判断

現行フェーズは、**dynamic gate の本質把握**までで区切るのが最もおさまりがよい。

理由は次の通り。

- 2048 では `min_score_margin=20, min_gate_step=7` が cross-style で `matched_all_generated_tokens = true` を達成している
- 成功 run では `predictor_effective_block_steps = []` であり、現時点の成功要因は predictor ではなく **temporal routing** と解釈できる
- structured の failure は block selection そのものではなく **early injection** に起因している理解がかなり強く支持された
- 一方 4096 は gate policy の良し悪し以前に、baseline / apply 比較系の整合確認が必要な状態にある

したがって、このフェーズで 4096 policy まで完成させようとするよりも、

**2048 で得られた temporal gate の知見を確定させて現フェーズを閉じ、次フェーズでは設計主語を retrieval / memory architecture に切り替える**

ほうが自然である。

---

## 2. 現行フェーズの主結論

現時点の主結論は次の3点。

### 2.1 2048 における暫定共通 policy 候補
`medium_2048` に対して、現時点の暫定共通 policy 候補は

- `min_score_margin = 20`
- `min_gate_step = 7`

である。

### 2.2 failure の本質
structured の不安定性は、主として

- どの block を使ったか

ではなく、

- **いつ apply したか**

に支配されている。

より凝縮すると、

- block 7 is not unsafe
- early block 7 is unsafe
- late block 7 is safe

という理解が現在の best explanation である。

### 2.3 predictor の位置づけ
predictor は

- hazard signal としては動いている
- divergence summary 上でも requested / effective を分けて観測できる

が、現時点の successful runs では control 本体ではなく、

**補助観測系**
として位置づけるのが妥当である。

---

## 3. ここで現行実験を切り上げる理由

現時点でこれ以上この系を深追いすると、研究の主問いが拡散しやすい。

### 3.1 2048 では十分な知見が出ている
- temporal routing が支配的
- early-step suppress が強く効く
- prose への副作用は小さい
- cross-style でも暫定共通 policy 候補が見えている

このため、2048 については「dynamic gate の核心は temporal gate」にあると十分言える。

### 3.2 4096 は今は policy 探索フェーズではない
structured / prose で `first_divergence_step = 0` が出ており、比較系の非対称が疑われるため、ここで gate 改善を積むのは順番が悪い。

4096 はこのフェーズでは

**gate policy 評価対象**
ではなく
**比較前提の sanity check 対象**

にとどめるべきである。

---

## 4. 現フェーズを閉じるための最小作業

現行フェーズを閉じるために必要な作業は、次の4点に絞る。

### 4.1 実動成功ケースを1本固定する
代表ケースを1本決める。

候補:
- 2048
- `medium_2048`
- `min_score_margin = 20`
- `min_gate_step = 7`
- cross-style の代表 run

### 4.2 既知制約を「未解決問題」ではなく「既知制約」に落とす
現時点での制約を整理する。

- early step は危険
- temporal routing が支配的
- predictor は補助観測段階
- 4096 は比較前提の整合確認が先
- retrieval 指標・三層設計・GPU live budget は未導入

### 4.3 現行ログを最小セットだけ残す
残すものは以下で十分。

- 実動成功 run
- baseline 対照 run
- early divergence が見える run
- layer 14 / 27 の代表比較 run

### 4.4 次フェーズへの不足物を明示する
次フェーズに必要だが未実装なものを明記する。

- retrieval recall / precision
- fixed GPU live KV budget
- recent / anchor / retrieval の三層設計
- coarse-to-fine retrieval
- warmup adaptive budget

---

## 5. 4096 の扱い

4096 は現フェーズでは深追いしない。

### やること
比較前提の整合確認だけを行う。

- step 0 の token / top5 一致
- `generated_tokens`
- `step_logs | length`
- baseline ファイル取り違え確認

### やらないこと
- 4096 用 gate policy の探索
- predictor threshold 探索の再開
- 追加の dynamic gate 微調整

### この判断の意味
4096 はこの時点では「研究を進める対象」ではなく、「fast track plan に入れる前提を壊していないか確認する対象」として扱う。

---

## 6. fast track plan への移行工程

現フェーズの次は、実験主語を変える。

これまでの主語:
- dynamic gate
- temporal routing
- predictor の意味分離

次の主語:
- fixed GPU live KV budget
- recent / anchor / retrieval 三層設計
- retrieval quality
- coarse-to-fine retrieval

### 移行順
1. 現フェーズを「temporal gate の本質把握」で閉じる
2. 4096 は sanity check のみに限定する
3. `B_total` を主語とする fixed GPU live KV budget 設計に入る
4. recent / anchor / retrieval の三層設計を導入する
5. retrieval 指標を導入する
6. 二段階 retrieval / rerank を最初の本命改善にする
7. warmup 限定 adaptive budget を入れる
8. 最後に非同期プリフェッチへ進む

---

## 7. 次フェーズの入口条件

fast track plan に入る条件は次の通り。

- 2048 の代表成功ケースが固定されている
- divergence summary が現仕様で安定している
- 4096 が policy 探索対象ではなく sanity check 対象だと整理されている
- 現フェーズの主結論が temporal routing にあると文書化されている

---

## 8. 締めの一文（devlog 用）

現フェーズでは、2048 条件において dynamic gate の成功要因が predictor ではなく temporal routing にあることが強く支持され、`min_score_margin=20, min_gate_step=7` が暫定共通 policy 候補として得られた。一方 4096 は gate policy 評価フェーズではなく比較前提の整合確認フェーズと判断する。したがって本フェーズは **「temporal gate の本質把握」** で区切り、次フェーズでは fixed GPU live KV budget と recent / anchor / retrieval 三層設計を土台に fast track plan へ移行する。
