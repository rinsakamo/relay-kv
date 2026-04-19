# RelayKV Gate v1 仕様メモ

## 目的

RelayKV の始動直後は、近似 KV の導入によって early divergence が起きやすい。特に decode 初期では、生成された token 列そのものが後続 step から見た recent window を形成するため、この形成途中で近似を入れると、壊れた token が直近文脈へ入り、以降の生成を連鎖的に不安定化しやすい。

このため Gate は、単に「最初の数 step を閉じる」だけではなく、**安定した recent window が形成されるまでを保護する仕組み**として設計する。

本仕様では、

- 始動前は RelayKV を使わない
- 始動直後の助走期間だけ recent-window 安定度を丁寧に判定する
- 完全稼働後は固定 step 数または軽量ルールへ蒸留して運用する

という 3 状態 Gate を定義する。

---

## 設計コンセプト

Gate は以下の 3 状態で管理する。

1. **未始動 (OFF)**
2. **助走 (WARMUP)**
3. **完全稼働 (ACTIVE)**

狙いは、**危険な始動直後だけ高価な安定度判定を行い、その後は軽量ポリシーで回す**ことである。

---

## 状態定義

### 1. 未始動 (OFF)

まだ RelayKV を使わない状態。

#### 役割
- full / baseline 相当で安全運転する
- Gate 判定はまだ行わない
- RelayKV 始動条件が満たされるまで待機する

#### 典型条件
- context length が始動しきい値未満
- cold/offload を使う意味がまだ小さい
- recent context が十分育っていない

---

### 2. 助走 (WARMUP)

RelayKV を使い始めるが、Gate は慎重に制御する状態。

#### 役割
- recent-window 安定度を観測する
- predictor danger を観測する
- 近似導入による系列分岐の危険を監視する
- 「完全稼働してよいか」を判定する

#### 特徴
- 一番高価な判定をこの期間だけ行う
- 実験段階ではログを厚く取る

---

### 3. 完全稼働 (ACTIVE)

recent window が十分安定したとみなし、軽量ポリシーで RelayKV を運用する状態。

#### 役割
- 固定 step Gate または軽量 predictor で回す
- 毎 step の重い安定度計算は行わない
- RelayKV の本来の効率を出す

#### 特徴
- 実運用向き
- 助走で得た知見を固定ルールへ蒸留する

---

## 状態遷移

基本の遷移は以下。

```text
OFF -> WARMUP -> ACTIVE
```

将来的には安全のための一時逆戻り

```text
ACTIVE -> WARMUP
```

も考えられるが、v1 では省略可能。

---

### 遷移1: OFF -> WARMUP

RelayKV の始動条件を満たしたら助走へ入る。

#### 条件例
- `seq_len >= relay_start_len`
- cold range が十分長い
- offload / retrieval を使う意味がある長さに達した

#### 実装イメージ

```python
if seq_len >= relay_start_len:
    state = "WARMUP"
    warmup_step = 0
```

---

### 遷移2: WARMUP -> ACTIVE

recent-window 安定度が十分と判定できたら完全稼働へ移る。

#### 条件の考え方
- 助走開始後に最低限の step 数が経過している
- 出力候補が安定している
- predictor danger が低い
- token 分岐が直近で起きていない

#### 代表条件
- `warmup_step >= min_warmup_steps`
- `avg_margin_recent >= margin_threshold`
- `predictor_danger_count_recent <= danger_threshold`
- `top1_changed_count_recent == 0`

---

### 遷移3: ACTIVE -> WARMUP（将来拡張）

v1 では省略してよいが、将来は以下のような条件で一時的に慎重モードへ戻す設計もありうる。

#### 条件例
- predictor danger が急増
- logits margin が急低下
- top-k overlap が急激に崩れる

---

## 助走期間で観測する判定量

判定量は以下の 4 群に整理する。

---

### 群A: 出力安定性

#### 1. logits margin
Top-1 と Top-2 の差。

##### 解釈
- 大きい: 現在の出力選択が安定
- 小さい: 少しの近似で token が変わりやすい

##### 使い方
- 直近 `m` step の平均を取る

---

#### 2. top-k overlap
近似あり/なし、または apply/baseline の top-k 候補集合の一致度。

##### 解釈
- 高い: 候補順位が安定
- 低い: 出力面で不安定

---

#### 3. top-1 changed flag
近似を入れたとき top-1 が変化したか。

##### 解釈
- early に一度でも出ると危険度が高い
- warmup 完了条件では「直近数 step 未発生」を使える

---

### 群B: predictor 系

#### 4. predictor danger
既存の predictor danger フラグ。

##### 解釈
- true が多い: まだ不安定
- 連続 false: 安定化の兆候

---

#### 5. predictor_gate_overlap
predictor が危険と見た場所と実際の gate / replacement の重なり。

##### 解釈
- overlap 高 + divergence 発生: predictor が弱い可能性
- overlap 低: Gate policy 側見直しの余地

---

### 群C: 直近文脈形成の proxy

#### 6. recent-token consistency
直近数 step で生成 token の文法・文体方向が急変していないかを見る軽量 proxy。

##### 例
- punctuation / newline の暴れ
- list 開始 / prose 開始の急変
- 同一文法フレームの継続性

##### 解釈
- recent context がまだ揺れているなら危険

---

#### 7. recent residual difference proxy
直近生成 token に対し、近似あり/なしの residual 差 proxy を取る。

##### 解釈
- 直近文脈の中核が揺れているなら危険
- recent window が安定していれば差は小さくなるはず

##### 備考
- 高価なら v1 では研究用ログのみに留める

---

### 群D: retrieval 系

#### 8. selected-block concentration
選択された block の score 分布の鋭さ・集中度。

##### 解釈
- 平たい: retrieval が曖昧で危険
- 集中している: 参照先が比較的定まっている

---

#### 9. recent vs non-recent reliance
現 step がどれだけ recent 側に依存しているか。

##### 解釈
- recent 側で十分: 近似の危険は相対的に低い
- long-range 側依存が急に高い: 慎重にしたい

---

## v1 で使う最小判定量

最初から増やしすぎないため、v1 では以下の 3 つを主判定に採用する。

1. `mean logits margin over recent m steps`
2. `predictor danger rate over recent m steps`
3. `top-1 changed in recent m steps`

これで「出力安定」「予兆」「実際の分岐」の 3 軸を押さえられる。

---

## WARMUP -> ACTIVE 判定ルール案

### 条件
- `warmup_step >= min_warmup_steps`
- `avg_margin_recent >= margin_threshold`
- `predictor_danger_count_recent <= danger_threshold`
- `top1_changed_count_recent == 0`

### 解釈
- 出力候補が安定している
- predictor 的にも危険が少ない
- 実際に token 分岐も直近で起きていない

なら、recent window が十分育ったとみなして ACTIVE へ移行する。

---

## 完全稼働後の軽量ポリシー

ACTIVE に入った後は、重い安定度計算を止める。選択肢は以下。

### 案A: 固定 step 数
- 例: `warmup_steps = 6`
- 最も軽い

### 案B: 条件別固定 step 数
- prose: 8
- structured: 4
- repetitive: 5

より現実的。

### 案C: 軽量 predictor のみ継続
- full な安定度計算はしない
- cheap な predictor danger だけ継続観測

安全性を少し残せる。

---

## 擬似コード

```python
if state == "OFF":
    if seq_len >= relay_start_len:
        state = "WARMUP"
        warmup_step = 0

elif state == "WARMUP":
    warmup_step += 1

    compute_margin_stats()
    compute_predictor_danger_stats()
    compute_top1_change_stats()

    if warmup_step >= min_warmup_steps and is_recent_window_stable():
        state = "ACTIVE"

elif state == "ACTIVE":
    run_relaykv_with_light_policy()
```

### 最小判定関数

```python
def is_recent_window_stable():
    return (
        avg_margin_recent >= margin_threshold
        and predictor_danger_count_recent <= danger_threshold
        and top1_changed_count_recent == 0
    )
```

---

## 研究段階で先にやるべきこと

いきなり本番 policy にせず、まずは WARMUP 中に以下をログ収集する。

- logits margin
- predictor danger
- top1 changed
- first divergence step
- divergence lag
- 必要なら recent residual proxy
- 必要なら top-k overlap

### 目的
「どの条件なら、その後安定に Gate を開けるか」を後から分析し、固定 step 数や条件別固定ルールへ蒸留するため。

---

## 推奨実験順

### 比較1: 固定 warmup step 数のみ
最も単純なベースライン。

### 比較2: 動的 recent-window 安定度のみ
readiness-based gate の有効性確認。

### 比較3: 動的判定から蒸留した条件別固定 step 数
実運用向け cheap policy の確認。

---

## 期待される読み方

### ケースA: 動的判定がかなり良い
- fixed gate より readiness-based gate が本質的に必要

### ケースB: 動的判定と条件別固定 step 数がほぼ同等
- 実運用では固定ルールで十分

### ケースC: prose だけ長め warmup が必要
- prompt_type 別 policy が必要

### ケースD: layer 27 だけ長め warmup が必要
- layer-specific gate の根拠になる

---

## v1 の要約

RelayKV Gate v1 は以下で構成する。

### 状態
- OFF
- WARMUP
- ACTIVE

### 遷移
- context length が閾値を超えたら WARMUP
- recent-window 安定度が十分なら ACTIVE

### 主判定量
- logits margin
- predictor danger
- top-1 changed

### 実運用方針
- 助走期間だけ高価な安定度判定を行う
- その結果を固定 step 数や条件別固定ルールへ蒸留する

---

## 一言まとめ

**RelayKV Gate v1 は、始動直後の危険な期間だけ recent-window 安定度を観測し、安定した直近文脈が形成されたら軽量ポリシーへ切り替える 3 状態 Gate である。**
