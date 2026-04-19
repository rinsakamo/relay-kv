# RelayKV Global Mode Integration: OFF / WARMUP / ACTIVE

## 1. 基本方針

RelayKV 全体の運転状態（非稼働 → 助走 → 稼働）と、Gate / 予算制御は、どちらも

**「今どれだけ近似を入れてよいか」**

を決めている。

したがって、これらは1つの**運転モード制御**として統合できる。

統合の中心は次の2軸である。

- **recent window 安定性**
- **計算コスト圧**

---

## 2. 統合の考え方

これまで別々に見えていたものを並べると次の通り。

### A. RelayKV 全体状態
- 非稼働
- 助走
- 稼働

### B. Gate
- close / open
- 危険度に応じて近似許可

### C. 予算制御
- recent full の大きさ
- top-k
- working ratio
- rerank 有無

これらは全部、本質的には

**「この時点で、どれくらい aggressive に KV 近似してよいか」**
を決めている。

したがって、1つの**運転モード制御**として統合できる。

---

## 3. 統合後の中心変数

統合後に中心になるのは次の2つ。

### ① recent window 安定性
- 生成された直近 token 列が、後続生成の土台として十分安定しているか
- early divergence を起こしにくいか

### ② 近似コスト圧
- もう RelayKV を使わないとコスト的に重いか
- long context 化で full-KV 維持が高すぎるか

直感的には、

- 安定性が低いなら近似を抑える
- コスト圧が高いなら近似を進めたい

という2つの力の釣り合いで状態を決める。

---

## 4. 状態定義

### State 0: 非稼働
#### 状態の意味
- recent window 安定性がまだ低い
- もしくはコスト圧がまだ低い
- なので RelayKV を使う必要も、使う安全性も十分でない

#### 典型例
- コンテキストがまだ短い
- 直近生成列が育っていない
- full のままでも十分軽い

#### 直感
**まだ RelayKV を起動する理由が弱い**

---

### State 1: 助走
#### 状態の意味
- コスト圧は上がってきたので RelayKV を使いたい
- ただし recent window 安定性がまだ十分ではない
- だから慎重に近似を導入する

#### 直感
**使いたいが、まだ全開にはできない**

---

### State 2: 稼働
#### 状態の意味
- recent window が十分安定
- コスト圧も高く、RelayKV を本格利用する意味がある
- ここで初めて軽量 policy に落とせる

#### 直感
**安定したので、効率モードへ移行できる**

---

## 5. 状態遷移

### 非稼働 → 助走
これは主に**コスト圧**で決めるのが自然。

例:
- seq_len がしきい値を超える
- active KV メモリがしきい値を超える
- full decode の TPOT 悪化が見えてくる
- cold range が十分長くなる

つまり、

**もう RelayKV を始める価値がある**
で入る。

---

### 助走 → 稼働
これは主に**recent window 安定性**で決めるのが自然。

例:
- 直近 m step の logits margin が高い
- top-1 changed が消えた
- predictor danger が落ち着いた
- recent token 列の方針が安定

つまり、

**もう近似を本格導入しても系列分岐しにくい**
で入る。

---

## 6. 状態ごとの計算コスト設計

### 非稼働
- RelayKV なし
- 安定度計算も最小
- full で安全運転

→ コストは高いが、まだコンテキストが短くて耐えられる

---

### 助走
- 安定度判定あり
- predictor あり
- budget も保守的
- 場合によっては rerank もあり

→ 一番判断コストが高くてもよい区間

理由:
- 危険期間は短い
- ここを誤ると後ろ全部に響く
- 稼働に入ればその後は cheap に回せる

---

### 稼働
- 安定度の重い計算はやめる
- cheap feature のみ
- 固定または準固定 policy
- aggressive relay 許可

→ 本格的に速度・メモリ改善を取りにいく区間

---

## 7. readiness × pressure という見方

状態は、

- **readiness** = recent window 安定性
- **pressure** = RelayKV を使いたいコスト圧

で決まると考えると整理しやすい。

### readiness 低 / pressure 低
→ 非稼働

### readiness 低 / pressure 高
→ 助走

### readiness 高 / pressure 高
→ 稼働

### readiness 高 / pressure 低
→ 理論上はありえるが、実用上は非稼働寄りでもよい

本質は、

**使う準備ができたか**
と
**使う必要があるか**
の掛け合わせである。

---

## 8. Gate と budget は state の中の出力

この統合では、Gate や budget は独立主体ではなく、

**現在 state に応じたサブ制御**
になる。

### 非稼働
- gate: closed
- recent full: max
- top-k: unused
- working ratio: full

### 助走
- gate: conditional
- recent full: large
- top-k: conservative
- working ratio: cautious

### 稼働
- gate: open by default
- recent full: standard / adaptive
- top-k: adaptive
- working ratio: normal or aggressive

---

## 9. 実装イメージ

```python
state = decide_global_mode(readiness, pressure)
policy = policy_for_state(state, local_features)
```

ここで

- `decide_global_mode(...)`
  が 非稼働 / 助走 / 稼働 を決める
- `policy_for_state(...)`
  が gate, recent_full, topk, working_ratio を決める

つまり、

**上位で運転状態を決め、下位でその状態に応じた近似強度を決める**
二層構造である。

---

## 10. この構造の利点

### 利点1
RelayKV を「いつ使うか」と「どう使うか」を分けられる。

### 利点2
高価な安定度計算を助走だけに閉じ込められる。

### 利点3
稼働後は cheap policy に蒸留しやすい。

### 利点4
評価も整理しやすい。
- state 遷移の妥当性
- state 内 policy の妥当性
を別に見られる。

---

## 11. 研究実験としての切り方

### 実験A
readiness 指標が state 遷移に使えるか

### 実験B
pressure 指標で始動タイミングが妥当か

### 実験C
各 state 内での policy table が妥当か

### 実験D
統合前後で品質 / コスト Pareto が改善するか

この見方では、RelayKV は単なる retrieval 手法ではなく、

**stateful approximation controller**
として扱える。

---

## 12. 一番短いまとめ

非稼働〜助走〜稼働は、

- **recent window 安定性 = 近似を入れても壊れにくいか**
- **計算コスト圧 = もう RelayKV を使いたいか**

の2軸で決まる**上位運転状態**として定義できる。

そして Gate や top-k などの予算制御は、その状態の中で決まる**下位近似強度制御**として統合できる。
