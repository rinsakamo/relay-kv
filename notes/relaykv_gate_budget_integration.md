# RelayKV Gate and Budget Control Integration

## 1. 基本方針

統合後は、Gate を単なる

- 開ける / 閉じる

ではなく、

**その時点で許してよい近似強度を決める制御器**

として扱う。

つまり出力は二値ではなく、

- full fallback
- safe relay
- normal relay
- aggressive relay

のような**段階的ポリシー**になる。

---

## 2. 制御の全体像

### 入力
その step・layer・query の危険度を表す特徴。

### 内部
risk 推定または policy table。

### 出力
- gate mode
- recent full 予算
- retrieval top-k
- working ratio
- recency bias 強度
- 必要なら rerank 実行有無

---

## 3. 状態

### State A: OFF
未始動。RelayKV を使わない。

### State B: WARMUP
助走期間。危険度を丁寧に見ながら保守的に動く。

### State C: ACTIVE
完全稼働。軽量 policy で回す。

ここに加えて、各状態内で**近似強度レベル**を持たせる。

---

## 4. 近似強度レベル

### Level 0: Full
- gate closed
- 近似なし
- full KV

### Level 1: Safe Relay
- gate open
- recent full を厚め
- retrieval top-k 大きめ
- conservative

### Level 2: Normal Relay
- 標準 recent full
- 標準 top-k
- 標準 working ratio

### Level 3: Aggressive Relay
- recent full 小さめ
- top-k 小さめ
- working ratio 小さめ
- 最大効率狙い

つまり、実際の policy は

**状態 × 強度レベル**
で決まる。

---

## 5. 入力特徴

まずは次の6つくらいがよい。

### ① step位置
- decode の何 step 目か
- early step かどうか

**役割**  
初期分岐リスクを見る。

---

### ② recent-window 安定度
- recent 数step の平均 logits margin
- top-1 changed の有無
- predictor danger の頻度

**役割**  
直近生成列が安定して recent context を形成したかを見る。

---

### ③ retrieval 分布の鋭さ
- score entropy
- top1/top5 比
- cumulative score concentration

**役割**  
retrieval の曖昧さを見る。

---

### ④ predictor danger
既存 predictor の危険信号。

**役割**  
full fallback が必要か、予算を厚くすべきかを見る。

---

### ⑤ layer ID / layer group
- 14
- 27
- あるいは shallow / mid / deep

**役割**  
層ごとの脆さを反映する。

---

### ⑥ prompt / task type
- prose
- structured
- repetitive
- 将来的には JSON / code など

**役割**  
品質許容度や warmup 長さを変える。

---

## 6. 出力制御量

少なくとも4本あると強い。

### A. gate mode
- close
- open

最も粗い制御。

---

### B. recent full budget
- recent window をどこまで full に残すか

例:
- 256
- 384
- 512

---

### C. retrieval top-k
- non-recent 側から何 block / token を回収するか

例:
- 3
- 6
- 10

---

### D. working ratio / relay strength
- 最終的な working KV の総量

例:
- 0.15
- 0.25
- 0.40

---

### E. オプション出力
必要なら追加で

- rerank 実行有無
- recency bias の強さ
- layer別 budget 配分

も制御できる。

---

## 7. 一番簡単な統合方法

最初は、risk score を学習しなくても  
**policy table** で十分。

### high risk
条件:
- early step
- recent-window 不安定
- predictor danger 高

出力:
- gate = close
- recent full = max
- retrieval top-k = safe max
- working ratio = full or near-full

---

### medium risk
条件:
- warmup 中
- margin は中程度
- danger は低下中

出力:
- gate = open
- recent full = large
- retrieval top-k = high
- working ratio = conservative

---

### low risk
条件:
- recent-window 安定
- predictor danger 低
- retrieval concentration 良好

出力:
- gate = open
- recent full = standard
- retrieval top-k = standard
- working ratio = normal

---

### very low risk
条件:
- active 後半
- margin 高
- danger ほぼなし
- concentration 高い

出力:
- gate = open
- recent full = reduced
- retrieval top-k = small
- working ratio = aggressive

---

## 8. 擬似コード

```python
def choose_policy(features):
    risk = estimate_risk(features)

    if risk == "high":
        return {
            "gate_open": False,
            "recent_full_budget": MAX_RECENT,
            "retrieval_topk": SAFE_TOPK,
            "working_ratio": FULL_RATIO,
            "rerank": False,
        }

    elif risk == "medium":
        return {
            "gate_open": True,
            "recent_full_budget": LARGE_RECENT,
            "retrieval_topk": LARGE_TOPK,
            "working_ratio": CONSERVATIVE_RATIO,
            "rerank": False,
        }

    elif risk == "low":
        return {
            "gate_open": True,
            "recent_full_budget": NORMAL_RECENT,
            "retrieval_topk": NORMAL_TOPK,
            "working_ratio": NORMAL_RATIO,
            "rerank": False,
        }

    else:  # very low
        return {
            "gate_open": True,
            "recent_full_budget": SMALL_RECENT,
            "retrieval_topk": SMALL_TOPK,
            "working_ratio": AGGRESSIVE_RATIO,
            "rerank": True,
        }
```

---

## 9. WARMUP との接続

この統合案は warmup と特に相性がいい。

### OFF
- full
- RelayKV なし

### WARMUP
- risk を毎step観測
- しばらくは high / medium 寄り
- recent-window 安定度が上がると low へ移る

### ACTIVE
- 基本は low / very low のみで回す
- 重い判定はやめる
- cheap features だけで table を引く

つまり、

**WARMUP 中に得た知見を ACTIVE 用の cheap policy に蒸留する**
構図になる。

---

## 10. 段階的な導入順

### v1
- binary gate
- fixed top-k
- fixed recent full

### v2
- binary gate
- adaptive top-k
- recent full は固定

### v3
- binary gate
- adaptive top-k
- adaptive recent full

### v4
- gate / top-k / recent full / working ratio を policy table で統合

### v5
- risk score 学習化
- layer別 / task別 policy

---

## 11. 評価の仕方

統合案の評価は、少なくとも3軸必要。

### ① 内部近似
- mean_abs_diff
- top-1 changed
- first divergence
- retrieval recall / precision

### ② 最終品質
- 要約一致
- 重要点抽出一致
- 指示追従
- フォーマット維持

### ③ コスト
- working ratio
- retrieval overhead
- CPU-GPU transfer
- decode time

統合したからこそ、  
**品質とコストの Pareto を見る**
のが大事。

---

## 12. この統合案の本質

一番大事なのは、

**Gate と budget を別々に最適化するのではなく、risk に応じて近似強度全体を制御する**

こと。

つまり最終的な制御対象は

- 近似するか
- どれだけ recent を守るか
- どれだけ non-recent を拾うか
- どれだけ aggressive に削るか

全部まとめて1つの policy にする、ということ。

---

## 13. 一番短いまとめ

設計としては、

### 入力特徴
- step位置
- recent-window 安定度
- predictor danger
- retrieval concentration
- layer
- task type

### 出力制御量
- gate open/close
- recent full budget
- retrieval top-k
- working ratio
- 必要なら rerank

を持つ  
**risk-aware approximation policy**
として扱うのがよい。
