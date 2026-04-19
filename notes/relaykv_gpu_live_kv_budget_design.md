# RelayKV Design Centered on GPU Live KV Budget

## 1. 設計の中心思想

RelayKV をメモリ制御の観点で見ると、中心に置くべきなのは

**GPU live KV budget**

である。

これは、

**その時点で GPU 上に常駐し、実際に attention 計算に使える KV の総量上限**

を意味する。

設計としては、

- 総コンテキスト長は長くてよい
- ただし GPU 上の live KV 総量は固定
- その固定予算の中で recent / retrieved / safety buffer を配分する

という形になる。

---

## 2. まず分けるべき3つ

RelayKV 全体設計では、KVを少なくとも3種類に分けるとわかりやすい。

### A. GPU live KV
今まさに計算に使えるKV。  
最重要で、予算管理の対象。

### B. Offloaded KV
CPUや別メモリにある cold 側KV。  
総コンテキスト保持の本体。

### C. In-flight / staging KV
GPUへ戻す途中、または一時的に置くKV。  
転送や再配置のためのバッファ。

このうち、**A を固定する**のが肝である。

---

## 3. 全体制約

一番上の制約は次の形になる。

**GPU live KV + staging buffer + その他 runtime buffer ≤ GPU安全上限**

この「GPU安全上限」は、理論上のVRAM上限そのものではなく、

- モデル本体
- activations
- workspace
- allocator 余裕
- fragmentation 余裕

を引いた後の、安全に使ってよい枠である。

RelayKV 的には、

**GPUに残る自由度の大半を live KV budget に明示的に割り当てる**
のがよい。

---

## 4. live KV budget の中身

GPU live KV budget は、さらに分けたほうがよい。

### ① recent full budget
直近トークンを full で保持する最低保証枠。

### ② retrieved working budget
non-recent 側から回収して実計算に使う枠。

### ③ safety / transient budget
転送中や policy 変更時の一時バッファ。

つまり、

**total_live_budget = recent_full + retrieved_working + transient**
である。

---

## 5. recent full を最低保証にする理由

recent full は、単なる1候補ではなく**床**として持つのがよい。

理由は、recent は

- 局所文法
- 直近の構文
- 出力の接続
- 直前の意味整合

に効きやすく、削ると壊れ方が急になりやすいからである。

したがって設計としては、

**recent full budget は最低保証**
にして、
残りを retrieved working 側で争う形が自然である。

---

## 6. retrieved working budget は可変配分にする

一方、non-recent 側は query 依存性が高い。

- ほとんど recent だけでよい step
- 遠距離参照が強く必要な step
- 層によって重要度が違う step

があるので、retrieved working budget は

**固定でもよいが、本命は可変配分**
である。

全体上限は固定しつつ、

- recent を厚くするか
- retrieval を厚くするか

を step / state / risk に応じて動かすのがよい。

---

## 7. state 制御との接続

前に議論した

- 非稼働
- 助走
- 稼働

は、この配分問題と自然に統合できる。

### 非稼働
- full 寄りで運転
- live budget 制約をまだきつく使わない

### 助走
- recent full を厚く取る
- retrieval は保守的
- stability 判定に少しコストを使う

### 稼働
- total live budget は固定
- その中で recent / retrieval を柔軟配分
- aggressive relay を許す

つまり、

**state は固定上限内の配分ポリシーを切り替える上位制御**
である。

---

## 8. 一番自然な全体アーキテクチャ

全体像を文章で書くとこうなる。

### 上位層
global controller  
→ 非稼働 / 助走 / 稼働 を決める

### 中位層
risk-aware allocator  
→ total live KV budget を recent / retrieved / transient に配分

### 下位層
retriever / selector  
→ retrieval top-k、working set、rerank を決める

### 実行層
writeback / offload / attention compute  
→ 実際に GPU 上の KV を入れ替えて計算する

---

## 9. コア制約は「総量固定、内訳可変」

この設計で一番大事なのは、

**総量は固定**
で、
**内訳だけ可変**
にすることである。

完全に自由にすると OOM 管理が難しい。  
完全固定だと query 依存性に弱くなる。

したがってちょうどよいのは、

- GPU total live budget: 固定
- recent floor: 固定
- retrieval reserve: 固定または下限あり
- recent/retrieval の追加配分: 可変

である。

---

## 10. 設計式っぽく書くと

たとえば次のように置ける。

- **B_total** = GPU live KV 総予算
- **B_recent_min** = recent full 最低保証
- **B_retrieval_min** = retrieval 最低保証
- **B_transient** = staging / 一時バッファ

Then:

**B_total ≥ B_recent + B_retrieval + B_transient**

with

- **B_recent ≥ B_recent_min**
- **B_retrieval ≥ B_retrieval_min**

残りを state や risk で配分する。

---

## 11. risk-aware allocation のイメージ

### high risk
- B_recent を増やす
- B_retrieval も安全側で厚め
- aggressive 削減はしない

### medium risk
- B_recent をやや厚め
- B_retrieval 標準
- top-k 大きめ

### low risk
- B_recent を標準
- B_retrieval を絞る
- top-k 小さめ

### long-range demand 高
- B_recent は floor だけ維持
- B_retrieval を増やす

したがって、

**risk と demand の両方で配分を変える**
のが自然である。

---

## 12. 超長コンテキストに対して何が起こるか

この設計のうれしい点は、総コンテキストがさらに伸びても、

**GPU上で増えるのは基本的に B_total まで**
に抑えられることである。

増えるのは主に

- CPU / offload 側の保存量
- retrieval index 的な管理量
- 転送頻度

である。

つまり、

**総長さが伸びても、GPUメモリ面では上限付きで扱える**
ようになる。

---

## 13. その代わり何がボトルネックになるか

メモリ制御ができるようになると、次の3つが主戦場になる。

### ① 転送帯域
CPU→GPU の戻しが間に合うか。

### ② retrieval quality
必要KVを限られた B_retrieval 内で拾えるか。

### ③ allocator overhead
出し入れや再配置が重すぎないか。

つまり設計の焦点は、

**OOM 回避**
から
**帯域・選別・配分の最適化**
へ移る。

---

## 14. Gate と budget はこの設計のどこに入るか

ここでは Gate と budget は中位層に入る。

### Gate
- そもそも retrieval / relay を使うか
- risk が高ければ full fallback へ戻すか

### Budget
- B_recent をどこまで厚くするか
- B_retrieval をどれだけ使うか
- top-k をいくつにするか

したがってこの全体設計では、

**Gate と budget は GPU live KV budget 配分器の一部**
である。

---

## 15. 実験段階でまず固定すべきもの

最初の実験では、全部可変にしすぎないほうがよい。  
まず固定したいのは次の3つ。

### 1. B_total
GPU live KV 総予算。  
最重要。

### 2. B_recent_min
recent floor。  
局所安定性の床。

### 3. B_transient
安全バッファ。  
OOM防止。

その上で最初は

- B_retrieval = B_total - B_recent - B_transient

として扱うとわかりやすい。

---

## 16. その後に可変化する順番

順番としてはこうなる。

### v1
- B_total 固定
- B_recent 固定
- B_retrieval 固定

### v2
- B_total 固定
- B_recent は risk-aware 可変
- B_retrieval は残差

### v3
- B_total 固定
- B_recent / B_retrieval 両方可変
- top-k も可変

### v4
- state-aware global mode と統合
- cheap runtime policy へ蒸留

---

## 17. 本質的な見方

RelayKV をこの観点で見ると、やっていることは

**超長コンテキストを、固定サイズの GPU 作業記憶に投影して計算すること**

である。

つまり、

- コンテキスト全体 = 長期記憶
- GPU live KV = 作業記憶
- retrieval = 長期記憶からの再想起

という見方ができる。

---

## 18. 一番大事なまとめ

GPU live KV budget を中心にした RelayKV 全体設計はこうである。

- **総コンテキスト長**は長く持ってよい
- **GPU上の live KV 総量**だけは固定上限で管理する
- その上限の中を
  - recent full
  - retrieved working
  - transient buffer
  に分ける
- 非稼働 / 助走 / 稼働 は、その固定上限内での配分ポリシーを切り替える上位状態
- Gate と top-k などは、その配分を決める下位制御

つまり RelayKV は、

**超長文脈を、固定サイズのGPU作業記憶に載せ替えながら扱う制御系**
として設計できる、ということである。
