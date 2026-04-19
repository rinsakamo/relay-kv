# RelayKVに統合候補となる新規アイデア（論文調査ベース）

## 評価軸

各案を次の3軸で見る。

- **導入難易度**
- **期待効果**
- **RelayKV親和性**

結論として、今すぐ試す価値が高いのは次の5つ。

1. 二段階 retrieval / rerank
2. adaptive budget
3. anchor bank（recentと別保持）
4. 固定 GPU live KV budget 設計
5. 非同期プリフェッチ

---

## 優先度表

| アイデア | 導入難易度 | 期待効果 | RelayKV親和性 | ひとこと |
|---|---:|---:|---:|---|
| 二段階 retrieval / rerank | 中 | 高 | 高 | まず広く拾って後で精密化 |
| adaptive budget | 中 | 高 | 高 | 危険 step だけ厚く守れる |
| anchor bank（recent と別保持） | 低〜中 | 中〜高 | 高 | 指示・形式保持に効く |
| 固定 GPU live KV budget 設計 | 低 | 高 | 高 | OOM 管理の主軸にできる |
| 非同期プリフェッチ | 中〜高 | 高 | 高 | offload 前提なら特に有効 |
| semantic / block-aware chunking | 中 | 中〜高 | 高 | block 設計の質が上がる |
| layer-wise reuse / layer-specific policy | 中 | 中〜高 | 高 | 14層と27層差に直結 |
| KV quantization 併用 | 中〜高 | 高 | 中〜高 | 保持数を増やせるが原因切り分けが難化 |
| 再構成型メモリ（checkpoint→KV再構成） | 高 | 中〜高 | 中 | 面白いが今は重い |
| drift-robust retrieval | 高 | 中〜高 | 中〜高 | 長い decode で効くが検証が難しい |

---

## 1. 二段階 retrieval / rerank

### 概要
軽量 score で候補 block を広く取り、その後により精密な score で再順位付けする。

### RelayKVへの落とし方
- stage 1: `mean_plus_norm` など軽量 score
- stage 2: `query_to_block_max`, `mean_plus_max`, headwise max などで rerank

### メリット
- 同じ working budget でも重要 block の recall を上げやすい
- layer 27 のような「平均では埋もれるが尖って重要」な block に効きやすい
- retrieval 指標との相性がよい

### デメリット
- 再ランキング計算のオーバーヘッドが増える
- 短文や軽いケースではコスト負けの可能性がある
- 実装がやや複雑になる

### 総評
**最優先候補。**  
RAG的発想を最も自然に RelayKV へ持ち込める。

---

## 2. adaptive budget

### 概要
固定 top-k / fixed working ratio をやめ、step・risk・query 分布に応じて budget を動かす。

### RelayKVへの落とし方
- `retrieval_topk`
- `working_ratio`
- `recent_full_budget`

を動的化する。

### 使える signal
- recent-window 安定度
- logits margin
- predictor danger
- score entropy
- top1/top5 比
- cumulative score concentration

### メリット
- 危険 step だけ budget を厚くできる
- 平均メモリと平均転送量を抑えやすい
- Gate と予算の統合設計と相性がよい

### デメリット
- controller が不安定だと再現性やデバッグ性が悪化
- 閾値や policy 設計が難しい
- 同じ設定でも挙動が読みづらくなる

### 総評
**二段階 retrieval の次に有力。**  
Gate 統合設計の本丸になりやすい。

---

## 3. anchor bank（recent と別保持）

### 概要
recent とは別に、文頭や前半の指示・形式・schema・主題定義などを固定保持する。

### RelayKVへの落とし方
GPU live KV budget を

- `B_recent`
- `B_anchor`
- `B_retrieval`
- `B_transient`

に分ける。

### メリット
- 長出力でのフォーマット崩れを抑えやすい
- 指示忘れや schema 崩壊に強くなりうる
- recent と prefix 制御を分離できる

### デメリット
- anchor を増やしすぎると retrieval 予算を圧迫する
- 何を anchor とみなすかの設計が必要
- prefix 以外の途中更新制約への対応は別途考える必要がある

### 総評
**低コストで導入しやすく、効果が見えやすい。**  
`recent / anchor / retrieval` の3層設計と相性がよい。

---

## 4. 固定 GPU live KV budget 設計

### 概要
総コンテキスト長ではなく、GPU 上の live KV working set に明示的な上限を置く。

### RelayKVへの落とし方
- `B_total = B_recent + B_anchor + B_retrieval + B_transient`
- 総コンテキスト長は伸ばしてよい
- GPU live KV 総量だけ固定して管理する

### メリット
- OOM をかなり避けやすい
- 超長コンテキストを GPU 上限近くで制御しやすい
- state 制御・Gate・予算配分を整理しやすい

### デメリット
- 品質問題の主戦場が retrieval quality と配分設計に移る
- budget 配分が悪いと品質低下の原因切り分けが難しい

### 総評
**設計の主語として非常に重要。**  
アルゴリズムそのものより、RelayKV 全体の土台として効く。

---

## 5. 非同期プリフェッチ

### 概要
次 step で必要になりそうな block を先読みして staging しておく。

### RelayKVへの落とし方
- CPU cold offload と組み合わせる
- 1 step 先の候補 block を予測して GPU staging
- attention/MLP 計算と転送を overlap させる

### メリット
- CPU→GPU 転送待ちを隠しやすい
- offload 前提の runtime 効率改善に効く可能性が高い
- 実運用で TTFT / decode latency 改善が期待できる

### デメリット
- 予測を外すと無駄転送が増える
- single stream や短い入力では複雑さに見合わない
- 実装とデバッグが少し難しい

### 総評
**CPU offload を本気で使うなら強い候補。**  
ただしアルゴリズム改善後に入れるほうが順番として自然。

---

## 6. semantic / block-aware chunking

### 概要
固定長 block だけでなく、意味境界寄りの chunk/block にする。

### RelayKVへの落とし方
- 文境界
- 見出し
- schema 境界
- 論理区切り

を block 設計に反映する。

### メリット
- block 境界で重要情報が分断されにくい
- retrieval precision が上がる可能性がある
- RAG の chunking 改善発想と対応が良い

### デメリット
- block サイズが不均一になる
- 転送・writeback・index 管理が面倒
- GPU 側の配列管理が複雑になる

### 総評
**block 設計を次の段階に進める候補。**  
ただし導入にはやや設計整理が必要。

---

## 7. layer-wise reuse / layer-specific policy

### 概要
層ごとに retrieval 戦略や保持方針を変える。似た層では index reuse も検討する。

### RelayKVへの落とし方
- layer 14 は軽量 score
- layer 27 は精密 retrieval や rerank
- 安定層では reuse 多め

### メリット
- 難しい層だけ計算を厚くできる
- 全体コストを抑えつつ性能を上げやすい
- 今の観測と整合しやすい

### デメリット
- policy が層ごとに分かれると管理が難しい
- ハイパーパラメータが増える
- 汎化確認の負担が増す

### 総評
**layer差がはっきり見えた時点で有力。**  
特に layer 27 問題への直接策になりやすい。

---

## 8. KV quantization 併用

### 概要
retrieval だけでなく KV 精度そのものも落として保持量を増やす。

### RelayKVへの落とし方
- anchor は高精度
- retrieval 側は低 bit
- cold 側はさらに圧縮

などの mixed precision 設計。

### メリット
- 同じ GPU budget でより多くの KV を保持できる
- budget 配分の自由度が上がる
- メモリ効率がさらに改善する

### デメリット
- 誤差の出所が retrieval 由来か quantization 由来か混ざる
- Gate や評価が難しくなる
- 実験の切り分けが複雑化する

### 総評
**強いが、今は二段階 retrieval より後。**  
土台が安定してからの拡張がよい。

---

## 9. 再構成型メモリ

### 概要
cold KV をそのまま保存するのではなく、軽い checkpoint 表現から必要時に再構成する。

### メリット
- CPU 側メモリ量まで減らせる可能性がある
- 超長文脈向けには魅力がある

### デメリット
- 再構成計算が重い
- 実装難度が高い
- 現行の retrieval + writeback とは設計がかなり変わる

### 総評
**面白いが今は優先度低。**  
RelayKV の主軸が安定してから考えるべき。

---

## 10. drift-robust retrieval

### 概要
decode が長くなる中で retrieval scorer 自体を drift に強くする。

### RelayKVへの落とし方
- prefill 時と decode 後半で scorer を変える
- 現在 query 分布への追随を強める
- 長出力時のみ drift-aware scorer を有効にする

### メリット
- 長い生成での retrieval 劣化を防ぎやすい
- long-generation 特化の改善が期待できる

### デメリット
- scorer 自体の更新や適応が必要
- 静的 score より検証が難しい
- デバッグしづらい

### 総評
**長出力問題が顕在化してからでよい。**  
今すぐの優先度はそこまで高くない。

---

## いまのおすすめ順

現時点の RelayKV に入れる優先順は次の通り。

1. **二段階 retrieval / rerank**
2. **adaptive budget**
3. **anchor bank**
4. **固定 GPU live KV budget 設計**
5. **非同期プリフェッチ**

---

## 実験導入の順番案

### 第1段階
- retrieval 指標を導入
- 二段階 retrieval を試す

### 第2段階
- adaptive budget を warmup 中だけ導入
- ACTIVE 後は cheap policy に蒸留

### 第3段階
- anchor bank を追加
- `recent / anchor / retrieval` の3層設計へ進む

### 第4段階
- `B_total` 固定の GPU live budget 設計を明示化
- state 制御と統合

### 第5段階
- CPU offload が本格運用に入るなら非同期プリフェッチ

---

## 一番大事なまとめ

今の RelayKV に最も噛み合うのは、

- **検索の頑健化（coarse-to-fine retrieval）**
- **危険度に応じた予算の動的化**
- **recent とは別の anchor 常駐**
- **GPU live KV 総量の明示管理**
- **offload 前提の非同期化**

である。

この順で導入すれば、設計の一貫性を保ちつつ、品質とコストの両面で前進しやすい。
