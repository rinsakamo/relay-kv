# RelayKV Three-Tier Memory Design: recent / anchor / retrieval

## 全体像

RelayKV の live KV を、役割で3つに分ける。

### 1. recent
**直近整合のための記憶**

### 2. anchor
**指示・形式・主題を固定するための記憶**

### 3. retrieval
**遠距離の内容参照を必要時に取り出す記憶**

この3つを分けると、

- recent だけでは弱い
- prefix 全保持は重い
- retrieval だけでは制御条件が不安定

という問題をかなりきれいに分担できる。

---

## 1. recent の役割

recent は、いわゆる recent window である。

### 主に守るもの
- 局所文法
- 直前の構文
- 出力のつながり
- 直近の意味整合
- 直近の自己生成トークン列

### 特徴
- decode では最重要
- 壊れると文法や接続が崩れやすい
- floor として最低保証したい

### イメージ
**作業記憶の最前線**

---

## 2. anchor の役割

anchor は、recent とは別に持つ  
**長く効く制御条件・枠組み**の保持領域である。

### 主に守るもの
- 出力形式指定
- JSON / XML / schema
- 「3点で答える」などの制約
- system 的な役割指定
- 最初の質問の主題
- 固有名詞や対象定義
- タスクの前提条件

### 特徴
- 位置的には文頭や前半に多い
- 直近ではないが、生成全体に効く
- retrieval に毎回頼ると不安定になりやすい
- pinned / persistent に持つ価値がある

### イメージ
**ルールブックや設計図**

---

## 3. retrieval の役割

retrieval は、常駐させない cold 側情報から  
**その step で必要なものだけ回収する領域**である。

### 主に守るもの
- 遠距離内容参照
- 長文前半の詳細
- 途中に出た重要説明
- 例外条件
- 長大履歴の特定部分

### 特徴
- query 依存性が高い
- 毎step必要とは限らない
- 動的選択が本質
- RelayKV のコアそのもの

### イメージ
**長期記憶からの検索結果**

---

## 4. 3者の違いを一言で

- **recent** = いま直前で効く
- **anchor** = ずっと効く
- **retrieval** = 必要なときだけ効く

この分け方がかなり大事である。

---

## 5. なぜ recent と anchor を分けるのか

文頭の重要指示は、長く効くけれど recent ではない。  
一方 recent は強く効くけれど、主に局所整合用である。

したがって文頭の制約を recent に期待すると、

- 長出力で薄れる
- retrieval に頼ると取り逃す
- フォーマット崩れが起きる

ことがある。

だから、

**recent は局所整合**
**anchor は全体制御**
で分けるのが自然である。

---

## 6. なぜ anchor を retrieval とも分けるのか

anchor を全部 retrieval 任せにすると、

- 毎step 取れる保証がない
- score が低いと落ちる
- 指示や形式の保持が不安定になる

からである。

例えば JSON schema や「箇条書きで」のような制約は、  
内容参照とは違って**常時有効な制御情報**である。

したがってそれを毎回 retrieval するのは危ない。

anchor は、

**retrieval すべき内容**
ではなく
**常駐させるべき制御情報**
として扱う価値がある。

---

## 7. 予算構成

GPU live KV budget を4つに分ける。

### B_recent
recent full 用

### B_anchor
anchor pinned 用

### B_retrieval
動的 retrieval 用

### B_transient
転送・一時バッファ用

つまり、

**B_total = B_recent + B_anchor + B_retrieval + B_transient**
である。

---

## 8. 各予算の性格

### B_recent
- 最低保証が必要
- risk 高いと厚くしたい

### B_anchor
- 比較的小さい固定枠でよい
- 一度決めたら大きく揺らさない

### B_retrieval
- 最も可変
- step / layer / risk / demand で動かす

### B_transient
- 安全用
- OOM防止

この性格の違いが重要である。

---

## 9. 状態制御との統合

非稼働 / 助走 / 稼働とつなげるとこうなる。

### 非稼働
- RelayKVなし
- anchor も必須でなければ特別扱いしなくてよい
- full 寄り

### 助走
- recent を厚く
- anchor を固定化
- retrieval は保守的

### 稼働
- recent は floor を守る
- anchor は常駐
- retrieval を本格運用

つまり助走では、

**recent を育てつつ、anchor を先に安定させる**
感じになる。

---

## 10. anchor の選び方

最初は単純でよい。

### v1
文頭 N token をそのまま anchor とする

### v2
文頭のうち
- system 相当
- task 定義
- format 指定
- schema 部分
だけを block 単位で pin

### v3
prefix importance scoring で選ぶ

実装順としてはこれが自然である。

---

## 11. anchor 候補

最初に pin しやすいものは次のあたり。

### 強い候補
- 「JSONで出力」
- 「3点で」
- 「表形式で」
- 「コードのみ」
- 「以下のフォーマット」
- schema 定義
- タスク対象の定義文
- 問題文の中心命題

### 弱い候補
- 挨拶
- 冗長な前置き
- 内容に直接効かない背景説明

---

## 12. この設計の利点

### 利点1
recent と prefix 制約を混同しなくてよい

### 利点2
長出力でのフォーマット崩れに強くなりうる

### 利点3
retrieval 予算を「内容参照」に集中できる

### 利点4
GPU live budget の意味が整理しやすい

---

## 13. 注意点

### 注意1
anchor を増やしすぎると retrieval 枠を圧迫する

### 注意2
prefix 以外にも途中の重要制約更新がある  
将来的には prefix anchor だけでなく global anchor も考えたい

### 注意3
anchor の選び方が雑だと無駄保持が増える

---

## 14. 一番自然な最初の形

最初にやるならこれがよい。

- **recent**: 現行 recent full
- **anchor**: 文頭の小さい固定枠
- **retrieval**: 現行 cold retrieval
- **budget**: `B_total = B_recent + B_anchor + B_retrieval + B_transient`

この形なら、今の設計にも載せやすい。

---

## 15. 一番大事なまとめ

`recent / anchor / retrieval` の3層メモリ設計では、

- **recent** が直近整合を守る
- **anchor** が指示・形式・主題を固定する
- **retrieval** が遠距離内容を必要時に回収する

という役割分担になる。

つまり RelayKV は、

**局所整合・全体制御・遠距離参照を分けて管理するメモリ制御系**
として設計できる。
