# RelayKV devlog — PyTorch側 三層 selection 整理時点

Date: 2026-04-25
Branch/Repo: `relay-kv` / local working branch assumed `work`

## 1. 今日の位置づけ

PyTorch側の RelayKV prototype では、KV selection を単純な `recent + retrieved` から、実用化を意識した **三層 selection** に整理した。

三層の基本形は以下。

1. **recent**  
   直近 window。decode 品質の最低保証として常に厚めに残す。

2. **anchor**  
   冒頭・指示・フォーマット・system 的な影響を持ちやすい固定ブロック。RAG 的な検索対象ではなく、文脈の土台として保持する。

3. **retrieved**  
   cold KV blocks から scoring / retrieval で選ばれる可変ブロック。長文中の関連情報を取り戻す層。

この整理により、RelayKV は「RAGの単純置換」ではなく、**長文コンテキストに対する固定長 working KV cache** として扱う方向が明確になった。

## 2. 現在の実装理解

主要 pipeline は `scripts/run_relaykv_pipeline.py` に統合されている。

確認済みの流れは以下。

```text
KV split
  -> CPU cold offload
  -> blockify
  -> metadata
  -> scoring
  -> retrieval
  -> candidate KV
  -> working KV
  -> attention comparison
```

三層 selection 整理後は、working KV の内訳を次のように見る。

```text
working KV = recent KV + anchor KV + retrieved KV
```

予算の代表例は以下。

```json
{
  "B_total_tokens": 4096,
  "B_recent_tokens": 256,
  "B_anchor_tokens": 256,
  "B_retrieval_tokens": 768,
  "B_transient_tokens": 0
}
```

このとき selection breakdown の代表例は以下。

```json
{
  "recent_tokens": 256,
  "anchor_tokens": 256,
  "retrieved_tokens": 768,
  "live_tokens": 1280,
  "prototype_materialized_tokens": 1024,
  "selection_ratio": 0.3125,
  "working_ratio": 0.25
}
```

## 3. 整理できたこと

### 3.1 recent と anchor は役割が違う

`recent` は直近の局所文脈を守る層。

`anchor` は冒頭指示・出力形式・system 的な初期文脈を守る層。

この2つを混同すると、retrieval の性能評価がブレる。特に、冒頭トークンやフォーマット指示の影響が強い prompt では、anchor を別枠で保持した方が設計意図が明確になる。

### 3.2 retrieved は品質を上げる可変層

retrieved は cold blocks から score に基づいて選ぶ。

現時点では、coverage_ratio を上げるほど mean_abs_diff が下がる傾向が確認されている。これは seq_len 1024 / 2048 / 4096、prompt type repetitive / prose / structured の広い範囲で概ね一貫している。

ただし layer によって難易度差があり、seq_len 4096 の prose では layer 27 が難しく、layer 14 は比較的扱いやすい傾向がある。

### 3.3 三層化は product 化の出口にも合う

RelayKV の実用化イメージは、OpenAI-compatible proxy で表面的に RAG を置き換えるよりも、LLM runtime / inference engine 側に近い。

つまり、現実的には以下の方向。

```text
超長コンテキスト / 長時間対話 / ローカルLLM runtime
  -> full KV を全部 materialize し続けない
  -> recent + anchor + retrieved の working KV だけで attention
  -> 固定長 memory cache 的に扱う
```

この意味で、三層 selection は実験プロトタイプとしても、実装方針としても筋が良い。

## 4. 現時点の問題点

### 4.1 scoring はまだ弱い

これまで試した軽量 scoring variants は、retrieval behavior を大きく変えるほどではなかった。

試した / 検討済みの例。

- `mean_plus_norm`
- `mean_plus_vnorm`
- `headwise_max_mean`

次に試すなら、軽い重み付け変更よりも、構造的な scoring が有力。

候補は以下。

- `query_to_block_max`
- `mean_plus_max`
- `recency_bias`
- layer-aware budget
- prompt-type aware retrieval

### 4.2 pipeline と sweep の prompt 生成ロジックを揃える必要がある

`run_relaykv_pipeline.py` は `prompt_type` を受けられるようになっているが、sweep 比較では prompt 生成ロジックがズレると比較が壊れる。

今後の比較では、以下を揃える必要がある。

- seq_len
- prompt_type
- layer_idx
- block_size
- hot_window
- anchor_blocks
- top_k / retrieval budget
- scoring variant

### 4.3 三層 selection のログを標準化したい

今後は結果 JSON に selection breakdown を必ず残したい。

最低限ほしい項目。

```json
{
  "anchor_blocks": 1,
  "budget": {
    "B_total_tokens": 4096,
    "B_recent_tokens": 256,
    "B_anchor_tokens": 256,
    "B_retrieval_tokens": 768,
    "B_transient_tokens": 0
  },
  "selection_breakdown": {
    "recent_tokens": 256,
    "anchor_tokens": 256,
    "retrieved_tokens": 768,
    "live_tokens": 1280,
    "prototype_materialized_tokens": 1024
  },
  "selection_ratio": 0.3125,
  "working_ratio": 0.25
}
```

## 5. 最有力の次ステップ

三層 selection が入った状態で、次は **retrieval 層の質** を上げる。

優先順は以下。

### Step 1: baseline 三層 selection を固定

まず、以下を固定する。

```text
recent = fixed recent window
anchor = first N blocks
retrieved = current scoring top-k
```

この状態を `three_tier_baseline` として扱う。

### Step 2: selection breakdown を全実験ログに出す

`results/raw/...json` に selection breakdown を残す。

目的は、mean_abs_diff が改善したときに、単に working KV が増えたのか、anchor/retrieval の構成が効いたのかを後から見分けること。

### Step 3: scoring を構造的に変える

次に試す候補。

```text
query_to_block_max
mean_plus_max
recency_bias
```

最初は layer 14 / seq_len 4096 / prompt_type prose で小さく見る。

### Step 4: layer 27 を hard case として確認

layer 14 で改善が見えたら、layer 27 prose 4096 に持っていく。

layer 27 は難しいので、ここで改善が出るなら価値が高い。

## 6. 実行コマンド案

### 三層 baseline の確認

```bash
python scripts/run_relaykv_pipeline.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --seq-len 4096 \
  --prompt-type prose \
  --layer-idx 14 \
  --hot-window 256 \
  --block-size 128 \
  --top-k 6 \
  --anchor-blocks 1 \
  --output results/raw/three_tier/prose_4096_layer14_anchor1_top6.json
```

### jq で selection breakdown を見る

```bash
jq '{
  anchor_blocks,
  budget,
  selection_breakdown,
  selection_ratio,
  working_ratio,
  attention_compare
}' results/raw/three_tier/prose_4096_layer14_anchor1_top6.json
```

### layer 比較

```bash
for L in 0 14 27; do
  python scripts/run_relaykv_pipeline.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda \
    --seq-len 4096 \
    --prompt-type prose \
    --layer-idx "$L" \
    --hot-window 256 \
    --block-size 128 \
    --top-k 6 \
    --anchor-blocks 1 \
    --output "results/raw/three_tier/prose_4096_layer${L}_anchor1_top6.json"
done
```

## 7. commit 案

三層 selection 整理がコードに反映済みなら、commit message は以下がよい。

```bash
git add scripts/run_relaykv_pipeline.py relaykv docs notes results/processed

git commit -m "Organize three-tier KV selection in PyTorch pipeline"
```

ログやメモだけを追加する場合。

```bash
git add notes/devlog_2026-04-25_relaykv_pytorch_three_tier_selection.ja.md

git commit -m "Add devlog for PyTorch three-tier selection"
```

## 8. 次チャットへの引き継ぎ

RelayKV は PyTorch prototype 側で、KV selection を `recent + anchor + retrieved` の三層に整理した段階。

`run_relaykv_pipeline.py` は unified pipeline として、KV split -> CPU cold offload -> blockify -> metadata -> scoring -> retrieval -> candidate KV -> working KV -> attention comparison を通す。

現在の重要な設計判断は、recent と anchor を分けること。recent は局所文脈の保護、anchor は冒頭指示・フォーマット・system 的文脈の保護、retrieved は cold blocks からの関連情報復元。

これにより、RelayKV は単純な RAG 置換ではなく、超長コンテキストや長時間対話で使う固定長 working KV cache として整理される。

次にやるべきことは、三層 baseline を固定し、selection breakdown をログ標準化し、その上で retrieval scoring を `query_to_block_max` / `mean_plus_max` / `recency_bias` などの構造的 scoring に進めること。

最初の評価対象は layer 14 / seq_len 4096 / prompt_type prose。改善が見えたら hard case として layer 27 prose 4096 に進める。
