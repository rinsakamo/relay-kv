# Devlog — RelayKV
Date: 2026-04-11

## 今日やったこと

### 1. 実験運用ルールを整理
- `docs/experiment_protocol.md` を追加
- 標準比較条件、標準出力、selection log の扱いを明文化
- `relaykv_pipeline_summary.json` の `top_scores` を標準 selection log として使う方針を整理

### 2. pipeline summary を比較向けに拡張
- `run_relaykv_pipeline.py` に selection logging を追加
- 保存項目:
  - `selected_block_ranks`
  - `selected_block_ids`
  - `selected_block_scores`
  - `selected_block_spans`
- これで scoring variant 間の block selection 比較がしやすくなった

### 3. `mean_plus_max` を追加
- `block_scoring.py` に `mean_plus_max` を追加
- 実装は「mean + headwise max」相当の軽量 variant
- まずは最小差分で比較ラインを動かす目的で導入

### 4. `query_to_block_max` を追加
- `ColdBlock.k` を直接参照する query-aware scoring を追加
- block 内 token ごとの `q·k` を計算し、
  - head 平均
  - token 最大
 で block score を作る版を実装
- `score_blocks_with_query(..., all_blocks=all_blocks)` で pipeline から block 本体を渡すよう変更

## 今日の実験結果

### A. `4096 / prose / layer 27 / block_size=256 / top_k=3`
比較:
- `mean_plus_norm`
- `mean_plus_max`
- `query_to_block_max`

結果:
- 3者とも同一 selection
- `selected_block_ids = [14, 13, 12]`
- `mean_abs_diff = 0.023519519716501236`
- candidate / working KV も同一

解釈:
- 軽量 scoring 変更だけでなく、単純な query-aware scoring を入れても ranking は変わらなかった
- この条件では recency dominance がかなり強い

### B. `4096 / prose / layer 27 / block_size=128 / top_k=3`
比較:
- `mean_plus_norm`
- `query_to_block_max`

結果:
- 両者とも同一 selection
- `selected_block_ids = [29, 28, 26]`
- `mean_abs_diff = 0.023545723408460617`
- `block_id=27` を飛ばして `26` を拾う非連続選択になった

解釈:
- `block_size=128` では selection 自体は変わる
- ただしその変化は `query_to_block_max` 固有ではなく、`mean_plus_norm` でも同じ
- したがって、この inspected setting では **scoring variant より block granularity の方が効いている** とみるのが安全

## 今日わかったこと

1. 現在の top-ranked cold blocks はかなり安定している
- `mean_plus_norm`
- `mean_plus_max`
- `query_to_block_max`
で、`block_size=256` では完全一致だった

2. `query_to_block_max` は今の条件では ranking を動かす主因ではなかった
- token-level query-aware scoring を入れても、少なくとも inspected case では selection は変わらなかった

3. block granularity の影響が大きい
- `block_size=256` → `[14, 13, 12]`
- `block_size=128` → `[29, 28, 26]`
- 変化の主要因は scoring ではなく block の切り方

4. `CandidateKV` summary の表現に注意点が見つかった
- 非連続選択でも `build_candidate_kv()` は `start=first.start`, `end=last.end` を使う
- そのため summary 上は連続範囲に見えるが、実 tensor 長とは一致しない場合がある

## 現時点の安全な解釈

- top-ranked cold blocks are stable under several lightweight scoring modifications
- simple query-aware scoring is still not sufficient to reliably alter retrieval behavior in the current inspected setup
- block granularity may currently matter more than small scoring changes
- more structural changes may need to target either:
  - block construction / granularity
  - layer-wise allocation
  - stronger retrieval formulations

## 次回の候補

### 優先候補
1. `experimental_findings.md` に今日の scoring / granularity 結果を追記
2. `CandidateKV` summary に非連続選択を表せる情報を追加
   - 例: `selected_spans` または `is_contiguous`
3. layer-wise budget の検討へ進む
   - layer 27 を厚く
   - layer 14 を薄く
   のような配分実験

### 現時点で深追い優先度が下がったもの
- `mean_plus_max` の追加探索
- 単純な `query_to_block_max` の更なる微修正

## 一言まとめ
今日は、
**比較基盤を整備し、scoring variant を複数比較した結果、現状の inspected setting では scoring より block granularity の影響が大きい**
ところまで整理できた。
