# RelayKV devlog — HF coding probe eval / score 整理

Date: 2026-05-09 JST
Branch/Repo: `relay-kv` / `hf-coding-probe-docs-v1`

## 1. 概要

HF ベースの coding probe workflow を一通り揃えた。
現在は **smoke -> coding probe -> eval runner -> score** の流れで、
Qwen2.5-Coder-7B-Instruct-AWQ を使った軽量な評価と比較ができる。

今回の目的は runtime 置換ではなく、**HF 上で coding probe 出力の品質と安定性を見るための reporting path** を作ること。

## 2. HF AWQ 環境 baseline

- `torch 2.11.0+cu128`
- `transformers 5.8.0`
- `gptqmodel 7.0.0`
- AWQ kernel: `AwqMarlinLinear`
- GPU: `RTX 3060 12GB`
- local model path: `~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ`

## 3. 実装済み script

- `scripts/hf_model_smoke.py`
- `scripts/hf_context_length_smoke.py`
- `scripts/hf_coding_probe_v0.py`
- `scripts/run_hf_coding_probe_eval.py`
- `scripts/score_hf_coding_probe_eval.py`

## 4. 現在の workflow

### 4.1 smoke

まず model load と generate が通るかを見る。

    python scripts/hf_model_smoke.py \
      --model ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ \
      --max-new-tokens 128

### 4.2 context length smoke

次に 4K / 8K / 16K の tokenization と generation の成立を見る。

    python scripts/hf_context_length_smoke.py \
      --model ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ \
      --lengths 4096,8192,16384 \
      --max-new-tokens 64

### 4.3 coding probe

repo grounding 済み prompt で単発の coding probe を実行する。

    python scripts/hf_coding_probe_v0.py \
      --model ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ \
      --context-tokens 4096 \
      --max-new-tokens 512 \
      --probe-name relaykv_repo_entry

### 4.4 eval runner

複数 context length に対して probe を回し、raw JSON と summary を残す。

    python scripts/run_hf_coding_probe_eval.py \
      --model ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ \
      --lengths 4096,8192,16384 \
      --max-new-tokens 512 \
      --summary-out results/processed/hf_coding_probe_eval_summary.json \
      --summary-md results/processed/hf_coding_probe_eval_summary.md

### 4.5 score

eval summary を読み、per-length score と stability、recommendation を出す。

    python scripts/score_hf_coding_probe_eval.py \
      --summary-in results/processed/hf_coding_probe_eval_summary.json \
      --score-out results/processed/hf_coding_probe_eval_score.json \
      --score-md results/processed/hf_coding_probe_eval_score.md

## 5. 推奨の使い分け

- `4096`
  - default の focused coding probe
  - まずはこれで十分
- `8192`
  - 安定な代替
  - 少し広めの文脈を見たいときに使う
- `16384`
  - repo-wide inspection 用
  - default にはしない

## 6. 現時点の観察結果

- `4096` と `8192` は score `1.0`
- `16384` は score `0.833`
- `16384` は `relaykv` 配下の file を多く選びやすく、VRAM 使用量も増える

現時点の recommendation は、tie-break を含めて **4096 を default** と見るのが妥当。
8192 は十分に安定しているが、常用 default にするほどの明確な利得はまだない。
16K は repo 全体を広く見せたいときだけ使う。

## 7. 重要な safeguard

- JSON fenced parsing
  - fenced / mixed 出力でも JSON object をできるだけ抽出する
- repo grounding
  - `scripts/` と `relaykv/` の実ファイル一覧を prompt に入れる
- smoke command validation
  - `smoke_commands` の script path と `--model` 値を保守的に検証する
- context truncation
  - target token 内に prompt を収める
- stale output deletion before per-length eval
  - per-length 実行前に古い output JSON を削除し、summary 汚染を防ぐ
- 0.05 tie-break across all rows
  - best score と 0.05 以内の row 全体を見て、最小 length を推奨する

## 8. 実務上の結論

HF coding probe は、長文 runtime の再設計ではなく、**AWQ model 上で coding-task 的な repo inspection 出力を安全に比較するための path** として十分使える状態になった。

運用上は次で足りる。

1. `4096` を default にする
2. 必要なら `8192` を比較対象に追加する
3. `16384` は repo-wide inspection に限定する
4. summary と score を見て recommendation を確認する
