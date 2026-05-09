# RelayKV devlog — HF coding probe milestone

Date: 2026-05-09 JST
Branch/Repo: `relay-kv` / `hf-coding-probe-devlog-20260509`

## 1. 日付確認

この devlog の日付は **2026-05-09 JST**。

## 2. 要約

HF Transformers / PyTorch 側で、Qwen2.5-Coder-7B-Instruct-AWQ を使った
coding probe 評価基盤が一通り成立した。

local validation で使った model path は以下。

    ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ

環境 baseline は以下。

- `torch 2.11.0+cu128`
- `transformers 5.8.0`
- `gptqmodel 7.0.0`
- `triton 3.6.0`
- AWQ kernel: `AwqMarlinLinear`
- GPU: `RTX 3060 12GB`

## 3. 実装済み / merge 済み要素

- `scripts/hf_model_smoke.py`
- `scripts/hf_context_length_smoke.py`
- `scripts/hf_coding_probe_v0.py`
- `scripts/run_hf_coding_probe_eval.py`
- `scripts/score_hf_coding_probe_eval.py`
- repo grounding
- command validation
- context truncation
- stale output handling
- score tie-break across all rows
- multi-profile task set
- profile grouped scoring
- profile quality warnings
- `--trust-remote-code` allowlist fix

## 4. 観測結果

- `4096` / `8192` は focused coding probe として安定
- `16384` は repo-wide inspection には有用だが、`relevant_files` が広がりやすく VRAM も重い
- task profile 群は `4K` / `8K` で実行でき、strict quality warning 導入前は `parse_ok` / `validation_ok` ともに通る条件があった
- 現在は profile quality warnings により、**`parse_ok` と `validation_ok` を分離**して見られる

## 5. 現在の推奨運用

- default focused probe: `4096`
- stable alternative: `8192`
- repo-wide inspection: `16384`
- `validation_ok=false` は **quality warning** と見なす。runtime failure と同義ではない

## 6. 重要な設計境界

- これは **HF 側の evaluation / support infrastructure** である
- RelayKV core internals 自体は変更していない
- SGLang / vLLM / FlashInfer integration はここでは再開していない

## 7. 次の候補

- known-script option validation の refinement
- mentioned-file と `relevant_files` の整合 warning
- profile-specific scoring
- `bug_triage` / `result_interpretation` に実際の failure / result snippet を入れる運用
- 将来的に probe 出力を safe な Codex prompt generation に接続する。ただし automatic patching には直結させない
