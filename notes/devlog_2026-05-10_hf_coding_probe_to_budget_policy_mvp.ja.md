# RelayKV Devlog — HF Coding Probe 安定化から Budget Policy MVP へ

- Date basis: **2026-05-10 JST**
- Scope: `relay-kv` repo
- Phase: HF coding probe evaluation foundation completed; next phase starts RelayKV budget policy MVP
- Status: HF coding probe is now usable as a practical local model-output evaluation harness. Next work should move back to RelayKV core policy.

---

## 1. 日付確認

この devlog は **JST 2026-05-10** 基準で作成する。

前回までの HF coding probe 系 PR 群は、Qwen2.5-Coder-7B-Instruct-AWQ を使った RelayKV 開発補助評価基盤を整備する目的で進めた。現時点では、coding probe 側は一応の評価に使える段階まで到達したため、次は RelayKV 本体の実用性判断に必要な **budget policy MVP** に進む。

---

## 2. 今回までに完了したこと

### 2.1 HF / AWQ 実行基盤

HF Transformers 側で Qwen2.5-Coder-7B-Instruct-AWQ を使うための smoke / context smoke を整備した。

完了済み:

- `scripts/hf_model_smoke.py`
- `scripts/hf_context_length_smoke.py`
- Qwen2.5-Coder-7B-Instruct-AWQ のロード・生成確認
- RTX 3060 12GB 上で 4K / 8K / 16K 入力の smoke 確認
- `results/raw/` / `results/processed/` を gitignore 対象として整理

この段階で、ローカルLLMを RelayKV repo 作業の補助評価に使う実行環境は成立した。

---

### 2.2 HF coding probe v0

`hf_coding_probe_v0.py` を中心に、LLM出力を JSON schema で評価する probe を整備した。

完了済み:

- JSON-only output
- `parse_ok`
- `validation_ok`
- profile別 `required_keys`
- repo grounding
- `relevant_files` の存在確認
- `smoke_commands` の script path / option validation
- context truncation fix
- `--trust-remote-code` allowlist 対応
- fenced JSON parsing

これにより、LLMの回答を「自然文として良さそう」ではなく、機械的に検査できる状態になった。

---

### 2.3 eval runner / scorer

単発 probe から、profile / length ごとの評価に進めるための runner と scorer を整備した。

完了済み:

- `scripts/run_hf_coding_probe_eval.py`
- `scripts/score_hf_coding_probe_eval.py`
- `--probe-names` による複数profile評価
- length別評価
- profile別 scoring
- stale per-length output 対策
- `MissingOutputAfterSubprocess`
- stability metrics
- score markdown 出力

これにより、4K / 8K / 16K の比較や、profile別の評価が可能になった。

---

### 2.4 probe profiles

現在の主要 profiles:

- `relaykv_repo_entry`
- `relaykv_bug_triage`
- `relaykv_smoke_plan`
- `relaykv_result_interpretation`
- `relaykv_safe_next_change`

各profileは required keys を持ち、出力の構造と期待役割を分けている。

---

### 2.5 real-case input

repo-only の評価から、実際のログ / Codex review / jq結果を渡す real-case 評価へ拡張した。

完了済み:

- `--case-text`
- `--case-file`
- `--case-name`
- eval runner から case input を pass-through
- summary rows に case metadata を追加
- long case input で task guidance が落ちない prompt ordering fix

これにより、実際の失敗ログや review 指摘を入力して、`bug_triage` / `result_interpretation` を評価できるようになった。

---

### 2.6 case-aware grounding

real-case input から関連ファイルを推定する軽量 grounding を追加した。

完了済み:

- `case_related_files`
- exact repo path / exact script name / keyword mapping
- primary case file validation
- `CaseRelatedFileMissing`
- `CaseRelatedPrimaryFileMissing`

代表例:

- `stale_output_review`
  - primary: `scripts/run_hf_coding_probe_eval.py`
  - secondary: `scripts/hf_coding_probe_v0.py`

以前は `relaykv/cold_cache.py` や存在しない `scripts/relaykv_decode_prototype.py` にズレることがあったが、現在は eval-runner case を `scripts/run_hf_coding_probe_eval.py` に寄せられる。

---

### 2.7 command quality guidance

`smoke_commands` の品質を改善した。

完了済み:

- allowed command patterns
- profile-specific command guidance
- eval-runner / scorer command examples
- `--help` / `-h` allowlist 整合
- angle-bracket placeholder 除去
- concrete filename 化
- `ShellPlaceholderInCommand`
- quoted comparison text の誤検出修正

解消済みの問題:

- `--output-path` / `--input-path` のような存在しない option を生成
- `results/processed/<name>.json` のような shell redirection 的に危険な placeholder
- `--help` を prompt で推奨しているのに validator で弾く不整合
- `--case-text 'seq_len > hot_window'` を shell placeholder と誤検出

---

### 2.8 risk wording quality

ローカル評価プロトタイプに対して強すぎる risk 表現を抑制した。

完了済み:

- `data loss`
- `data corruption`
- `security breach`
- `production outage`

のような過剰表現を避け、以下のような表現へ誘導:

- stale result file
- misleading validation result
- missed warning
- incomplete triage
- wrong relevant file selection
- unsupported command option
- hallucinated script path
- prompt truncation

確認済みの代表例:

- `Stale result may hide subprocess failure`
- `Stale result file if the subprocess fails before rewriting output_path.`

`OverstatedRiskWording` validation は維持したまま、prompt側で過剰表現を減らした。

---

## 3. 現在できる評価

現時点で、HF coding probe は以下を評価できる。

- modelが required JSON schema に従えるか
- profile別 required keys を満たせるか
- repo context に基づいて実在ファイルを選べるか
- real-case input を反映できるか
- primary case file を拾えるか
- smoke command が実在script / supported option に基づくか
- shell placeholder や unsupported option を出さないか
- risk wording がローカル評価文脈に合っているか
- 4K / 8K / 16K の context length 差で出力品質がどう変わるか

これは RelayKV 本体の最終性能評価ではないが、**RelayKV 開発補助としてのローカルLLM出力品質評価器**としては一応使える状態に達した。

---

## 4. 現在まだ弱いこと

まだ弱い、または今後の課題:

- 実際にコード変更を正しく作れるか
- 修正案がテストを通すか
- bug triage の原因推定が本当に正しいか
- result interpretation が数値・ログの意味をどこまで理解しているか
- safe next change が本当に最小差分になるか
- profile-specific scoring がまだ粗い

このため、HF coding probe は今後も補助基盤として使いつつ、主作業は RelayKV 本体の実用性判断に戻す。

---

## 5. 次フェーズ: RelayKV Budget Policy MVP

次に進むべき作業は、RelayKV 本体に **budget-driven working KV policy** を入れること。

目的:

```text
RelayKV本体に、
「固定 working KV budget の中で RECENT / ANCHOR / RETRIEVED をどう配分して残すか」
を明示的に扱う最小機能を追加する。
```

短期の budget model:

```text
B_total_working_kv =
  B_recent
+ B_anchor
+ B_retrieval
+ optional B_transient
```

ただし MVP では、まず VRAM bytes ではなく token/block budget で扱う。

---

## 6. Budget Policy MVP で最初に実装するもの

最小実装:

- `total_working_blocks`
- `recent_blocks`
- `anchor_blocks`
- `retrieval_blocks`
- block id selection
- deduplication
- deterministic ordering
- budget enforcement
- `budget_ok`
- `fallback_reason`
- JSON-serializable policy decision

想定出力例:

```json
{
  "policy_name": "budget_mvp",
  "seq_len": 4096,
  "block_size": 128,
  "budgets": {
    "total_working_blocks": 12,
    "recent_blocks": 6,
    "anchor_blocks": 2,
    "retrieval_blocks": 4
  },
  "selected": {
    "recent_block_ids": [26, 27, 28, 29, 30, 31],
    "anchor_block_ids": [0, 1],
    "retrieved_block_ids": [12, 15, 18, 21],
    "working_block_ids": [0, 1, 12, 15, 18, 21, 26, 27, 28, 29, 30, 31]
  },
  "budget_ok": true,
  "fallback_reason": null
}
```

---

## 7. Budget Policy MVP の境界

今回やらないこと:

- SGLang KV pool read/write
- attention backend接続
- scheduler変更
- runtime writeback
- vLLM adapter
- HiCache / RadixTree 連携
- FlashInfer / Triton attention接続
- 実GPU KV materialization

今回やるのは、engine-independent な budget/policy/logging 層のみ。

---

## 8. Budget Policy MVP の評価軸

MVP後に見るべき評価:

1. same seq_len / same prompt_type で budget を変えたとき、mean_abs_diff がどう変わるか
2. recent / anchor / retrieval 配分を変えたとき、品質がどう変わるか
3. same working_ratio で recent-only より良いケースがあるか
4. structured / prose / repetitive のどれで破綻しやすいか
5. 4096 → 8192 → 16384 で policy が破綻しないか

比較対象:

- recent-only
- anchor+recent
- retrieval-only寄り
- budget policy MVP

---

## 9. 次の推奨branch

```bash
git switch main
git pull origin main
git status --short

git switch -c relaykv-budget-policy-mvp
```

---

## 10. 次にCodexへ渡す作業概要

次の作業では、まず `scripts/` と `relaykv/` を確認し、既存の block selection / retrieval / working KV assembly に最小差分で budget policy を差し込む。

重要な制約:

- repo事実優先
- `scripts/` → `relaykv/` の順に確認
- 既存pipelineを壊さない
- budget flags 未指定時は既存挙動維持
- result artifacts は commit しない
- attention接続やengine統合には進まない

---

## 11. コミット方針

devlogをcommitする場合:

```bash
git add notes/devlog_2026-05-10_hf_coding_probe_to_budget_policy_mvp.ja.md
git commit -m "Document HF coding probe milestone and budget policy next step"
git push -u origin <branch-name>
```

Budget Policy MVP の実装コミットは devlog とは分けるのが望ましい。

---

## 12. 現在の結論

HF coding probe は、RelayKV開発補助として一応の評価に使える状態になった。

次は RelayKV 本体に戻り、**固定 working KV budget の中で RECENT / ANCHOR / RETRIEVED を選ぶ最小 budget policy** を実装する。

この MVP により、RelayKV の実用性判断に必要な「budgetを守れるか」「recent-onlyより良い配分があるか」「長文で破綻しないか」を検証できるようにする。
