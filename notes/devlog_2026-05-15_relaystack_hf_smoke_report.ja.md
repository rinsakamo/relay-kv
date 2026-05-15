# Devlog: RelayStack HF smoke report

- Date: 2026-05-15 JST
- Date basis: JSTで確認。PR #57 merge commit は 2026-05-15 22:29:54 +09:00。
- Scope: RelayStack Phase 9
- Repository: `rinsakamo/relay-kv`
- PR: #57 `Add RelayStack HF smoke report`

## Summary

Phase 9 では、実モデル/HF context-length smoke の実測JSONと RelayStack dry-run artifact を結合する report layer を追加した。

これにより、以下を1つのJSON reportで比較できるようになった。

- HF smoke の model load 状態
- context length ごとの成功/失敗
- OOM の有無
- peak allocated / reserved VRAM
- tokens/sec
- RelayStack final routing decision
- RelayKV VRAM reservation / memory pressure readiness

今回も report / join の範囲に限定し、model loading 自体、runtime path、attention、KV pool、scheduler、RelayKV apply は変更していない。

## What changed

### `relaykv/relaystack_hf_smoke_report.py` を追加

新しく lightweight な report module を追加した。

主な要素:

- `RelayStackHFContextRowSummary`
- `RelayStackHFSmokeReport`
- `build_relaystack_hf_smoke_report`

この module は、既存の HF context smoke JSON と RelayStack dry-run JSON を入力にして、JSON-safe な統合reportを返す。

### `scripts/run_relaystack_hf_smoke_report.py` を追加

新しい script で以下を行う。

    HF context-length smoke JSON
      + RelayStack dry-run JSON
      -> RelayStack HF smoke report JSON

CLI の基本形:

    python scripts/run_relaystack_hf_smoke_report.py \
      --hf-smoke-json /path/to/hf_context_smoke.json \
      --relaystack-json /path/to/relaystack_dry_run.json \
      --output /path/to/relaystack_hf_smoke_report.json

このscriptは model / torch / transformers を import せず、既存artifactを読むだけにした。

### Derived fields

HF smoke 側から以下を集約する。

- `hf_load_ok`
- `max_ok_context_tokens`
- `first_failed_context_tokens`
- `any_oom`
- `max_peak_allocated_mib`
- `max_peak_reserved_mib`
- `measured_vram_pressure_level`
- per-context row summary

RelayStack 側から以下を引き継ぐ。

- `final_routing_state`
- `relaymem_apply_allowed`
- `relaykv_routing_allowed`
- `fallback_reason`
- `blocking_reasons`
- `vram_reservation_status`
- `available_working_kv_budget_mib`

### Synthetic fixture-based tests

`tests/test_relaystack_hf_smoke_report.py` を追加した。

テストは synthetic JSON fixture のみを使い、以下を要求しない。

- torch
- transformers
- GPU
- model download
- actual HF model execution

これにより、Phase 9 の report parsing / aggregation を通常CIや軽量環境で検証できる。

## Review-driven fix

PR #57 では Codex review により、load-time OOM の扱いを修正した。

### load-time OOM を `any_oom` / pressure summary に反映

`hf_context_length_smoke.py` は、model load 中に失敗した場合、context row を出さずに `load.error` に例外を記録する。

当初の report は per-context row の OOM だけを見ていたため、model load 中の OOM が以下のように見える可能性があった。

    any_oom: false
    measured_vram_pressure_level: no_cuda_measurement
    report_notes: ["hf_load_failed"]

これは 12GB GPU で実際に起きやすい重要な失敗モードを隠す。

修正後は、`load.error` も OOM判定に含める。

対象例:

- `OutOfMemoryError`
- `torch.cuda.OutOfMemoryError`
- type / message に `oom` を含むケース

load-time OOM の場合:

- `any_oom == true`
- `measured_vram_pressure_level == "oom_observed"`
- `report_notes` に `hf_load_failed` と `hf_oom_observed` を含める
- load-time CUDA snapshot の peak memory も集計に含める

## Validation

PR #57 の validation は以下。

    python -m compileall relaykv tests scripts
    python -m pytest -q tests/test_relaystack_hf_smoke_report.py
    python -m pytest -q tests/test_relaystack_dry_run.py
    python scripts/run_relaystack_dry_run.py --disable-approval-gate --output /tmp/relaystack_dry_run_no_gate.json
    python -m json.tool /tmp/relaystack_dry_run_no_gate.json >/dev/null
    python scripts/run_relaystack_hf_smoke_report.py --hf-smoke-json /tmp/hf_context_smoke_synthetic.json --relaystack-json /tmp/relaystack_dry_run_no_gate.json --output /tmp/relaystack_hf_smoke_report.json
    python -m json.tool /tmp/relaystack_hf_smoke_report.json >/dev/null

## Current state after Phase 9

RelayStack は、以下の2段階のartifactを扱える状態になった。

### Planning artifact

    RelayMEM prompt preview
      + RelayKV VRAM reservation / memory pressure
      + runtime policy
      -> relaystack.final_routing_decision

### Measurement/report artifact

    HF context-length smoke result
      + RelayStack dry-run result
      -> relaystack HF smoke report

この段階で、RelayStack の判断と実機の memory envelope を並べて評価できる。

## Still not connected

Phase 9 でも以下には進んでいない。

- actual RelayKV apply
- attention / KV pool / scheduler
- runtime request path
- model internals
- automatic HF smoke execution
- real user approval UI
- external recall backend

## Design notes

Phase 9 の重要点は、**実モデル実行そのものと report parsing を分離したこと**。

実モデル実行は既存の `scripts/hf_context_length_smoke.py` が担当する。

Phase 9 の新規層は、その結果JSONを読み、RelayStack dry-run JSON と結合するだけにした。

これにより、以下を両立できる。

- 実機では Qwen2.5-Coder-7B-AWQ などを使って memory envelope を測る。
- CI / no-GPU 環境では synthetic JSON で report logic を検証する。
- RelayStack の final decision と実測 peak VRAM / OOM / context length を同じ形式で比較できる。

## Next step

次の自然な step は Phase 10。

候補:

1. 実機で `hf_context_length_smoke.py` を Qwen2.5-Coder-7B-AWQ に対して実行する。
2. その実測JSONを `run_relaystack_hf_smoke_report.py` に流す。
3. 実測reportを `results/processed/` に保存し、12GB GPU上での memory envelope を確認する。
4. RelayStack final decision と実測の乖離を整理する。
5. まだ runtime apply には進まず、まず実測reportの devlog / docs 化まで行う。

Phase 10 でも、最初は no attention / no KV pool mutation / no scheduler change / no RelayKV apply を維持する。
