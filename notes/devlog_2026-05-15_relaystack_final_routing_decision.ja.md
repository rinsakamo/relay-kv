# Devlog: RelayStack final routing decision dry-run

- Date: 2026-05-15 JST
- Date basis: JSTで確認。PR #55 merge commit は 2026-05-15 21:52:52 +09:00。
- Scope: RelayStack Phase 8
- Repository: `rinsakamo/relay-kv`
- PR: #55 `Add RelayStack final routing decision dry run`

## Summary

Phase 8 では、RelayStack dry-run artifact に `relaystack.final_routing_decision` を追加した。

これにより、RelayMEM prompt preview / user gate / fallback reason / RelayKV VRAM reservation / memory pressure readiness / runtime policy を、1つの最終判断として読めるようになった。

今回も schema / smoke / dry-run planning の範囲に限定し、model / GPU / runtime / attention / KV / scheduler には接続していない。

## What changed

### `relaykv/relaystack_final_decision.py` を追加

新しく lightweight な final decision module を追加した。

主な要素:

- `RelayStackFinalRoutingState`
- `RelayStackFinalRoutingDecision`
- `decide_relaystack_final_routing`

この module は RelayStack の最終 routing 方針を JSON-safe summary として返すための dry-run planning layer であり、runtime 実行や scheduler 変更は行わない。

### RelayStack dry-run artifact に final decision を追加

`scripts/run_relaystack_dry_run.py` に以下の section を追加した。

    relaystack.final_routing_decision

これにより、dry-run JSON 上で以下の流れが一つの artifact にまとまった。

    RelayMEM retrieval results
      -> context_assembly_plan
      -> prompt_preview_plan
      -> user_gated_fallback compatibility summary
      -> RelayKV VRAM reservation / memory pressure decision
      -> relaystack.final_routing_decision

### Summary / stdout に final decision を追加

CLI出力と summary に、最終判断の主要項目を追加した。

主な項目:

- final_routing_state
- relaymem_apply_allowed
- final_relaykv_routing_allowed
- final_fallback_reason
- final_blocking_reasons

## Decision policy

Phase 8 時点の final decision は、実行ではなく dry-run planning として deterministic な優先順位で判断する。

重要な方針:

1. Prompt-preview fallback は user approval より優先する。
2. `token_budget_exceeded` など、承認しても適用不能な状態では `waiting_for_user_approval` にしない。
3. no-KV-budget は RelayKV を止める理由だが、RelayMEM-only が可能なら RelayMEM-only path を維持する。
4. Approval gate が残っている場合は、適用可能な plan に対してだけ approval 待ちにする。
5. RelayKV readiness と RelayMEM apply 可否を分離して扱う。

## Review-driven fixes

PR #55 では Codex review により、final decision の優先順位を修正した。

### 1. Prompt-preview fallback を approval より優先

当初、`--token-budget 1` のように prompt preview が `fallback_reason == "token_budget_exceeded"` を持つ場合でも、approval gate が有効だと `waiting_for_user_approval` になり得た。

しかし、preview items が落ちている場合、ユーザーが承認しても適用できない。

修正後は、fallback/blocking reason を approval より先に評価する。

### 2. no-KV-budget でも RelayMEM-only path を維持

当初、KV budget が不足すると final state が `blocked_no_kv_budget` になり、RelayMEM-only が可能なケースまで全体 block と読める可能性があった。

修正後は、no-KV-budget は RelayKV routing を止める理由として扱い、Prompt Preview が approval なしで適用可能なら `relaymem_only` を維持する。

## Validation

PR #55 の validation は以下。

    python -m compileall relaykv tests scripts
    python -m pytest -q tests/test_relaymem_fast_recall.py
    python -m pytest -q tests/test_relaymem_prompt_preview.py
    python -m pytest -q tests/test_relaymem_context_assembly_smoke.py
    python -m pytest -q tests/test_relaystack_dry_run.py
    python scripts/run_relaystack_dry_run.py --output /tmp/relaystack_dry_run.json
    python -m json.tool /tmp/relaystack_dry_run.json >/dev/null
    python scripts/run_relaystack_dry_run.py --disable-approval-gate --output /tmp/relaystack_dry_run_no_gate.json
    python -m json.tool /tmp/relaystack_dry_run_no_gate.json >/dev/null
    python scripts/run_relaystack_dry_run.py --token-budget 1 --output /tmp/relaystack_dry_run_tight_budget.json
    python -m json.tool /tmp/relaystack_dry_run_tight_budget.json >/dev/null
    python scripts/run_relaystack_dry_run.py --model-weights-reserved-mib 11000 --output /tmp/relaystack_dry_run_no_kv_budget.json
    python -m json.tool /tmp/relaystack_dry_run_no_kv_budget.json >/dev/null

## Current state after Phase 8

RelayStack dry-run artifact は、以下をまとめる統合 planning artifact になった。

    RelayMEM preview gate
    + RelayMEM fallback reason
    + RelayKV VRAM reservation
    + RelayKV memory pressure readiness
    + runtime policy
    = relaystack.final_routing_decision

まだ未接続のもの:

- actual model loading
- HF model smoke integration
- GPU/runtime inspection
- actual user approval interaction
- actual RelayMEM external backend
- attention / KV pool / scheduler
- runtime RelayKV apply

## Design notes

Phase 8 の重要点は、RelayMEM と RelayKV を同じ最終状態で扱いながらも、両者の可否を混同しないこと。

- RelayMEM apply が可能でも RelayKV が不可の場合がある。
- RelayKV budget が不可でも RelayMEM-only は成立し得る。
- Prompt-preview fallback がある場合は、approval gate より fallback/blocking が優先される。
- Final state だけでなく、`relaymem_apply_allowed` と `relaykv_routing_allowed` を併記する。

これにより、将来の UI / runner / HF smoke / runtime adapter が final state だけでなく、どの層が apply 可能かを安全に読める。

## Next step

次の自然な step は Phase 9。

Phase 9 では、実モデル/HF smoke 側の artifact を RelayStack dry-run と接続する。

候補:

1. `scripts/hf_context_length_smoke.py` または既存HF smoke結果を RelayStack artifact と並べる。
2. モデルロード結果、context length、peak VRAM、tokens/sec を RelayStack planning artifact の外側に添付する。
3. まだ runtime apply には進まず、HF smoke artifact + RelayStack dry-run artifact の join / report に留める。
4. 12GB VRAM 前提で、RelayStack がどう判断するかを実測 memory envelope と比較できるようにする。

Phase 9 でも、最初は no attention / no KV pool mutation / no scheduler change / no RelayKV apply を維持する。
