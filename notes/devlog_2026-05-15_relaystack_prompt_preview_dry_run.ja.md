# Devlog: RelayMEM Prompt Preview を RelayStack dry-run artifact に接続

- Date: 2026-05-15 JST
- Date basis: JSTで確認。PR #53 merge commit は 2026-05-15 21:06:23 +09:00。
- Scope: RelayStack Phase 7
- Repository: `rinsakamo/relay-kv`
- PR: #53 `Connect RelayMEM prompt preview to RelayStack dry run`

## Summary

Phase 7 では、Phase 6.5 で追加した RelayMEM Prompt Preview / User-Gated Fallback planning を RelayStack dry-run artifact に接続した。

これにより、RelayStack の dry-run JSON 上で以下を同時に見られるようになった。

- RelayMEM retrieval results
- RelayMEM context assembly plan
- RelayMEM prompt preview plan
- user-gated fallback compatibility summary
- RelayKV VRAM reservation / memory pressure state

今回も引き続き schema / smoke / dry-run planning の範囲に限定し、model / GPU / runtime / attention / KV / scheduler には接続していない。

## What changed

### RelayStack dry-run artifact に `prompt_preview_plan` を追加

`scripts/run_relaystack_dry_run.py` の RelayMEM section に、既存の `context_assembly_plan` に加えて `prompt_preview_plan` を追加した。

概念上の流れは以下。

    synthetic RelayMEM retrieval results
      -> context_assembly_plan
      -> prompt_preview_plan
      -> user_gated_fallback compatibility summary
      -> RelayKV VRAM reservation / memory pressure decision
      -> one RelayStack dry-run artifact

### `user_gated_fallback` を互換summaryとして維持

`user_gated_fallback` は既存の `relaystack_dry_run_v1` consumer 互換のため残した。

ただし、手作りの別状態として増殖させるのではなく、`prompt_preview_plan` と整合する compatibility view として扱う方針にした。

保持する主な情報:

- approval_required
- approval_reason
- proposed_retrieval_mode
- fallback_if_denied
- fallback_reason
- user_visible_message
- can_apply_without_user_approval

### Summary / stdout に Prompt Preview 状態を追加

dry-run summary と CLI stdout から、Prompt Preview の主要状態を確認できるようにした。

追加された主な確認項目:

- preview_item_count
- prompt_preview_dropped_memory_count
- prompt_preview_approval_required
- prompt_preview_can_apply_without_user_approval
- prompt_preview_fallback_reason

## Review-driven fixes

PR #53 では、Codex review により RelayStack artifact の整合性を複数回修正した。

### 1. `proposed_retrieval_mode` を互換viewに保持

`relaystack_dry_run_v1` では、approval gate が何を承認させるのかを `user_gated_fallback.proposed_retrieval_mode` から読める必要がある。

そのため、default approval-gated dry-run では以下を維持した。

    user_gated_fallback.proposed_retrieval_mode == "deep_recall"
    user_gated_fallback.fallback_if_denied == "fast_recall"

### 2. Deep Recall / Fast Recall の user-facing message を mode-aware 化

Prompt Preview integration 後、`retrieval_mode == deep_recall` なのに `Fast Recall prepared...` と表示される経路が複数見つかった。

修正方針:

- `retrieval_mode == deep_recall` なら Deep Recall / deeper memory recall 系の文言を使う。
- `retrieval_mode == fast_recall` なら Fast Recall 文言を使う。
- approval gate 有無に関係なく mode-aware にする。

### 3. tight-budget fallback でも mode-aware message を維持

`--token-budget 1` のような tight budget では preview items が落ちる。

この場合でも、`prompt_preview_plan.retrieval_mode == deep_recall` なら、fallback/budget message も Deep Recall 系にする必要がある。

修正後の優先順位:

1. fallback_reason / token_budget_exceeded
2. dropped_memory_ids
3. empty preview
4. approval_required
5. normal no-approval preview

### 4. `context_assembly_plan.user_visible_message` の互換性を維持

default approval-gated dry-run では、`context_assembly_plan.approval_required=True` かつ `approval_reason` があるにもかかわらず `user_visible_message=None` になると、context assembly plan を直接読む consumer から見て artifact が自己完結しない。

そのため、context assembly plan にも approval message を保持するよう修正した。

## Validation

PR #53 の validation は以下。

    python -m compileall relaykv tests scripts
    python -m pytest -q tests/test_relaymem_fast_recall.py
    python -m pytest -q tests/test_relaymem_prompt_preview.py
    python -m pytest -q tests/test_relaymem_context_assembly_smoke.py
    python -m pytest -q tests/test_relaystack_dry_run.py
    python scripts/run_relaymem_fast_recall_smoke.py --output /tmp/relaymem_fast_recall_smoke.json
    python -m json.tool /tmp/relaymem_fast_recall_smoke.json >/dev/null
    python scripts/run_relaymem_prompt_preview_smoke.py --output /tmp/relaymem_prompt_preview_smoke.json
    python -m json.tool /tmp/relaymem_prompt_preview_smoke.json >/dev/null
    python scripts/run_relaystack_dry_run.py --output /tmp/relaystack_dry_run.json
    python -m json.tool /tmp/relaystack_dry_run.json >/dev/null
    python scripts/run_relaystack_dry_run.py --disable-approval-gate --output /tmp/relaystack_dry_run_no_gate.json
    python -m json.tool /tmp/relaystack_dry_run_no_gate.json >/dev/null
    python scripts/run_relaystack_dry_run.py --token-budget 1 --output /tmp/relaystack_dry_run_tight_budget.json
    python -m json.tool /tmp/relaystack_dry_run_tight_budget.json >/dev/null

## Current state after Phase 7

RelayStack dry-run は、以下の統合 planning artifact になった。

    RelayMEM records / retrieval results
      -> context assembly
      -> prompt preview
      -> user-gated fallback compatibility summary
      -> VRAM reservation
      -> RelayKV memory pressure decision
      -> one JSON artifact

まだ未接続のもの:

- actual user interaction
- external recall backend
- model loading
- GPU inspection
- runtime request path
- attention / KV pool
- scheduler decision
- actual RelayKV apply path

## Design notes

今回のレビューで、RelayStack artifact の責務境界がより明確になった。

- `prompt_preview_plan`: user-facing preview / fallback message の本体
- `context_assembly_plan`: selected context / token budget / approval metadata の既存計画
- `user_gated_fallback`: `relaystack_dry_run_v1` 互換の summary view
- `relaykv`: VRAM reservation / memory pressure / routing readiness

重要なのは、同じ artifact 内で `retrieval_mode`, `proposed_retrieval_mode`, `fallback_if_denied`, `user_visible_message`, `can_apply_without_user_approval` が矛盾しないこと。

## Next step

次の自然な step は Phase 8。

Phase 8 では、RelayMEM preview gate / RelayKV readiness / VRAM reservation をまとめて、RelayStack final routing decision dry-run を追加する。

候補:

1. `relaystack_final_decision` の lightweight schema を追加する。
2. RelayMEM approval state / preview state / fallback reason を decision input にする。
3. RelayKV VRAM reservation status / memory pressure state を decision input にする。
4. Final state を例として以下のように整理する。

    - `apply_relaymem_and_relaykv`
    - `apply_relaymem_only`
    - `relaykv_shadow_only`
    - `fallback_full_context_recent_only`
    - `blocked_waiting_for_user_approval`
    - `blocked_no_kv_budget`

ただし、Phase 8 も引き続き no model / no GPU / no runtime / no attention / no KV / no scheduler の dry-run planning に限定する。
