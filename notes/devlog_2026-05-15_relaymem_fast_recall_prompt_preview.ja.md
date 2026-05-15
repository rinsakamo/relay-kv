# Devlog: RelayMEM Fast Recall backend と Prompt Preview / User-Gated Fallback planning

- Date: 2026-05-15 JST
- Date basis: JSTで確認。PR #50 merge commit は 2026-05-15 15:18:33 +09:00、PR #51 は 2026-05-15 夜の作業として整理。
- Scope: RelayMEM Phase 6 / Phase 6.5
- Repository: `rinsakamo/relay-kv`

## Summary

RelayStack の memory/context assembly 側を、設計文書だけでなく実際の stdlib-only planning path へ進めた。

今回のまとまりでは、RelayMEM を単なる RAG ではなく、RelayStack の階層的 memory / active context assembly layer として扱い、次の2段を実装した。

1. Phase 6: RelayMEM Fast Recall backend v1
2. Phase 6.5: Fast Recall results -> Prompt Preview / User-Gated Fallback planning

どちらも model / GPU / runtime / attention / KV / scheduler には接続していない。現段階では schema / smoke / dry-run planning の範囲に限定した。

## Completed PRs

### PR #50: Add RelayMEM Fast Recall backend

Fast Recall backend v1 を追加した。

主な追加内容:

- `relaykv/relaymem_fast_recall.py`
- `scripts/run_relaymem_fast_recall_smoke.py`
- `tests/test_relaymem_fast_recall.py`
- `relaykv/__init__.py` からの lightweight export

Fast Recall は以下の RelayMEM record を lightweight retrieval candidate に変換する。

- `RelayMEMProfileRecord`
- `RelayMEMEpisodeRecord`
- `RelayMEMSummaryRecord`
- `RelayMEMStructuredRecord`
- `RelayMEMKVCheckpointMetadata`

返り値は既存の `RelayMEMRetrievalResult` を使う。外部 vector DB / embedding / model 呼び出しは行わない。

当初の scoring では raw overlap count が強すぎたため、review を受けて `overlap_ratio` と memory class priority / importance を使う形へ調整した。Fast Recall は pure keyword ranking ではなく、Profile / Structured / Summary / Episode などの memory class priority を反映する軽量 recall として扱う。

### PR #51: Add RelayMEM prompt preview fallback planning

Fast Recall results を、ユーザーに提示できる preview plan へ変換する planning layer を追加した。

主な追加内容:

- `relaykv/relaymem_prompt_preview.py`
- `scripts/run_relaymem_prompt_preview_smoke.py`
- `tests/test_relaymem_prompt_preview.py`
- `docs/current_status.md` の Phase 6 / 6.5 更新
- `relaykv/__init__.py` からの lightweight export

追加した主な schema:

- `RelayMEMPromptPreviewItem`
- `RelayMEMPromptPreviewPlan`
- `build_relaymem_prompt_preview_plan(...)`

Prompt Preview plan は以下を保持する。

- query
- retrieval mode / backend kind
- preview items
- dropped memory IDs
- total estimated tokens
- token budget
- approval required flag
- approval reason
- user-visible message
- fallback-if-denied
- fallback reason
- can-apply-without-user-approval flag

## Important fixes during review

PR #51 では、Codex review により fallback / user approval 周りの矛盾を修正した。

### 1. fallback_reason がある場合は auto-apply しない

`fallback_reason` が明示されているのに `can_apply_without_user_approval=True` になると、将来の consumer が apply flag だけを見て誤適用するリスクがある。

そのため、auto-apply 条件を次のように整理した。

    can_apply_without_user_approval = (
        not approval_required
        and not bool(context_plan.dropped_memory_ids)
        and resolved_fallback_reason is None
    )

### 2. budget fallback message を empty-preview message より優先

retrieval result は存在するが token budget により全件 dropped された場合、単に “no memory found” と表示すると誤解を招く。

そのため、user-visible message の優先順位を以下の方向へ整理した。

1. fallback / budget
2. dropped memory
3. empty preview
4. approval required
5. normal preview

### 3. `--no-approval-required` smoke の UX 矛盾を修正

`--no-approval-required` なのに approval prompt や approval reason が artifact に残ると、non-gated preview planning の確認として矛盾する。

そのため、smoke script では approval が必要な場合だけ approval reason / fallback-if-denied を渡すよう整理した。

### 4. tight budget 時に apply prompt で fallback message を上書きしない

approval gate が有効でも、token budget が小さく preview item が残らない場合は、`Apply these Fast Recall memories...` のような prompt を出すべきではない。

そのため、smoke script 側で固定 `user_visible_message` を渡すのを避け、planner に message priority を集約する方針にした。

## Validation

PR #50 / #51 の範囲では、主に以下を確認した。

    python -m compileall relaykv tests scripts
    python -m pytest -q tests/test_relaymem_fast_recall.py
    python -m pytest -q tests/test_relaymem_prompt_preview.py
    python -m pytest -q tests/test_relaymem_context_assembly_smoke.py
    python -m pytest -q tests/test_relaystack_dry_run.py
    python scripts/run_relaymem_fast_recall_smoke.py --output /tmp/relaymem_fast_recall_smoke.json
    python -m json.tool /tmp/relaymem_fast_recall_smoke.json >/dev/null
    python scripts/run_relaymem_prompt_preview_smoke.py --output /tmp/relaymem_prompt_preview_smoke.json
    python -m json.tool /tmp/relaymem_prompt_preview_smoke.json >/dev/null

追加で、prompt preview smoke では以下のような edge case も確認対象にした。

    python scripts/run_relaymem_prompt_preview_smoke.py --no-approval-required --output /tmp/relaymem_prompt_preview_no_approval.json
    python scripts/run_relaymem_prompt_preview_smoke.py --token-budget 1 --output /tmp/relaymem_prompt_preview_tight_budget.json

## Current state

RelayMEM 側は、次の最小 chain まで到達した。

    RelayMEM records
      -> Fast Recall retrieval results
      -> Context assembly plan
      -> Prompt preview plan
      -> User-gated fallback planning artifact

まだ実施していないこと:

- model loading
- GPU path
- embedding / vector DB
- external RAG backend
- runtime connection
- attention / KV pool connection
- scheduler change
- actual UI / user interaction

## Design notes

今回の実装で、RelayStack における RelayMEM の役割がより明確になった。

RelayMEM は RAG の置換そのものではなく、以下を統合する memory/context assembly layer として扱う。

- project/profile memory
- episodic memory
- structured facts
- summaries
- future external recall
- future KV checkpoint metadata
- user-gated prompt insertion

RelayKV は引き続き post-prefill / decode-time KV working set control を担う。RelayMEM はその前段で active context に何を入れるか、また user approval / fallback をどう扱うかを計画する。

## Next step

次の自然な step は Phase 7 として、Prompt Preview plan を RelayStack dry-run planning artifact に接続すること。

候補:

1. `run_relaystack_dry_run.py` に RelayMEM prompt preview summary を統合する。
2. RelayStack artifact に `relaymem_preview_plan` / `memory_context_plan` を追加する。
3. VRAM reservation / RelayKV readiness / User-Gated Fallback と同じ planning artifact 上で見られるようにする。
4. ただし、引き続き model / GPU / runtime / attention / KV / scheduler には接続しない。

## Commit / branch hygiene

PR #50 / #51 は Phase 6 / 6.5 の区切りとして十分に大きい単位でまとまった。

次PRでは小さすぎる commit 分割を避け、`RelayMEM Prompt Preview -> RelayStack dry-run artifact integration` のような phase-level PR として扱うのがよい。
