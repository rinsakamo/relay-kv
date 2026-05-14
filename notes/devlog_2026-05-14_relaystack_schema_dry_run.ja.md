# Devlog: RelayStack schema / dry-run consolidation

- Date: 2026-05-14 JST
- Repository: `rinsakamo/relay-kv`
- Scope: RelayMEM / RelayStack schema, User-Gated Fallback, VRAM reservation, and combined dry-run planning
- Status: PR #41〜#47 merged

## Date confirmation

This devlog is dated **2026-05-14 JST**. The milestone covers the merged PR sequence from PR #41 through PR #47.

## Summary

This milestone consolidates RelayKV from a KV-cache experiment into a broader **RelayStack** architecture:

- **RelayMEM**: max-context-external memory and context assembly layer
- **RelayKV**: active-context KV routing and VRAM-aware working-set layer
- **Runtime policy**: User-Gated Fallback, VRAM reservation, and planning dry-run

The key outcome is that RelayMEM, User-Gated Fallback, VRAM reservation, and RelayKV memory-pressure planning can now be represented together in a single **no-model / no-GPU RelayStack dry-run JSON**.

## Merged PRs

### PR #41: Document RelayStack target and memory architecture

Purpose:

- Reframe RelayKV as a VRAM-aware active-context KV routing layer and tiered KV memory manager.
- Introduce documentation-level RelayMEM and RelayStack concepts.
- Document Open-LLM-VTuber as a low-VRAM Japanese AI character / AI Vtuber integration target.
- Document User-Gated Fallback, pluggable RAG, Deep Recall, and NeocorRAG-style evidence-chain retrieval as design concepts.

Main files:

- `README.md`
- `docs/README.md`
- `docs/current_status.md`
- `docs/experimental_findings.md`
- `docs/evaluation_targets.md`
- `docs/relaymem.md`
- `docs/open_llm_vtuber_target.md`
- `docs/user_gated_fallback.md`
- `docs/max_context_external_memory.md`
- `notes/devlog_2026-05-13_relaystack_target_consolidation.ja.md`

Boundary:

- Documentation-only.
- No runtime behavior change.

### PR #42: Add RelayMEM schema v0

Purpose:

- Add stdlib-only RelayMEM schema objects for retrieval results and context assembly plans.
- Keep concrete RAG implementations pluggable and out of scope.

Added / updated:

- `relaykv/relaymem.py`
- `relaykv/__init__.py`
- `tests/test_relaymem_schema.py`
- `docs/relaymem.md`

Key objects:

- `RelayMEMRetrievalMode`
- `RelayMEMBackendKind`
- `RelayMEMMemorySource`
- `RelayMEMRetrievalResult`
- `RelayMEMContextItem`
- `RelayMEMContextAssemblyPlan`
- `build_relaymem_context_assembly_plan(...)`

Validation:

- `python -m compileall relaykv tests`
- `pytest -q tests/test_relaymem_schema.py`
- Observed: `7 passed`

Boundary:

- Schema/log-only.
- No RAG runtime.
- No NeocorRAG dependency.
- No model loading.
- No attention/KV/scheduler changes.

### PR #43: Add RelayMEM context assembly smoke

Purpose:

- Add a repository-native offline smoke path showing how synthetic RelayMEM retrieval results become a context assembly plan.
- Prove Profile Memory, Episode Memory, Summary Memory, RAG Chunk, and Evidence Chain examples can be represented without backend dependencies.

Added:

- `scripts/run_relaymem_context_assembly_smoke.py`
- `tests/test_relaymem_context_assembly_smoke.py`

Important review fix:

- The default token budget was raised so RAG Chunk and Evidence Chain examples are selected in `plan.selected_items`, not merely present in raw retrieval inputs.
- A separate low-budget test keeps `dropped_memory_ids` coverage.

Validation:

- `python -m compileall relaykv scripts tests`
- `pytest -q tests/test_relaymem_context_assembly_smoke.py`
- `python scripts/run_relaymem_context_assembly_smoke.py --output /tmp/relaymem_context_assembly_smoke.json`
- `python -m json.tool /tmp/relaymem_context_assembly_smoke.json >/dev/null`

Boundary:

- Offline smoke only.
- No model loading.
- No concrete RAG backend.

### PR #44: Add RelayMEM record schema v1

Purpose:

- Add memory-side record schemas that RelayMEM can later retrieve from.

Added / updated:

- `relaykv/relaymem_records.py`
- `relaykv/__init__.py`
- `tests/test_relaymem_records.py`
- `docs/relaymem.md`

Key record types:

- `RelayMEMProfileRecord`
- `RelayMEMEpisodeRecord`
- `RelayMEMSummaryRecord`
- `RelayMEMStructuredRecord`
- `RelayMEMKVCheckpointMetadata`
- `summarize_relaymem_records(...)`

Design decision:

- Use **Profile Memory** as the generic concept, not viewer-specific profile memory.
- AI Vtuber viewer profiles are treated as one use case of Profile Memory.
- `RelayMEMEpisodeRecord` uses `episode_summary` as the text field so `summary()` remains a consistent method across schema objects.

Validation:

- `python -m compileall relaykv tests`
- `pytest -q tests/test_relaymem_records.py`
- Observed: `14 passed`

Boundary:

- Schema/log-only.
- No storage, DB, embedding, vector search, RAG backend, model loading, or runtime retrieval.

### PR #45: Add User-Gated Fallback schema fields

Purpose:

- Move User-Gated Fallback from docs-only into schema/log representation for RelayKV and RelayMEM.

Updated:

- `relaykv/routing_decision.py`
- `relaykv/relaymem.py`
- `tests/test_routing_decision.py`
- `tests/test_relaymem_schema.py`
- `docs/user_gated_fallback.md`

RelayKV additions:

- `ExecutionMode.PROPOSE_FALLBACK`
- `ExecutionMode.WAIT_USER_APPROVAL`
- `ExecutionMode.APPLY_RAM_BACKED`
- `ExecutionMode.FALLBACK_FULLKV_RAM`
- `ExecutionMode.FALLBACK_FULLKV_TIERED`
- `ExecutionMode.FALLBACK_RECENT_ANCHOR`

`RelayKVDecision` fields:

- `approval_required`
- `approval_reason`
- `proposed_fallback_mode`
- `user_visible_message`
- `fallback_if_denied`

RelayMEM additions:

- `RelayMEMContextAssemblyPlan.proposed_retrieval_mode`
- `RelayMEMContextAssemblyPlan.fallback_if_denied`
- `build_relaymem_context_assembly_plan(...)` support for those fields

Validation:

- `python -m compileall relaykv tests scripts`
- `pytest -q tests/test_routing_decision.py tests/test_relaymem_schema.py tests/test_relaymem_context_assembly_smoke.py`
- Observed: `15 passed`

Boundary:

- Schema/log-only.
- No runtime approval UI.
- No fallback execution path.

### PR #46: Add VRAM reservation schema

Purpose:

- Add upstream residual VRAM accounting for local multimodal stacks.
- Reserve VRAM for model weights, TTS, ASR, avatar, safety margin, and other fixed costs before estimating RelayKV working KV budget.

Added / updated:

- `relaykv/vram_reservation.py`
- `relaykv/__init__.py`
- `scripts/run_vram_reservation_smoke.py`
- `tests/test_vram_reservation.py`
- `tests/test_vram_reservation_smoke.py`
- `docs/open_llm_vtuber_target.md`

Key objects:

- `RelayKVVramReservationStatus`
- `RelayKVVramReservation`
- `RelayKVVramBudgetDecision`
- `build_vram_budget_decision(...)` inside `relaykv/vram_reservation.py`

Top-level export compatibility:

- Existing `relaykv.vram_budget` already exported `RelayKVVramBudgetDecision` and `build_vram_budget_decision`.
- To avoid breaking public API, reservation-side exports use aliases:
  - `RelayKVVramReservationBudgetDecision`
  - `build_vram_reservation_budget_decision`

Validation:

- `python -m compileall relaykv tests scripts`
- `pytest -q tests/test_vram_reservation.py`
- `pytest -q tests/test_vram_reservation_smoke.py`
- `python scripts/run_vram_reservation_smoke.py --output /tmp/relaykv_vram_reservation_smoke.json`
- `python -m json.tool /tmp/relaykv_vram_reservation_smoke.json >/dev/null`
- Observed:
  - `tests/test_vram_reservation.py`: `16 passed`
  - `tests/test_vram_reservation_smoke.py`: `2 passed`

Boundary:

- No GPU inspection.
- No `nvidia-smi`.
- No model loading.
- No runtime scheduler/attention/KV changes.

### PR #47: Add RelayStack dry-run smoke

Purpose:

- Add the first combined RelayStack planning artifact.
- Combine RelayMEM, User-Gated Fallback, VRAM reservation, lightweight memory-pressure planning, and runtime policy into one no-model / no-GPU JSON.

Added / updated:

- `scripts/run_relaystack_dry_run.py`
- `tests/test_relaystack_dry_run.py`
- `docs/open_llm_vtuber_target.md`

Top-level JSON sections:

- `metadata`
- `runtime_policy`
- `relaymem`
- `relaykv`
- `user_gated_fallback`
- `summary`

Included planning layers:

- Synthetic RelayMEM retrieval results
- Profile Memory / Episode Memory / Summary Memory / RAG Chunk / Evidence Chain
- RelayMEM context assembly plan
- User-Gated Fallback fields
- VRAM reservation decision
- Lightweight RelayKV memory-pressure planning via torch-free helpers

Important review fix:

- RelayKV routing is blocked when the VRAM reservation status is not `OK`.
- If VRAM reservation reports `no_kv_budget` or `over_budget`, the dry-run emits a blocked/skip state instead of reporting `relaykv_routed_ready`.
- This prevents contradictory planning JSON such as “no usable KV budget, but RelayKV routing ready.”

Validation:

- `python -m compileall relaykv tests scripts`
- `pytest -q tests/test_relaystack_dry_run.py`
- `python scripts/run_relaystack_dry_run.py --output /tmp/relaystack_dry_run.json`
- `python -m json.tool /tmp/relaystack_dry_run.json >/dev/null`
- `pytest -q tests/test_relaymem_context_assembly_smoke.py tests/test_vram_reservation_smoke.py`
- Observed:
  - `tests/test_relaystack_dry_run.py`: `3 passed`
  - nearby smoke tests: `5 passed`

Boundary:

- Dry-run / schema / log-only.
- No model loading.
- No GPU inspection.
- No retrieval backend.
- No runtime approval flow.
- No attention/KV materialization/scheduler changes.

## Architecture after this milestone

```text
RelayStack
├─ RelayMEM
│  ├─ Retrieval result schema
│  ├─ Context assembly plan
│  ├─ Profile / Episode / Summary / Structured / KV Checkpoint records
│  └─ Pluggable backend representation
│
├─ RelayKV
│  ├─ Memory-pressure decision helpers
│  ├─ VRAM reservation accounting
│  ├─ Working KV budget planning
│  └─ User-Gated Fallback fields
│
└─ Runtime policy
   ├─ live_low_latency mode representation
   ├─ approval gate fields
   ├─ fallback-if-denied fields
   └─ no-model/no-GPU planning JSON
```

## Current implementation boundary

This milestone intentionally remains conservative.

Not implemented:

- Real RAG backend
- NeocorRAG dependency
- Embeddings or vector DB
- Storage layer
- Runtime approval UI
- Actual memory recall execution
- GPU inspection
- `nvidia-smi`
- Model loading
- KV materialization
- Attention backend connection
- Scheduler changes
- Runtime fallback execution

Implemented:

- Schema/log-only representation
- Offline smoke scripts
- JSON-serializable plans
- Tests for budget, fallback, and context assembly behavior
- Combined RelayStack planning artifact

## Why this matters

Before this milestone, RelayKV had memory-pressure planning and documentation for RelayMEM/RelayStack, but the pieces were not yet connected.

After this milestone:

- RelayMEM can represent memory records, retrieval results, and context assembly plans.
- RelayKV can represent fallback approval and residual VRAM reservation decisions.
- RelayStack can emit a single planning JSON for a local AI Vtuber-style scenario.
- The architecture now has a no-model/no-GPU path that can be used for future adapter work and policy iteration.

## Recommended next step

Next recommended phase:

```text
Phase 6: Fast Recall backend prototype
```

Goal:

- Add a minimal stdlib-only Fast Recall backend for RelayMEM.
- No external DB.
- No embeddings.
- Use keyword/token-overlap or BM25-like scoring.
- Return `RelayMEMRetrievalResult` objects.
- Feed those results into the existing context assembly and RelayStack dry-run path.

Suggested branch:

```bash
git switch -c relaymem-fast-recall-backend-v1
```

Suggested validation direction:

```bash
python -m compileall relaykv tests scripts
pytest -q tests/test_relaymem_fast_recall.py
pytest -q tests/test_relaystack_dry_run.py
```

## Commit / push command for adding this devlog

```bash
git switch main
git pull origin main

git switch -c relaystack-devlog-20260514

git add notes/devlog_2026-05-14_relaystack_schema_dry_run.ja.md

git commit -m "Add devlog for RelayStack schema dry-run milestone"

git push -u origin relaystack-devlog-20260514
```
