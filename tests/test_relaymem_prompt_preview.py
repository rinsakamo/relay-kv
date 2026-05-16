import subprocess
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMBackendKind,
    RelayMEMEpisodeKind,
    RelayMEMEpisodeRecord,
    RelayMEMProfileKind,
    RelayMEMProfileRecord,
    RelayMEMRetrievalMode,
    RelayMEMStructuredRecord,
    RelayMEMSummaryKind,
    RelayMEMSummaryRecord,
    build_relaymem_prompt_preview_plan,
    search_relaymem_fast_recall,
)


def make_records() -> list[object]:
    return [
        RelayMEMProfileRecord(
            profile_id="profile:project",
            profile_kind=RelayMEMProfileKind.PROJECT,
            display_name="RelayStack project",
            facts={"definition": "RelayStack coordinates RelayMEM RelayKV VRAM fallback"},
            preferences={"workflow": "no GPU smoke tests first"},
            relationships={},
            importance=0.9,
            source_refs=["doc://profile"],
            updated_at="2026-05-15",
        ),
        RelayMEMEpisodeRecord(
            episode_id="episode:design-pr",
            episode_kind=RelayMEMEpisodeKind.PROJECT_EVENT,
            title="Fixed-budget design PR",
            episode_summary=(
                "RelayKV was documented as fixed VRAM working set control and "
                "RelayMEM prompt preview should stay planning-only in Phase 6.5."
            ),
            participants=["user", "assistant"],
            key_events=["merged design PR", "selected Fast Recall"],
            source_refs=["pr://49"],
            importance=0.8,
            ended_at="2026-05-15",
        ),
        RelayMEMSummaryRecord(
            summary_id="summary:vtuber",
            summary_kind=RelayMEMSummaryKind.LONG_TERM_SUMMARY,
            text=(
                "Open-LLM-VTuber requires VRAM planning for model TTS ASR avatar and KV. "
                "Fast Recall should feed a safe prompt preview before deeper recall."
            ),
            covered_source_refs=["doc://vtuber"],
            token_estimate=24,
            importance=0.85,
            updated_at="2026-05-15",
        ),
        RelayMEMStructuredRecord(
            record_id="structured:phase",
            namespace="relay.phase",
            key="next",
            value="Phase 6.5 implements RelayMEM prompt preview fallback planning.",
            attributes={"kind": "phase_plan"},
            source_refs=["doc://current_status"],
            confidence=0.95,
            updated_at="2026-05-15",
        ),
    ]


def make_fast_recall_results():
    return search_relaymem_fast_recall(
        "RelayStack VRAM Fast Recall phase preview",
        make_records(),
    )


def test_fast_recall_results_feed_prompt_preview_plan() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        retrieval_mode=RelayMEMRetrievalMode.FAST_RECALL,
        token_budget=96,
    )

    assert [item.memory_id for item in plan.preview_items]
    assert plan.backend_kind is RelayMEMBackendKind.BM25
    assert plan.total_estimated_tokens <= 96


def test_prompt_preview_truncation_does_not_mutate_source_text() -> None:
    results = make_fast_recall_results()
    original_text = results[0].text

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        max_preview_chars=40,
    )

    assert plan.preview_items[0].preview_text.endswith("...")
    assert len(plan.preview_items[0].preview_text) <= 40
    assert results[0].text == original_text


def test_approval_required_blocks_can_apply_without_user_approval() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        approval_required=True,
        approval_reason="user confirmation required",
    )

    assert plan.approval_required is True
    assert plan.can_apply_without_user_approval is False


def test_deep_recall_approval_message_mentions_deeper_memory_recall() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        approval_required=True,
        approval_reason="user confirmation required",
    )

    assert "deep" in plan.user_visible_message.lower()
    assert "fast recall prepared" not in plan.user_visible_message.lower()


def test_no_approval_allows_apply_when_results_exist_and_no_blocking_fallback() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        approval_required=False,
        token_budget=200,
    )

    assert plan.preview_items
    assert plan.dropped_memory_ids == []
    assert plan.fallback_reason is None
    assert plan.can_apply_without_user_approval is True


def test_deep_recall_no_approval_message_mentions_deeper_memory_recall() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        approval_required=False,
        token_budget=200,
    )

    assert plan.can_apply_without_user_approval is True
    assert "deep" in plan.user_visible_message.lower()
    assert "fast recall prepared" not in plan.user_visible_message.lower()


def test_explicit_fallback_reason_blocks_auto_apply_without_budget_drop() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        approval_required=False,
        token_budget=200,
        fallback_reason="caller_requested_manual_review",
    )

    assert plan.dropped_memory_ids == []
    assert plan.fallback_reason == "caller_requested_manual_review"
    assert plan.can_apply_without_user_approval is False


def test_token_budget_records_dropped_memory_ids_and_fallback_reason() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        token_budget=30,
    )

    assert plan.dropped_memory_ids
    assert plan.fallback_reason == "token_budget_exceeded"
    assert plan.can_apply_without_user_approval is False
    assert "fast recall" in plan.user_visible_message.lower()


def test_drop_all_results_does_not_report_no_memory_found() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        token_budget=1,
    )

    assert plan.preview_items == []
    assert plan.dropped_memory_ids
    assert "found no memory" not in plan.user_visible_message.lower()


def test_drop_all_results_mentions_budget_or_fallback() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        token_budget=1,
    )

    assert plan.preview_items == []
    assert plan.fallback_reason == "token_budget_exceeded"
    assert (
        "budget" in plan.user_visible_message.lower()
        or "fallback" in plan.user_visible_message.lower()
    )
    assert "fast recall" in plan.user_visible_message.lower()


def test_empty_retrieval_results_produce_empty_preview_plan() -> None:
    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=[],
        approval_required=False,
    )

    assert plan.preview_items == []
    assert plan.dropped_memory_ids == []
    assert plan.fallback_reason == "no_retrieval_results"
    assert plan.can_apply_without_user_approval is False
    assert plan.user_visible_message == "Fast Recall found no memory to preview for this query."


def test_deep_recall_empty_preview_uses_deep_recall_wording() -> None:
    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=[],
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        approval_required=False,
    )

    assert plan.fallback_reason == "no_retrieval_results"
    assert "deep recall" in plan.user_visible_message.lower()
    assert "fast recall" not in plan.user_visible_message.lower()


def test_deep_recall_tight_budget_uses_deep_recall_budget_wording() -> None:
    results = make_fast_recall_results()

    plan = build_relaymem_prompt_preview_plan(
        query="RelayStack VRAM Fast Recall phase preview",
        retrieval_results=results,
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        approval_required=True,
        token_budget=1,
    )

    assert plan.fallback_reason == "token_budget_exceeded"
    assert plan.preview_items == []
    assert "deep recall" in plan.user_visible_message.lower()
    assert "budget" in plan.user_visible_message.lower()
    assert "fast recall" not in plan.user_visible_message.lower()


def test_import_from_relaykv_with_prompt_preview_stays_torch_free() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import relaykv; "
            "assert 'torch' not in sys.modules; "
            "assert callable(relaykv.build_relaymem_prompt_preview_plan); "
            "print('ok')"
        ),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_prompt_preview_smoke_no_approval_omits_approval_ux(tmp_path: Path) -> None:
    output_path = tmp_path / "relaymem_prompt_preview_no_approval.json"
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaymem_prompt_preview_smoke.py",
            "--no-approval-required",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    backend_capabilities = payload["metadata"]["backend_capabilities"]
    plan = payload["preview_plan"]
    assert backend_capabilities["backend_type"] == "fast_recall"
    assert backend_capabilities["runs_on_cpu"] is True
    assert backend_capabilities["requires_gpu"] is False
    assert backend_capabilities["uses_vram"] is False
    assert plan["approval_required"] is False
    assert plan["can_apply_without_user_approval"] is True
    assert not plan["approval_reason"]
    assert plan["fallback_if_denied"] is None
    assert "Apply these Fast Recall memories" not in plan["user_visible_message"]


def test_prompt_preview_smoke_tight_budget_uses_budget_message(tmp_path: Path) -> None:
    output_path = tmp_path / "relaymem_prompt_preview_tight_budget.json"
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaymem_prompt_preview_smoke.py",
            "--token-budget",
            "1",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    backend_capabilities = payload["metadata"]["backend_capabilities"]
    plan = payload["preview_plan"]
    assert backend_capabilities["backend_type"] == "fast_recall"
    assert backend_capabilities["runs_on_cpu"] is True
    assert backend_capabilities["requires_gpu"] is False
    assert backend_capabilities["uses_vram"] is False
    assert plan["approval_required"] is True
    assert plan["can_apply_without_user_approval"] is False
    assert plan["fallback_reason"] == "token_budget_exceeded"
    assert plan["preview_items"] == []
    assert "Apply these Fast Recall memories" not in plan["user_visible_message"]
    assert (
        "budget" in plan["user_visible_message"].lower()
        or "fallback" in plan["user_visible_message"].lower()
    )
