import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMBackendKind,
    RelayMEMEpisodeKind,
    RelayMEMEpisodeRecord,
    RelayMEMMemorySource,
    RelayMEMProfileKind,
    RelayMEMProfileRecord,
    RelayMEMRecordStatus,
    RelayMEMStructuredRecord,
    RelayMEMSummaryKind,
    RelayMEMSummaryRecord,
    build_relaymem_context_assembly_plan,
    relaymem_record_to_fast_recall_candidate,
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
            episode_summary="RelayKV was documented as fixed VRAM working set control.",
            participants=["user", "assistant"],
            key_events=["merged design PR", "selected Fast Recall"],
            source_refs=["pr://49"],
            importance=0.8,
            ended_at="2026-05-15",
        ),
        RelayMEMSummaryRecord(
            summary_id="summary:vtuber",
            summary_kind=RelayMEMSummaryKind.LONG_TERM_SUMMARY,
            text="Open-LLM-VTuber requires VRAM planning for model TTS ASR avatar and KV.",
            covered_source_refs=["doc://vtuber"],
            token_estimate=18,
            importance=0.85,
            updated_at="2026-05-15",
        ),
        RelayMEMStructuredRecord(
            record_id="structured:phase",
            namespace="relay.phase",
            key="next",
            value="Phase 6 implements RelayMEM Fast Recall backend.",
            attributes={"kind": "phase_plan"},
            source_refs=["doc://current_status"],
            confidence=0.95,
            updated_at="2026-05-15",
        ),
    ]


def test_fast_recall_returns_relaymem_retrieval_results_in_deterministic_order() -> None:
    results = search_relaymem_fast_recall(
        "RelayStack VRAM Fast Recall phase",
        make_records(),
    )

    assert [result.memory_id for result in results] == [
        "profile:project",
        "structured:phase",
        "summary:vtuber",
        "episode:design-pr",
    ]
    assert all(result.retrieval_backend is RelayMEMBackendKind.BM25 for result in results)
    assert results[0].memory_source is RelayMEMMemorySource.PROFILE
    assert results[0].confidence is not None
    assert results[0].evidence[0].startswith("matched_tokens=")


def test_fast_recall_respects_max_results() -> None:
    results = search_relaymem_fast_recall(
        "RelayStack VRAM Fast Recall phase",
        make_records(),
        max_results=2,
    )

    assert [result.memory_id for result in results] == [
        "profile:project",
        "structured:phase",
    ]


def test_fast_recall_filters_inactive_records_by_default() -> None:
    records = make_records()
    records.append(
        RelayMEMSummaryRecord(
            summary_id="summary:archived",
            summary_kind=RelayMEMSummaryKind.SESSION_SUMMARY,
            text="Archived RelayStack VRAM plan",
            covered_source_refs=["old://plan"],
            token_estimate=10,
            importance=1.0,
            status=RelayMEMRecordStatus.ARCHIVED,
        )
    )

    results = search_relaymem_fast_recall("RelayStack VRAM plan", records)

    assert "summary:archived" not in [result.memory_id for result in results]


def test_fast_recall_can_include_inactive_records_when_requested() -> None:
    records = make_records()
    records.append(
        RelayMEMSummaryRecord(
            summary_id="summary:archived",
            summary_kind=RelayMEMSummaryKind.SESSION_SUMMARY,
            text="Archived RelayStack VRAM plan",
            covered_source_refs=["old://plan"],
            token_estimate=10,
            importance=1.0,
            status=RelayMEMRecordStatus.ARCHIVED,
        )
    )

    results = search_relaymem_fast_recall(
        "RelayStack VRAM plan",
        records,
        include_inactive=True,
    )

    assert "summary:archived" in [result.memory_id for result in results]


def test_fast_recall_results_feed_context_assembly() -> None:
    results = search_relaymem_fast_recall(
        "RelayStack VRAM Fast Recall phase",
        make_records(),
        max_results=3,
    )
    plan = build_relaymem_context_assembly_plan(
        query="RelayStack VRAM Fast Recall phase",
        retrieval_mode=__import__("relaykv").RelayMEMRetrievalMode.FAST_RECALL,
        backend_kind=RelayMEMBackendKind.BM25,
        retrieval_results=results,
        token_budget=64,
    )

    assert [item.memory_id for item in plan.selected_items]
    assert plan.total_estimated_tokens <= 64
    assert plan.summary()["backend_kind"] == "bm25"


def test_record_to_fast_recall_candidate_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="unsupported RelayMEM record type"):
        relaymem_record_to_fast_recall_candidate(object())


def test_fast_recall_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query"):
        search_relaymem_fast_recall("", make_records())


def test_fast_recall_rejects_negative_max_results() -> None:
    with pytest.raises(ValueError, match="max_results"):
        search_relaymem_fast_recall("RelayStack", make_records(), max_results=-1)


def test_import_from_relaykv_with_fast_recall_stays_torch_free() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import relaykv; "
            "assert 'torch' not in sys.modules; "
            "assert callable(relaykv.search_relaymem_fast_recall); "
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
