import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMEpisodeKind,
    RelayMEMEpisodeRecord,
    RelayMEMKVCheckpointMetadata,
    RelayMEMProfileKind,
    RelayMEMProfileRecord,
    RelayMEMRecordStatus,
    RelayMEMStructuredRecord,
    RelayMEMSummaryKind,
    RelayMEMSummaryRecord,
    summarize_relaymem_records,
)


def test_relaymem_record_summaries_are_json_friendly() -> None:
    records = [
        RelayMEMProfileRecord(
            profile_id="profile-user-1",
            profile_kind=RelayMEMProfileKind.USER,
            display_name="User One",
            facts={"locale": "ja-JP"},
            preferences={"tone": "concise"},
            relationships={"assistant": "trusted"},
            importance=0.9,
            source_refs=["profile://user/1"],
            updated_at="2026-05-14T10:00:00Z",
        ),
        RelayMEMEpisodeRecord(
            episode_id="episode-1",
            episode_kind=RelayMEMEpisodeKind.CONVERSATION,
            title="Long memory planning",
            episode_summary="Planned RelayMEM schema expansion.",
            participants=["user", "assistant"],
            key_events=["defined Profile Memory", "agreed schema-first rollout"],
            source_refs=["episode://conversation/1"],
            importance=0.8,
        ),
        RelayMEMSummaryRecord(
            summary_id="summary-1",
            summary_kind=RelayMEMSummaryKind.ROLLING_SUMMARY,
            text="Rolling summary of the recent RelayMEM design decisions.",
            covered_source_refs=["episode://conversation/1"],
            token_estimate=42,
            importance=0.7,
        ),
        RelayMEMStructuredRecord(
            record_id="structured-1",
            namespace="project.memory",
            key="target_demo",
            value="Open-LLM-VTuber",
            attributes={"language": "ja"},
            source_refs=["doc://target"],
            confidence=0.95,
        ),
        RelayMEMKVCheckpointMetadata(
            checkpoint_id="kvcp-1",
            label="Character intro prefix",
            model_id="llm-jp-4-8b",
            tokenizer_id="llm-jp-4-tokenizer",
            token_span=(0, 512),
            source_refs=["checkpoint://prefix/1"],
            storage_uri="file:///tmp/kvcp-1",
            rope_status="native",
            precision_level="fp16",
        ),
    ]

    for record in records:
        summary = record.summary()
        assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: RelayMEMProfileRecord(
                profile_id="",
                profile_kind=RelayMEMProfileKind.USER,
                display_name="User One",
                facts={},
                preferences={},
                relationships={},
                importance=0.5,
            ),
            "profile_id",
        ),
        (
            lambda: RelayMEMEpisodeRecord(
                episode_id="",
                episode_kind=RelayMEMEpisodeKind.CONVERSATION,
                title="Session",
                episode_summary="Summary",
                participants=[],
                key_events=[],
                source_refs=[],
                importance=0.5,
            ),
            "episode_id",
        ),
        (
            lambda: RelayMEMSummaryRecord(
                summary_id="",
                summary_kind=RelayMEMSummaryKind.SESSION_SUMMARY,
                text="Summary",
                covered_source_refs=[],
                token_estimate=1,
                importance=0.5,
            ),
            "summary_id",
        ),
        (
            lambda: RelayMEMStructuredRecord(
                record_id="",
                namespace="ns",
                key="k",
                value="v",
                attributes={},
                source_refs=[],
            ),
            "record_id",
        ),
        (
            lambda: RelayMEMKVCheckpointMetadata(
                checkpoint_id="",
                label="Label",
                model_id="model",
                tokenizer_id="tokenizer",
                token_span=(0, 8),
                source_refs=[],
            ),
            "checkpoint_id",
        ),
    ],
)
def test_relaymem_records_reject_empty_ids(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_relaymem_records_reject_invalid_importance_and_confidence() -> None:
    with pytest.raises(ValueError, match="importance"):
        RelayMEMProfileRecord(
            profile_id="profile-1",
            profile_kind=RelayMEMProfileKind.CHARACTER,
            display_name="Character One",
            facts={},
            preferences={},
            relationships={},
            importance=1.5,
        )

    with pytest.raises(ValueError, match="confidence"):
        RelayMEMStructuredRecord(
            record_id="structured-1",
            namespace="ns",
            key="k",
            value="v",
            attributes={},
            source_refs=[],
            confidence=-0.1,
        )


@pytest.mark.parametrize(
    "token_span",
    [
        (0,),
        ("0", 8),
        (-1, 8),
        (8, 8),
    ],
)
def test_kv_checkpoint_metadata_rejects_invalid_token_span(token_span) -> None:
    with pytest.raises(ValueError, match="token_span"):
        RelayMEMKVCheckpointMetadata(
            checkpoint_id="kvcp-1",
            label="Label",
            model_id="model",
            tokenizer_id="tokenizer",
            token_span=token_span,  # type: ignore[arg-type]
            source_refs=[],
        )


def test_summarize_relaymem_records_counts_types_statuses_and_kinds() -> None:
    records = [
        RelayMEMProfileRecord(
            profile_id="profile-1",
            profile_kind=RelayMEMProfileKind.USER,
            display_name="User One",
            facts={},
            preferences={},
            relationships={},
            importance=0.9,
            status=RelayMEMRecordStatus.ACTIVE,
        ),
        RelayMEMProfileRecord(
            profile_id="profile-2",
            profile_kind=RelayMEMProfileKind.PROJECT,
            display_name="Project One",
            facts={},
            preferences={},
            relationships={},
            importance=0.8,
            status=RelayMEMRecordStatus.STALE,
        ),
        RelayMEMEpisodeRecord(
            episode_id="episode-1",
            episode_kind=RelayMEMEpisodeKind.STREAM,
            title="Stream 1",
            episode_summary="Summary",
            participants=[],
            key_events=[],
            source_refs=[],
            importance=0.6,
            status=RelayMEMRecordStatus.ARCHIVED,
        ),
        RelayMEMSummaryRecord(
            summary_id="summary-1",
            summary_kind=RelayMEMSummaryKind.LONG_TERM_SUMMARY,
            text="Long-term summary",
            covered_source_refs=[],
            token_estimate=20,
            importance=0.7,
            status=RelayMEMRecordStatus.ACTIVE,
        ),
        RelayMEMStructuredRecord(
            record_id="structured-1",
            namespace="ns",
            key="k",
            value="v",
            attributes={},
            source_refs=[],
            confidence=0.8,
            status=RelayMEMRecordStatus.DELETED,
        ),
        RelayMEMKVCheckpointMetadata(
            checkpoint_id="kvcp-1",
            label="Prefix",
            model_id="model",
            tokenizer_id="tokenizer",
            token_span=(0, 64),
            source_refs=[],
            status=RelayMEMRecordStatus.ACTIVE,
        ),
    ]

    summary = summarize_relaymem_records(records)

    assert summary["total_records"] == 6
    assert summary["unknown_records"] == 0
    assert summary["profile_records"] == 2
    assert summary["episode_records"] == 1
    assert summary["summary_records"] == 1
    assert summary["structured_records"] == 1
    assert summary["kv_checkpoint_metadata_records"] == 1
    assert summary["status_counts"] == {
        "active": 3,
        "stale": 1,
        "archived": 1,
        "deleted": 1,
    }
    assert summary["profile_kind_counts"] == {
        "user": 1,
        "project": 1,
    }
    assert summary["episode_kind_counts"] == {
        "stream": 1,
    }
    assert summary["summary_kind_counts"] == {
        "long_term_summary": 1,
    }


def test_summarize_relaymem_records_tolerates_unknown_objects() -> None:
    summary = summarize_relaymem_records(
        [
            RelayMEMSummaryRecord(
                summary_id="summary-1",
                summary_kind=RelayMEMSummaryKind.SESSION_SUMMARY,
                text="Session summary",
                covered_source_refs=[],
                token_estimate=12,
                importance=0.5,
            ),
            object(),
        ]
    )

    assert summary["total_records"] == 2
    assert summary["summary_records"] == 1
    assert summary["unknown_records"] == 1


def test_relaymem_records_import_from_relaykv_stays_torch_free() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import relaykv; "
            "assert 'torch' not in sys.modules; "
            "assert relaykv.RelayMEMProfileKind.USER.value == 'user'; "
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
