from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RelayMEMProfileKind(str, Enum):
    USER = "user"
    CHARACTER = "character"
    ORGANIZATION = "organization"
    PROJECT = "project"
    RELATIONSHIP = "relationship"


class RelayMEMEpisodeKind(str, Enum):
    CONVERSATION = "conversation"
    STREAM = "stream"
    WRITING_SESSION = "writing_session"
    TASK_SESSION = "task_session"
    PROJECT_EVENT = "project_event"


class RelayMEMSummaryKind(str, Enum):
    SESSION_SUMMARY = "session_summary"
    ROLLING_SUMMARY = "rolling_summary"
    LONG_TERM_SUMMARY = "long_term_summary"


class RelayMEMRecordStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"
    DELETED = "deleted"


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _require_unit_interval(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _require_dict_of_str(value: dict[str, str], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dict")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only str keys and values")


def _require_list(value: list[Any], field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")


@dataclass(frozen=True)
class RelayMEMProfileRecord:
    profile_id: str
    profile_kind: RelayMEMProfileKind
    display_name: str
    facts: dict[str, str]
    preferences: dict[str, str]
    relationships: dict[str, str]
    importance: float
    status: RelayMEMRecordStatus = RelayMEMRecordStatus.ACTIVE
    source_refs: list[str] = field(default_factory=list)
    updated_at: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.profile_id, "profile_id")
        _require_non_empty(self.display_name, "display_name")
        _require_unit_interval(self.importance, "importance")
        _require_dict_of_str(self.facts, "facts")
        _require_dict_of_str(self.preferences, "preferences")
        _require_dict_of_str(self.relationships, "relationships")
        _require_list(self.source_refs, "source_refs")

    def summary(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind.value,
            "display_name": self.display_name,
            "facts": dict(self.facts),
            "preferences": dict(self.preferences),
            "relationships": dict(self.relationships),
            "importance": self.importance,
            "status": self.status.value,
            "source_refs": list(self.source_refs),
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class RelayMEMEpisodeRecord:
    episode_id: str
    episode_kind: RelayMEMEpisodeKind
    title: str
    episode_summary: str
    participants: list[str]
    key_events: list[str]
    source_refs: list[str]
    importance: float
    status: RelayMEMRecordStatus = RelayMEMRecordStatus.ACTIVE
    started_at: str | None = None
    ended_at: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.episode_id, "episode_id")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.episode_summary, "episode_summary")
        _require_unit_interval(self.importance, "importance")
        _require_list(self.participants, "participants")
        _require_list(self.key_events, "key_events")
        _require_list(self.source_refs, "source_refs")

    def summary(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "episode_kind": self.episode_kind.value,
            "title": self.title,
            "episode_summary": self.episode_summary,
            "participants": list(self.participants),
            "key_events": list(self.key_events),
            "source_refs": list(self.source_refs),
            "importance": self.importance,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True)
class RelayMEMSummaryRecord:
    summary_id: str
    summary_kind: RelayMEMSummaryKind
    text: str
    covered_source_refs: list[str]
    token_estimate: int
    importance: float
    status: RelayMEMRecordStatus = RelayMEMRecordStatus.ACTIVE
    updated_at: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.summary_id, "summary_id")
        _require_non_empty(self.text, "text")
        _require_list(self.covered_source_refs, "covered_source_refs")
        _require_non_negative(self.token_estimate, "token_estimate")
        _require_unit_interval(self.importance, "importance")

    def summary(self) -> dict:
        return {
            "summary_id": self.summary_id,
            "summary_kind": self.summary_kind.value,
            "text": self.text,
            "covered_source_refs": list(self.covered_source_refs),
            "token_estimate": self.token_estimate,
            "importance": self.importance,
            "status": self.status.value,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class RelayMEMStructuredRecord:
    record_id: str
    namespace: str
    key: str
    value: str
    attributes: dict[str, str]
    source_refs: list[str]
    confidence: float | None = None
    status: RelayMEMRecordStatus = RelayMEMRecordStatus.ACTIVE
    updated_at: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.record_id, "record_id")
        _require_non_empty(self.namespace, "namespace")
        _require_non_empty(self.key, "key")
        _require_non_empty(self.value, "value")
        _require_dict_of_str(self.attributes, "attributes")
        _require_list(self.source_refs, "source_refs")
        _require_unit_interval(self.confidence, "confidence")

    def summary(self) -> dict:
        return {
            "record_id": self.record_id,
            "namespace": self.namespace,
            "key": self.key,
            "value": self.value,
            "attributes": dict(self.attributes),
            "source_refs": list(self.source_refs),
            "confidence": self.confidence,
            "status": self.status.value,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class RelayMEMKVCheckpointMetadata:
    checkpoint_id: str
    label: str
    model_id: str
    tokenizer_id: str
    token_span: tuple[int, int]
    source_refs: list[str]
    storage_uri: str | None = None
    rope_status: str | None = None
    precision_level: str | None = None
    status: RelayMEMRecordStatus = RelayMEMRecordStatus.ACTIVE

    def __post_init__(self) -> None:
        _require_non_empty(self.checkpoint_id, "checkpoint_id")
        _require_non_empty(self.label, "label")
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.tokenizer_id, "tokenizer_id")
        _require_list(self.source_refs, "source_refs")
        if len(self.token_span) != 2:
            raise ValueError("token_span must contain exactly two ints")
        start, end = self.token_span
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("token_span must contain exactly two ints")
        if start < 0:
            raise ValueError("token_span start must be >= 0")
        if end <= start:
            raise ValueError("token_span end must be > start")

    def summary(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "label": self.label,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "token_span": list(self.token_span),
            "source_refs": list(self.source_refs),
            "storage_uri": self.storage_uri,
            "rope_status": self.rope_status,
            "precision_level": self.precision_level,
            "status": self.status.value,
        }


def summarize_relaymem_records(records: list[object]) -> dict:
    summary = {
        "total_records": 0,
        "unknown_records": 0,
        "profile_records": 0,
        "episode_records": 0,
        "summary_records": 0,
        "structured_records": 0,
        "kv_checkpoint_metadata_records": 0,
        "status_counts": {},
        "profile_kind_counts": {},
        "episode_kind_counts": {},
        "summary_kind_counts": {},
    }

    for record in records:
        summary["total_records"] += 1

        if isinstance(record, RelayMEMProfileRecord):
            summary["profile_records"] += 1
            status_key = record.status.value
            kind_key = record.profile_kind.value
            summary["status_counts"][status_key] = (
                summary["status_counts"].get(status_key, 0) + 1
            )
            summary["profile_kind_counts"][kind_key] = (
                summary["profile_kind_counts"].get(kind_key, 0) + 1
            )
        elif isinstance(record, RelayMEMEpisodeRecord):
            summary["episode_records"] += 1
            status_key = record.status.value
            kind_key = record.episode_kind.value
            summary["status_counts"][status_key] = (
                summary["status_counts"].get(status_key, 0) + 1
            )
            summary["episode_kind_counts"][kind_key] = (
                summary["episode_kind_counts"].get(kind_key, 0) + 1
            )
        elif isinstance(record, RelayMEMSummaryRecord):
            summary["summary_records"] += 1
            status_key = record.status.value
            kind_key = record.summary_kind.value
            summary["status_counts"][status_key] = (
                summary["status_counts"].get(status_key, 0) + 1
            )
            summary["summary_kind_counts"][kind_key] = (
                summary["summary_kind_counts"].get(kind_key, 0) + 1
            )
        elif isinstance(record, RelayMEMStructuredRecord):
            summary["structured_records"] += 1
            status_key = record.status.value
            summary["status_counts"][status_key] = (
                summary["status_counts"].get(status_key, 0) + 1
            )
        elif isinstance(record, RelayMEMKVCheckpointMetadata):
            summary["kv_checkpoint_metadata_records"] += 1
            status_key = record.status.value
            summary["status_counts"][status_key] = (
                summary["status_counts"].get(status_key, 0) + 1
            )
        else:
            summary["unknown_records"] += 1

    return summary
