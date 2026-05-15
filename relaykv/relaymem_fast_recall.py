from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .relaymem import (
    RelayMEMBackendKind,
    RelayMEMMemorySource,
    RelayMEMRetrievalResult,
)
from .relaymem_records import (
    RelayMEMEpisodeRecord,
    RelayMEMKVCheckpointMetadata,
    RelayMEMProfileRecord,
    RelayMEMRecordStatus,
    RelayMEMStructuredRecord,
    RelayMEMSummaryRecord,
)

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_OVERLAP_WEIGHT = 0.35

_SOURCE_BASE_PRIORITY = {
    RelayMEMMemorySource.PROFILE: 0.55,
    RelayMEMMemorySource.STRUCTURED: 0.28,
    RelayMEMMemorySource.SUMMARY: 0.22,
    RelayMEMMemorySource.EPISODE: 0.02,
    RelayMEMMemorySource.RAG_CHUNK: 0.04,
    RelayMEMMemorySource.KV_CHECKPOINT_METADATA: 0.01,
}


@dataclass(frozen=True)
class RelayMEMFastRecallCandidate:
    memory_id: str
    memory_source: RelayMEMMemorySource
    text: str
    source_refs: list[str]
    estimated_tokens: int
    importance: float
    status: RelayMEMRecordStatus | None = None
    recency_hint: str | None = None

    def summary(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_source": self.memory_source.value,
            "text": self.text,
            "source_refs": list(self.source_refs),
            "estimated_tokens": self.estimated_tokens,
            "importance": self.importance,
            "status": self.status.value if self.status is not None else None,
            "recency_hint": self.recency_hint,
        }


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _estimate_tokens(text: str) -> int:
    word_count = len(_tokenize(text))
    if word_count > 0:
        return max(1, word_count)
    return max(1, len(text) // 4)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _join_labelled_fields(label: str, values: dict[str, str]) -> str:
    if not values:
        return ""
    joined = "; ".join(f"{key}: {value}" for key, value in sorted(values.items()))
    return f"{label}: {joined}"


def relaymem_record_to_fast_recall_candidate(
    record: object,
) -> RelayMEMFastRecallCandidate:
    if isinstance(record, RelayMEMProfileRecord):
        parts = [
            f"Profile: {record.display_name}",
            _join_labelled_fields("Facts", record.facts),
            _join_labelled_fields("Preferences", record.preferences),
            _join_labelled_fields("Relationships", record.relationships),
        ]
        text = ". ".join(part for part in parts if part)
        return RelayMEMFastRecallCandidate(
            memory_id=record.profile_id,
            memory_source=RelayMEMMemorySource.PROFILE,
            text=text,
            source_refs=list(record.source_refs),
            estimated_tokens=_estimate_tokens(text),
            importance=record.importance,
            status=record.status,
            recency_hint=record.updated_at,
        )

    if isinstance(record, RelayMEMEpisodeRecord):
        key_events = "; ".join(record.key_events)
        participants = ", ".join(record.participants)
        parts = [
            f"Episode: {record.title}",
            record.episode_summary,
            f"Participants: {participants}" if participants else "",
            f"Key events: {key_events}" if key_events else "",
        ]
        text = ". ".join(part for part in parts if part)
        return RelayMEMFastRecallCandidate(
            memory_id=record.episode_id,
            memory_source=RelayMEMMemorySource.EPISODE,
            text=text,
            source_refs=list(record.source_refs),
            estimated_tokens=_estimate_tokens(text),
            importance=record.importance,
            status=record.status,
            recency_hint=record.ended_at or record.started_at,
        )

    if isinstance(record, RelayMEMSummaryRecord):
        return RelayMEMFastRecallCandidate(
            memory_id=record.summary_id,
            memory_source=RelayMEMMemorySource.SUMMARY,
            text=record.text,
            source_refs=list(record.covered_source_refs),
            estimated_tokens=record.token_estimate,
            importance=record.importance,
            status=record.status,
            recency_hint=record.updated_at,
        )

    if isinstance(record, RelayMEMStructuredRecord):
        parts = [
            f"Structured record: {record.namespace}.{record.key}",
            record.value,
            _join_labelled_fields("Attributes", record.attributes),
        ]
        text = ". ".join(part for part in parts if part)
        return RelayMEMFastRecallCandidate(
            memory_id=record.record_id,
            memory_source=RelayMEMMemorySource.STRUCTURED,
            text=text,
            source_refs=list(record.source_refs),
            estimated_tokens=_estimate_tokens(text),
            importance=record.confidence if record.confidence is not None else 0.5,
            status=record.status,
            recency_hint=record.updated_at,
        )

    if isinstance(record, RelayMEMKVCheckpointMetadata):
        start, end = record.token_span
        parts = [
            f"KV checkpoint: {record.label}",
            f"model: {record.model_id}",
            f"tokenizer: {record.tokenizer_id}",
            f"token span: {start}-{end}",
            f"precision: {record.precision_level}" if record.precision_level else "",
            f"rope status: {record.rope_status}" if record.rope_status else "",
        ]
        text = ". ".join(part for part in parts if part)
        return RelayMEMFastRecallCandidate(
            memory_id=record.checkpoint_id,
            memory_source=RelayMEMMemorySource.KV_CHECKPOINT_METADATA,
            text=text,
            source_refs=list(record.source_refs),
            estimated_tokens=_estimate_tokens(text),
            importance=0.4,
            status=record.status,
            recency_hint=None,
        )

    raise TypeError(f"unsupported RelayMEM record type: {type(record).__name__}")


def _score_candidate(
    query_tokens: set[str],
    candidate: RelayMEMFastRecallCandidate,
) -> tuple[float, list[str]]:
    candidate_tokens = set(_tokenize(candidate.text))
    overlap_tokens = sorted(query_tokens & candidate_tokens)
    overlap_ratio = len(overlap_tokens) / max(1, len(query_tokens))
    source_priority = _SOURCE_BASE_PRIORITY.get(candidate.memory_source, 0.0)
    importance_bonus = _clamp_unit(candidate.importance) * 0.15
    score = source_priority + importance_bonus + (_OVERLAP_WEIGHT * overlap_ratio)
    return score, overlap_tokens


def search_relaymem_fast_recall(
    query: str,
    records: Iterable[object],
    *,
    max_results: int | None = None,
    include_inactive: bool = False,
) -> list[RelayMEMRetrievalResult]:
    """Return deterministic stdlib-only RelayMEM retrieval results.

    This is a lightweight Fast Recall backend for smoke tests and early policy
    experiments. It does not call embeddings, a vector DB, a model, or external
    services.
    """

    _require_non_empty(query, "query")
    if max_results is not None and max_results < 0:
        raise ValueError("max_results must be >= 0")

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, RelayMEMFastRecallCandidate, list[str]]] = []
    for record in records:
        candidate = relaymem_record_to_fast_recall_candidate(record)
        if (
            not include_inactive
            and candidate.status is not None
            and candidate.status is not RelayMEMRecordStatus.ACTIVE
        ):
            continue

        score, overlap_tokens = _score_candidate(query_tokens, candidate)
        if not overlap_tokens:
            continue
        scored.append((score, candidate, overlap_tokens))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].memory_source.value,
            item[1].memory_id,
        )
    )

    if max_results is not None:
        scored = scored[:max_results]

    results: list[RelayMEMRetrievalResult] = []
    for score, candidate, overlap_tokens in scored:
        confidence = _clamp_unit(score / 1.5)
        evidence = [f"matched_tokens={','.join(overlap_tokens)}"]
        if candidate.recency_hint:
            evidence.append(f"recency_hint={candidate.recency_hint}")
        results.append(
            RelayMEMRetrievalResult(
                memory_id=candidate.memory_id,
                memory_source=candidate.memory_source,
                text=candidate.text,
                score=score,
                evidence=evidence,
                source_refs=list(candidate.source_refs),
                estimated_tokens=candidate.estimated_tokens,
                retrieval_backend=RelayMEMBackendKind.BM25,
                confidence=confidence,
            )
        )

    return results
