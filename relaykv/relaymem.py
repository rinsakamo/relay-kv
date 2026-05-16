from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RelayMEMRetrievalMode(str, Enum):
    FAST_RECALL = "fast_recall"
    DEEP_RECALL = "deep_recall"
    OFFLINE_REFINEMENT = "offline_refinement"


class RelayMEMBackendKind(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    GRAPH = "graph"
    EVIDENCE_CHAIN = "evidence_chain"
    EXTERNAL_SERVICE = "external_service"


class RelayMEMMemorySource(str, Enum):
    PROFILE = "profile"
    EPISODE = "episode"
    SUMMARY = "summary"
    RAG_CHUNK = "rag_chunk"
    STRUCTURED = "structured"
    KV_CHECKPOINT_METADATA = "kv_checkpoint_metadata"


@dataclass(frozen=True)
class RelayMEMBackendCapabilities:
    backend_id: str
    backend_type: str
    runs_on_cpu: bool = True
    requires_gpu: bool = False
    uses_vram: bool = False
    disk_backed: bool = False
    local_first: bool = True
    remote_allowed: bool = False
    supports_hybrid_search: bool = False
    supports_graph: bool = False
    supports_timeline: bool = False
    supports_compaction: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.backend_id, "backend_id")
        _require_non_empty(self.backend_type, "backend_type")

    def summary(self) -> dict:
        return {
            "backend_id": self.backend_id,
            "backend_type": self.backend_type,
            "runs_on_cpu": self.runs_on_cpu,
            "requires_gpu": self.requires_gpu,
            "uses_vram": self.uses_vram,
            "disk_backed": self.disk_backed,
            "local_first": self.local_first,
            "remote_allowed": self.remote_allowed,
            "supports_hybrid_search": self.supports_hybrid_search,
            "supports_graph": self.supports_graph,
            "supports_timeline": self.supports_timeline,
            "supports_compaction": self.supports_compaction,
        }


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


def _require_numeric(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")


def _priority_from_retrieval_result(result: RelayMEMRetrievalResult) -> float:
    if result.confidence is not None:
        return result.confidence
    if result.score is not None and 0.0 <= float(result.score) <= 1.0:
        return float(result.score)
    return 0.0


@dataclass(frozen=True)
class RelayMEMRetrievalResult:
    memory_id: str
    memory_source: RelayMEMMemorySource
    text: str
    score: float | None
    evidence: list[str]
    source_refs: list[str]
    estimated_tokens: int
    retrieval_backend: RelayMEMBackendKind
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.memory_id, "memory_id")
        _require_non_empty(self.text, "text")
        _require_non_negative(self.estimated_tokens, "estimated_tokens")
        _require_numeric(self.score, "score")
        _require_unit_interval(self.confidence, "confidence")

    def summary(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_source": self.memory_source.value,
            "text": self.text,
            "score": self.score,
            "evidence": list(self.evidence),
            "source_refs": list(self.source_refs),
            "estimated_tokens": self.estimated_tokens,
            "retrieval_backend": self.retrieval_backend.value,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RelayMEMContextItem:
    context_item_id: str
    memory_id: str
    memory_source: RelayMEMMemorySource
    text: str
    estimated_tokens: int
    priority: float
    pinned: bool
    insertion_reason: str | None
    source_refs: list[str]

    def __post_init__(self) -> None:
        _require_non_empty(self.context_item_id, "context_item_id")
        _require_non_empty(self.memory_id, "memory_id")
        _require_non_empty(self.text, "text")
        _require_non_negative(self.estimated_tokens, "estimated_tokens")
        _require_unit_interval(self.priority, "priority")

    def summary(self) -> dict:
        return {
            "context_item_id": self.context_item_id,
            "memory_id": self.memory_id,
            "memory_source": self.memory_source.value,
            "text": self.text,
            "estimated_tokens": self.estimated_tokens,
            "priority": self.priority,
            "pinned": self.pinned,
            "insertion_reason": self.insertion_reason,
            "source_refs": list(self.source_refs),
        }


@dataclass(frozen=True)
class RelayMEMContextAssemblyPlan:
    retrieval_mode: RelayMEMRetrievalMode
    backend_kind: RelayMEMBackendKind
    query: str
    selected_items: list[RelayMEMContextItem]
    dropped_memory_ids: list[str]
    total_estimated_tokens: int
    token_budget: int | None
    approval_required: bool
    approval_reason: str | None
    user_visible_message: str | None
    proposed_retrieval_mode: RelayMEMRetrievalMode | None = None
    fallback_if_denied: RelayMEMRetrievalMode | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.query, "query")
        _require_non_negative(self.total_estimated_tokens, "total_estimated_tokens")
        if self.token_budget is not None:
            _require_non_negative(self.token_budget, "token_budget")

    def summary(self) -> dict:
        return {
            "retrieval_mode": self.retrieval_mode.value,
            "backend_kind": self.backend_kind.value,
            "query": self.query,
            "selected_items": [item.summary() for item in self.selected_items],
            "dropped_memory_ids": list(self.dropped_memory_ids),
            "total_estimated_tokens": self.total_estimated_tokens,
            "token_budget": self.token_budget,
            "approval_required": self.approval_required,
            "approval_reason": self.approval_reason,
            "user_visible_message": self.user_visible_message,
            "proposed_retrieval_mode": (
                self.proposed_retrieval_mode.value
                if self.proposed_retrieval_mode is not None
                else None
            ),
            "fallback_if_denied": (
                self.fallback_if_denied.value
                if self.fallback_if_denied is not None
                else None
            ),
        }


def build_relaymem_context_assembly_plan(
    query: str,
    retrieval_mode: RelayMEMRetrievalMode,
    backend_kind: RelayMEMBackendKind,
    retrieval_results: list[RelayMEMRetrievalResult],
    token_budget: int | None = None,
    approval_required: bool = False,
    approval_reason: str | None = None,
    user_visible_message: str | None = None,
    proposed_retrieval_mode: RelayMEMRetrievalMode | None = None,
    fallback_if_denied: RelayMEMRetrievalMode | None = None,
) -> RelayMEMContextAssemblyPlan:
    _require_non_empty(query, "query")
    if token_budget is not None:
        _require_non_negative(token_budget, "token_budget")

    selected_items: list[RelayMEMContextItem] = []
    dropped_memory_ids: list[str] = []
    total_estimated_tokens = 0

    for result in retrieval_results:
        next_total_estimated_tokens = total_estimated_tokens + result.estimated_tokens
        if token_budget is not None and next_total_estimated_tokens > token_budget:
            dropped_memory_ids.append(result.memory_id)
            continue

        priority = _priority_from_retrieval_result(result)
        selected_items.append(
            RelayMEMContextItem(
                context_item_id=f"context_item:{result.memory_id}",
                memory_id=result.memory_id,
                memory_source=result.memory_source,
                text=result.text,
                estimated_tokens=result.estimated_tokens,
                priority=priority,
                pinned=False,
                insertion_reason="retrieval_result",
                source_refs=list(result.source_refs),
            )
        )
        total_estimated_tokens = next_total_estimated_tokens

    return RelayMEMContextAssemblyPlan(
        retrieval_mode=retrieval_mode,
        backend_kind=backend_kind,
        query=query,
        selected_items=selected_items,
        dropped_memory_ids=dropped_memory_ids,
        total_estimated_tokens=total_estimated_tokens,
        token_budget=token_budget,
        approval_required=approval_required,
        approval_reason=approval_reason,
        user_visible_message=user_visible_message,
        proposed_retrieval_mode=proposed_retrieval_mode,
        fallback_if_denied=fallback_if_denied,
    )


def build_default_fast_recall_backend_capabilities() -> RelayMEMBackendCapabilities:
    return RelayMEMBackendCapabilities(
        backend_id="relaymem.fast_recall.default",
        backend_type=RelayMEMRetrievalMode.FAST_RECALL.value,
        runs_on_cpu=True,
        requires_gpu=False,
        uses_vram=False,
        disk_backed=False,
        local_first=True,
        remote_allowed=False,
        supports_hybrid_search=False,
        supports_graph=False,
        supports_timeline=False,
        supports_compaction=False,
    )
