from __future__ import annotations

from dataclasses import dataclass

from .relaymem import (
    RelayMEMBackendKind,
    RelayMEMContextAssemblyPlan,
    RelayMEMMemorySource,
    RelayMEMRetrievalMode,
    RelayMEMRetrievalResult,
    _require_non_empty,
    _require_non_negative,
    _require_unit_interval,
    build_relaymem_context_assembly_plan,
)

_DEFAULT_MAX_PREVIEW_CHARS = 160


def _truncate_preview_text(text: str, max_preview_chars: int) -> str:
    if len(text) <= max_preview_chars:
        return text
    if max_preview_chars <= 3:
        return text[:max_preview_chars]

    truncated = text[: max_preview_chars - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    truncated = truncated.rstrip(".,;: ")
    if not truncated:
        truncated = text[: max_preview_chars - 3]
    return f"{truncated}..."


def _default_user_visible_message(
    *,
    retrieval_mode: RelayMEMRetrievalMode,
    approval_required: bool,
    preview_item_count: int,
    dropped_memory_ids: list[str],
    fallback_reason: str | None,
) -> str:
    if fallback_reason == "no_retrieval_results":
        return "Fast Recall found no memory to preview for this query."
    if fallback_reason is not None:
        if dropped_memory_ids:
            return "Fast Recall found matching memory, but it was omitted from the prompt preview because the token budget would be exceeded."
        return "Fast Recall prepared a prompt preview with a fallback condition."
    if dropped_memory_ids:
        return "Fast Recall prepared a partial prompt preview because the token budget would be exceeded."
    if preview_item_count == 0:
        return "Fast Recall found no memory to preview for this query."
    if approval_required:
        if retrieval_mode is RelayMEMRetrievalMode.DEEP_RECALL:
            return (
                "RelayMEM prepared a deeper memory recall preview. "
                "User approval is required before applying it."
            )
        return "Fast Recall prepared a prompt preview. User approval is required before applying it."
    return "Fast Recall prepared a prompt preview that can be applied to active context."


@dataclass(frozen=True)
class RelayMEMPromptPreviewItem:
    context_item_id: str
    memory_id: str
    memory_source: RelayMEMMemorySource
    preview_text: str
    estimated_tokens: int
    priority: float
    source_refs: list[str]
    insertion_reason: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.context_item_id, "context_item_id")
        _require_non_empty(self.memory_id, "memory_id")
        _require_non_empty(self.preview_text, "preview_text")
        _require_non_negative(self.estimated_tokens, "estimated_tokens")
        _require_unit_interval(self.priority, "priority")

    def summary(self) -> dict:
        return {
            "context_item_id": self.context_item_id,
            "memory_id": self.memory_id,
            "memory_source": self.memory_source.value,
            "preview_text": self.preview_text,
            "estimated_tokens": self.estimated_tokens,
            "priority": self.priority,
            "source_refs": list(self.source_refs),
            "insertion_reason": self.insertion_reason,
        }


@dataclass(frozen=True)
class RelayMEMPromptPreviewPlan:
    query: str
    retrieval_mode: RelayMEMRetrievalMode
    backend_kind: RelayMEMBackendKind
    preview_items: list[RelayMEMPromptPreviewItem]
    dropped_memory_ids: list[str]
    total_estimated_tokens: int
    token_budget: int | None
    approval_required: bool
    approval_reason: str | None
    user_visible_message: str | None
    fallback_if_denied: RelayMEMRetrievalMode | None
    fallback_reason: str | None
    can_apply_without_user_approval: bool

    def __post_init__(self) -> None:
        _require_non_empty(self.query, "query")
        _require_non_negative(self.total_estimated_tokens, "total_estimated_tokens")
        if self.token_budget is not None:
            _require_non_negative(self.token_budget, "token_budget")

    def summary(self) -> dict:
        return {
            "query": self.query,
            "retrieval_mode": self.retrieval_mode.value,
            "backend_kind": self.backend_kind.value,
            "preview_items": [item.summary() for item in self.preview_items],
            "dropped_memory_ids": list(self.dropped_memory_ids),
            "total_estimated_tokens": self.total_estimated_tokens,
            "token_budget": self.token_budget,
            "approval_required": self.approval_required,
            "approval_reason": self.approval_reason,
            "user_visible_message": self.user_visible_message,
            "fallback_if_denied": (
                self.fallback_if_denied.value
                if self.fallback_if_denied is not None
                else None
            ),
            "fallback_reason": self.fallback_reason,
            "can_apply_without_user_approval": self.can_apply_without_user_approval,
        }


def build_relaymem_prompt_preview_plan(
    query: str,
    retrieval_results: list[RelayMEMRetrievalResult],
    *,
    retrieval_mode: RelayMEMRetrievalMode = RelayMEMRetrievalMode.FAST_RECALL,
    backend_kind: RelayMEMBackendKind | None = None,
    token_budget: int | None = None,
    approval_required: bool = False,
    approval_reason: str | None = None,
    user_visible_message: str | None = None,
    fallback_if_denied: RelayMEMRetrievalMode | None = None,
    fallback_reason: str | None = None,
    max_preview_chars: int = _DEFAULT_MAX_PREVIEW_CHARS,
) -> RelayMEMPromptPreviewPlan:
    _require_non_empty(query, "query")
    if token_budget is not None:
        _require_non_negative(token_budget, "token_budget")
    if max_preview_chars <= 0:
        raise ValueError("max_preview_chars must be > 0")

    resolved_backend_kind = backend_kind
    if resolved_backend_kind is None:
        if retrieval_results:
            resolved_backend_kind = retrieval_results[0].retrieval_backend
        else:
            resolved_backend_kind = RelayMEMBackendKind.BM25

    context_plan: RelayMEMContextAssemblyPlan = build_relaymem_context_assembly_plan(
        query=query,
        retrieval_mode=retrieval_mode,
        backend_kind=resolved_backend_kind,
        retrieval_results=retrieval_results,
        token_budget=token_budget,
        approval_required=approval_required,
        approval_reason=approval_reason,
        user_visible_message=user_visible_message,
        fallback_if_denied=fallback_if_denied,
    )

    preview_items = [
        RelayMEMPromptPreviewItem(
            context_item_id=item.context_item_id,
            memory_id=item.memory_id,
            memory_source=item.memory_source,
            preview_text=_truncate_preview_text(item.text, max_preview_chars),
            estimated_tokens=item.estimated_tokens,
            priority=item.priority,
            source_refs=list(item.source_refs),
            insertion_reason=item.insertion_reason,
        )
        for item in context_plan.selected_items
    ]

    resolved_fallback_reason = fallback_reason
    if context_plan.dropped_memory_ids and resolved_fallback_reason is None:
        resolved_fallback_reason = "token_budget_exceeded"
    elif not retrieval_results and resolved_fallback_reason is None:
        resolved_fallback_reason = "no_retrieval_results"

    resolved_user_visible_message = user_visible_message
    if resolved_user_visible_message is None:
        resolved_user_visible_message = _default_user_visible_message(
            retrieval_mode=retrieval_mode,
            approval_required=approval_required,
            preview_item_count=len(preview_items),
            dropped_memory_ids=context_plan.dropped_memory_ids,
            fallback_reason=resolved_fallback_reason,
        )

    can_apply_without_user_approval = (
        not approval_required
        and not bool(context_plan.dropped_memory_ids)
        and resolved_fallback_reason is None
    )

    return RelayMEMPromptPreviewPlan(
        query=query,
        retrieval_mode=retrieval_mode,
        backend_kind=resolved_backend_kind,
        preview_items=preview_items,
        dropped_memory_ids=list(context_plan.dropped_memory_ids),
        total_estimated_tokens=context_plan.total_estimated_tokens,
        token_budget=context_plan.token_budget,
        approval_required=approval_required,
        approval_reason=approval_reason,
        user_visible_message=resolved_user_visible_message,
        fallback_if_denied=fallback_if_denied,
        fallback_reason=resolved_fallback_reason,
        can_apply_without_user_approval=can_apply_without_user_approval,
    )
