from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .block_scoring import BlockScore

SpanKind = Literal["anchor", "retrieval", "recent"]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: SpanKind

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)

    def summary(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "kind": self.kind,
        }


@dataclass
class ThreeTierSelection:
    layer_idx: int
    full_token_count: int
    recent_spans: list[Span]
    anchor_spans: list[Span]
    retrieval_spans: list[Span]
    selected_block_ids: list[int]
    selected_token_count: int
    policy_name: str = "three_tier_v0"

    def all_spans(self) -> list[Span]:
        # physical order for PyTorch prototype / future index mapping
        return sorted(
            [*self.anchor_spans, *self.retrieval_spans, *self.recent_spans],
            key=lambda s: (s.start, s.end, s.kind),
        )

    def summary(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "layer_idx": self.layer_idx,
            "full_token_count": self.full_token_count,
            "selected_token_count": self.selected_token_count,
            "selected_ratio": (
                self.selected_token_count / self.full_token_count
                if self.full_token_count > 0
                else 0.0
            ),
            "recent_spans": [s.summary() for s in self.recent_spans],
            "anchor_spans": [s.summary() for s in self.anchor_spans],
            "retrieval_spans": [s.summary() for s in self.retrieval_spans],
            "selected_block_ids": self.selected_block_ids,
            "all_spans": [s.summary() for s in self.all_spans()],
        }


def _clip_span(start: int, end: int, seq_len: int, kind: SpanKind) -> Span | None:
    clipped_start = max(0, min(start, seq_len))
    clipped_end = max(0, min(end, seq_len))
    if clipped_end <= clipped_start:
        return None
    return Span(start=clipped_start, end=clipped_end, kind=kind)


def _subtract_span(span: Span, blockers: list[Span]) -> list[Span]:
    """
    Remove blocker ranges from span.
    Used so retrieval does not duplicate anchor/recent tokens.
    """
    pieces = [(span.start, span.end)]

    for blocker in blockers:
        next_pieces: list[tuple[int, int]] = []
        for start, end in pieces:
            if blocker.end <= start or end <= blocker.start:
                next_pieces.append((start, end))
                continue

            if start < blocker.start:
                next_pieces.append((start, blocker.start))
            if blocker.end < end:
                next_pieces.append((blocker.end, end))

        pieces = next_pieces

    return [Span(start=s, end=e, kind=span.kind) for s, e in pieces if e > s]


def build_three_tier_selection(
    *,
    seq_len: int,
    hot_window: int,
    anchor_blocks: int,
    block_size: int,
    selected_scores: list[BlockScore],
    layer_idx: int,
) -> ThreeTierSelection:
    """
    Build a runtime-neutral RelayKV selection.

    This function intentionally returns token spans / block ids, not KV tensors.
    PyTorch prototype code may later materialize these spans as tensors, while
    SGLang integration should map them to KV page/slot/radix-cache indices.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if hot_window < 0:
        raise ValueError("hot_window must be >= 0")
    if anchor_blocks < 0:
        raise ValueError("anchor_blocks must be >= 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    recent_start = max(0, seq_len - hot_window)
    recent = _clip_span(recent_start, seq_len, seq_len, "recent")
    recent_spans = [recent] if recent is not None else []

    anchor_end = min(anchor_blocks * block_size, recent_start)
    anchor = _clip_span(0, anchor_end, seq_len, "anchor")
    anchor_spans = [anchor] if anchor is not None else []

    blockers = [*anchor_spans, *recent_spans]

    retrieval_spans: list[Span] = []
    selected_block_ids: list[int] = []

    for score in selected_scores:
        if score.layer_idx != layer_idx:
            continue

        raw_span = _clip_span(score.start, score.end, seq_len, "retrieval")
        if raw_span is None:
            continue

        pieces = _subtract_span(raw_span, blockers)
        if not pieces:
            continue

        retrieval_spans.extend(pieces)
        selected_block_ids.append(score.block_id)

    selected_token_count = sum(s.length for s in anchor_spans)
    selected_token_count += sum(s.length for s in retrieval_spans)
    selected_token_count += sum(s.length for s in recent_spans)

    return ThreeTierSelection(
        layer_idx=layer_idx,
        full_token_count=seq_len,
        recent_spans=recent_spans,
        anchor_spans=anchor_spans,
        retrieval_spans=retrieval_spans,
        selected_block_ids=selected_block_ids,
        selected_token_count=selected_token_count,
    )