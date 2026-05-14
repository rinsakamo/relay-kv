import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMBackendKind,
    RelayMEMMemorySource,
    RelayMEMRetrievalMode,
    RelayMEMRetrievalResult,
    build_relaymem_context_assembly_plan,
)


def make_result(
    *,
    memory_id: str = "mem-1",
    memory_source: RelayMEMMemorySource = RelayMEMMemorySource.SUMMARY,
    text: str = "summary text",
    score: float | None = 0.75,
    evidence: list[str] | None = None,
    source_refs: list[str] | None = None,
    estimated_tokens: int = 32,
    retrieval_backend: RelayMEMBackendKind = RelayMEMBackendKind.VECTOR,
    confidence: float | None = None,
) -> RelayMEMRetrievalResult:
    return RelayMEMRetrievalResult(
        memory_id=memory_id,
        memory_source=memory_source,
        text=text,
        score=score,
        evidence=list(evidence or []),
        source_refs=list(source_refs or []),
        estimated_tokens=estimated_tokens,
        retrieval_backend=retrieval_backend,
        confidence=confidence,
    )


def test_retrieval_result_summary_uses_enum_values() -> None:
    result = make_result(
        memory_source=RelayMEMMemorySource.RAG_CHUNK,
        retrieval_backend=RelayMEMBackendKind.HYBRID,
        evidence=["fact-a"],
        source_refs=["doc://a"],
        confidence=0.6,
    )

    assert result.summary() == {
        "memory_id": "mem-1",
        "memory_source": "rag_chunk",
        "text": "summary text",
        "score": 0.75,
        "evidence": ["fact-a"],
        "source_refs": ["doc://a"],
        "estimated_tokens": 32,
        "retrieval_backend": "hybrid",
        "confidence": 0.6,
    }


def test_retrieval_result_rejects_empty_memory_id() -> None:
    with pytest.raises(ValueError, match="memory_id"):
        make_result(memory_id="")


def test_retrieval_result_rejects_confidence_above_one() -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_result(confidence=1.1)


def test_retrieval_result_allows_bm25_like_score_greater_than_one() -> None:
    result = make_result(
        retrieval_backend=RelayMEMBackendKind.BM25,
        score=12.5,
    )

    assert result.score == 12.5


def test_retrieval_result_allows_out_of_range_vector_like_score() -> None:
    result = make_result(
        retrieval_backend=RelayMEMBackendKind.VECTOR,
        score=-3.25,
    )

    assert result.score == -3.25


def test_context_assembly_respects_token_budget() -> None:
    plan = build_relaymem_context_assembly_plan(
        query="character continuity",
        retrieval_mode=RelayMEMRetrievalMode.FAST_RECALL,
        backend_kind=RelayMEMBackendKind.VECTOR,
        retrieval_results=[
            make_result(memory_id="mem-1", estimated_tokens=30, score=0.8),
            make_result(memory_id="mem-2", estimated_tokens=40, score=0.7),
            make_result(memory_id="mem-3", estimated_tokens=20, score=0.6),
        ],
        token_budget=50,
    )

    assert [item.memory_id for item in plan.selected_items] == ["mem-1", "mem-3"]
    assert plan.total_estimated_tokens == 50


def test_context_assembly_records_dropped_memory_ids() -> None:
    plan = build_relaymem_context_assembly_plan(
        query="older episode facts",
        retrieval_mode=RelayMEMRetrievalMode.FAST_RECALL,
        backend_kind=RelayMEMBackendKind.BM25,
        retrieval_results=[
            make_result(memory_id="mem-1", estimated_tokens=20),
            make_result(memory_id="mem-2", estimated_tokens=50),
            make_result(memory_id="mem-3", estimated_tokens=10),
        ],
        token_budget=30,
    )

    assert plan.dropped_memory_ids == ["mem-2"]
    assert [item.memory_id for item in plan.selected_items] == ["mem-1", "mem-3"]


def test_context_assembly_uses_zero_priority_for_out_of_range_score() -> None:
    plan = build_relaymem_context_assembly_plan(
        query="older sparse evidence",
        retrieval_mode=RelayMEMRetrievalMode.FAST_RECALL,
        backend_kind=RelayMEMBackendKind.BM25,
        retrieval_results=[
            make_result(
                memory_id="mem-1",
                retrieval_backend=RelayMEMBackendKind.BM25,
                score=8.0,
                confidence=None,
            ),
            make_result(
                memory_id="mem-2",
                retrieval_backend=RelayMEMBackendKind.VECTOR,
                score=-0.5,
                confidence=None,
            ),
        ],
    )

    assert [item.priority for item in plan.selected_items] == [0.0, 0.0]


def test_deep_recall_evidence_chain_can_be_represented_without_backend_dependency() -> None:
    plan = build_relaymem_context_assembly_plan(
        query="trace older supporting evidence",
        retrieval_mode=RelayMEMRetrievalMode.DEEP_RECALL,
        backend_kind=RelayMEMBackendKind.EVIDENCE_CHAIN,
        retrieval_results=[
            make_result(
                memory_id="mem-chain-1",
                memory_source=RelayMEMMemorySource.RAG_CHUNK,
                retrieval_backend=RelayMEMBackendKind.EVIDENCE_CHAIN,
                score=None,
                confidence=0.9,
                evidence=["claim-a", "claim-b"],
                source_refs=["note://1", "note://2"],
            )
        ],
        approval_required=True,
        approval_reason="deeper recall requested",
        user_visible_message="May I search older memory?",
    )

    assert plan.retrieval_mode is RelayMEMRetrievalMode.DEEP_RECALL
    assert plan.backend_kind is RelayMEMBackendKind.EVIDENCE_CHAIN
    assert plan.selected_items[0].priority == 0.9
    assert plan.summary()["backend_kind"] == "evidence_chain"


def test_import_from_relaykv_stays_torch_free() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import relaykv; "
            "assert 'torch' not in sys.modules; "
            "assert relaykv.RelayMEMBackendKind.VECTOR.value == 'vector'; "
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
