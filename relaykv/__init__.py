from __future__ import annotations

from importlib import import_module

from .demotion_policy import (
    RelayKVDemotionDecision,
    build_demotion_decision,
)
from .vram_budget import (
    RelayKVVramBudgetDecision,
    build_vram_budget_decision,
)
from .memory_block import (
    KVClass,
    ResidencyLevel,
    PrecisionLevel,
    RopeStatus,
    CachePositionPolicy,
    RelayKVMemoryBlock,
)
from .routing_decision import (
    ExecutionMode,
    RelayKVDecision,
)
from .routing_policy import (
    build_routing_decision_from_demotion,
)
from .memory_pressure import (
    RelayKVMemoryPressureState,
    RelayKVMemoryPressureDecision,
    decide_memory_pressure_state,
)
from .memory_pressure_summary import (
    summarize_memory_pressure_decisions,
)
from .relaymem import (
    RelayMEMBackendCapabilities,
    RelayMEMBackendKind,
    RelayMEMContextAssemblyPlan,
    RelayMEMContextItem,
    RelayMEMMemorySource,
    RelayMEMRetrievalMode,
    RelayMEMRetrievalResult,
    build_default_fast_recall_backend_capabilities,
    build_relaymem_context_assembly_plan,
)
from .relaymem_fast_recall import (
    RelayMEMFastRecallCandidate,
    relaymem_record_to_fast_recall_candidate,
    search_relaymem_fast_recall,
)
from .relaymem_prompt_preview import (
    RelayMEMPromptPreviewItem,
    RelayMEMPromptPreviewPlan,
    build_relaymem_prompt_preview_plan,
)
from .relaymem_records import (
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
from .relaystack_final_decision import (
    RelayStackFinalRoutingDecision,
    RelayStackFinalRoutingState,
    decide_relaystack_final_routing,
)
from .relaystack_hf_smoke_report import (
    RelayStackHFContextRowSummary,
    RelayStackHFSmokeReport,
    build_relaykv_shadow_quality_test_plan_fields,
    build_relaystack_hf_smoke_report,
)
from .relaykv_pressure_shadow_quality_report import (
    RelayKVPressureShadowQualityReport,
    RelayKVShadowQualitySummary,
    build_relaykv_pressure_shadow_quality_report,
)
from .vram_reservation import (
    RelayKVVramReservation,
    RelayKVVramReservationStatus,
    RelayKVVramBudgetDecision as RelayKVVramReservationBudgetDecision,
    build_vram_budget_decision as build_vram_reservation_budget_decision,
)

_LAZY_EXPORTS = {
    "TierManager": ("relaykv.tier_manager", "TierManager"),
    "build_blocks": ("relaykv.block_index", "build_blocks"),
    "ColdCache": ("relaykv.cold_cache", "ColdCache"),
    "ColdSegment": ("relaykv.cold_cache", "ColdSegment"),
    "ColdBlock": ("relaykv.cold_cache", "ColdBlock"),
    "HotKV": ("relaykv.kv_extract", "HotKV"),
    "split_dynamic_cache_layers": (
        "relaykv.kv_extract",
        "split_dynamic_cache_layers",
    ),
    "BlockMetadata": ("relaykv.block_metadata", "BlockMetadata"),
    "build_block_metadata": ("relaykv.block_metadata", "build_block_metadata"),
    "build_metadata_for_blocks": (
        "relaykv.block_metadata",
        "build_metadata_for_blocks",
    ),
    "BlockScore": ("relaykv.block_scoring", "BlockScore"),
    "score_blocks_with_query": (
        "relaykv.block_scoring",
        "score_blocks_with_query",
    ),
    "top_k_blocks": ("relaykv.block_scoring", "top_k_blocks"),
    "RetrievedBlock": ("relaykv.block_retrieval", "RetrievedBlock"),
    "retrieve_blocks": ("relaykv.block_retrieval", "retrieve_blocks"),
    "retrieve_blocks_by_ids": (
        "relaykv.block_retrieval",
        "retrieve_blocks_by_ids",
    ),
    "CandidateKV": ("relaykv.candidate_kv", "CandidateKV"),
    "build_candidate_kv": ("relaykv.candidate_kv", "build_candidate_kv"),
    "build_empty_candidate_kv": (
        "relaykv.candidate_kv",
        "build_empty_candidate_kv",
    ),
    "WorkingKV": ("relaykv.working_kv", "WorkingKV"),
    "build_working_kv": ("relaykv.working_kv", "build_working_kv"),
    "WorkingBlockBudgets": ("relaykv.budget_policy", "WorkingBlockBudgets"),
    "WorkingBlockSelection": ("relaykv.budget_policy", "WorkingBlockSelection"),
    "WorkingBlockBudgetDecision": (
        "relaykv.budget_policy",
        "WorkingBlockBudgetDecision",
    ),
    "build_working_block_budget_decision": (
        "relaykv.budget_policy",
        "build_working_block_budget_decision",
    ),
    "RelayKVActivationDecision": (
        "relaykv.activation_policy",
        "RelayKVActivationDecision",
    ),
    "build_activation_decision": (
        "relaykv.activation_policy",
        "build_activation_decision",
    ),
    "AttentionCompareResult": (
        "relaykv.attention_compare",
        "AttentionCompareResult",
    ),
    "scaled_dot_product_attention": (
        "relaykv.attention_compare",
        "scaled_dot_product_attention",
    ),
    "compare_attention_outputs": (
        "relaykv.attention_compare",
        "compare_attention_outputs",
    ),
    "Span": ("relaykv.three_tier_policy", "Span"),
    "ThreeTierSelection": (
        "relaykv.three_tier_policy",
        "ThreeTierSelection",
    ),
    "build_three_tier_selection": (
        "relaykv.three_tier_policy",
        "build_three_tier_selection",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "TierManager",
    "build_blocks",
    "ColdCache",
    "ColdSegment",
    "ColdBlock",
    "HotKV",
    "split_dynamic_cache_layers",
    "BlockMetadata",
    "build_block_metadata",
    "build_metadata_for_blocks",
    "BlockScore",
    "score_blocks_with_query",
    "top_k_blocks",
    "RetrievedBlock",
    "retrieve_blocks",
    "retrieve_blocks_by_ids",
    "CandidateKV",
    "build_candidate_kv",
    "build_empty_candidate_kv",
    "WorkingKV",
    "build_working_kv",
    "WorkingBlockBudgets",
    "WorkingBlockSelection",
    "WorkingBlockBudgetDecision",
    "build_working_block_budget_decision",
    "RelayKVActivationDecision",
    "build_activation_decision",
    "RelayKVDemotionDecision",
    "build_demotion_decision",
    "RelayKVVramBudgetDecision",
    "build_vram_budget_decision",
    "RelayKVVramReservationStatus",
    "RelayKVVramReservation",
    "RelayKVVramReservationBudgetDecision",
    "build_vram_reservation_budget_decision",
    "KVClass",
    "ResidencyLevel",
    "PrecisionLevel",
    "RopeStatus",
    "CachePositionPolicy",
    "RelayKVMemoryBlock",
    "ExecutionMode",
    "RelayKVDecision",
    "build_routing_decision_from_demotion",
    "RelayKVMemoryPressureState",
    "RelayKVMemoryPressureDecision",
    "decide_memory_pressure_state",
    "summarize_memory_pressure_decisions",
    "RelayMEMRetrievalMode",
    "RelayMEMBackendKind",
    "RelayMEMBackendCapabilities",
    "RelayMEMMemorySource",
    "RelayMEMRetrievalResult",
    "RelayMEMContextItem",
    "RelayMEMContextAssemblyPlan",
    "build_default_fast_recall_backend_capabilities",
    "build_relaymem_context_assembly_plan",
    "RelayMEMFastRecallCandidate",
    "relaymem_record_to_fast_recall_candidate",
    "search_relaymem_fast_recall",
    "RelayMEMPromptPreviewItem",
    "RelayMEMPromptPreviewPlan",
    "build_relaymem_prompt_preview_plan",
    "RelayMEMProfileKind",
    "RelayMEMEpisodeKind",
    "RelayMEMSummaryKind",
    "RelayMEMRecordStatus",
    "RelayMEMProfileRecord",
    "RelayMEMEpisodeRecord",
    "RelayMEMSummaryRecord",
    "RelayMEMStructuredRecord",
    "RelayMEMKVCheckpointMetadata",
    "summarize_relaymem_records",
    "RelayStackFinalRoutingDecision",
    "RelayStackFinalRoutingState",
    "decide_relaystack_final_routing",
    "RelayStackHFContextRowSummary",
    "RelayStackHFSmokeReport",
    "build_relaykv_shadow_quality_test_plan_fields",
    "build_relaystack_hf_smoke_report",
    "RelayKVShadowQualitySummary",
    "RelayKVPressureShadowQualityReport",
    "build_relaykv_pressure_shadow_quality_report",
    "AttentionCompareResult",
    "scaled_dot_product_attention",
    "compare_attention_outputs",
    "Span",
    "ThreeTierSelection",
    "build_three_tier_selection",
]
