# RelayStack Evaluation Attribution Plan

Design status: 2026-05-17 JST.

This document records the design-level evaluation attribution plan for RelayStack. It is a design/evaluation document only. It does not claim that the described runtime integration is implemented.

## Purpose

The purpose of this plan is to avoid mixed-layer regressions where memory selection, context transformation, and KV working-set errors cannot be separated.

RelayStack runtime activation may be KV-first under VRAM pressure, but evaluation attribution should remain layered:

```text
MEM → CTX → KV
```

## Stage definitions

### MEM-only

Evaluates:

- memory retrieval and recall relevance
- ranking quality
- source coverage
- freshness and privacy-scope behavior
- hallucination risk from wrong memory proposal

Does not evaluate:

- tokenizer packing
- compression quality
- KV working-set quality
- engine runtime behavior

Metrics and examples:

- recall@k
- source precision
- context item usefulness
- attribution coverage
- rejected or irrelevant memory count
- privacy or scope violation count

### MEM+CTX

Evaluates:

- context packing
- token budget fitting
- compression plan metadata
- prompt layout role
- token span mapping
- source attribution preservation

Does not evaluate:

- KV working-set quality
- runtime adapter behavior
- scheduler or KV pool mutation

Metrics and examples:

- token budget fit rate
- source span preservation
- compression loss proxy
- layout stability
- attribution completeness
- context reduction quality

### MEM+CTX+KV

Evaluates:

- RelayKV working-set decision quality
- fixed-budget RECENT / ANCHOR / RETRIEVED allocation
- shadow quality against FullKV reference where FullKV is available
- practical low-VRAM quality against sliding-window, truncation, and recent+anchor baselines
- safe degrade, block, and context-reduction recommendations

Does not evaluate:

- memory backend relevance alone
- tool execution
- product UI approval flow
- production engine performance before the adapter phase

Metrics and examples:

- attention diff or logit diff where available
- output equivalence or first divergence
- top-k overlap
- task answer correctness
- budget compliance
- selected, rejected, and overflow block analysis
- quality under fixed VRAM budget

## Reference vs practical baselines

### Reference baselines

- FullKV when available
- used for shadow compare and quality-teacher evaluation
- not always available after apply under pressure

### Practical low-VRAM baselines

- sliding window
- truncation
- recent + static anchor
- summary compression
- RAG-style retrieval or context stuffing
- engine default context or KV policy when available

Reference quality and practical low-VRAM quality should not be collapsed into one score.

## Attribution rules

- If MEM-only fails, do not blame RelayCTX or RelayKV.
- If MEM-only passes but MEM+CTX fails, likely classify as a context packing, compression, or span-attribution issue.
- If MEM+CTX passes but MEM+CTX+KV fails, likely classify as a KV working-set, budget, or selection issue.
- If output differs from FullKV but task quality remains acceptable, classify as behavioral divergence, not necessarily failure.
- If fixed-budget compliance fails, classify as a budget policy issue.
- If lineage is missing, classify as a trace or data-contract issue before quality attribution.
- If runtime adapter capability is missing, classify as an adapter capability issue, not a core policy quality issue.

## Artifact mapping

Existing and near-term artifacts:

- RelayMEM retrieval and context-assembly JSON
- RelayCTX span or packing-plan JSON (future)
- `relaykv_candidates.json`
- `relaykv_fixed_budget_block_selection.json`
- `chain_summary.json`
- pressure shadow quality report
- RelayStackTraceEvent-compatible JSONL (future)

Suggested stage mapping:

```text
MEM-only:
  RelayMEM retrieval results
  context assembly plans

MEM+CTX:
  RelayCTX span / packing plan artifacts
  source attribution and token-span metadata

MEM+CTX+KV:
  relaykv_candidates.json
  relaykv_fixed_budget_block_selection.json
  chain_summary.json
  pressure shadow quality report
```

## Phase mapping

- Phase 11.5-D: docs and evaluation contract only
- Phase 12: adapter target selection
- Phase 13: safe materialization and shadow attention compare
- Phase 14: gated apply and safe degrade / block / context reduction
- Phase 15: RelayCTX budgeted context integration
- Phase 16: MEM+CTX+KV attribution evaluation

## Compact stage table

| Stage | Primary owner | Inputs | Outputs | Main metrics | Do not blame this stage for |
| --- | --- | --- | --- | --- | --- |
| `MEM-only` | RelayMEM | memory records, retrieval query, scope metadata | context-item proposals | recall@k, source precision, usefulness, privacy/scope violations | tokenizer packing, compression, KV working-set selection |
| `MEM+CTX` | RelayCTX | RelayMEM proposals, token budget, tokenizer metadata | packed context, token spans, attribution metadata | budget fit, span preservation, compression proxy, attribution completeness | KV working-set quality, runtime adapter behavior |
| `MEM+CTX+KV` | RelayKV | packed context, span metadata, KV budget, candidate blocks | working-set decision artifacts, quality reports, selected/rejected/overflow analysis | attention or logit diff, task quality, budget compliance, block analysis | memory relevance alone, product UI flow, tool execution |

## Minimal examples

### Example 1

Memory recall returns the wrong item.

Classification:

- MEM issue

### Example 2

The right memory item is selected, but compression or packing loses the key fact.

Classification:

- CTX issue

### Example 3

The right span enters the prompt, but the needed KV block is not selected under the fixed budget.

Classification:

- KV issue

### Example 4

The engine lacks shadow-compare capability.

Classification:

- adapter capability issue

### Example 5

FullKV and RelayKV outputs differ, but the task answer remains correct and acceptable.

Classification:

- divergence accepted with note

## Near-term usage

Phase 11.5-D is evaluation-contract consolidation only.

This document does not add:

- runtime adapter behavior
- KV materialization
- attention connection
- scheduler changes
- tool execution inside RelayStack Core
