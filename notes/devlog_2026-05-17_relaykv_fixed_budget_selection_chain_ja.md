# RelayKV Devlog - 2026-05-17 JST - Phase 11-E fixed-budget selection chain

## Date basis

- Date basis: JST 2026-05-17.

## Summary

- Phase 11-A から Phase 11-D までに、RelayKV fixed-budget working-set dry-run policy、block candidate selection dry-run、pipeline/scoring artifact からの candidate export、fixed-budget selection artifact chain が揃った。
- 今回の Phase 11-E では、その fixed-budget dry-run chain を JST 2026-05-17 時点の devlog として整理し、設計境界と次段の contract consolidation に接続する状態を記録した。
- Phase 11 の実装スコープは引き続き dry-run/schema/CLI/report-only であり、runtime adapter、attention connection、KV materialization、scheduler change には進んでいない。

## What changed

- PR #67:
  - fixed-budget working-set dry-run policy schema / CLI
  - RECENT / ANCHOR / RETRIEVED / TRANSIENT reservation の token budget と block-aligned plan
- PR #68:
  - fixed-budget block candidate selection dry-run
  - selected / rejected / overflow block ids の artifact 化
- PR #69:
  - pipeline/scoring artifact から Phase 11-B candidates-json への exporter
  - pipeline-style keys と filtered `top_scores` guard の整備
- PR #70:
  - pipeline candidate export → fixed-budget block selection → chain_summary の Phase 11-D chain CLI

## Artifacts

Phase 11-D chain が出力する主 artifact:

- `relaykv_candidates.json`
- `relaykv_fixed_budget_block_selection.json`
- `chain_summary.json`

artifact flow:

```text
pipeline/scoring artifact
  → relaykv_candidates.json
  → relaykv_fixed_budget_block_selection.json
  → chain_summary.json
```

## Safety boundary

- no model loading by default
- no actual KV materialization
- no attention connection
- no runtime adapter
- no scheduler changes

Phase 11 chain は artifact conversion と dry-run decision/report だけを扱う。実 engine object、runtime scheduler、tool execution、approval、UI、heavy RAG、worker execution は扱わない。

## Design alignment

- RelayKV は fixed-VRAM-budget decode-time KV working-set controller として扱う。
- RelayMEM は max-context-external memory layer として扱う。
- RelayCTX は context transform / packing / token-span / attribution layer として扱う。
- RelayStack Core は decision / metadata / budget / lineage / trace を扱う。
- Tool execution / approval / UI / heavy RAG / model-worker execution / engine runtime internals は RelayStack Core 外に置く。
- RelayKV after apply under VRAM pressure では FullKV fallback を前提にしない。
- after apply の安全系は safe degrade / block / request context reduction を前提にする。
- Phase 11 は引き続き dry-run/schema/CLI/report-only であり、materialization / attention connection / runtime adapter / scheduler changes はまだ対象外である。

## Validation commands

```bash
python -m compileall relaykv tests scripts

python -m pytest -q tests/test_relaykv_fixed_budget_selection_chain.py
python -m pytest -q tests/test_relaykv_pipeline_candidate_export.py
python -m pytest -q tests/test_relaykv_fixed_budget_block_selection.py
python -m pytest -q tests/test_relaykv_fixed_budget_working_set.py
python -m pytest -q tests/test_relaykv_pressure_shadow_quality_report.py
python -m pytest -q tests/test_relaystack_dry_run.py
```

## Notes

- Phase 11-D chain により、pipeline/scoring artifact から fixed-budget selection summary までを no-model/no-GPU で再現できるようになった。
- current design では、Phase 11 の成果は runtime activation ではなく contract と attribution を明確にするための dry-run substrate として扱う。
- RelayKV working-set budget、candidate lineage、selected/rejected/overflow outcomes を artifact として固定できたため、次段では adapter 実装より先に contract consolidation を進めやすくなった。

## Next

- Phase 11.5:
  - RelayStack design contract consolidation
  - data contract
  - adapter contract
  - runtime modes
  - evaluation attribution plan
- runtime adapter にはまだ進まない。
