# RelayKV Devlog - 2026-05-16 JST - Phase 10-E pressure shadow quality real smoke

## Date basis

- Date is JST 2026-05-16.

## Summary

- Phase 10-A/B/C/D までに、pressure-triggered shadow quality recommendation、RelayKV quality join report、synthetic artifact chain、existing pipeline artifact validation guide が揃っていた。
- 今回の Phase 10-E では、synthetic chain を再確認したうえで、real RelayKV pipeline artifact の fresh generation を 1 回試行した。
- fresh generation は失敗したため、その失敗を記録した。
- そのうえで、workspace に既存で存在していた `results/raw/prototype_checks/relaykv_pipeline_summary.json` を pressure shadow quality chain に流し、real-artifact path の report 振る舞いを確認した。

## Commands

```bash
git switch main
git pull origin main
git status --short
git switch -c devlog-relaykv-pressure-shadow-quality-real-smoke-2026-05-16

python scripts/run_relaykv_pressure_shadow_quality_chain.py \
  --output-dir /tmp/relaykv_phase10_chain_synthetic

python -m json.tool /tmp/relaykv_phase10_chain_synthetic/chain_summary.json >/dev/null
python -m json.tool /tmp/relaykv_phase10_chain_synthetic/relaykv_pressure_shadow_quality_report.json >/dev/null

python scripts/run_relaykv_pipeline.py --help
python scripts/run_relaykv_pipeline.py --seq-len 8192

python scripts/run_relaykv_pressure_shadow_quality_chain.py \
  --output-dir /tmp/relaykv_phase10_chain_real \
  --relaykv-pipeline-json results/raw/prototype_checks/relaykv_pipeline_summary.json

python -m json.tool /tmp/relaykv_phase10_chain_real/chain_summary.json >/dev/null
python -m json.tool /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json >/dev/null

jq '{
  quality_status,
  shadow_quality_test_recommended,
  pressure_reason,
  notes,
  quality_summary
}' /tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json
```

## Artifacts

- synthetic chain output path: `/tmp/relaykv_phase10_chain_synthetic`
- synthetic final report path: `/tmp/relaykv_phase10_chain_synthetic/relaykv_pressure_shadow_quality_report.json`
- attempted fresh real pipeline summary path: `results/raw/prototype_checks/relaykv_pipeline_summary.json`
- real chain output path: `/tmp/relaykv_phase10_chain_real`
- real final pressure shadow quality report path: `/tmp/relaykv_phase10_chain_real/relaykv_pressure_shadow_quality_report.json`

## Observed result

### Synthetic chain

- `quality_status`: `recommended_quality_within_threshold`
- `shadow_quality_test_recommended`: `true`
- `pressure_reason`: `oom_observed`
- model mismatch: none
- context mismatch: none
- `mean_abs_diff`: `0.005`
- `max_abs_diff`: `0.05`
- `coverage_ratio`: `0.5`
- `working_ratio`: `0.375`
- `seq_len`: `8192`
- `layer_idx`: `14`
- `prompt_type`: `structured`

### Fresh real pipeline generation attempt

- command: `python scripts/run_relaykv_pipeline.py --seq-len 8192`
- result: failed
- observed device banner: `device=cpu`
- failure class: `httpx.ConnectError`
- failure detail: `Temporary failure in name resolution`
- subsequent failure class: `ImportError`
- subsequent failure detail: `requires the protobuf library but it was not found in your environment`

This means a fresh `8192` pipeline artifact could not be produced in the current environment during this smoke run.

### Existing real pipeline artifact joined into the chain

- input artifact: `results/raw/prototype_checks/relaykv_pipeline_summary.json`
- pipeline model: `Qwen/Qwen2.5-1.5B-Instruct`
- pipeline `seq_len_actual`: `1024`
- `quality_status`: `recommended_quality_context_mismatch`
- `shadow_quality_test_recommended`: `true`
- `pressure_reason`: `oom_observed`
- model mismatch: none observed
- context mismatch: observed
- context mismatch note: `pressure_context_mismatch:expected=8192:observed=1024`
- `mean_abs_diff`: `0.01823497749865055`
- `max_abs_diff`: `0.09314870834350586`
- `coverage_ratio`: `0.2857142857142857`
- `working_ratio`: `0.375`
- `layer_idx`: `27`
- `prompt_type`: `null`

## Interpretation

- synthetic chain は引き続き正常で、Phase 10 report path 自体は `within_threshold` まで確認できた。
- fresh real pipeline generation は、この環境では network name resolution と `protobuf` 欠落により停止した。今回は smoke/devlog 目的のため、重い依存修正には進んでいない。
- 既存 real pipeline artifact を chain に流した結果、`recommended_quality_context_mismatch` になった。これは guard が期待通りに効いていることを示す。
- 今回の real-artifact path は threshold-ready validation には到達していない。理由は pressure target context `8192` に対して pipeline artifact 側が `1024` だからである。
- 一方で、quality metrics 自体は report に保持されており、`mean_abs_diff` はしきい値 `0.01` を超え、`max_abs_diff` は `0.10` 未満だった。だが context mismatch のため threshold claim には使っていない。
- threshold calibration の初期示唆としては、まず `seq_len_actual=8192` で model-consistent な fresh pipeline artifact を 1 件以上確保し、その後に `within_threshold` / `exceeds_threshold` の判定分布を見る必要がある。

## Notes

- No actual shadow attention execution beyond existing pipeline artifact generation.
- No RelayKV apply.
- No runtime adapter.
- No attention / KV pool / scheduler changes.

## Next

- Phase 10-F candidate:
  - threshold calibration across multiple seq_len / prompt_type artifacts
  - at least one fresh `seq_len=8192` real pipeline artifact under a resolved local dependency setup
- Phase 11 candidate:
  - fixed-budget working-set dry-run policy
