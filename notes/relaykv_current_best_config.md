# RelayKV Current Best Tested Configuration

## Current best tested setting
- seq_len=4096
- block_size=256
- hot_window=256
- scoring_variant=mean_plus_norm
- layer-wise allocation: very-heavy (1,1,7)

## Current evidence
- lightweight scoring changes were not very effective
- block granularity mattered
- layer-wise allocation mattered more
- very-heavy performed best across repetitive, prose, and structured prompts

## Important limitation
Current quick benchmark compares:
- full generation speed
- RelayKV approximation summaries

It does not yet measure true RelayKV-assisted generation speed.

## Next implementation goal
Implement a decode-time RelayKV path that reuses hot/cold split and layer-wise budget allocation during incremental generation.

---