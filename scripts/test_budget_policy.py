import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import BlockScore, build_working_block_budget_decision


def main() -> None:
    scored_blocks = [
        BlockScore(layer_idx=0, block_id=2, start=32, end=48, score=0.92),
        BlockScore(layer_idx=0, block_id=1, start=16, end=32, score=0.88),
        BlockScore(layer_idx=0, block_id=4, start=64, end=80, score=0.75),
        BlockScore(layer_idx=0, block_id=0, start=0, end=16, score=0.40),
    ]

    ok_decision = build_working_block_budget_decision(
        seq_len=96,
        block_size=16,
        total_working_blocks=4,
        recent_blocks=1,
        anchor_blocks=1,
        retrieval_blocks=2,
        scored_blocks=scored_blocks,
    )

    assert ok_decision.budget_ok
    assert ok_decision.fallback_reason is None
    assert ok_decision.selected.recent_block_ids == [5]
    assert ok_decision.selected.anchor_block_ids == [0]
    assert ok_decision.selected.retrieved_block_ids == [2, 1]
    assert ok_decision.selected.working_block_ids == [0, 1, 2, 5]
    assert (
        len(ok_decision.selected.working_block_ids)
        <= ok_decision.budgets.total_working_blocks
    )

    fallback_decision = build_working_block_budget_decision(
        seq_len=64,
        block_size=16,
        total_working_blocks=2,
        recent_blocks=1,
        anchor_blocks=1,
        retrieval_blocks=2,
        scored_blocks=scored_blocks,
    )

    assert not fallback_decision.budget_ok
    assert fallback_decision.fallback_reason is not None
    assert (
        len(fallback_decision.selected.working_block_ids)
        <= fallback_decision.budgets.total_working_blocks
    )

    zero_total_recent_decision = build_working_block_budget_decision(
        seq_len=64,
        block_size=16,
        total_working_blocks=0,
        recent_blocks=1,
        anchor_blocks=0,
        retrieval_blocks=0,
        scored_blocks=scored_blocks,
    )

    assert not zero_total_recent_decision.budget_ok
    assert zero_total_recent_decision.fallback_reason is not None
    assert zero_total_recent_decision.selected.recent_block_ids == []
    assert zero_total_recent_decision.selected.anchor_block_ids == []
    assert zero_total_recent_decision.selected.retrieved_block_ids == []
    assert zero_total_recent_decision.selected.working_block_ids == []
    assert len(zero_total_recent_decision.selected.working_block_ids) == 0

    zero_total_neutral_decision = build_working_block_budget_decision(
        seq_len=64,
        block_size=16,
        total_working_blocks=0,
        recent_blocks=0,
        anchor_blocks=0,
        retrieval_blocks=0,
        scored_blocks=scored_blocks,
    )

    assert zero_total_neutral_decision.budget_ok
    assert zero_total_neutral_decision.fallback_reason is None
    assert zero_total_neutral_decision.selected.recent_block_ids == []
    assert zero_total_neutral_decision.selected.anchor_block_ids == []
    assert zero_total_neutral_decision.selected.retrieved_block_ids == []
    assert zero_total_neutral_decision.selected.working_block_ids == []

    print(
        json.dumps(
            {
                "budget_ok_case": ok_decision.summary(),
                "fallback_case": fallback_decision.summary(),
                "zero_total_recent_case": zero_total_recent_decision.summary(),
                "zero_total_neutral_case": zero_total_neutral_decision.summary(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
