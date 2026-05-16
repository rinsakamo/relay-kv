#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayKVBlockCandidate,
    RelayKVFixedBudgetConfig,
    build_relaykv_fixed_budget_block_selection_decision,
    build_relaykv_fixed_budget_working_set_decision,
    build_synthetic_block_candidates,
)


def _load_candidates(path: Path) -> list[RelayKVBlockCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidates-json must contain a list of candidate objects")
    return [
        RelayKVBlockCandidate(
            block_id=int(item["block_id"]),
            token_start=int(item["token_start"]),
            token_end=int(item["token_end"]),
            score=float(item["score"]) if item.get("score") is not None else None,
            is_recent=bool(item.get("is_recent", False)),
            is_anchor=bool(item.get("is_anchor", False)),
            is_retrieval_candidate=bool(item.get("is_retrieval_candidate", True)),
            layer_id=(
                int(item["layer_id"])
                if item.get("layer_id") is not None
                else None
            ),
            kv_head_group=(
                int(item["kv_head_group"])
                if item.get("kv_head_group") is not None
                else None
            ),
        )
        for item in payload
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run fixed-budget RelayKV block selection. "
            "candidates-json format: "
            '[{"block_id":0,"token_start":0,"token_end":64,"score":1.0,'
            '"is_recent":false,"is_anchor":true,"is_retrieval_candidate":true}]'
        )
    )
    parser.add_argument("--total-working-budget-tokens", type=int, required=True)
    parser.add_argument("--recent-budget-tokens", type=int, default=None)
    parser.add_argument("--anchor-budget-tokens", type=int, default=None)
    parser.add_argument("--retrieved-budget-tokens", type=int, default=None)
    parser.add_argument("--transient-budget-tokens", type=int, default=0)
    parser.add_argument("--recent-ratio", type=float, default=0.50)
    parser.add_argument("--anchor-ratio", type=float, default=0.10)
    parser.add_argument("--retrieved-ratio", type=float, default=0.40)
    parser.add_argument("--min-recent-tokens", type=int, default=128)
    parser.add_argument("--min-anchor-tokens", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--candidates-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed_budget_config = RelayKVFixedBudgetConfig(
        total_working_budget_tokens=args.total_working_budget_tokens,
        recent_budget_tokens=args.recent_budget_tokens,
        anchor_budget_tokens=args.anchor_budget_tokens,
        retrieved_budget_tokens=args.retrieved_budget_tokens,
        transient_budget_tokens=args.transient_budget_tokens,
        recent_ratio=args.recent_ratio,
        anchor_ratio=args.anchor_ratio,
        retrieved_ratio=args.retrieved_ratio,
        min_recent_tokens=args.min_recent_tokens,
        min_anchor_tokens=args.min_anchor_tokens,
        block_size=args.block_size,
    )
    fixed_budget_decision = build_relaykv_fixed_budget_working_set_decision(
        config=fixed_budget_config
    )
    candidates = (
        _load_candidates(args.candidates_json)
        if args.candidates_json is not None
        else build_synthetic_block_candidates(
            num_blocks=args.num_blocks,
            block_size=args.block_size,
        )
    )
    selection_decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=fixed_budget_decision,
        candidates=candidates,
        block_size=args.block_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection_decision.summary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "decision_state": selection_decision.decision_state,
                "estimated_working_tokens": selection_decision.estimated_working_tokens,
                "selected_block_count_by_class": (
                    selection_decision.selected_block_count_by_class
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
