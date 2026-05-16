#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayKVFixedBudgetConfig,
    build_relaykv_fixed_budget_working_set_decision,
)


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = RelayKVFixedBudgetConfig(
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
    decision = build_relaykv_fixed_budget_working_set_decision(config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision.summary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "decision_state": decision.decision_state,
                "estimated_working_tokens": decision.estimated_working_tokens,
                "dry_run_only": decision.dry_run_only,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
