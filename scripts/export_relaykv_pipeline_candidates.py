#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import export_pipeline_candidates_from_json_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export RelayKV pipeline/scoring artifacts into Phase 11-B candidates-json. "
            "Supports list input or dict input with top_scores, block_scores, candidates, "
            "or top_blocks. Accepted row aliases include block_idx/idx, start/end, "
            "layer_idx, and block_score/importance_score. "
            "Tail recent marking requires a full block inventory and is not allowed on top_scores."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--input-key",
        type=str,
        default="auto",
        choices=("auto", "top_scores", "block_scores", "candidates", "top_blocks"),
    )
    parser.add_argument("--default-layer-id", type=int, default=None)
    parser.add_argument("--mark-recent-tail-blocks", type=int, default=0)
    parser.add_argument("--mark-anchor-head-blocks", type=int, default=0)
    args = parser.parse_args()

    candidates = export_pipeline_candidates_from_json_file(
        args.input,
        block_size=args.block_size,
        input_key=args.input_key,
        default_layer_id=args.default_layer_id,
        mark_recent_tail_blocks=args.mark_recent_tail_blocks,
        mark_anchor_head_blocks=args.mark_anchor_head_blocks,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "num_candidates": len(candidates),
                "block_size": args.block_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
