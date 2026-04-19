#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("results/raw/prototype_checks")

FILES = [
    BASE / "next_step_apply_divergence_summary_medium.json",
    BASE / "next_step_apply_divergence_summary_medium_2048.json",
]

def main() -> None:
    rows: list[dict] = []
    for path in FILES:
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.load(f))

    headers = [
        "seq_len",
        "gate_policy",
        "prompt_type",
        "first_consumed_step",
        "first_divergence_step",
        "divergence_lag",
        "top5_overlap_count",
        "top5_jaccard",
        "top5_change_type",
        "apply_top1_in_baseline_top5",
        "baseline_top1_in_apply_top5",
        "change_subtype",
    ]

    print("\t".join(headers))
    for row in sorted(rows, key=lambda r: (r["prompt_type"], r["seq_len"])):
        print("\t".join(str(row.get(h)) for h in headers))

    out_path = BASE / "next_step_apply_divergence_compare.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {out_path}")

if __name__ == "__main__":
    main()