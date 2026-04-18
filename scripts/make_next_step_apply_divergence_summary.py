#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASE = Path("results/raw/prototype_checks")
PROMPTS = ["prose", "repetitive", "structured"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_consumed_step(apply_data: dict) -> int | None:
    for step in apply_data["step_logs"]:
        if step["layer14"].get("replacement_next_step_apply_consumed") is True:
            return step["step_idx"]
    return None


def first_divergence_step(apply_data: dict, baseline_data: dict) -> int | None:
    for a, b in zip(apply_data["step_logs"], baseline_data["step_logs"]):
        if a["generated_token_id"] != b["generated_token_id"]:
            return a["step_idx"]
    return None


def step_by_idx(data: dict, step_idx: int | None) -> dict | None:
    if step_idx is None:
        return None
    for step in data["step_logs"]:
        if step["step_idx"] == step_idx:
            return step
    return None


def main() -> None:
    rows: list[dict] = []

    for prompt in PROMPTS:
        apply_path = BASE / f"{prompt}_apply_next_step_apply_medium.json"
        baseline_path = BASE / f"{prompt}_baseline_next_step_apply_medium.json"

        apply_data = load_json(apply_path)
        baseline_data = load_json(baseline_path)

        fc = first_consumed_step(apply_data)
        fd = first_divergence_step(apply_data, baseline_data)

        apply_div = step_by_idx(apply_data, fd)
        baseline_div = step_by_idx(baseline_data, fd)

        apply_top5 = apply_div.get("top5_token_ids") if apply_div is not None else None
        baseline_top5 = baseline_div.get("top5_token_ids") if baseline_div is not None else None

        if apply_top5 is not None and baseline_top5 is not None:
            apply_top5_set = set(apply_top5)
            baseline_top5_set = set(baseline_top5)
            top5_overlap_count = len(apply_top5_set & baseline_top5_set)
            same_top5_set = apply_top5_set == baseline_top5_set
        else:
            top5_overlap_count = None
            same_top5_set = None

        divergence_lag = (
            fd - fc
            if fd is not None and fc is not None
            else None
        )

        rows.append(
            {
                "prompt_type": prompt,
                "first_consumed_step": fc,
                "first_divergence_step": fd,
                "divergence_lag": divergence_lag,
                "divergence_step_top1_apply": (
                    apply_div["generated_token_id"] if apply_div is not None else None
                ),
                "divergence_step_top1_baseline": (
                    baseline_div["generated_token_id"] if baseline_div is not None else None
                ),
                "divergence_step_top5_apply": apply_top5,
                "divergence_step_top5_baseline": baseline_top5,
                "top5_overlap_count": top5_overlap_count,
                "same_top5_set": same_top5_set,
            }
        )

    headers = [
        "prompt_type",
        "first_consumed_step",
        "first_divergence_step",
        "divergence_lag",
        "divergence_step_top1_apply",
        "divergence_step_top1_baseline",
        "top5_overlap_count",
        "same_top5_set",
    ]

    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[h]) for h in headers))

    out_path = BASE / "next_step_apply_divergence_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
