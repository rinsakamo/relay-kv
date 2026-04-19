#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

BASE = Path("results/raw/prototype_checks")
PROMPTS = ["prose", "repetitive", "structured"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--gate-policy", type=str, required=True)
    p.add_argument("--apply-name-suffix", type=str, default="")
    p.add_argument("--baseline-name-suffix", type=str, default="")
    p.add_argument(
        "--prompt-types",
        nargs="+",
        default=["prose", "repetitive", "structured"],
    )
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_path(candidates: list[Path]) -> Path:
    if not candidates:
        raise FileNotFoundError("No matching file found. Candidate list is empty.")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No matching file found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def build_candidates(
    *,
    prompt: str,
    mode: str,   # "apply" or "baseline"
    suffix: str,
    gate_policy: str,
    seq_len: int,
) -> list[Path]:
    if suffix:
        return [
            BASE / f"{prompt}_{mode}_next_step_apply_{suffix}.json",
            BASE / f"{prompt}_{mode}_next_step_apply_{suffix}_v2.json",
            BASE / f"{prompt}_{mode}_next_step_apply_{suffix}_v5.json",
        ]

    return [
        BASE / f"{prompt}_{mode}_next_step_apply_{gate_policy}_v2.json",
        BASE / f"{prompt}_{mode}_next_step_apply_{gate_policy}_{seq_len}_v2.json",
        BASE / f"{prompt}_{mode}_next_step_apply_{gate_policy}.json",
        BASE / f"{prompt}_{mode}_next_step_apply_{gate_policy}_{seq_len}.json",
    ]


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


def raw_gate_pass_steps(data: dict) -> list[int]:
    return [
        step["step_idx"]
        for step in data["step_logs"]
        if step.get("layer14", {}).get("replacement_gate_passed_raw") is True
    ]


def final_gate_pass_steps(data: dict) -> list[int]:
    return [
        step["step_idx"]
        for step in data["step_logs"]
        if step.get("layer14", {}).get("replacement_gate_passed") is True
    ]


def predictor_danger_steps(data: dict) -> list[int]:
    return [
        step["step_idx"]
        for step in data["step_logs"]
        if step.get("layer14", {}).get("predictor_danger") is True
    ]


def predictor_block_requested_steps(data: dict) -> list[int]:
    return [
        step["step_idx"]
        for step in data["step_logs"]
        if step.get("layer14", {}).get("predictor_block_requested") is True
    ]


def predictor_effective_block_steps(data: dict) -> list[int]:
    return [
        step["step_idx"]
        for step in data["step_logs"]
        if step.get("layer14", {}).get("predictor_effective_block") is True
    ]


def first_predictor_danger_step(data: dict) -> int | None:
    steps = predictor_danger_steps(data)
    return steps[0] if steps else None


def layer14_metric(step: dict | None, key: str):
    if step is None:
        return None
    return step.get("layer14", {}).get(key)


def layer14_mean_abs_diff(step: dict | None):
    if step is None:
        return None
    return (
        step.get("layer14", {})
        .get("attention_compare", {})
        .get("mean_abs_diff")
    )

def classify_top5_change(
    apply_top5: list[int] | None,
    baseline_top5: list[int] | None,
    apply_top1: int | None,
    baseline_top1: int | None,
) -> str | None:
    if apply_top5 is None or baseline_top5 is None:
        return None

    apply_set = set(apply_top5)
    baseline_set = set(baseline_top5)
    overlap = len(apply_set & baseline_set)

    if apply_set == baseline_set and apply_top1 != baseline_top1:
        return "rank_flip_same_set"
    if overlap >= 2 and apply_top1 != baseline_top1:
        return "rank_flip_partial_overlap"
    if overlap <= 1:
        return "candidate_shift"
    return "mixed"

def main() -> None:
    args = parse_args()
    if not args.prompt_types:
        raise ValueError("prompt_types is empty")
    rows: list[dict] = []

    for prompt in args.prompt_types:
        apply_suffix = args.apply_name_suffix.strip()
        baseline_suffix = args.baseline_name_suffix.strip()

        apply_candidates = build_candidates(
            prompt=prompt,
            mode="apply",
            suffix=apply_suffix,
            gate_policy=args.gate_policy,
            seq_len=args.seq_len,
        )
        baseline_candidates = build_candidates(
            prompt=prompt,
            mode="baseline",
            suffix=baseline_suffix,
            gate_policy=args.gate_policy,
            seq_len=args.seq_len,
        )

        print(f"[candidate-build] prompt={prompt}")
        print(f"  apply_candidates={apply_candidates}")
        print(f"  baseline_candidates={baseline_candidates}")

        apply_path = resolve_existing_path(apply_candidates)
        baseline_path = resolve_existing_path(baseline_candidates)

        print(f"[load] prompt={prompt} apply={apply_path.name} baseline={baseline_path.name}")

        apply_data = load_json(apply_path)
        baseline_data = load_json(baseline_path)

        gate_info = apply_data.get("replacement_gate", {})
        min_score_margin = gate_info.get("min_score_margin")
        min_gate_step = gate_info.get("min_gate_step")

        raw_passed_steps = raw_gate_pass_steps(apply_data)
        final_passed_steps = final_gate_pass_steps(apply_data)
        predictor_steps = predictor_danger_steps(apply_data)
        predictor_req_steps = predictor_block_requested_steps(apply_data)
        predictor_eff_steps = predictor_effective_block_steps(apply_data)

        fc = first_consumed_step(apply_data)
        fd = first_divergence_step(apply_data, baseline_data)
        fpd = first_predictor_danger_step(apply_data)

        matched_all_generated_tokens = (fd is None)

        apply_div = step_by_idx(apply_data, fd)
        baseline_div = step_by_idx(baseline_data, fd)

        pre_div_step = (fd - 1) if fd is not None and fd > 0 else None
        pre_div_apply = step_by_idx(apply_data, pre_div_step)

        predictor_precedes_divergence = (
            fpd is not None and fd is not None and fpd < fd
            if fd is not None else None
        )
        predictor_hits_pre_div_step = (
            pre_div_apply is not None
            and pre_div_apply.get("layer14", {}).get("predictor_danger") is True
            if fd is not None else None
        )

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

        apply_set = set(apply_top5) if apply_top5 is not None else None
        baseline_set = set(baseline_top5) if baseline_top5 is not None else None
        top5_union_count = (
            len(apply_set | baseline_set)
            if apply_set is not None and baseline_set is not None
            else None
        )
        top5_jaccard = (
            top5_overlap_count / top5_union_count
            if top5_union_count not in (None, 0)
            else None
        )
        top1_changed = (
            apply_div["generated_token_id"] != baseline_div["generated_token_id"]
            if apply_div is not None and baseline_div is not None
            else None
        )
        apply_top1 = apply_div["generated_token_id"] if apply_div is not None else None
        baseline_top1 = baseline_div["generated_token_id"] if baseline_div is not None else None

        apply_top1_in_baseline_top5 = (
            apply_top1 in baseline_top5
            if apply_top1 is not None and baseline_top5 is not None
            else None
        )
        baseline_top1_in_apply_top5 = (
            baseline_top1 in apply_top5
            if baseline_top1 is not None and apply_top5 is not None
            else None
        )
        if apply_top1_in_baseline_top5 and baseline_top1_in_apply_top5:
            change_subtype = "mutual_retention"
        elif apply_top1_in_baseline_top5 or baseline_top1_in_apply_top5:
            change_subtype = "one_sided_retention"
        else:
            change_subtype = "mutual_drop"

        rows.append(
            {
                "seq_len": args.seq_len,
                "gate_policy": args.gate_policy,
                "prompt_type": prompt,
                "min_score_margin": min_score_margin,
                "min_gate_step": min_gate_step,
                "raw_gate_pass_steps": raw_passed_steps,
                "final_gate_pass_steps": final_passed_steps,
                "predictor_danger_steps": predictor_steps,
                "predictor_block_requested_steps": predictor_req_steps,
                "predictor_effective_block_steps": predictor_eff_steps,
                "predictor_effective_block_count": len(predictor_eff_steps),
                "matched_all_generated_tokens": matched_all_generated_tokens,
                "predictor_first_danger_step": fpd,
                "predictor_precedes_divergence": predictor_precedes_divergence,
                "predictor_hits_pre_div_step": predictor_hits_pre_div_step,
                "first_consumed_step": fc,
                "first_divergence_step": fd,
                "divergence_lag": divergence_lag,
                "pre_div_step": pre_div_step,
                "pre_div_mean_abs_diff": layer14_mean_abs_diff(pre_div_apply),
                "pre_div_score_margin": layer14_metric(pre_div_apply, "score_margin"),
                "div_step_mean_abs_diff": layer14_mean_abs_diff(apply_div),
                "div_step_score_margin": layer14_metric(apply_div, "score_margin"),
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
                "top5_change_type": classify_top5_change(
                    apply_top5,
                    baseline_top5,
                    apply_div["generated_token_id"] if apply_div is not None else None,
                    baseline_div["generated_token_id"] if baseline_div is not None else None,
                ),
                "top5_union_count": top5_union_count,
                "top5_jaccard": top5_jaccard,
                "top1_changed": top1_changed,
                "apply_top1_in_baseline_top5": apply_top1_in_baseline_top5,
                "baseline_top1_in_apply_top5": baseline_top1_in_apply_top5,
                "change_subtype": change_subtype,
                "apply_name_suffix": apply_suffix if apply_suffix else None,
                "baseline_name_suffix": baseline_suffix if baseline_suffix else None,

            }
        )

    headers = [
        "seq_len",
        "gate_policy",
        "prompt_type",
        "min_score_margin",
        "min_gate_step",
        "raw_gate_pass_steps",
        "final_gate_pass_steps",
        "predictor_danger_steps",
        "predictor_block_requested_steps",
        "predictor_effective_block_steps",
        "predictor_effective_block_count",
        "matched_all_generated_tokens",
        "predictor_first_danger_step",
        "predictor_precedes_divergence",
        "predictor_hits_pre_div_step",
        "first_consumed_step",
        "first_divergence_step",
        "divergence_lag",
        "pre_div_step",
        "pre_div_mean_abs_diff",
        "pre_div_score_margin",
        "div_step_mean_abs_diff",
        "div_step_score_margin",
        "divergence_step_top1_apply",
        "divergence_step_top1_baseline",
        "top5_overlap_count",
        "same_top5_set",
        "top5_change_type",
        "top5_union_count",
        "top5_jaccard",
        "top1_changed",
        "apply_top1_in_baseline_top5",
        "baseline_top1_in_apply_top5",
        "change_subtype",
        "apply_name_suffix",
        "baseline_name_suffix",
    ]

    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[h]) for h in headers))

    out_path = BASE / f"next_step_apply_divergence_summary_{args.gate_policy}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
