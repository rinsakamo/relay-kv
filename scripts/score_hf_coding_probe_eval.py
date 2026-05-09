#!/usr/bin/env python3
"""Score HF coding probe eval summaries across context lengths."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_IN = Path("results/processed/hf_coding_probe_eval_summary.json")
DEFAULT_SCORE_OUT = Path("results/processed/hf_coding_probe_eval_score.json")
DEFAULT_SCORE_MD = Path("results/processed/hf_coding_probe_eval_score.md")


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"input summary not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"input summary is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("input summary must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("input summary must contain a top-level 'rows' list")
    return payload


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def as_bool(value: Any) -> bool:
    return bool(value)


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
    return out


def average_non_null(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def score_warning_count(warning_count: int | None) -> float:
    if warning_count is None:
        return 1.0
    return max(0.0, 1.0 - (0.25 * warning_count))


def score_file_focus(file_count: int) -> float:
    if file_count == 0:
        return 0.0
    if 1 <= file_count <= 4:
        return 1.0
    if 5 <= file_count <= 8:
        return 0.75
    if 9 <= file_count <= 12:
        return 0.5
    return 0.25


def score_command(smoke_commands: list[str], validation_ok: bool) -> float:
    if not smoke_commands:
        return 0.0
    if validation_ok:
        return 1.0
    return 0.5


def score_vram(peak_reserved_mib: float | None) -> float | None:
    if peak_reserved_mib is None:
        return None
    if peak_reserved_mib <= 8192:
        return 1.0
    if peak_reserved_mib <= 10240:
        return 0.75
    if peak_reserved_mib <= 11264:
        return 0.5
    return 0.25


def score_speed(tokens_per_sec_new: float | None) -> float | None:
    if tokens_per_sec_new is None:
        return None
    if tokens_per_sec_new >= 15:
        return 1.0
    if tokens_per_sec_new >= 10:
        return 0.75
    if tokens_per_sec_new >= 5:
        return 0.5
    if tokens_per_sec_new > 0:
        return 0.25
    return 0.0


def jaccard(items_a: list[str], items_b: list[str]) -> float | None:
    set_a = set(items_a)
    set_b = set(items_b)
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    probe_name = row.get("probe_name")
    if not isinstance(probe_name, str) or not probe_name:
        probe_name = "default"
    length = as_int(row.get("length"))
    ok = as_bool(row.get("ok"))
    parse_ok = as_bool(row.get("parse_ok"))
    validation_ok = as_bool(row.get("validation_ok"))
    warning_count = as_int(row.get("validation_warning_count"))
    relevant_files = as_str_list(row.get("relevant_files"))
    smoke_commands = as_str_list(row.get("smoke_commands"))
    peak_reserved_mib = as_float(row.get("peak_reserved_mib"))
    tokens_per_sec_new = as_float(row.get("tokens_per_sec_new"))

    runtime_ok_score = 1.0 if (ok and parse_ok and validation_ok) else 0.0
    warning_score = score_warning_count(warning_count)
    file_focus_score = score_file_focus(len(relevant_files))
    command_score = score_command(smoke_commands, validation_ok)
    vram_score = score_vram(peak_reserved_mib)
    speed_score = score_speed(tokens_per_sec_new)
    overall_per_length_score = average_non_null(
        [
            runtime_ok_score,
            warning_score,
            file_focus_score,
            command_score,
            vram_score,
            speed_score,
        ]
    )

    return {
        "probe_name": probe_name,
        "length": length,
        "output_path": row.get("output_path"),
        "ok": ok,
        "parse_ok": parse_ok,
        "validation_ok": validation_ok,
        "validation_warning_count": warning_count if warning_count is not None else 0,
        "error_type": row.get("error_type"),
        "peak_reserved_mib": peak_reserved_mib,
        "tokens_per_sec_new": tokens_per_sec_new,
        "relevant_files": relevant_files,
        "relevant_files_count": len(relevant_files),
        "smoke_commands": smoke_commands,
        "smoke_commands_count": len(smoke_commands),
        "runtime_ok_score": runtime_ok_score,
        "warning_score": warning_score,
        "file_focus_score": file_focus_score,
        "command_score": command_score,
        "vram_score": vram_score,
        "speed_score": speed_score,
        "overall_per_length_score": overall_per_length_score,
    }


def group_rows_by_probe_name(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        probe_name = row.get("probe_name")
        if not isinstance(probe_name, str) or not probe_name:
            probe_name = "default"
            row["probe_name"] = probe_name
        grouped.setdefault(probe_name, []).append(row)
    return grouped


def build_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    file_jaccards: list[float] = []
    command_jaccards: list[float] = []

    for left, right in combinations(rows, 2):
        file_jaccard = jaccard(left.get("relevant_files", []), right.get("relevant_files", []))
        command_jaccard = jaccard(left.get("smoke_commands", []), right.get("smoke_commands", []))
        if file_jaccard is not None:
            file_jaccards.append(file_jaccard)
        if command_jaccard is not None:
            command_jaccards.append(command_jaccard)
        pair_rows.append(
            {
                "length_a": left.get("length"),
                "length_b": right.get("length"),
                "relevant_files_jaccard": file_jaccard,
                "smoke_commands_jaccard": command_jaccard,
            }
        )

    return {
        "pairs": pair_rows,
        "average_file_jaccard": average_non_null(file_jaccards),
        "average_command_jaccard": average_non_null(command_jaccards),
    }


def choose_recommendation(rows: list[dict[str, Any]], stability: dict[str, Any]) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.get("overall_per_length_score") is not None]
    if not scored_rows:
        return {
            "best_length_by_score": None,
            "best_length_reason": "No rows had enough information to compute a score.",
            "notes": [
                "Check that the input summary contains at least one scored row.",
            ],
        }

    top_score = max(row["overall_per_length_score"] for row in scored_rows)
    candidate_rows = [
        row
        for row in scored_rows
        if (top_score - row["overall_per_length_score"]) <= 0.05
    ]
    best = min(
        candidate_rows,
        key=lambda row: row["length"] if row.get("length") is not None else sys.maxsize,
    )

    notes = [
        f"Average file Jaccard: {format_float(stability.get('average_file_jaccard'))}.",
        f"Average command Jaccard: {format_float(stability.get('average_command_jaccard'))}.",
    ]
    if best.get("runtime_ok_score") == 0.0:
        notes.append("Recommended row is still degraded on runtime/parse/validation success.")
    if best.get("validation_warning_count", 0) > 0:
        notes.append("Recommended row still has validation warnings.")
    if best.get("speed_score") is None:
        notes.append("Speed score was unavailable for the recommended row.")
    if best.get("vram_score") is None:
        notes.append("VRAM score was unavailable for the recommended row.")

    return {
        "best_length_by_score": best.get("length"),
        "best_length_reason": (
            f"Top overall score was {format_float(top_score)}; among all rows within 0.05 of that score, "
            f"selected the smallest context length with score {format_float(best.get('overall_per_length_score'))}."
        ),
        "notes": notes,
    }


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def make_markdown_section(
    scored_rows: list[dict[str, Any]],
    stability: dict[str, Any],
    recommendation: dict[str, Any],
    *,
    heading: str | None,
) -> list[str]:
    lines: list[str] = []
    if heading is not None:
        lines.extend([heading, ""])
    lines.extend(
        [
            "## Per-Length Scores",
            "",
            "| length | overall | runtime_ok | warnings | files | commands | file_focus | command | vram | speed | error_type |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in scored_rows:
        lines.append(
            f"| {row.get('length') if row.get('length') is not None else '-'} "
            f"| {format_float(row.get('overall_per_length_score'))} "
            f"| {format_float(row.get('runtime_ok_score'))} "
            f"| {format_float(row.get('warning_score'))} "
            f"| {row.get('relevant_files_count', 0)} "
            f"| {row.get('smoke_commands_count', 0)} "
            f"| {format_float(row.get('file_focus_score'))} "
            f"| {format_float(row.get('command_score'))} "
            f"| {format_float(row.get('vram_score'))} "
            f"| {format_float(row.get('speed_score'))} "
            f"| {row.get('error_type') or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Stability",
            "",
            "| length_a | length_b | relevant_files_jaccard | smoke_commands_jaccard |",
            "|---:|---:|---:|---:|",
        ]
    )
    for pair in stability.get("pairs", []):
        lines.append(
            f"| {pair.get('length_a') if pair.get('length_a') is not None else '-'} "
            f"| {pair.get('length_b') if pair.get('length_b') is not None else '-'} "
            f"| {format_float(pair.get('relevant_files_jaccard'))} "
            f"| {format_float(pair.get('smoke_commands_jaccard'))} |"
        )

    lines.extend(
        [
            "",
            f"Average file Jaccard: {format_float(stability.get('average_file_jaccard'))}",
            "",
            f"Average command Jaccard: {format_float(stability.get('average_command_jaccard'))}",
            "",
            "## Recommendation",
            "",
            f"Best length by score: {recommendation.get('best_length_by_score') if recommendation.get('best_length_by_score') is not None else '-'}",
            "",
            recommendation.get("best_length_reason", ""),
            "",
            "Notes:",
        ]
    )
    for note in recommendation.get("notes", []):
        lines.append(f"- {note}")
    return lines


def make_markdown(
    scored_rows: list[dict[str, Any]],
    stability: dict[str, Any] | None,
    recommendation: dict[str, Any] | None,
    profiles: dict[str, dict[str, Any]] | None,
) -> str:
    lines = [
        "# HF Coding Probe Eval Score",
        "",
    ]
    if profiles is None:
        lines.extend(make_markdown_section(scored_rows, stability or {}, recommendation or {}, heading=None))
        return "\n".join(lines)

    first = True
    for probe_name, payload in profiles.items():
        if not first:
            lines.append("")
        first = False
        lines.extend(
            make_markdown_section(
                payload["rows"],
                payload["stability"],
                payload["recommendation"],
                heading=f"## Profile: {probe_name}",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score HF coding probe eval summaries.")
    parser.add_argument("--summary-in", type=Path, default=DEFAULT_SUMMARY_IN)
    parser.add_argument("--score-out", type=Path, default=DEFAULT_SCORE_OUT)
    parser.add_argument("--score-md", type=Path, default=DEFAULT_SCORE_MD)
    args = parser.parse_args()

    try:
        summary = load_summary(args.summary_in)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ensure_parent_dir(args.score_out)
    ensure_parent_dir(args.score_md)

    raw_rows = summary.get("rows", [])
    scored_rows = [score_row(row if isinstance(row, dict) else {}) for row in raw_rows]
    grouped_rows = group_rows_by_probe_name(scored_rows)
    multi_profile = len(grouped_rows) > 1

    profiles_output: dict[str, dict[str, Any]] | None = None
    stability: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None

    if multi_profile:
        profiles_output = {}
        for probe_name, profile_rows in grouped_rows.items():
            profile_stability = build_stability(profile_rows)
            profile_recommendation = choose_recommendation(profile_rows, profile_stability)
            profiles_output[probe_name] = {
                "rows": profile_rows,
                "stability": profile_stability,
                "recommendation": profile_recommendation,
            }
        stability = {
            "type": "multi_profile",
            "message": "Per-profile stability is available under profiles[<probe_name>].stability.",
        }
        recommendation = {
            "type": "multi_profile",
            "message": "Per-profile recommendations are available under profiles[<probe_name>].recommendation.",
        }
    else:
        stability = build_stability(scored_rows)
        recommendation = choose_recommendation(scored_rows, stability)

    output: dict[str, Any] = {
        "input_summary": str(args.summary_in),
        "rows": scored_rows,
        "stability": stability,
        "recommendation": recommendation,
    }
    if profiles_output is not None:
        output["profiles"] = profiles_output

    args.score_out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = make_markdown(scored_rows, stability, recommendation, profiles_output)
    args.score_md.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"\nsaved json: {args.score_out}")
    print(f"saved markdown: {args.score_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
