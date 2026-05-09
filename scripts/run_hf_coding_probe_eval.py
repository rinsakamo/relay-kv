#!/usr/bin/env python3
"""Run the HF coding probe across multiple context lengths and summarize results."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
DEFAULT_LENGTHS = "4096,8192,16384"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_PROBE_NAME = "relaykv_repo_entry"
DEFAULT_OUT_DIR = Path("results/raw/hf_coding_probe")
DEFAULT_SUMMARY_OUT = Path("results/processed/hf_coding_probe_eval_summary.json")
DEFAULT_SUMMARY_MD = Path("results/processed/hf_coding_probe_eval_summary.md")
PROBE_SCRIPT = Path("scripts/hf_coding_probe_v0.py")


def parse_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        length = int(token)
        if length <= 0:
            raise ValueError(f"context length must be positive, got {length}")
        lengths.append(length)
    if not lengths:
        raise ValueError("at least one context length is required")
    return lengths


def parse_probe_names(value: str) -> list[str]:
    probe_names = [item.strip() for item in value.split(",") if item.strip()]
    if not probe_names:
        raise ValueError("at least one probe_name is required")
    return probe_names


def normalize_case_name(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.strip().lower()
    if lowered in {"", "none"}:
        return ""
    return value.strip()


def sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "case"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_failure_row(
    *,
    probe_name: str,
    case_name: str,
    case_input_present: bool,
    length: int,
    output_path: Path,
    error_type: str,
    subprocess_returncode: int | None,
) -> dict[str, Any]:
    return {
        "probe_name": probe_name,
        "case_name": case_name,
        "case_input_present": case_input_present,
        "length": length,
        "output_path": str(output_path),
        "ok": False,
        "input_tokens": None,
        "new_tokens": None,
        "elapsed_sec": None,
        "tokens_per_sec_new": None,
        "parse_ok": False,
        "validation_ok": False,
        "validation_warning_count": 0,
        "validation_warnings": [],
        "error_type": error_type,
        "subprocess_returncode": subprocess_returncode,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "relevant_files": [],
        "smoke_commands": [],
    }


def summarize_probe_result(
    probe_name: str,
    case_name: str,
    case_input_present: bool,
    length: int,
    output_path: Path,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if payload is None:
        return make_failure_row(
            probe_name=probe_name,
            case_name=case_name,
            case_input_present=case_input_present,
            length=length,
            output_path=output_path,
            error_type="MissingOutput",
            subprocess_returncode=None,
        )

    parsed = payload.get("parsed")
    validation = payload.get("validation") or {}
    warnings = validation.get("warnings") or []
    error = payload.get("error") or {}

    relevant_files = []
    smoke_commands = []
    if isinstance(parsed, dict):
        relevant_files = parsed.get("relevant_files") or []
        smoke_commands = parsed.get("smoke_commands") or []

    return {
        "probe_name": payload.get("probe_name") or probe_name,
        "case_name": payload.get("case_name") or case_name,
        "case_input_present": bool(payload.get("case_input_present", case_input_present)),
        "length": length,
        "output_path": str(output_path),
        "ok": bool(payload.get("ok", False)),
        "input_tokens": payload.get("input_tokens"),
        "new_tokens": payload.get("new_tokens"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "tokens_per_sec_new": payload.get("tokens_per_sec_new"),
        "parse_ok": bool(payload.get("parse_ok", False)),
        "validation_ok": bool(validation.get("ok", False)),
        "validation_warning_count": len(warnings),
        "validation_warnings": warnings,
        "error_type": error.get("type") or (payload.get("parse_error") or {}).get("type"),
        "subprocess_returncode": None,
        "peak_allocated_mib": payload.get("peak_allocated_mib"),
        "peak_reserved_mib": payload.get("peak_reserved_mib"),
        "relevant_files": relevant_files,
        "smoke_commands": smoke_commands,
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def make_markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| case_name | probe_name | length | ok | parse_ok | validation_ok | warnings | input_tokens | new_tokens | elapsed_sec | new_tok_s | error_type | output_path |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_name'] or '-'} "
            f"| {row['probe_name']} "
            f"| {row['length']} "
            f"| {format_number(row['ok'])} "
            f"| {format_number(row['parse_ok'])} "
            f"| {format_number(row['validation_ok'])} "
            f"| {format_number(row['validation_warning_count'])} "
            f"| {format_number(row['input_tokens'])} "
            f"| {format_number(row['new_tokens'])} "
            f"| {format_number(row['elapsed_sec'])} "
            f"| {format_number(row['tokens_per_sec_new'])} "
            f"| {format_number(row['error_type'])} "
            f"| {row['output_path']} |"
        )
    return "\n".join(lines)


def build_output_path(out_dir: Path, case_name: str, probe_name: str, length: int) -> Path:
    if case_name:
        safe_case_name = sanitize_filename_component(case_name)
        filename = f"qwen25_coder_7b_awq_probe_eval_{safe_case_name}_{probe_name}_{length}.json"
    else:
        filename = f"qwen25_coder_7b_awq_probe_eval_{probe_name}_{length}.json"
    return out_dir / filename


def run_one_eval(
    *,
    python_bin: str,
    model: str,
    probe_name: str,
    case_name: str,
    case_text: str | None,
    case_file: str | None,
    length: int,
    max_new_tokens: int,
    output_path: Path,
) -> tuple[dict[str, Any], int]:
    ensure_parent_dir(output_path)
    if output_path.exists():
        output_path.unlink()

    case_input_present = bool((case_text and case_text.strip()) or (case_file and case_file.strip()))
    command = [
        python_bin,
        str(PROBE_SCRIPT),
        "--model",
        model,
        "--out",
        str(output_path),
        "--max-new-tokens",
        str(max_new_tokens),
        "--context-tokens",
        str(length),
        "--probe-name",
        probe_name,
    ]
    if case_text:
        command.extend(["--case-text", case_text])
    if case_file:
        command.extend(["--case-file", case_file])
    if case_name:
        command.extend(["--case-name", case_name])

    print(
        f"[run] case_name={case_name or 'none'} probe_name={probe_name} length={length} command={shlex.join(command)}",
        flush=True,
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.stdout:
        print(
            f"[stdout] case_name={case_name or 'none'} probe_name={probe_name} length={length}\n{completed.stdout}",
            end="" if completed.stdout.endswith("\n") else "\n",
            flush=True,
        )
    if completed.stderr:
        print(
            f"[stderr] case_name={case_name or 'none'} probe_name={probe_name} length={length}\n{completed.stderr}",
            end="" if completed.stderr.endswith("\n") else "\n",
            flush=True,
        )
    print(
        f"[done] case_name={case_name or 'none'} probe_name={probe_name} length={length} returncode={completed.returncode} output={output_path}",
        flush=True,
    )

    if not output_path.exists():
        row = make_failure_row(
            probe_name=probe_name,
            case_name=case_name,
            case_input_present=case_input_present,
            length=length,
            output_path=output_path,
            error_type="MissingOutputAfterSubprocess",
            subprocess_returncode=completed.returncode,
        )
        return row, completed.returncode

    try:
        payload = read_json_if_present(output_path)
    except json.JSONDecodeError:
        row = make_failure_row(
            probe_name=probe_name,
            case_name=case_name,
            case_input_present=case_input_present,
            length=length,
            output_path=output_path,
            error_type="OutputJsonDecodeError",
            subprocess_returncode=completed.returncode,
        )
        return row, completed.returncode
    except OSError:
        row = make_failure_row(
            probe_name=probe_name,
            case_name=case_name,
            case_input_present=case_input_present,
            length=length,
            output_path=output_path,
            error_type="OutputJsonReadError",
            subprocess_returncode=completed.returncode,
        )
        return row, completed.returncode
    except Exception:
        row = make_failure_row(
            probe_name=probe_name,
            case_name=case_name,
            case_input_present=case_input_present,
            length=length,
            output_path=output_path,
            error_type="OutputJsonReadError",
            subprocess_returncode=completed.returncode,
        )
        return row, completed.returncode

    try:
        row = summarize_probe_result(probe_name, case_name, case_input_present, length, output_path, payload)
    except Exception:
        row = make_failure_row(
            probe_name=probe_name,
            case_name=case_name,
            case_input_present=case_input_present,
            length=length,
            output_path=output_path,
            error_type="OutputJsonReadError",
            subprocess_returncode=completed.returncode,
        )
        return row, completed.returncode

    row["subprocess_returncode"] = completed.returncode
    if row["error_type"] is None and completed.returncode != 0:
        row["error_type"] = "SubprocessNonZeroExit"
    return row, completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the HF coding probe across multiple context lengths.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--probe-name", default=DEFAULT_PROBE_NAME)
    parser.add_argument("--probe-names", default=None)
    parser.add_argument("--case-text", default=None)
    parser.add_argument("--case-file", default=None)
    parser.add_argument("--case-name", default="")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)
    probe_names = parse_probe_names(args.probe_names) if args.probe_names else [args.probe_name]
    case_name = normalize_case_name(args.case_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_dir(args.summary_out)
    ensure_parent_dir(args.summary_md)

    rows: list[dict[str, Any]] = []
    return_codes: dict[str, int] = {}

    for probe_name in probe_names:
        for length in lengths:
            output_path = build_output_path(args.out_dir, case_name, probe_name, length)
            row, return_code = run_one_eval(
                python_bin=args.python,
                model=args.model,
                probe_name=probe_name,
                case_name=case_name,
                case_text=args.case_text,
                case_file=args.case_file,
                length=length,
                max_new_tokens=args.max_new_tokens,
                output_path=output_path,
            )
            rows.append(row)
            return_codes[f"{case_name or 'none'}:{probe_name}:{length}"] = return_code

    summary: dict[str, Any] = {
        "script": "run_hf_coding_probe_eval.py",
        "probe_script": str(PROBE_SCRIPT),
        "model": args.model,
        "lengths": lengths,
        "max_new_tokens": args.max_new_tokens,
        "probe_name": args.probe_name,
        "probe_names": probe_names,
        "case_name": case_name,
        "case_input_present": bool((args.case_text and args.case_text.strip()) or (args.case_file and args.case_file.strip())),
        "out_dir": str(args.out_dir),
        "summary_md": str(args.summary_md),
        "rows": rows,
        "return_codes": return_codes,
    }

    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = make_markdown_summary(rows)
    args.summary_md.write_text(markdown + "\n", encoding="utf-8")

    print(f"saved json: {args.summary_out}")
    print(f"saved markdown: {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
