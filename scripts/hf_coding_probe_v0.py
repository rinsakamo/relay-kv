#!/usr/bin/env python3
"""Minimal HF coding probe for Qwen2.5-Coder AWQ.

Loads a local or Hub model with Hugging Face Transformers, builds a
deterministic repo-oriented prompt, runs one generation pass, and records
timing, VRAM, and parsed JSON output.

This script intentionally does not import or modify RelayKV internals.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import gc
import json
import os
import platform
import shlex
import time
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
DEFAULT_OUT = "results/raw/hf_coding_probe/qwen25_coder_7b_awq_probe_v0.json"
MAX_SCRIPT_FILES = 32
MAX_RELAYKV_FILES = 16
MAX_DEVLOG_FILES = 16
PROBE_PROFILES: dict[str, dict[str, Any]] = {
    "relaykv_repo_entry": {
        "required_keys": (
            "relevant_files",
            "change_plan",
            "smoke_commands",
            "risks",
        ),
        "schema_example": {
            "relevant_files": ["path"],
            "change_plan": ["step"],
            "smoke_commands": ["command"],
            "risks": ["risk"],
        },
        "task_lines": [
            "- Identify the most relevant files to inspect first.",
            "- Propose a minimal implementation plan for a coding probe.",
            "- Provide smoke commands only, not full experiments.",
            "- Mention the main failure or ambiguity risks.",
        ],
    },
    "relaykv_bug_triage": {
        "required_keys": (
            "relevant_files",
            "likely_causes",
            "minimal_fix_plan",
            "smoke_commands",
            "risks",
        ),
        "schema_example": {
            "relevant_files": ["path"],
            "likely_causes": ["cause"],
            "minimal_fix_plan": ["step"],
            "smoke_commands": ["command"],
            "risks": ["risk"],
        },
        "task_lines": [
            "- Focus on likely causes for a failed smoke or eval behavior.",
            "- Keep the proposed fix plan minimal and conservative.",
            "- Provide smoke commands that confirm the suspected cause.",
            "- Mention the main debugging risks or ambiguities.",
        ],
    },
    "relaykv_smoke_plan": {
        "required_keys": (
            "relevant_files",
            "smoke_commands",
            "expected_observations",
            "failure_modes",
            "risks",
        ),
        "schema_example": {
            "relevant_files": ["path"],
            "smoke_commands": ["command"],
            "expected_observations": ["observation"],
            "failure_modes": ["failure"],
            "risks": ["risk"],
        },
        "task_lines": [
            "- Plan safe validation commands only, not broad experiments.",
            "- State the expected observation for each smoke direction.",
            "- Mention the main failure modes to watch for.",
            "- Keep all commands runnable from the repo root.",
        ],
    },
    "relaykv_result_interpretation": {
        "required_keys": (
            "relevant_files",
            "interpretation",
            "next_checks",
            "smoke_commands",
            "risks",
        ),
        "schema_example": {
            "relevant_files": ["path"],
            "interpretation": ["finding"],
            "next_checks": ["check"],
            "smoke_commands": ["command"],
            "risks": ["risk"],
        },
        "task_lines": [
            "- Interpret an eval or result JSON conservatively.",
            "- Suggest the most relevant follow-up checks.",
            "- Keep smoke commands narrow and runnable from repo root.",
            "- Mention the main interpretation risks or ambiguities.",
        ],
    },
    "relaykv_safe_next_change": {
        "required_keys": (
            "relevant_files",
            "change_plan",
            "safety_constraints",
            "smoke_commands",
            "risks",
        ),
        "schema_example": {
            "relevant_files": ["path"],
            "change_plan": ["step"],
            "safety_constraints": ["constraint"],
            "smoke_commands": ["command"],
            "risks": ["risk"],
        },
        "task_lines": [
            "- Suggest the smallest safe next code change.",
            "- Make safety constraints explicit before proposing commands.",
            "- Provide smoke commands that validate the safe next step.",
            "- Mention the main regression risks or unknowns.",
        ],
    },
}


def get_probe_profile(probe_name: str) -> dict[str, Any]:
    try:
        return PROBE_PROFILES[probe_name]
    except KeyError as exc:
        supported = ", ".join(sorted(PROBE_PROFILES))
        raise ValueError(f"Unsupported probe_name: {probe_name}. Supported probe names: {supported}") from exc


def make_warning(
    warning_type: str,
    message: str,
    *,
    command: str | None = None,
    value: str | None = None,
) -> dict[str, Any]:
    warning: dict[str, Any] = {
        "type": warning_type,
        "message": message,
    }
    if command is not None:
        warning["command"] = command
    if value is not None:
        warning["value"] = value
    return warning


def mib(n: int | float) -> float:
    return float(n) / 1024.0 / 1024.0


def cuda_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    idx = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "allocated_mib": mib(torch.cuda.memory_allocated(idx)),
        "reserved_mib": mib(torch.cuda.memory_reserved(idx)),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(idx)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(idx)),
    }


def collect_repo_context(repo_root: Path) -> dict[str, Any]:
    scripts_dir = repo_root / "scripts"
    relaykv_dir = repo_root / "relaykv"
    notes_dir = repo_root / "notes"
    docs_path = repo_root / "docs" / "experimental_findings.md"

    scripts_all = sorted(
        str(path.relative_to(repo_root))
        for path in scripts_dir.glob("*.py")
        if path.is_file()
    )
    relaykv_all = sorted(
        str(path.relative_to(repo_root))
        for path in relaykv_dir.glob("*.py")
        if path.is_file()
    )
    devlog_all = sorted(
        str(path.relative_to(repo_root))
        for path in notes_dir.glob("devlog*")
        if path.is_file()
    )

    included_files: list[str] = []
    included_files.extend(scripts_all[:MAX_SCRIPT_FILES])
    included_files.extend(relaykv_all[:MAX_RELAYKV_FILES])
    if docs_path.is_file():
        included_files.append(str(docs_path.relative_to(repo_root)))
    included_files.extend(devlog_all[:MAX_DEVLOG_FILES])

    return {
        "repo_root": str(repo_root),
        "num_repo_files": len(scripts_all) + len(relaykv_all) + (1 if docs_path.is_file() else 0) + len(devlog_all),
        "scripts_count": len(scripts_all),
        "relaykv_files_count": len(relaykv_all),
        "included_files": included_files,
        "all_repo_files": sorted(
            set(scripts_all + relaykv_all + ([str(docs_path.relative_to(repo_root))] if docs_path.is_file() else []) + devlog_all)
        ),
        "scripts_all": scripts_all,
        "relaykv_all": relaykv_all,
        "devlog_all": devlog_all,
        "docs_present": docs_path.is_file(),
    }


def repo_context_lines(repo_context: dict[str, Any]) -> list[str]:
    lines = [
        "Repo grounding context:",
        "Actual script files under scripts/:",
    ]
    lines.extend(f"- {path}" for path in repo_context["scripts_all"][:MAX_SCRIPT_FILES])
    lines.append("Actual top-level relaykv/*.py files:")
    lines.extend(f"- {path}" for path in repo_context["relaykv_all"][:MAX_RELAYKV_FILES])
    if repo_context["docs_present"]:
        lines.append("Docs file included:")
        lines.append("- docs/experimental_findings.md")
    if repo_context["devlog_all"]:
        lines.append("notes/devlog files list only:")
        lines.extend(f"- {path}" for path in repo_context["devlog_all"][:MAX_DEVLOG_FILES])
    return lines


def is_plausible_model_value(value: str, repo_root: Path) -> bool:
    if not value:
        return False
    if value.startswith("~/") or value.startswith("/"):
        expanded = Path(value).expanduser()
        return expanded.exists() or value.startswith("~/") or value.startswith("/")
    if "/" in value:
        return True
    candidate = repo_root / value
    return candidate.exists()


def validate_smoke_command(command: object, repo_context: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    repo_root = Path(repo_context["repo_root"])
    known_scripts = set(repo_context["scripts_all"])

    if not isinstance(command, str):
        warnings.append(
            make_warning(
                "SmokeCommandNonString",
                "smoke_commands contains a non-string entry.",
                value=repr(command),
            )
        )
        return warnings

    if not command.strip():
        warnings.append(
            make_warning(
                "SmokeCommandEmpty",
                "smoke_commands contains an empty command string.",
                command=command,
            )
        )
        return warnings

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        warnings.append(
            make_warning(
                "SmokeCommandParseError",
                f"smoke_commands could not be parsed with shlex: {exc}",
                command=command,
            )
        )
        return warnings

    if not parts:
        warnings.append(
            make_warning(
                "SmokeCommandEmpty",
                "smoke_commands contains an empty command after parsing.",
                command=command,
            )
        )
        return warnings

    first = parts[0]
    script_token: str | None = None
    if first in {"python", "python3"}:
        if len(parts) >= 2 and parts[1].startswith("scripts/"):
            script_token = parts[1]
    elif first.startswith("scripts/"):
        script_token = first
        warnings.append(
            make_warning(
                "PythonScriptWithoutPython",
                "Python scripts must be invoked from repo root as python scripts/<existing_script>.py ...",
                command=command,
            )
        )

    if script_token is not None and script_token not in known_scripts:
        warnings.append(
            make_warning(
                "UnknownScriptPath",
                "smoke_commands references a script path that does not exist.",
                command=command,
                value=script_token,
            )
        )

    for idx, part in enumerate(parts):
        if not part.startswith("scripts/"):
            continue
        if part not in known_scripts:
            warnings.append(
                make_warning(
                    "UnknownScriptPath",
                    "smoke_commands references a script path that does not exist.",
                    command=command,
                    value=part,
                )
            )
        if idx == 0:
            continue
        if parts[0] not in {"python", "python3"} and idx == 1:
            warnings.append(
                make_warning(
                    "PythonScriptWithoutPython",
                    "Python scripts must be invoked from repo root as python scripts/<existing_script>.py ...",
                    command=command,
                )
            )
        break

    for idx, part in enumerate(parts):
        if part != "--model":
            continue
        if idx + 1 >= len(parts):
            warnings.append(
                make_warning(
                    "ModelValueMissing",
                    "--model must be followed by a model id or path.",
                    command=command,
                )
            )
            continue
        model_value = parts[idx + 1]
        if not is_plausible_model_value(model_value, repo_root):
            warnings.append(
                make_warning(
                    "InvalidModelValue",
                    "--model value must be an existing local path, a path beginning with ~/ or /, or a Hub id containing '/'.",
                    command=command,
                    value=model_value,
                )
            )

    return warnings


def validate_generated_output(parsed: object, repo_context: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return {"ok": True, "warnings": warnings}

    known_repo_files = set(repo_context["all_repo_files"])
    relevant_files = parsed.get("relevant_files", [])
    if not isinstance(relevant_files, list):
        warnings.append(
            make_warning(
                "RelevantFilesNonList",
                "relevant_files must be a list of strings.",
                value=repr(relevant_files),
            )
        )
        relevant_files = []
    for path in relevant_files:
        if not isinstance(path, str):
            warnings.append(
                make_warning(
                    "RelevantFileNonString",
                    "relevant_files contains a non-string entry.",
                    value=repr(path),
                )
            )
            continue
        if path not in known_repo_files:
            warnings.append(
                make_warning(
                    "UnknownRelevantFile",
                    "relevant_files mentions a path not present in the grounded repo file list.",
                    value=path,
                )
            )

    smoke_commands = parsed.get("smoke_commands", [])
    if not isinstance(smoke_commands, list):
        warnings.append(
            make_warning(
                "SmokeCommandsNonList",
                "smoke_commands must be a list of strings.",
                value=repr(smoke_commands),
            )
        )
        smoke_commands = []
    for command in smoke_commands:
        warnings.extend(validate_smoke_command(command, repo_context))

    return {
        "ok": len(warnings) == 0,
        "warnings": warnings,
    }


def make_schema_text(schema_example: dict[str, list[str]]) -> str:
    lines = ["{"]
    items = list(schema_example.items())
    for idx, (key, value) in enumerate(items):
        suffix = "," if idx < len(items) - 1 else ""
        lines.append(f'  "{key}": {json.dumps(value)}{suffix}')
    lines.append("}")
    return "\n".join(lines)


def make_probe_prompt(
    tokenizer: Any,
    probe_name: str,
    target_tokens: int,
    repo_context: dict[str, Any],
) -> tuple[str, int]:
    profile = get_probe_profile(probe_name)
    instruction = (
        "You are inspecting the RelayKV repository. "
        "Return JSON only. Do not use markdown fences. "
        "Do not add explanatory text before or after the JSON.\n\n"
        "Required JSON object schema:\n"
        f"{make_schema_text(profile['schema_example'])}\n\n"
        "Task constraints:\n"
        "- Prefer minimal diffs.\n"
        "- Do not modify relaykv/ internals.\n"
        "- Reuse existing HF smoke patterns.\n"
        "- Keep the main comparison path easy to reason about.\n"
        "- Only mention files that appear in the provided repo file list.\n"
        "- Only suggest smoke commands using scripts that appear in the provided scripts list.\n"
        "- Do not invent test directories, CLI flags, or files.\n"
        "- If unsure, use existing scripts and conservative commands.\n"
        "- smoke_commands must be runnable from the repo root.\n"
        "- Python scripts must be invoked as: python scripts/<existing_script>.py ...\n"
        "- Do not use source file names as --model values.\n"
        "- For HF model commands, --model must be either Qwen/Qwen2.5-Coder-7B-Instruct-AWQ or a local path placeholder like ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ.\n"
    )

    repo_lines = repo_context_lines(repo_context)
    seed_block = "\n".join(
        [
            f"Probe name: {probe_name}",
            *repo_lines,
            "Requested output:",
            *profile["task_lines"],
            "Conservative smoke command examples:",
            "- python scripts/hf_model_smoke.py --model ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ --max-new-tokens 128",
            "- python scripts/hf_context_length_smoke.py --model ~/work/hf-models/Qwen2.5-Coder-7B-Instruct-AWQ --lengths 4096,8192,16384 --max-new-tokens 64",
            "- python scripts/run_relaykv_pipeline.py --help",
        ]
    )

    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    seed_ids = tokenizer(seed_block, add_special_tokens=False)["input_ids"]
    if not instruction_ids or not seed_ids:
        raise RuntimeError("Tokenizer produced empty prompt ids.")

    available_seed_tokens = target_tokens - len(instruction_ids) - 128
    repeated_seed_text = ""
    if available_seed_tokens <= 0:
        repeated_seed_text = ""
    elif len(seed_ids) >= available_seed_tokens:
        repeated_seed_text = tokenizer.decode(seed_ids[:available_seed_tokens], skip_special_tokens=True)
    else:
        repeated_seed_text = seed_block
        padding_block = "\n".join(
            [
                "Grounding reminder:",
                "- Use only listed files.",
                "- Use only listed scripts in smoke_commands.",
                "- Do not invent repo paths, test directories, or CLI flags.",
                "- Commands must be runnable from repo root.",
                "- Use python scripts/<existing_script>.py ... for Python scripts.",
                "- Do not use source file names as --model values.",
            ]
        )
        padding_ids = tokenizer(padding_block, add_special_tokens=False)["input_ids"]
        if not padding_ids:
            raise RuntimeError("Tokenizer produced empty padding ids.")
        remaining_tokens = available_seed_tokens - len(seed_ids)
        repeat = max(1, (remaining_tokens // len(padding_ids)) + 1)
        repeated_padding_ids = (padding_ids * repeat)[:remaining_tokens]
        repeated_seed_text = seed_block + "\n" + tokenizer.decode(repeated_padding_ids, skip_special_tokens=True)

    user_content = instruction + "\n" + repeated_seed_text
    messages = [
        {"role": "system", "content": "You are a precise coding assistant that obeys output schemas exactly."},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, input_tokens


def extract_json_object(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            first_line = lines[0].strip()
            last_line = lines[-1].strip()
            if first_line in {"```", "```json"} and last_line == "```":
                candidate = "\n".join(lines[1:-1]).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return candidate[start : end + 1]


def parse_generated_json(
    text: str,
    required_keys: tuple[str, ...],
) -> tuple[object | None, bool, dict[str, Any] | str | None]:
    candidate = text.strip()
    if not candidate:
        return None, False, {"type": "EmptyOutput", "message": "Model returned empty text."}

    for payload in (candidate, extract_json_object(candidate)):
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error: dict[str, Any] | str | None = {
                "type": "JSONDecodeError",
                "message": str(exc),
            }
            continue
        if parsed is None:
            return None, False, {
                "type": "ParsedNull",
                "message": "JSON parsed to null instead of an object.",
            }
        if not isinstance(parsed, dict):
            return None, False, {
                "type": "ParsedNonObject",
                "message": f"JSON parsed to {type(parsed).__name__}, expected object.",
            }

        missing_keys = [key for key in required_keys if key not in parsed]
        if missing_keys:
            return parsed, False, {
                "type": "ParsedMissingKeys",
                "message": "Parsed object is missing required keys.",
                "missing_keys": missing_keys,
                "required_keys": list(required_keys),
            }
        return parsed, True, None

    return None, False, last_error if "last_error" in locals() else {
        "type": "JSONDecodeError",
        "message": "No JSON object found in generated text.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--probe-name", default="relaykv_repo_entry")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    profile = get_probe_profile(args.probe_name)
    required_keys = tuple(profile["required_keys"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()
    repo_context = collect_repo_context(repo_root)

    result: dict[str, Any] = {
        "script": "hf_coding_probe_v0.py",
        "probe_name": args.probe_name,
        "required_keys": list(required_keys),
        "model": args.model,
        "out": str(out_path),
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens,
        "prompt_context_paths": [
            *repo_context["included_files"],
        ],
        "repo_context": {
            "repo_root": repo_context["repo_root"],
            "num_repo_files": repo_context["num_repo_files"],
            "scripts_count": repo_context["scripts_count"],
            "relaykv_files_count": repo_context["relaykv_files_count"],
            "included_files": repo_context["included_files"],
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_version_torch": torch.version.cuda,
            "transformers": None,
            "pid": os.getpid(),
        },
        "before": {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        },
        "ok": False,
        "parse_ok": False,
        "parsed": None,
        "parsed_json": None,
        "parse_error": None,
        "generated_text": None,
        "validation": {"ok": True, "warnings": []},
        "error": None,
    }

    try:
        import transformers

        result["env"]["transformers"] = transformers.__version__

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        load_t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        result["load"] = {
            "ok": True,
            "elapsed_sec": time.perf_counter() - load_t0,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
            "hf_device_map": getattr(model, "hf_device_map", None),
            "config": {
                "model_type": getattr(model.config, "model_type", None),
                "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
                "num_attention_heads": getattr(model.config, "num_attention_heads", None),
                "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            },
            "cuda": cuda_snapshot(),
        }

        prompt, prompt_tokens = make_probe_prompt(
            tokenizer=tokenizer,
            probe_name=args.probe_name,
            target_tokens=args.context_tokens,
            repo_context=repo_context,
        )
        result["prompt_preview"] = prompt[:2000]
        result["prompt_preview_truncated"] = len(prompt) > 2000
        result["prompt_target_context_tokens"] = args.context_tokens
        result["actual_prompt_tokens_pre_tensor"] = prompt_tokens

        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_tokens = int(inputs["input_ids"].shape[-1])
        result["input_tokens"] = input_tokens

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - gen_t0

        output_tokens_total = int(outputs.shape[-1])
        new_tokens = max(0, output_tokens_total - input_tokens)
        generated_text = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)

        parsed, parse_ok, parse_error = parse_generated_json(generated_text, required_keys)
        validation = validate_generated_output(parsed, repo_context)

        result.update({
            "ok": True,
            "generated_text": generated_text,
            "parsed": parsed,
            "parsed_json": parsed,
            "parse_ok": parse_ok and parsed is not None,
            "parse_error": None if (parse_ok and parsed is not None) else parse_error,
            "validation": validation,
            "new_tokens": new_tokens,
            "elapsed_sec": elapsed,
            "tokens_per_sec_new": (new_tokens / elapsed) if elapsed > 0 else None,
            "peak_allocated_mib": cuda_snapshot().get("peak_allocated_mib"),
            "peak_reserved_mib": cuda_snapshot().get("peak_reserved_mib"),
            "after": {
                "cuda": cuda_snapshot(),
                "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
            },
        })
    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = {"type": "torch.cuda.OutOfMemoryError", "message": str(exc)}
        result["after_error"] = {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }
        result["peak_allocated_mib"] = cuda_snapshot().get("peak_allocated_mib")
        result["peak_reserved_mib"] = cuda_snapshot().get("peak_reserved_mib")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        result["after_error"] = {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }
        result["peak_allocated_mib"] = cuda_snapshot().get("peak_allocated_mib")
        result["peak_reserved_mib"] = cuda_snapshot().get("peak_reserved_mib")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "parse_ok": result["parse_ok"],
                "probe_name": result["probe_name"],
                "out": str(out_path),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
