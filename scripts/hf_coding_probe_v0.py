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
import time
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
DEFAULT_OUT = "results/raw/hf_coding_probe/qwen25_coder_7b_awq_probe_v0.json"
REQUIRED_PARSED_KEYS = (
    "relevant_files",
    "change_plan",
    "smoke_commands",
    "risks",
)


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


def repo_context_lines() -> list[str]:
    return [
        "Repo context:",
        "- scripts/run_relaykv_pipeline.py: main prototype comparison path from model KV extraction through retrieval and attention comparison.",
        "- scripts/hf_model_smoke.py: minimal HF model/tokenizer/CUDA/generation smoke with JSON logging.",
        "- scripts/hf_context_length_smoke.py: deterministic HF context-length smoke with per-run OOM capture.",
        "- relaykv/: core prototype package for KV split, block metadata, retrieval, working KV assembly, and attention comparison.",
        "- docs/experimental_findings.md: current prototype findings emphasize coverage-driven error trends and keeping comparisons easy to reason about.",
    ]


def make_probe_prompt(
    tokenizer: Any,
    probe_name: str,
    target_tokens: int,
) -> tuple[str, int]:
    instruction = (
        "You are inspecting the RelayKV repository. "
        "Return JSON only. Do not use markdown fences. "
        "Do not add explanatory text before or after the JSON.\n\n"
        "Required JSON object schema:\n"
        "{\n"
        '  "relevant_files": ["path"],\n'
        '  "change_plan": ["step"],\n'
        '  "smoke_commands": ["command"],\n'
        '  "risks": ["risk"]\n'
        "}\n\n"
        "Task constraints:\n"
        "- Prefer minimal diffs.\n"
        "- Do not modify relaykv/ internals.\n"
        "- Reuse existing HF smoke patterns.\n"
        "- Keep the main comparison path easy to reason about.\n"
    )

    repo_lines = repo_context_lines()
    seed_block = "\n".join(
        [
            f"Probe name: {probe_name}",
            *repo_lines,
            "Requested output:",
            "- Identify the most relevant files to inspect first.",
            "- Propose a minimal implementation plan for a coding probe.",
            "- Provide smoke commands only, not full experiments.",
            "- Mention the main failure or ambiguity risks.",
        ]
    )

    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    seed_ids = tokenizer(seed_block, add_special_tokens=False)["input_ids"]
    if not instruction_ids or not seed_ids:
        raise RuntimeError("Tokenizer produced empty prompt ids.")

    available_seed_tokens = max(1, target_tokens - len(instruction_ids) - 128)
    repeat = max(1, (available_seed_tokens // len(seed_ids)) + 2)
    repeated_seed_ids = (seed_ids * repeat)[:available_seed_tokens]
    repeated_seed_text = tokenizer.decode(repeated_seed_ids, skip_special_tokens=True)

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


def parse_generated_json(text: str) -> tuple[object | None, bool, dict[str, Any] | str | None]:
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

        missing_keys = [key for key in REQUIRED_PARSED_KEYS if key not in parsed]
        if missing_keys:
            return parsed, False, {
                "type": "ParsedMissingKeys",
                "message": "Parsed object is missing required keys.",
                "missing_keys": missing_keys,
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "script": "hf_coding_probe_v0.py",
        "probe_name": args.probe_name,
        "model": args.model,
        "out": str(out_path),
        "max_new_tokens": args.max_new_tokens,
        "context_tokens": args.context_tokens,
        "prompt_context_paths": [
            "scripts/run_relaykv_pipeline.py",
            "scripts/hf_model_smoke.py",
            "scripts/hf_context_length_smoke.py",
            "relaykv/",
            "docs/experimental_findings.md",
        ],
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

        parsed, parse_ok, parse_error = parse_generated_json(generated_text)

        result.update({
            "ok": True,
            "generated_text": generated_text,
            "parsed": parsed,
            "parsed_json": parsed,
            "parse_ok": parse_ok and parsed is not None,
            "parse_error": None if (parse_ok and parsed is not None) else parse_error,
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
