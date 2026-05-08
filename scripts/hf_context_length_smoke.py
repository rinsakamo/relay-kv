#!/usr/bin/env python3
"""Context length smoke for Qwen2.5-Coder AWQ.

Runs short generation at requested context lengths and records JSON.
OOM or other errors are captured per length without aborting the entire run.
This script intentionally does not import or modify RelayKV internals.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def make_token_target_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    # Build a deterministic coding-relevant context. Tokenize-and-repeat avoids relying on char/token guesses.
    seed = (
        "You are reviewing a Python repository.\n"
        "File: relaykv/example.py\n"
        "Task: identify the relevant files, propose a minimal fix, and give smoke commands.\n"
        "Code fragment:\n"
        "def process_blocks(blocks):\n"
        "    result = []\n"
        "    for block in blocks:\n"
        "        if block.get('score', 0) > 0:\n"
        "            result.append(block['id'])\n"
        "    return result\n\n"
    )
    seed_ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    if not seed_ids:
        raise RuntimeError("Tokenizer produced empty seed ids.")
    repeat = max(1, (target_tokens // len(seed_ids)) + 2)
    ids = (seed_ids * repeat)[:target_tokens]
    prompt_body = tokenizer.decode(ids, skip_special_tokens=True)
    messages = [
        {"role": "system", "content": "You are a concise coding assistant."},
        {"role": "user", "content": prompt_body + "\n\nQuestion: What file should be changed first and what smoke command should be run?"},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    final_ids = tokenizer(chat_text, add_special_tokens=False)["input_ids"]
    return chat_text, len(final_ids)


def run_one(model, tokenizer, target_tokens: int, max_new_tokens: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_context_tokens": target_tokens,
        "ok": False,
        "error": None,
    }

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        prompt, actual_input_tokens = make_token_target_prompt(tokenizer, target_tokens)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        row["input_tokens"] = int(inputs["input_ids"].shape[-1])
        row["actual_prompt_tokens_pre_tensor"] = actual_input_tokens
        row["max_new_tokens"] = max_new_tokens
        row["before"] = {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        total_tokens = int(outputs.shape[-1])
        new_tokens = max(0, total_tokens - int(inputs["input_ids"].shape[-1]))
        row.update({
            "ok": True,
            "elapsed_sec": elapsed,
            "output_tokens_total": total_tokens,
            "new_tokens": new_tokens,
            "tokens_per_sec_new": (new_tokens / elapsed) if elapsed > 0 else None,
            "generated_text_preview": tokenizer.decode(outputs[0][-min(128, total_tokens):], skip_special_tokens=True),
            "after": {
                "cuda": cuda_snapshot(),
                "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
            },
        })
    except torch.cuda.OutOfMemoryError as exc:
        row["error"] = {"type": "torch.cuda.OutOfMemoryError", "message": str(exc)}
        row["after_error"] = {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        row["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        row["after_error"] = {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    parser.add_argument("--out", default="results/raw/hf_smoke/qwen25_coder_7b_awq_context_smoke.json")
    parser.add_argument("--lengths", default="4096,8192,16384")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "script": "hf_context_length_smoke.py",
        "model": args.model,
        "lengths": [int(x.strip()) for x in args.lengths.split(",") if x.strip()],
        "max_new_tokens": args.max_new_tokens,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_version_torch": torch.version.cuda,
            "transformers": None,
            "pid": os.getpid(),
        },
        "load": {},
        "results": [],
    }

    try:
        import transformers
        result["env"]["transformers"] = transformers.__version__
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        result["load"] = {
            "ok": True,
            "elapsed_sec": time.perf_counter() - t0,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
            "hf_device_map": getattr(model, "hf_device_map", None),
            "config": {
                "model_type": getattr(model.config, "model_type", None),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
                "num_attention_heads": getattr(model.config, "num_attention_heads", None),
                "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
                "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
            },
            "cuda": cuda_snapshot(),
        }

        for target in result["lengths"]:
            row = run_one(model, tokenizer, target, args.max_new_tokens)
            result["results"].append(row)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"target": target, "ok": row["ok"], "error": row.get("error")}, ensure_ascii=False))

    except Exception as exc:
        result["load"] = {
            "ok": False,
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = result.get("results", [])
    overall_ok = bool(result.get("load", {}).get("ok")) and bool(rows) and all(
        r.get("ok") for r in rows
    )
    print(json.dumps({"ok": overall_ok, "out": str(out_path)}, ensure_ascii=False, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
