#!/usr/bin/env python3
"""Minimal HF model smoke for Qwen2.5-Coder AWQ.

Goals:
- tokenizer load
- model load
- CUDA recognition
- device_map / hf_device_map logging
- short generation
- VRAM usage logging
- JSON output

This script intentionally does not import or modify RelayKV internals.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    parser.add_argument("--out", default="results/raw/hf_smoke/qwen25_coder_7b_awq_model_smoke.json")
    parser.add_argument("--prompt", default="Write a Python function that returns the sum of even numbers in a list.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "script": "hf_model_smoke.py",
        "model": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
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
        "error": None,
    }

    try:
        import transformers
        result["env"]["transformers"] = transformers.__version__

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        result["tokenizer"] = {
            "class": tokenizer.__class__.__name__,
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        result["model_load_sec"] = time.perf_counter() - t0
        result["model"] = {
            "class": model.__class__.__name__,
            "device": str(getattr(model, "device", None)),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "config_model_type": getattr(model.config, "model_type", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
            "quantization_config": (
                model.config.quantization_config.to_dict()
                if hasattr(getattr(model.config, "quantization_config", None), "to_dict")
                else getattr(model.config, "quantization_config", None)
            ),
        }

        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[-1])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - gen_t0

        output_tokens_total = int(outputs.shape[-1])
        new_tokens = max(0, output_tokens_total - input_tokens)
        decoded = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)

        result.update({
            "ok": True,
            "input_tokens": input_tokens,
            "output_tokens_total": output_tokens_total,
            "new_tokens": new_tokens,
            "generate_elapsed_sec": elapsed,
            "tokens_per_sec_new": (new_tokens / elapsed) if elapsed > 0 else None,
            "generated_text": decoded,
            "after": {
                "cuda": cuda_snapshot(),
                "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
            },
        })
    except Exception as exc:
        result["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        result["after_error"] = {
            "cuda": cuda_snapshot(),
            "cpu_rss_mib": mib(psutil.Process(os.getpid()).memory_info().rss),
        }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "out": str(out_path), "error": result.get("error")}, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
