import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import os
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_LENGTHS = [2048, 4096, 8192]
MAX_NEW_TOKENS = 32
RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "metrics.csv"


def get_cpu_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def make_prompt(target_tokens: int) -> str:
    base = "RelayKV baseline test. "
    text = base * max(1, target_tokens // max(1, len(base.split())))
    return text


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def append_result(row: dict) -> None:
    file_exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "model",
                "context_length",
                "gpu_peak_mb",
                "cpu_rss_mb",
                "elapsed_ms",
                "output_tokens",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ensure_results_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    for context_length in CONTEXT_LENGTHS:
        prompt = make_prompt(context_length)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length)

        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            torch.cuda.reset_peak_memory_stats()

        cpu_before = get_cpu_rss_mb()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        cpu_after = get_cpu_rss_mb()
        gpu_peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device == "cuda"
            else 0.0
        )

        row = {
            "mode": "baseline",
            "model": MODEL_NAME,
            "context_length": context_length,
            "gpu_peak_mb": round(gpu_peak_mb, 2),
            "cpu_rss_mb": round(max(cpu_before, cpu_after), 2),
            "elapsed_ms": round(elapsed_ms, 2),
            "output_tokens": int(outputs.shape[-1]),
        }
        append_result(row)
        print(row)


if __name__ == "__main__":
    main()