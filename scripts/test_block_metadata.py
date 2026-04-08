import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from relaykv import (
    TierManager,
    split_dynamic_cache_layers,
    build_metadata_for_blocks,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "block_metadata_summary.json"
BLOCK_SIZE = 128


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_results_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    prompt = "RelayKV metadata test. " * 64
    inputs = tokenizer(prompt, return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    layers = outputs.past_key_values.layers
    seq_len = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=128)
    split = tier_manager.split_range(seq_len)

    _, cold_cache = split_dynamic_cache_layers(layers, split)
    blocks = cold_cache.blockify(block_size=BLOCK_SIZE)
    metadata = build_metadata_for_blocks(blocks)

    summary = {
        "seq_len": seq_len,
        "cold_range": list(split.cold_range),
        "block_size": BLOCK_SIZE,
        "num_blocks_total": len(blocks),
        "num_metadata_entries": len(metadata),
        "first_five_metadata": [m.summary() for m in metadata[:5]],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()