import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from relaykv import TierManager, split_dynamic_cache_layers


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "cold_cache_summary.json"


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

    prompt = "RelayKV KV split test. " * 64
    inputs = tokenizer(prompt, return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    layers = outputs.past_key_values.layers
    seq_len = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=128)
    split = tier_manager.split_range(seq_len)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)

    summary = {
        "seq_len": seq_len,
        "cold_range": list(split.cold_range),
        "hot_range": list(split.hot_range),
        "num_layers": len(layers),
        "num_cold_segments": len(cold_cache.segments),
        "first_hot_k_shape": list(hot_kv.keys[0].shape),
        "first_hot_v_shape": list(hot_kv.values[0].shape),
        "cold_segments": cold_cache.summary()[:3],  # 最初の3層だけ保存
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()