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
    score_blocks_with_query,
    top_k_blocks,
    retrieve_blocks,
    build_candidate_kv,
    build_working_kv,
    compare_attention_outputs,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "attention_compare_summary.json"
BLOCK_SIZE = 128
TOP_K = 2
LAYER_IDX = 0


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

    prompt = "RelayKV attention compare test. " * 64
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
    all_blocks = cold_cache.blockify(block_size=BLOCK_SIZE)
    metadata = build_metadata_for_blocks(all_blocks)

    # layer 0 の最後の token を query とする
    # keys shape: [1, heads, seq_len, head_dim]
    query = layers[LAYER_IDX].keys[:, :, -1:, :]   # [1, heads, 1, head_dim]

    layer_metadata = [m for m in metadata if m.layer_idx == LAYER_IDX]
    scores = score_blocks_with_query(layer_metadata, query[:, :, 0, :])
    top_scores = top_k_blocks(scores, k=TOP_K)
    retrieved = retrieve_blocks(all_blocks, top_scores)
    candidate_kv = build_candidate_kv(retrieved)

    hot_k = hot_kv.keys[LAYER_IDX]
    hot_v = hot_kv.values[LAYER_IDX]

    working_kv = build_working_kv(
        candidate_kv=candidate_kv,
        hot_k=hot_k,
        hot_v=hot_v,
        hot_range=split.hot_range,
    )

    # full KV は layer 0 の全 keys / values
    full_k = layers[LAYER_IDX].keys
    full_v = layers[LAYER_IDX].values

    result = compare_attention_outputs(
        query=query,
        full_k=full_k,
        full_v=full_v,
        approx_k=working_kv.k,
        approx_v=working_kv.v,
    )

    summary = {
        "seq_len": seq_len,
        "cold_range": list(split.cold_range),
        "hot_range": list(split.hot_range),
        "block_size": BLOCK_SIZE,
        "top_k": TOP_K,
        "layer_idx": LAYER_IDX,
        "full_k_shape": list(full_k.shape),
        "working_k_shape": list(working_kv.k.shape),
        "attention_compare": result.summary(),
        "top_scores": [s.summary() for s in top_scores],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()