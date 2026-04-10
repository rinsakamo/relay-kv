import sys
import csv
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
RESULTS_DIR = Path("results/raw/sweeps")
RESULTS_CSV = RESULTS_DIR / "attention_sweep_layers.csv"

SEQ_LEN_TARGETS = [1024, 4096]
HOT_WINDOW_VALUES = [128, 256]
BLOCK_SIZE_VALUES = [128, 256]
TOP_K_VALUES = [1, 2, 3]
LAYER_IDXS = [0, 14, 27]
PROMPT_TYPES = ["prose"]


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return tokenizer, model, device


def make_prompt_for_target_tokens(target_tokens: int, prompt_type: str) -> str:
    if prompt_type == "repetitive":
        base = "RelayKV long-context sweep test. "
        unit = base

    elif prompt_type == "prose":
        unit = (
            "RelayKV studies how tiered KV cache reconstruction behaves under long-context inference. "
            "The system keeps recent KV in a hot window, moves older KV to colder storage, "
            "retrieves useful blocks, and rebuilds a working KV set for approximate attention. "
        )

    elif prompt_type == "structured":
        unit = (
            "- topic: RelayKV\n"
            "- goal: tiered KV cache for long-context inference\n"
            "- components: hot KV, cold KV, block metadata, scoring, retrieval, working KV\n"
            "- evaluation: coverage ratio, working ratio, mean absolute difference\n"
        )

    else:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    text = ""
    while True:
        candidate = text + unit
        token_count = len(candidate.split())
        if token_count >= target_tokens:
            return candidate
        text = candidate


def run_once(model, tokenizer, device, seq_len_target: int, hot_window: int, block_size: int, top_k: int, layer_idx: int, prompt_type: str) -> dict:
    prompt = make_prompt_for_target_tokens(seq_len_target, prompt_type)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len_target)

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    layers = outputs.past_key_values.layers
    seq_len = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=hot_window)
    split = tier_manager.split_range(seq_len)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)
    all_blocks = cold_cache.blockify(block_size=block_size)
    metadata = build_metadata_for_blocks(all_blocks)

    query = layers[layer_idx].keys[:, :, -1:, :]  # [1, heads, 1, head_dim]
    layer_metadata = [m for m in metadata if m.layer_idx == layer_idx]

    scores = score_blocks_with_query(layer_metadata, query[:, :, 0, :])
    top_scores = top_k_blocks(scores, k=min(top_k, len(scores)))

    retrieved = retrieve_blocks(all_blocks, top_scores)
    candidate_kv = build_candidate_kv(retrieved)

    hot_k = hot_kv.keys[layer_idx]
    hot_v = hot_kv.values[layer_idx]

    working_kv = build_working_kv(
        candidate_kv=candidate_kv,
        hot_k=hot_k,
        hot_v=hot_v,
        hot_range=split.hot_range,
    )

    full_k = layers[layer_idx].keys
    full_v = layers[layer_idx].values

    result = compare_attention_outputs(
        query=query,
        full_k=full_k,
        full_v=full_v,
        approx_k=working_kv.k,
        approx_v=working_kv.v,
    )
    
    cold_k_len = split.cold_range[1] - split.cold_range[0]
    candidate_k_len = candidate_kv.k.shape[2]
    working_k_len = working_kv.k.shape[2]
    full_k_len = full_k.shape[2]

    coverage_ratio = (
        candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    )
    working_ratio = (
        working_k_len / full_k_len if full_k_len > 0 else 0.0
    )

    return {
        "model": MODEL_NAME,
        "seq_len_target": seq_len_target,
        "seq_len_actual": seq_len,
        "layer_idx": layer_idx,
        "prompt_type": prompt_type,
        "hot_window": hot_window,
        "block_size": block_size,
        "top_k": top_k,
        "num_layer_blocks": len(layer_metadata),
        "num_selected_blocks": len(top_scores),
        "cold_k_len": cold_k_len,
        "candidate_k_len": candidate_k_len,
        "full_k_len": full_k_len,
        "working_k_len": working_k_len,
        "coverage_ratio": coverage_ratio,
        "working_ratio": working_ratio,
        "mean_abs_diff": result.mean_abs_diff,
        "max_abs_diff": result.max_abs_diff,
        "l2_diff": result.l2_diff,
    }


def main() -> None:
    ensure_results_dir()
    tokenizer, model, device = load_model()

    rows = []

    for seq_len_target in SEQ_LEN_TARGETS:
        for prompt_type in PROMPT_TYPES:
            for hot_window in HOT_WINDOW_VALUES:
                for block_size in BLOCK_SIZE_VALUES:
                    for top_k in TOP_K_VALUES:
                        for layer_idx in LAYER_IDXS:
                            row = run_once(
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                seq_len_target=seq_len_target,
                                hot_window=hot_window,
                                block_size=block_size,
                                top_k=top_k,
                                layer_idx=layer_idx,
                                prompt_type=prompt_type,
                            )
                            rows.append(row)
                            print(row)

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {RESULTS_CSV}")


if __name__ == "__main__":
    main()