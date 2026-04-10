import sys
import json
import argparse
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


RESULTS_DIR = Path("results/raw/prototype_checks")


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return tokenizer, model, device


def make_prompt_for_target_tokens(target_tokens: int, prompt_type: str) -> str:
    base = "RelayKV pipeline test. "
    words = base.split()
    chunks = []
    while len(" ".join(chunks).split()) < target_tokens:
        chunks.extend(words)
    return " ".join(chunks)


def run_pipeline(
    model_name: str,
    seq_len_target: int,
    hot_window: int,
    block_size: int,
    top_k: int,
    layer_idx: int,
    scoring_variant: str,
    prompt_type: str,
) -> dict:
    tokenizer, model, device = load_model(model_name)

    prompt = make_prompt_for_target_tokens(seq_len_target, prompt_type)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len_target,
    )

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    layers = outputs.past_key_values.layers
    seq_len_actual = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=hot_window)
    split = tier_manager.split_range(seq_len_actual)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)
    all_blocks = cold_cache.blockify(block_size=block_size)
    metadata = build_metadata_for_blocks(all_blocks)

    query = layers[layer_idx].keys[:, :, -1:, :]  # [1, heads, 1, head_dim]
    layer_metadata = [m for m in metadata if m.layer_idx == layer_idx]

    scores = score_blocks_with_query(
        layer_metadata,
        query[:, :, 0, :],
        variant=scoring_variant,
        norm_weight=1e-3,
    )
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

    attention_result = compare_attention_outputs(
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

    coverage_ratio = candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    working_ratio = working_k_len / full_k_len if full_k_len > 0 else 0.0

    summary = {
        "model": model_name,
        "device": device,
        "seq_len_target": seq_len_target,
        "seq_len_actual": seq_len_actual,
        "layer_idx": layer_idx,
        "hot_window": hot_window,
        "block_size": block_size,
        "top_k": top_k,
        "cold_range": list(split.cold_range),
        "hot_range": list(split.hot_range),
        "num_layers": len(layers),
        "num_all_blocks": len(all_blocks),
        "num_layer_blocks": len(layer_metadata),
        "num_selected_blocks": len(top_scores),
        "cold_k_len": cold_k_len,
        "candidate_k_len": candidate_k_len,
        "full_k_len": full_k_len,
        "working_k_len": working_k_len,
        "coverage_ratio": coverage_ratio,
        "working_ratio": working_ratio,
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
        "attention_compare": attention_result.summary(),
        "top_scores": [s.summary() for s in top_scores],
        "scoring_variant": scoring_variant,
        "prompt_type": prompt_type,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RelayKV prototype pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Target input sequence length",
    )
    parser.add_argument(
        "--hot-window",
        type=int,
        default=128,
        help="Hot KV window size",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Cold block size",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k cold blocks to retrieve",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=27,
        help="Layer index for scoring and attention comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="relaykv_pipeline_summary.json",
        help="Output JSON filename under results/raw/prototype_checks/",
    )
    parser.add_argument(
        "--scoring-variant",
        type=str,
        default="mean_only",
        help="Scoring variant",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="prose",
        help="Prompt type: repetitive, prose, structured",
    )
    args = parser.parse_args()

    ensure_results_dir()

    summary = run_pipeline(
        model_name=args.model,
        seq_len_target=args.seq_len,
        hot_window=args.hot_window,
        block_size=args.block_size,
        top_k=args.top_k,
        layer_idx=args.layer_idx,
        scoring_variant=args.scoring_variant,
        prompt_type=args.prompt_type,
    )

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()