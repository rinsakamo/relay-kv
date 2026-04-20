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
    RetrievedBlock,
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
    repetitive_unit = (
        "RelayKV checks recent context retrieval behavior. "
        "RelayKV checks recent context retrieval behavior. "
        "RelayKV checks recent context retrieval behavior. "
    )

    prose_unit = (
        "RelayKV is a prototype for splitting KV cache into hot and cold regions, "
        "retrieving a smaller working set, and comparing approximate attention outputs "
        "against full attention. The current experiments examine how coverage ratio, "
        "block granularity, and layer difficulty affect approximation quality across "
        "different sequence lengths and prompt styles. "
    )

    structured_unit = (
        "Experiment summary:\n"
        "- system: RelayKV\n"
        "- goal: compare approximate attention against full attention\n"
        "- factors: coverage ratio, block size, hot window, layer index\n"
        "- observation: harder layers may require larger retrieval budgets\n"
        "- note: scoring changes and block granularity should be evaluated separately\n"
    )

    if prompt_type == "repetitive":
        base = repetitive_unit
    elif prompt_type == "prose":
        base = prose_unit
    elif prompt_type == "structured":
        base = structured_unit
    else:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    words = base.split()
    chunks = []

    while len(" ".join(chunks).split()) < target_tokens:
        chunks.extend(words)

    return " ".join(chunks[:target_tokens])


def run_pipeline(
    model_name: str,
    seq_len_target: int,
    hot_window: int,
    block_size: int,
    top_k: int,
    layer_idx: int,
    scoring_variant: str,
    prompt_type: str,
    anchor_blocks: int,
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
    layer_blocks = sorted(
        [b for b in all_blocks if b.layer_idx == layer_idx],
        key=lambda b: b.block_id,
    )

    anchor_count = max(0, min(anchor_blocks, len(layer_blocks)))
    anchor_layer_blocks = layer_blocks[:anchor_count]
    anchor_block_ids = [b.block_id for b in anchor_layer_blocks]
    anchor_block_spans = [[b.start, b.end] for b in anchor_layer_blocks]
    anchor_block_id_set = set(anchor_block_ids)

    retrieval_metadata = [
        m for m in layer_metadata if m.block_id not in anchor_block_id_set
    ]

    scores = score_blocks_with_query(
        retrieval_metadata,
        query[:, :, 0, :],
        variant=scoring_variant,
        norm_weight=1e-3,
        all_blocks=all_blocks,
    )

    top_scores = top_k_blocks(scores, k=min(top_k, len(scores)))
    retrieved = retrieve_blocks(all_blocks, top_scores)

    selected_summaries = [s.summary() for s in top_scores]
    selected_block_ranks = list(range(len(selected_summaries)))
    selected_block_ids = [s["block_id"] for s in selected_summaries]
    selected_block_scores = [s["score"] for s in selected_summaries]
    selected_block_spans = [[s["start"], s["end"]] for s in selected_summaries]

    if retrieved:
        candidate_kv = build_candidate_kv(retrieved)
    else:
        empty_k = query.new_zeros(
            (query.shape[0], query.shape[1], 0, query.shape[3]),
        )
        candidate_layer_idx = layer_idx
        candidate_start = split.cold_range[0]
        if layer_blocks:
            candidate_layer_idx = layer_blocks[0].layer_idx
        candidate_kv = build_candidate_kv(
            [
                RetrievedBlock(
                    layer_idx=candidate_layer_idx,
                    block_id=-1,
                    start=candidate_start,
                    end=candidate_start,
                    k=empty_k,
                    v=empty_k.clone(),
                )
            ]
        )
        candidate_kv.selected_spans = []
        candidate_kv.is_contiguous = True

    anchor_k = None
    anchor_v = None
    if anchor_layer_blocks:
        anchor_k = torch.cat([b.k for b in anchor_layer_blocks], dim=2)
        anchor_v = torch.cat([b.v for b in anchor_layer_blocks], dim=2)

    hot_k = hot_kv.keys[layer_idx]
    hot_v = hot_kv.values[layer_idx]

    working_kv = build_working_kv(
        candidate_kv=candidate_kv,
        hot_k=hot_k,
        hot_v=hot_v,
        hot_range=split.hot_range,
        anchor_k=anchor_k,
        anchor_v=anchor_v,
        anchor_spans=anchor_block_spans,
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
    anchor_k_len = int(anchor_k.shape[2]) if anchor_k is not None else 0
    candidate_k_len = candidate_kv.k.shape[2]
    hot_k_len = hot_k.shape[2]
    working_k_len = working_kv.k.shape[2]
    full_k_len = full_k.shape[2]

    coverage_ratio = candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    working_ratio = working_k_len / full_k_len if full_k_len > 0 else 0.0

    assert set(anchor_block_ids).isdisjoint(set(selected_block_ids))
    assert candidate_k_len == sum((e - s) for s, e in selected_block_spans)
    assert anchor_k_len == sum((e - s) for s, e in anchor_block_spans)
    assert working_k_len == anchor_k_len + candidate_k_len + hot_k_len

    budget = {
        "top_k": top_k,
        "anchor_blocks": anchor_blocks,
    }
    selection_breakdown = {
        "anchor_blocks": len(anchor_block_ids),
        "retrieved_blocks": len(selected_block_ids),
        "hot_tokens": hot_k_len,
        "anchor_tokens": anchor_k_len,
        "candidate_tokens": candidate_k_len,
    }

    summary = {
        "model": model_name,
        "device": device,
        "seq_len_target": seq_len_target,
        "seq_len_actual": seq_len_actual,
        "layer_idx": layer_idx,
        "hot_window": hot_window,
        "block_size": block_size,
        "top_k": top_k,
        "anchor_blocks": anchor_blocks,
        "cold_range": list(split.cold_range),
        "hot_range": list(split.hot_range),
        "num_layers": len(layers),
        "num_all_blocks": len(all_blocks),
        "num_layer_blocks": len(layer_metadata),
        "num_selected_blocks": len(top_scores),
        "cold_k_len": cold_k_len,
        "anchor_k_len": anchor_k_len,
        "candidate_k_len": candidate_k_len,
        "full_k_len": full_k_len,
        "working_k_len": working_k_len,
        "coverage_ratio": coverage_ratio,
        "working_ratio": working_ratio,
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
        "attention_compare": attention_result.summary(),
        "top_scores": selected_summaries,
        "selected_block_ranks": selected_block_ranks,
        "anchor_block_ids": anchor_block_ids,
        "anchor_block_spans": anchor_block_spans,
        "selected_block_ids": selected_block_ids,
        "selected_block_scores": selected_block_scores,
        "selected_block_spans": selected_block_spans,
        "budget": budget,
        "selection_breakdown": selection_breakdown,
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
    parser.add_argument(
        "--anchor-blocks",
        type=int,
        default=0,
        help="Number of leading cold blocks kept as anchors",
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
        anchor_blocks=args.anchor_blocks,
    )

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
