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
    build_three_tier_selection,
    plan_budget,
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


def infer_kv_bytes_per_token(layers) -> int | None:
    if not layers:
        return None
    sample = layers[0].keys
    if sample.ndim < 4:
        return None
    num_layers = len(layers)
    num_kv_heads = int(sample.shape[1])
    head_dim = int(sample.shape[-1])
    dtype_bytes = sample.element_size()
    return int(2 * num_layers * num_kv_heads * head_dim * dtype_bytes)


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
    available_kv_budget_mib: float,
    kv_working_budget_tokens: int,
    recent_window_tokens: int | None,
    budget_block_size: int,
    retrieval_top_k: int | None,
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
    kv_bytes_per_token = infer_kv_bytes_per_token(layers)

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
        all_blocks=all_blocks,
    )

    top_scores = top_k_blocks(scores, k=min(top_k, len(scores)))
    retrieval_top_k_requested = top_k if retrieval_top_k is None else retrieval_top_k
    budget_plan = plan_budget(
        available_kv_budget_mib=available_kv_budget_mib,
        kv_working_budget_tokens=kv_working_budget_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        recent_window_tokens=hot_window
        if recent_window_tokens is None
        else recent_window_tokens,
        anchor_blocks=anchor_blocks,
        budget_block_size=budget_block_size,
        retrieval_top_k_requested=retrieval_top_k_requested,
        fallback_working_budget_tokens=seq_len_actual,
    )

    selection = build_three_tier_selection(
        seq_len=seq_len_actual,
        hot_window=hot_window,
        anchor_blocks=anchor_blocks,
        block_size=block_size,
        selected_scores=top_scores,
        layer_idx=layer_idx,
    )

    retrieved = retrieve_blocks(all_blocks, top_scores)

    selected_summaries = [s.summary() for s in top_scores]
    selected_block_ranks = list(range(len(selected_summaries)))
    selected_block_ids = [s["block_id"] for s in selected_summaries]
    selected_block_scores = [s["score"] for s in selected_summaries]
    selected_block_spans = [[s["start"], s["end"]] for s in selected_summaries]

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

    # ---- Three-tier accounting / SGLang migration boundary ----

    working_k_len = working_kv.k.shape[2]
    candidate_k_len = candidate_kv.k.shape[2] if candidate_kv.k.numel() > 0 else 0
    full_k_len = full_k.shape[2]
    hot_k_len = hot_k.shape[2]
    cold_k_len = max(0, full_k_len - hot_k_len)

    recent_tokens = sum(s.length for s in selection.recent_spans)
    anchor_tokens = sum(s.length for s in selection.anchor_spans)
    retrieved_tokens = sum(s.length for s in selection.retrieval_spans)
    live_tokens = recent_tokens + anchor_tokens + retrieved_tokens

    # PyTorch prototype path currently materializes retrieval + recent only.
    # Anchor spans are tracked for the SGLang migration boundary, but not yet
    # concatenated into working_kv.
    prototype_materialized_tokens = retrieved_tokens + recent_tokens

    coverage_ratio = candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    working_ratio = working_k_len / full_k_len if full_k_len > 0 else 0.0

    selection_ratio = (
        selection.selected_token_count / full_k_len if full_k_len > 0 else 0.0
    )
    prototype_working_ratio = (
        working_k_len / full_k_len if full_k_len > 0 else 0.0
    )

    budget = {
        "B_total_tokens": full_k_len,
        "B_recent_tokens": recent_tokens,
        "B_anchor_tokens": anchor_tokens,
        "B_retrieval_tokens": retrieved_tokens,
        "B_transient_tokens": 0,
    }

    selection_breakdown = {
        "recent_tokens": recent_tokens,
        "anchor_tokens": anchor_tokens,
        "retrieved_tokens": retrieved_tokens,
        "live_tokens": live_tokens,
        "prototype_materialized_tokens": prototype_materialized_tokens,
    }

    assert budget["B_recent_tokens"] == recent_tokens, (
        f"B_recent_tokens mismatch: {budget['B_recent_tokens']} != {recent_tokens}"
    )

    assert budget["B_anchor_tokens"] == anchor_tokens, (
        f"B_anchor_tokens mismatch: {budget['B_anchor_tokens']} != {anchor_tokens}"
    )

    assert budget["B_retrieval_tokens"] == retrieved_tokens, (
        f"B_retrieval_tokens mismatch: {budget['B_retrieval_tokens']} != {retrieved_tokens}"
    )

    assert candidate_k_len + recent_tokens == working_k_len, (
        "prototype working_k_len mismatch: "
        f"candidate_k_len={candidate_k_len}, "
        f"recent_tokens={recent_tokens}, "
        f"working_k_len={working_k_len}"
    )

    assert (
        budget["B_recent_tokens"]
        + budget["B_anchor_tokens"]
        + budget["B_retrieval_tokens"]
        + budget["B_transient_tokens"]
        <= budget["B_total_tokens"]
    ), f"budget overflow: {budget}"

    summary = {
        "model": model_name,
        "device": device,
        "seq_len_target": seq_len_target,
        "seq_len_actual": seq_len_actual,
        "layer_idx": layer_idx,
        "hot_window": hot_window,
        "anchor_blocks": anchor_blocks,
        "block_size": block_size,
        "top_k": top_k,
        "retrieval_top_k": retrieval_top_k,
        "kv_bytes_per_token": kv_bytes_per_token,
        **budget_plan.summary(),
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
        "mean_abs_diff": attention_result.summary()["mean_abs_diff"],
        "top5_overlap": None,
        "first_divergence_step": None,
        "task_accuracy": None,
        "same_first_code": None,
        "selection_ratio": selection_ratio,
        "prototype_working_ratio": prototype_working_ratio,
        "budget": budget,
        "selection_breakdown": selection_breakdown,
        "three_tier_selection": selection.summary(),
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
        "attention_compare": attention_result.summary(),
        "top_scores": selected_summaries,
        "selected_block_ranks": selected_block_ranks,
        "selected_block_ids": selected_block_ids,
        "selected_block_scores": selected_block_scores,
        "selected_block_spans": selected_block_spans,
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
        "--recent-window",
        type=int,
        dest="hot_window",
        help="Alias for --hot-window.",
    )
    parser.add_argument(
        "--recent-window-tokens",
        type=int,
        default=None,
        help="Budget metadata recent window; does not alter --hot-window.",
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
        help="Number of initial cold blocks to keep as anchor/sink spans",
    )
    parser.add_argument(
        "--available-kv-budget-mib",
        type=float,
        default=0.0,
        help="Available KV working-set budget in MiB for metadata planning.",
    )
    parser.add_argument(
        "--kv-working-budget-tokens",
        type=int,
        default=0,
        help="Explicit KV working-set token budget; overrides MiB estimation.",
    )
    parser.add_argument(
        "--budget-block-size",
        type=int,
        default=128,
        help="Logical RelayKV budget block size in tokens.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Requested retrieval top-k for budget metadata; does not alter --top-k.",
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
        available_kv_budget_mib=args.available_kv_budget_mib,
        kv_working_budget_tokens=args.kv_working_budget_tokens,
        recent_window_tokens=args.recent_window_tokens,
        budget_block_size=args.budget_block_size,
        retrieval_top_k=args.retrieval_top_k,
    )

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
