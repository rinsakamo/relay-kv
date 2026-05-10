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
    retrieve_blocks_by_ids,
    build_candidate_kv,
    build_empty_candidate_kv,
    build_working_kv,
    compare_attention_outputs,
    build_three_tier_selection,
    build_working_block_budget_decision,
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
    working_budget_blocks: int | None = None,
    recent_budget_blocks: int | None = None,
    anchor_budget_blocks: int | None = None,
    retrieval_budget_blocks: int | None = None,
    retrieval_exclude_tail_blocks: int | None = None,
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

    budget_policy_enabled = working_budget_blocks is not None
    effective_recent_budget_blocks = recent_budget_blocks or 0
    effective_anchor_budget_blocks = anchor_budget_blocks or 0
    effective_retrieval_budget_blocks = retrieval_budget_blocks or 0
    effective_retrieval_exclude_tail_blocks = retrieval_exclude_tail_blocks or 0
    effective_hot_window = (
        effective_recent_budget_blocks * block_size
        if budget_policy_enabled
        else hot_window
    )
    total_sequence_blocks = (seq_len_actual + block_size - 1) // block_size
    retrieval_excluded_block_ids = (
        list(
            range(
                max(0, total_sequence_blocks - effective_retrieval_exclude_tail_blocks),
                total_sequence_blocks,
            )
        )
        if budget_policy_enabled and effective_retrieval_exclude_tail_blocks > 0
        else []
    )

    tier_manager = TierManager(hot_window=effective_hot_window)
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

    budget_policy_decision = None
    if budget_policy_enabled:
        budget_policy_decision = build_working_block_budget_decision(
            seq_len=seq_len_actual,
            block_size=block_size,
            total_working_blocks=working_budget_blocks,
            recent_blocks=effective_recent_budget_blocks,
            anchor_blocks=effective_anchor_budget_blocks,
            retrieval_blocks=effective_retrieval_budget_blocks,
            scored_blocks=scores,
            retrieval_exclude_block_ids=retrieval_excluded_block_ids,
        )
        selected_retrieval_ids = set(
            budget_policy_decision.selected.retrieved_block_ids
        )
        top_scores = [
            score
            for score in scores
            if score.block_id in selected_retrieval_ids
        ]
    else:
        top_scores = top_k_blocks(scores, k=min(top_k, len(scores)))

    selection = build_three_tier_selection(
        seq_len=seq_len_actual,
        hot_window=effective_hot_window,
        anchor_blocks=(
            effective_anchor_budget_blocks
            if budget_policy_enabled
            else anchor_blocks
        ),
        block_size=block_size,
        selected_scores=top_scores,
        layer_idx=layer_idx,
    )

    if budget_policy_enabled:
        selected_cold_block_ids = (
            budget_policy_decision.selected.anchor_block_ids
            + budget_policy_decision.selected.retrieved_block_ids
        )
        retrieved = retrieve_blocks_by_ids(
            all_blocks,
            layer_idx=layer_idx,
            block_ids=selected_cold_block_ids,
        )
    else:
        retrieved = retrieve_blocks(all_blocks, top_scores)

    selected_summaries = [s.summary() for s in top_scores]
    selected_block_ranks = list(range(len(selected_summaries)))
    selected_block_ids = [s["block_id"] for s in selected_summaries]
    selected_block_scores = [s["score"] for s in selected_summaries]
    selected_block_spans = [[s["start"], s["end"]] for s in selected_summaries]

    hot_k = hot_kv.keys[layer_idx]
    hot_v = hot_kv.values[layer_idx]

    if retrieved:
        candidate_kv = build_candidate_kv(retrieved)
    else:
        candidate_kv = build_empty_candidate_kv(
            layer_idx=layer_idx,
            like_k=hot_k,
            like_v=hot_v,
        )

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

    # PyTorch prototype path materializes selected cold blocks via candidate_kv
    # plus recent hot tokens in working_kv.
    prototype_materialized_tokens = candidate_k_len + recent_tokens

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
        "effective_hot_window": effective_hot_window,
        "anchor_blocks": anchor_blocks,
        "block_size": block_size,
        "top_k": top_k,
        "retrieval_exclude_tail_blocks": effective_retrieval_exclude_tail_blocks,
        "retrieval_excluded_block_ids": retrieval_excluded_block_ids,
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
        "selection_ratio": selection_ratio,
        "prototype_working_ratio": prototype_working_ratio,
        "budget": budget,
        "selection_breakdown": selection_breakdown,
        "three_tier_selection": selection.summary(),
        "budget_policy_decision": (
            budget_policy_decision.summary()
            if budget_policy_decision is not None
            else None
        ),
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
        "--working-budget-blocks",
        type=int,
        default=None,
        help="Total working-set budget in blocks for budget policy mode",
    )
    parser.add_argument(
        "--recent-budget-blocks",
        type=int,
        default=None,
        help="Recent working-set budget in blocks for budget policy mode",
    )
    parser.add_argument(
        "--anchor-budget-blocks",
        type=int,
        default=None,
        help="Anchor working-set budget in blocks for budget policy mode",
    )
    parser.add_argument(
        "--retrieval-budget-blocks",
        type=int,
        default=None,
        help="Retrieved working-set budget in blocks for budget policy mode",
    )
    args = parser.parse_args()

    if args.working_budget_blocks is None:
        budget_args_present = any(
            value is not None
            for value in (
                args.recent_budget_blocks,
                args.anchor_budget_blocks,
                args.retrieval_budget_blocks,
            )
        )
        if budget_args_present:
            parser.error("--working-budget-blocks is required when budget sub-flags are provided")

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
        working_budget_blocks=args.working_budget_blocks,
        recent_budget_blocks=args.recent_budget_blocks,
        anchor_budget_blocks=args.anchor_budget_blocks,
        retrieval_budget_blocks=args.retrieval_budget_blocks,
    )

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
