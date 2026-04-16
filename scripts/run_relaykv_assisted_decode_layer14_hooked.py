#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any

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

from dataclasses import dataclass
@dataclass
class Layer14InjectionState:
    enabled: bool = False
    injection_tensor: torch.Tensor | None = None

    attn_hook_seen: bool = False
    attn_last_seen_shape: list[int] | None = None
    attn_last_seen_dtype: str | None = None
    attn_last_seen_device: str | None = None

    layer_hook_seen: bool = False
    layer_last_seen_shape: list[int] | None = None
    layer_last_seen_dtype: str | None = None
    layer_last_seen_device: str | None = None

    oproj_pre_hook_seen: bool = False
    oproj_pre_last_seen_shape: list[int] | None = None
    oproj_pre_last_seen_dtype: str | None = None
    oproj_pre_last_seen_device: str | None = None

    oproj_post_hook_seen: bool = False
    oproj_post_last_seen_shape: list[int] | None = None
    oproj_post_last_seen_dtype: str | None = None
    oproj_post_last_seen_device: str | None = None


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
    chunks: list[str] = []
    while len(" ".join(chunks).split()) < target_tokens:
        chunks.extend(words)
    return " ".join(chunks[:target_tokens])


def run_prefill(
    *,
    model,
    tokenizer,
    device: str,
    prompt: str,
    seq_len: int,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    return inputs, outputs

def should_apply_layer14_replacement(
    *,
    selected_block_ids: list[int],
    candidate_is_contiguous: bool,
    mean_abs_diff: float,
    mean_abs_diff_threshold: float,
) -> bool:
    if not candidate_is_contiguous:
        return False

    if len(selected_block_ids) != 1:
        return False

    if selected_block_ids[0] not in (14, 15):
        return False

    if mean_abs_diff > mean_abs_diff_threshold:
        return False

    return True

def check_tensor_writeback_ready(
    *,
    full_attention_output: torch.Tensor,
    approx_attention_output: torch.Tensor,
) -> tuple[bool, str | None]:
    if full_attention_output.shape != approx_attention_output.shape:
        return False, "shape_mismatch"

    if full_attention_output.device != approx_attention_output.device:
        return False, "device_mismatch"

    if full_attention_output.dtype != approx_attention_output.dtype:
        return False, "dtype_mismatch"

    return True, None

def prepare_layer14_actual_injection(
    *,
    injection_tensor: torch.Tensor,
    injection_target: str,
) -> tuple[bool, str | None]:
    if injection_target != "layer14_attention_output":
        return False, "unsupported_injection_target"

    if injection_tensor.ndim != 4:
        return False, "unexpected_injection_tensor_rank"

    if injection_tensor.shape[-1] != 128:
        return False, "unexpected_hidden_dim"

    return True, None

def apply_layer14_actual_injection(
    *,
    injection_tensor: torch.Tensor,
    injection_target: str,
) -> tuple[bool, str | None]:
    if injection_target != "layer14_attention_output":
        return False, "unsupported_injection_target"

    # Actual forward-path wiring is not implemented yet.
    # This helper exists to define the callsite and wiring contract.
    return False, "not_wired_yet"

def kv_heads_to_hidden_state_bridge(
    kv_head_output: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> torch.Tensor:
    """
    kv_head_output: [batch, kv_heads, q_len, head_dim]
    returns:        [batch, q_len, hidden_size]
    """
    if kv_head_output.ndim != 4:
        raise ValueError(f"Expected 4D kv_head_output, got shape={list(kv_head_output.shape)}")

    batch_size, kv_heads, q_len, head_dim = kv_head_output.shape

    if kv_heads != num_key_value_heads:
        raise ValueError(
            f"KV head mismatch: tensor has {kv_heads}, expected {num_key_value_heads}"
        )

    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} must be divisible by "
            f"num_key_value_heads={num_key_value_heads}"
        )

    repeat_factor = num_attention_heads // num_key_value_heads

    # [B, kv_heads, Q, D] -> [B, attn_heads, Q, D]
    expanded = kv_head_output.repeat_interleave(repeat_factor, dim=1)

    # [B, attn_heads, Q, D] -> [B, Q, attn_heads, D]
    expanded = expanded.permute(0, 2, 1, 3).contiguous()

    # [B, Q, attn_heads, D] -> [B, Q, hidden_size]
    hidden_state = expanded.view(batch_size, q_len, num_attention_heads * head_dim)
    return hidden_state

def make_layer14_attention_probe_hook(state: Layer14InjectionState):
    def _hook(module, inputs, output):
        state.attn_hook_seen = True

        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output

        state.attn_last_seen_shape = list(tensor.shape)
        state.attn_last_seen_dtype = str(tensor.dtype)
        state.attn_last_seen_device = str(tensor.device)
        return output

    return _hook


def make_layer14_block_probe_hook(state: Layer14InjectionState):
    def _hook(module, inputs, output):
        state.layer_hook_seen = True

        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output

        state.layer_last_seen_shape = list(tensor.shape)
        state.layer_last_seen_dtype = str(tensor.dtype)
        state.layer_last_seen_device = str(tensor.device)
        return output

    return _hook

def make_layer14_oproj_pre_hook(state: Layer14InjectionState):
    def _hook(module, inputs):
        state.oproj_pre_hook_seen = True

        tensor = inputs[0]
        state.oproj_pre_last_seen_shape = list(tensor.shape)
        state.oproj_pre_last_seen_dtype = str(tensor.dtype)
        state.oproj_pre_last_seen_device = str(tensor.device)

    return _hook


def make_layer14_oproj_post_hook(state: Layer14InjectionState):
    def _hook(module, inputs, output):
        state.oproj_post_hook_seen = True

        tensor = output[0] if isinstance(output, tuple) else output
        state.oproj_post_last_seen_shape = list(tensor.shape)
        state.oproj_post_last_seen_dtype = str(tensor.dtype)
        state.oproj_post_last_seen_device = str(tensor.device)

        return output

    return _hook

def evaluate_layer14_step(
    *,
    layers,
    hot_window: int,
    block_size: int,
    scoring_variant: str,
    norm_weight: float = 1e-3,
    mean_abs_diff_threshold: float = 0.002,
) -> dict[str, Any]:
    """
    Fixed-prefill-style step evaluation for layer 14 only.
    This function computes:
    - RelayKV working set for layer 14
    - full vs approx attention comparison
    """

    layer_idx = 14
    top_k = 1

    seq_len_actual = layers[0].keys.shape[2]

    tier_manager = TierManager(hot_window=hot_window)
    split = tier_manager.split_range(seq_len_actual)

    hot_kv, cold_cache = split_dynamic_cache_layers(layers, split)
    all_blocks = cold_cache.blockify(block_size=block_size)
    metadata = build_metadata_for_blocks(all_blocks)

    layer_metadata = [m for m in metadata if m.layer_idx == layer_idx]
    query = layers[layer_idx].keys[:, :, -1:, :]
    full_k = layers[layer_idx].keys
    full_v = layers[layer_idx].values

    scores = score_blocks_with_query(
        layer_metadata,
        query[:, :, 0, :],
        variant=scoring_variant,
        norm_weight=norm_weight,
        all_blocks=all_blocks,
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

    attention_result = compare_attention_outputs(
        query=query,
        full_k=full_k,
        full_v=full_v,
        approx_k=working_kv.k,
        approx_v=working_kv.v,
    )

    attention_summary = attention_result.summary()
    full_attention_output = attention_result.full_output
    approx_attention_output = attention_result.approx_output

    full_shape = list(full_attention_output.shape)
    approx_shape = list(approx_attention_output.shape)

    full_num_heads = int(full_shape[1]) if len(full_shape) >= 4 else None
    full_head_dim = int(full_shape[-1]) if len(full_shape) >= 4 else None
    approx_num_heads = int(approx_shape[1]) if len(approx_shape) >= 4 else None
    approx_head_dim = int(approx_shape[-1]) if len(approx_shape) >= 4 else None

    full_flattened_last_dim = (
        int(full_num_heads * full_head_dim)
        if full_num_heads is not None and full_head_dim is not None
        else None
    )
    approx_flattened_last_dim = (
        int(approx_num_heads * approx_head_dim)
        if approx_num_heads is not None and approx_head_dim is not None
        else None
    )

    candidate_is_contiguous = bool(candidate_kv.summary()["is_contiguous"])
    selected_block_ids = [s.block_id for s in top_scores]

    replacement_gate_passed = should_apply_layer14_replacement(
        selected_block_ids=selected_block_ids,
        candidate_is_contiguous=candidate_is_contiguous,
        mean_abs_diff=float(attention_summary["mean_abs_diff"]),
        mean_abs_diff_threshold=mean_abs_diff_threshold,
    )

    cold_k_len = split.cold_range[1] - split.cold_range[0]
    candidate_k_len = int(candidate_kv.k.shape[2])
    working_k_len = int(working_kv.k.shape[2])
    full_k_len = int(full_k.shape[2])

    coverage_ratio = candidate_k_len / cold_k_len if cold_k_len > 0 else 0.0
    working_ratio = working_k_len / full_k_len if full_k_len > 0 else 0.0

    return {
        "layer_idx": layer_idx,
        "top_k": top_k,
        "selected_block_ids": selected_block_ids,
        "selected_block_spans": [[s.start, s.end] for s in top_scores],
        "selected_block_scores": [float(s.score) for s in top_scores],
        "candidate_k_len": candidate_k_len,
        "working_k_len": working_k_len,
        "coverage_ratio": coverage_ratio,
        "working_ratio": working_ratio,
        "candidate_kv": candidate_kv.summary(),
        "working_kv": working_kv.summary(),
        "attention_compare": attention_summary,
        "attention_tensor_breakdown": {
            "full_shape": full_shape,
            "approx_shape": approx_shape,
            "full_num_heads": full_num_heads,
            "full_head_dim": full_head_dim,
            "full_flattened_last_dim": full_flattened_last_dim,
            "approx_num_heads": approx_num_heads,
            "approx_head_dim": approx_head_dim,
            "approx_flattened_last_dim": approx_flattened_last_dim,
        },
        "replacement_ready": True,
        "replacement_gate_passed": replacement_gate_passed,
        "replacement_selected": replacement_gate_passed,
        "_full_attention_output": full_attention_output,
        "_approx_attention_output": approx_attention_output,
    }


def run_decode_loop_layer14_assisted_ready(
    *,
    model,
    tokenizer,
    prefill_inputs,
    prefill_outputs,
    max_new_tokens: int,
    hot_window: int,
    block_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    scoring_variant: str,
    mean_abs_diff_threshold: float,
    enable_replacement_hook: bool = False,
) -> dict[str, Any]:
    input_ids = prefill_inputs["input_ids"]
    attention_mask = prefill_inputs["attention_mask"]
    past_key_values = prefill_outputs.past_key_values

    generated_token_ids: list[int] = []
    step_logs: list[dict[str, Any]] = []

    start = time.perf_counter()

    current_input_ids = input_ids[:, -1:]
    current_attention_mask = attention_mask

    injection_state = Layer14InjectionState()

    layer14_attn_module = model.model.layers[14].self_attn
    layer14_block_module = model.model.layers[14]

    layer14_attn_probe_handle = layer14_attn_module.register_forward_hook(
        make_layer14_attention_probe_hook(injection_state)
    )
    layer14_block_probe_handle = layer14_block_module.register_forward_hook(
        make_layer14_block_probe_hook(injection_state)
    )
    injection_state = Layer14InjectionState()

    layer14_attn_module = model.model.layers[14].self_attn
    layer14_block_module = model.model.layers[14]
    layer14_oproj_module = model.model.layers[14].self_attn.o_proj

    layer14_attn_probe_handle = layer14_attn_module.register_forward_hook(
        make_layer14_attention_probe_hook(injection_state)
    )
    layer14_block_probe_handle = layer14_block_module.register_forward_hook(
        make_layer14_block_probe_hook(injection_state)
    )
    layer14_oproj_pre_handle = layer14_oproj_module.register_forward_pre_hook(
        make_layer14_oproj_pre_hook(injection_state)
    )
    layer14_oproj_post_handle = layer14_oproj_module.register_forward_hook(
        make_layer14_oproj_post_hook(injection_state)
    )

    for step_idx in range(max_new_tokens):
        injection_state.attn_hook_seen = False
        injection_state.attn_last_seen_shape = None
        injection_state.attn_last_seen_dtype = None
        injection_state.attn_last_seen_device = None

        injection_state.layer_hook_seen = False
        injection_state.layer_last_seen_shape = None
        injection_state.layer_last_seen_dtype = None
        injection_state.layer_last_seen_device = None

        injection_state.oproj_pre_hook_seen = False
        injection_state.oproj_pre_last_seen_shape = None
        injection_state.oproj_pre_last_seen_dtype = None
        injection_state.oproj_pre_last_seen_device = None

        injection_state.oproj_post_hook_seen = False
        injection_state.oproj_post_last_seen_shape = None
        injection_state.oproj_post_last_seen_dtype = None
        injection_state.oproj_post_last_seen_device = None

        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        generated_token_ids.append(next_token_id)

        past_key_values = outputs.past_key_values
        layers = past_key_values.layers

        layer14_eval = evaluate_layer14_step(
            layers=layers,
            hot_window=hot_window,
            block_size=block_size,
            scoring_variant=scoring_variant,
            mean_abs_diff_threshold=mean_abs_diff_threshold,
        )

        replacement_hook_enabled = enable_replacement_hook
        replacement_hook_triggered = (
            replacement_hook_enabled
            and layer14_eval["replacement_gate_passed"]
        )
        replacement_hook_mode = (
            "tensor_writeback_ready_only" if replacement_hook_enabled else "disabled"
        )

        layer14_eval["replacement_hook_enabled"] = replacement_hook_enabled
        layer14_eval["replacement_hook_triggered"] = replacement_hook_triggered
        layer14_eval["replacement_hook_mode"] = replacement_hook_mode

        if replacement_hook_triggered:
            tensor_writeback_ready, tensor_writeback_reason = check_tensor_writeback_ready(
                full_attention_output=layer14_eval["_full_attention_output"],
                approx_attention_output=layer14_eval["_approx_attention_output"],
            )

            layer14_eval["replacement_tensor_writeback_ready"] = tensor_writeback_ready
            layer14_eval["replacement_tensor_writeback_reason"] = tensor_writeback_reason

            if tensor_writeback_ready:
                writeback_tensor = layer14_eval["_approx_attention_output"]

                bridge_hidden_state = kv_heads_to_hidden_state_bridge(
                    writeback_tensor,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                )

                layer14_eval["replacement_bridge_hidden_state_shape"] = list(
                    bridge_hidden_state.shape
                )
                layer14_eval["replacement_bridge_hidden_state_device"] = str(
                    bridge_hidden_state.device
                )
                layer14_eval["replacement_bridge_hidden_state_dtype"] = str(
                    bridge_hidden_state.dtype
                )
                layer14_eval["replacement_bridge_compare"] = {
                    "writeback_tensor_shape": list(writeback_tensor.shape),
                    "bridge_hidden_state_shape": list(bridge_hidden_state.shape),
                    "expected_hidden_size": num_attention_heads * writeback_tensor.shape[-1],
                }

                layer14_eval["replacement_writeback_tensor_shape"] = list(writeback_tensor.shape)
                layer14_eval["replacement_writeback_tensor_device"] = str(writeback_tensor.device)
                layer14_eval["replacement_writeback_tensor_dtype"] = str(writeback_tensor.dtype)

                layer14_eval["replacement_injection_selected"] = True
                layer14_eval["replacement_injection_target"] = "layer14_attention_output"

                injection_ready, injection_reason = prepare_layer14_actual_injection(
                    injection_tensor=writeback_tensor,
                    injection_target=layer14_eval["replacement_injection_target"],
                )

                layer14_eval["replacement_injection_ready"] = injection_ready
                layer14_eval["replacement_injection_ready_reason"] = injection_reason
                layer14_eval["replacement_injection_tensor_shape"] = list(writeback_tensor.shape)
                layer14_eval["replacement_injection_tensor_device"] = str(writeback_tensor.device)
                layer14_eval["replacement_injection_tensor_dtype"] = str(writeback_tensor.dtype)

                if injection_ready:
                    injection_applied, injection_apply_reason = apply_layer14_actual_injection(
                        injection_tensor=writeback_tensor,
                        injection_target=layer14_eval["replacement_injection_target"],
                    )
                    layer14_eval["replacement_injection_apply_reason"] = injection_apply_reason

                    if injection_applied:
                        layer14_eval["replacement_injection_applied"] = True
                        layer14_eval["replacement_injection_mode"] = "actual_injection_wired"
                    else:
                        layer14_eval["replacement_injection_applied"] = False
                        layer14_eval["replacement_injection_mode"] = "actual_injection_callsite_defined"
                else:
                    layer14_eval["replacement_injection_apply_reason"] = injection_reason
                    layer14_eval["replacement_injection_applied"] = False
                    layer14_eval["replacement_injection_mode"] = "injection_not_ready"

                replacement_writeback_applied = True
                replacement_writeback_mode = "tensor_writeback_ready_only"
                replacement_fallback_reason = None

            else:
                layer14_eval["replacement_writeback_tensor_shape"] = None
                layer14_eval["replacement_writeback_tensor_device"] = None
                layer14_eval["replacement_writeback_tensor_dtype"] = None

                layer14_eval["replacement_injection_selected"] = False
                layer14_eval["replacement_injection_target"] = None
                layer14_eval["replacement_injection_mode"] = "tensor_not_ready"
                layer14_eval["replacement_injection_ready"] = False
                layer14_eval["replacement_injection_ready_reason"] = "tensor_not_ready"
                layer14_eval["replacement_injection_apply_reason"] = "tensor_not_ready"
                layer14_eval["replacement_injection_applied"] = False
                layer14_eval["replacement_injection_tensor_shape"] = None
                layer14_eval["replacement_injection_tensor_device"] = None
                layer14_eval["replacement_injection_tensor_dtype"] = None

                layer14_eval["replacement_bridge_hidden_state_shape"] = None
                layer14_eval["replacement_bridge_hidden_state_device"] = None
                layer14_eval["replacement_bridge_hidden_state_dtype"] = None
                layer14_eval["replacement_bridge_compare"] = None

                replacement_writeback_applied = False
                replacement_writeback_mode = "tensor_not_ready_fallback"
                replacement_fallback_reason = tensor_writeback_reason

        else:
            layer14_eval["replacement_tensor_writeback_ready"] = False
            layer14_eval["replacement_tensor_writeback_reason"] = "gate_not_passed"
            layer14_eval["replacement_writeback_tensor_shape"] = None
            layer14_eval["replacement_writeback_tensor_device"] = None
            layer14_eval["replacement_writeback_tensor_dtype"] = None

            layer14_eval["replacement_injection_selected"] = False
            layer14_eval["replacement_injection_target"] = None
            layer14_eval["replacement_injection_mode"] = "gate_not_passed"
            layer14_eval["replacement_injection_ready"] = False
            layer14_eval["replacement_injection_ready_reason"] = "gate_not_passed"
            layer14_eval["replacement_injection_apply_reason"] = "gate_not_passed"
            layer14_eval["replacement_injection_applied"] = False
            layer14_eval["replacement_injection_tensor_shape"] = None
            layer14_eval["replacement_injection_tensor_device"] = None
            layer14_eval["replacement_injection_tensor_dtype"] = None

            layer14_eval["replacement_bridge_hidden_state_shape"] = None
            layer14_eval["replacement_bridge_hidden_state_device"] = None
            layer14_eval["replacement_bridge_hidden_state_dtype"] = None
            layer14_eval["replacement_bridge_compare"] = None

            replacement_writeback_applied = False
            replacement_writeback_mode = "gate_not_passed"
            replacement_fallback_reason = "gate_not_passed"

        layer14_eval["replacement_attn_hook_seen"] = injection_state.attn_hook_seen
        layer14_eval["replacement_attn_hook_shape"] = injection_state.attn_last_seen_shape
        layer14_eval["replacement_attn_hook_dtype"] = injection_state.attn_last_seen_dtype
        layer14_eval["replacement_attn_hook_device"] = injection_state.attn_last_seen_device

        layer14_eval["replacement_layer_hook_seen"] = injection_state.layer_hook_seen
        layer14_eval["replacement_layer_hook_shape"] = injection_state.layer_last_seen_shape
        layer14_eval["replacement_layer_hook_dtype"] = injection_state.layer_last_seen_dtype
        layer14_eval["replacement_layer_hook_device"] = injection_state.layer_last_seen_device

        layer14_eval["replacement_oproj_pre_hook_seen"] = injection_state.oproj_pre_hook_seen
        layer14_eval["replacement_oproj_pre_hook_shape"] = injection_state.oproj_pre_last_seen_shape
        layer14_eval["replacement_oproj_pre_hook_dtype"] = injection_state.oproj_pre_last_seen_dtype
        layer14_eval["replacement_oproj_pre_hook_device"] = injection_state.oproj_pre_last_seen_device

        layer14_eval["replacement_oproj_post_hook_seen"] = injection_state.oproj_post_hook_seen
        layer14_eval["replacement_oproj_post_hook_shape"] = injection_state.oproj_post_last_seen_shape
        layer14_eval["replacement_oproj_post_hook_dtype"] = injection_state.oproj_post_last_seen_dtype
        layer14_eval["replacement_oproj_post_hook_device"] = injection_state.oproj_post_last_seen_device

        attn_hook_shape = injection_state.attn_last_seen_shape
        if attn_hook_shape is not None and len(attn_hook_shape) >= 3:
            layer14_eval["replacement_attn_hook_hidden_size"] = attn_hook_shape[-1]
        else:
            layer14_eval["replacement_attn_hook_hidden_size"] = None

        layer_hook_shape = injection_state.layer_last_seen_shape
        if layer_hook_shape is not None and len(layer_hook_shape) >= 3:
            layer14_eval["replacement_layer_hook_hidden_size"] = layer_hook_shape[-1]
        else:
            layer14_eval["replacement_layer_hook_hidden_size"] = None

        layer14_eval["replacement_shape_compare"] = {
            "writeback_tensor_shape": layer14_eval.get("replacement_writeback_tensor_shape"),
            "attn_hook_output_shape": layer14_eval["replacement_attn_hook_shape"],
            "oproj_pre_hook_input_shape": layer14_eval["replacement_oproj_pre_hook_shape"],
            "oproj_post_hook_output_shape": layer14_eval["replacement_oproj_post_hook_shape"],
            "layer_hook_output_shape": layer14_eval["replacement_layer_hook_shape"],
        }

        layer14_eval["replacement_representation_compare"] = {
            "writeback_device": layer14_eval.get("replacement_writeback_tensor_device"),
            "writeback_dtype": layer14_eval.get("replacement_writeback_tensor_dtype"),
            "attn_hook_device": layer14_eval["replacement_attn_hook_device"],
            "attn_hook_dtype": layer14_eval["replacement_attn_hook_dtype"],
            "oproj_pre_hook_device": layer14_eval["replacement_oproj_pre_hook_device"],
            "oproj_pre_hook_dtype": layer14_eval["replacement_oproj_pre_hook_dtype"],
            "oproj_post_hook_device": layer14_eval["replacement_oproj_post_hook_device"],
            "oproj_post_hook_dtype": layer14_eval["replacement_oproj_post_hook_dtype"],
            "layer_hook_device": layer14_eval["replacement_layer_hook_device"],
            "layer_hook_dtype": layer14_eval["replacement_layer_hook_dtype"],
        }

        layer14_eval["replacement_layout_compare"] = {
            "writeback_num_heads": layer14_eval["attention_tensor_breakdown"]["approx_num_heads"],
            "writeback_head_dim": layer14_eval["attention_tensor_breakdown"]["approx_head_dim"],
            "writeback_flattened_last_dim": layer14_eval["attention_tensor_breakdown"]["approx_flattened_last_dim"],
            "attn_hook_hidden_size": layer14_eval["replacement_attn_hook_hidden_size"],
            "layer_hook_hidden_size": layer14_eval["replacement_layer_hook_hidden_size"],
        }

        if replacement_hook_triggered:
            # Future actual replacement point:
            # write approx layer-14 attention output back into the decode path here.
            pass

        # Placeholder for future actual replacement
        # Future step:
        # - compute approx attention output for layer 14
        # - route it into the actual decode path
        # For now we only log that the step is replacement-ready.
        layer14_eval.pop("_full_attention_output", None)
        layer14_eval.pop("_approx_attention_output", None)
        step_logs.append(
            {
                "step_idx": step_idx,
                "generated_token_id": next_token_id,
                "layer14": layer14_eval,
            }
        )

        next_token = torch.tensor([[next_token_id]], device=input_ids.device)
        current_input_ids = next_token
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    device=current_attention_mask.device,
                    dtype=current_attention_mask.dtype,
                ),
            ],
            dim=1,
        )

    elapsed = time.perf_counter() - start
    tokens_per_sec = (len(generated_token_ids) / elapsed) if elapsed > 0 else 0.0
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    layer14_attn_probe_handle.remove()
    layer14_block_probe_handle.remove()
    layer14_oproj_pre_handle.remove()
    layer14_oproj_post_handle.remove()

    return {
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "generated_tokens": len(generated_token_ids),
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "step_logs": step_logs,
    }
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-14 assisted-ready decode prototype.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="prose",
        choices=["repetitive", "prose", "structured"],
    )
    parser.add_argument("--hot-window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--scoring-variant", type=str, default="mean_plus_norm")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--output",
        type=str,
        default="relaykv_assisted_decode_layer14_summary.json",
    )
    parser.add_argument(
        "--mean-abs-diff-threshold",
        type=float,
        default=0.002,
        help="Gate threshold for applying layer-14 replacement candidates.",
    )
    parser.add_argument(
        "--enable-replacement-hook",
        action="store_true",
        help="Enable dry-run replacement hook for gate-passed layer-14 steps.",
    )

    args = parser.parse_args()

    ensure_results_dir()

    tokenizer, model, device = load_model(args.model)

    model_config = model.config
    hidden_size = int(model_config.hidden_size)
    num_attention_heads = int(model_config.num_attention_heads)
    num_key_value_heads = int(
        getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    )
    head_dim = hidden_size // num_attention_heads
    kv_hidden_size = num_key_value_heads * head_dim

    prompt = make_prompt_for_target_tokens(args.seq_len, args.prompt_type)

    prefill_inputs, prefill_outputs = run_prefill(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        seq_len=args.seq_len,
    )

    seq_len_actual = int(prefill_outputs.past_key_values.layers[0].keys.shape[2])

    decode_result = run_decode_loop_layer14_assisted_ready(
        model=model,
        tokenizer=tokenizer,
        prefill_inputs=prefill_inputs,
        prefill_outputs=prefill_outputs,
        max_new_tokens=args.max_new_tokens,
        hot_window=args.hot_window,
        block_size=args.block_size,
        scoring_variant=args.scoring_variant,
        mean_abs_diff_threshold=args.mean_abs_diff_threshold,
        enable_replacement_hook=args.enable_replacement_hook,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )

    summary = {
        "mode": "layer14_assisted_ready_decode_prototype",
        "model": args.model,
        "model_attention_layout": {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "kv_hidden_size": kv_hidden_size,
        },
        "prompt_type": args.prompt_type,
        "seq_len_target": args.seq_len,
        "seq_len_actual": seq_len_actual,
        "hot_window": args.hot_window,
        "block_size": args.block_size,
        "scoring_variant": args.scoring_variant,
        "assisted_target_layers": [14],
        "decode_policy": {
            "cold_blocks_fixed_from_prefill": True,
            "decode_added_tokens_hot_only": True,
            "replacement_scope": "layer14_gated_hooked_dry_run",
            "writeback_stage": "tensor_writeback_ready_only",
            "injection_stage": "kv_head_bridge_ready",
        },
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": decode_result["generated_tokens"],
        "generated_text": decode_result["generated_text"],
        "elapsed_sec": decode_result["elapsed_sec"],
        "tokens_per_sec": decode_result["tokens_per_sec"],
        "step_logs": decode_result["step_logs"],
        "replacement_gate": {
            "mean_abs_diff_threshold": args.mean_abs_diff_threshold,
            "require_contiguous_candidate": True,
            "require_single_block": True,
            "allowed_block_ids": [14, 15],
        },
        "replacement_hook": {
            "enabled": args.enable_replacement_hook,
            "mode": "tensor_writeback_ready_only",
            "target_layers": [14],
        },
        "replacement_writeback": {
            "implemented": False,
            "mode": "tensor_writeback_ready_only",
            "target_layers": [14],
            "fallback_to_full_path": True,
        },
        "replacement_injection": {
            "implemented": False,
            "mode": "kv_head_bridge_ready",
            "target": "layer14_attention_output",
            "target_layers": [14],
        }
    }

    output_path = RESULTS_DIR / args.output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()