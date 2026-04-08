import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "kv_shapes.json"


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

    prompt = "RelayKV KV cache inspection test. " * 64
    inputs = tokenizer(prompt, return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values

    print("past_key_values type:", type(past_key_values))
    print("dir(past_key_values):")
    for name in dir(past_key_values):
        if not name.startswith("__"):
            print(" ", name)

    candidate_attrs = [
        "key_cache",
        "value_cache",
        "layers",
        "caches",
        "_layers",
        "_caches",
        "cache",
    ]

    attr_summary = {}

    for attr in candidate_attrs:
        if hasattr(past_key_values, attr):
            value = getattr(past_key_values, attr)
            print(f"attr {attr}: type={type(value)}")
            attr_summary[attr] = {"type": str(type(value))}
            try:
                value_len = len(value)
                print(f"  len={value_len}")
                attr_summary[attr]["len"] = value_len
            except Exception:
                pass

            if isinstance(value, (list, tuple)) and len(value) > 0:
                first = value[0]
                print(f"  first item type={type(first)}")
                attr_summary[attr]["first_item_type"] = str(type(first))
                if hasattr(first, "shape"):
                    print(f"  first item shape={list(first.shape)}")
                    attr_summary[attr]["first_item_shape"] = list(first.shape)

    available_methods = []
    for method_name in ["to_legacy_cache", "to_tuple", "to_list"]:
        if hasattr(past_key_values, method_name):
            print(f"has method: {method_name}")
            available_methods.append(method_name)

    summary = {
        "model": MODEL_NAME,
        "device": device,
        "past_key_values_type": str(type(past_key_values)),
        "available_attrs": attr_summary,
        "available_methods": available_methods,
    }

    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
        first_layer = past_key_values.layers[0]
        print("first_layer type:", type(first_layer))
        print("dir(first_layer):")
        for name in dir(first_layer):
            if not name.startswith("__"):
                print(" ", name)

        layer_candidate_attrs = [
            "keys",
            "values",
            "key",
            "value",
            "k",
            "v",
            "cache",
            "key_cache",
            "value_cache",
        ]

        layer_attr_summary = {}

        for attr in layer_candidate_attrs:
            if hasattr(first_layer, attr):
                value = getattr(first_layer, attr)
                print(f"first_layer attr {attr}: type={type(value)}")
                layer_attr_summary[attr] = {"type": str(type(value))}
                if hasattr(value, "shape"):
                    print(f"  shape={list(value.shape)}")
                    layer_attr_summary[attr]["shape"] = list(value.shape)

        summary["first_layer_attrs"] = layer_attr_summary

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()