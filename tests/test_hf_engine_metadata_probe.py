import json
import os
from pathlib import Path
import subprocess
import sys


def test_hf_engine_metadata_probe_default_run(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_engine_metadata_probe.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_engine_metadata_probe.py",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    stdout_summary = json.loads(result.stdout)
    assert stdout_summary["ok"] is True
    assert stdout_summary["artifact_kind"] == "hf_engine_metadata_probe"
    assert stdout_summary["no_model_loaded"] is True
    assert stdout_summary["no_tokenizer_loaded"] is True
    assert stdout_summary["no_gpu_inspection"] is True

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "relaystack.engine_metadata_probe.v0.1"
    assert loaded["phase"] == "12-F"
    assert loaded["artifact_kind"] == "hf_engine_metadata_probe"
    assert loaded["adapter_kind"] == "hf_prototype"
    assert loaded["runtime_target"] == "huggingface_transformers"
    assert loaded["model_ref"]["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    assert (
        loaded["tokenizer_ref"]["tokenizer_name_or_path"]
        == loaded["model_ref"]["model_id"]
    )
    assert loaded["engine_metadata"]["model_loaded"] is False
    assert loaded["engine_metadata"]["tokenizer_loaded"] is False
    assert loaded["engine_metadata"]["gpu_inspected"] is False
    assert loaded["safety_scope"]["no_model_loading_required"] is True
    assert loaded["safety_scope"]["no_tokenizer_loading_required"] is True
    assert loaded["safety_scope"]["no_attention_connection"] is True
    assert loaded["summary"]["no_model_loaded"] is True
    assert loaded["summary"]["no_tokenizer_loaded"] is True
    assert json.loads(json.dumps(loaded)) == loaded


def test_hf_engine_metadata_probe_cli_override(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_engine_metadata_probe_override.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_engine_metadata_probe.py",
            "--output",
            str(output_path),
            "--model-id",
            "test/model",
            "--tokenizer-name-or-path",
            "test/tokenizer",
            "--quantization-hint",
            "none",
            "--dtype-hint",
            "float16",
            "--configured-context-tokens",
            "8192",
            "--max-model-context-tokens",
            "32768",
            "--num-hidden-layers",
            "32",
            "--num-attention-heads",
            "32",
            "--num-key-value-heads",
            "8",
            "--head-dim",
            "128",
            "--hidden-size",
            "4096",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["model_ref"]["model_id"] == "test/model"
    assert loaded["tokenizer_ref"]["tokenizer_name_or_path"] == "test/tokenizer"
    assert loaded["model_ref"]["quantization_hint"] == "none"
    assert loaded["model_ref"]["dtype_hint"] == "float16"
    assert loaded["context_window_hint"]["configured_context_tokens"] == 8192
    assert loaded["context_window_hint"]["max_model_context_tokens"] == 32768
    assert loaded["engine_metadata"]["num_hidden_layers"] == 32
    assert loaded["engine_metadata"]["num_attention_heads"] == 32
    assert loaded["engine_metadata"]["num_key_value_heads"] == 8
    assert loaded["engine_metadata"]["head_dim"] == 128
    assert loaded["engine_metadata"]["hidden_size"] == 4096
    assert loaded["engine_metadata"]["attention_type_hint"] == "gqa"
    assert loaded["engine_metadata"]["kv_head_group_count"] == 8


def test_hf_engine_metadata_probe_rejects_non_positive_numeric_overrides(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaystack_engine_metadata_probe_invalid.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    invalid_cases = (
        ("--num-hidden-layers", "0"),
        ("--configured-context-tokens", "-1"),
    )
    for flag_name, invalid_value in invalid_cases:
        if output_path.exists():
            output_path.unlink()
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_hf_engine_metadata_probe.py",
                "--output",
                str(output_path),
                flag_name,
                invalid_value,
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert not output_path.exists()
        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        assert "positive" in combined_output or "non-positive" in combined_output
