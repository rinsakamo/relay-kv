import json
import os
from pathlib import Path
import subprocess
import sys


def test_hf_adapter_capability_smoke_default_run(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_adapter_capabilities.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_adapter_capability_smoke.py",
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
    assert stdout_summary["artifact_kind"] == "hf_adapter_capability_smoke"
    assert stdout_summary["no_model_loaded"] is True
    assert stdout_summary["no_gpu_inspection"] is True
    assert stdout_summary["supports_apply"] is False
    assert stdout_summary["supports_materialization"] is False

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "relaystack.adapter_capabilities.v0.1"
    assert loaded["phase"] == "12-C"
    assert loaded["adapter_kind"] == "hf_prototype"
    assert loaded["runtime_target"] == "huggingface_transformers"
    assert loaded["model_ref"]["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    assert (
        loaded["tokenizer_ref"]["tokenizer_name_or_path"]
        == loaded["model_ref"]["model_id"]
    )
    assert loaded["capabilities"]["supports_tokenizer_span_probe"] is True
    assert loaded["capabilities"]["supports_apply"] is False
    assert loaded["capabilities"]["supports_materialization"] is False
    assert loaded["safety_scope"]["no_model_loading_required"] is True
    assert loaded["safety_scope"]["no_attention_connection"] is True
    assert loaded["summary"]["no_model_loaded"] is True
    assert json.loads(json.dumps(loaded)) == loaded


def test_hf_adapter_capability_smoke_cli_override(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_adapter_capabilities_override.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_adapter_capability_smoke.py",
            "--output",
            str(output_path),
            "--model-id",
            "test/model",
            "--tokenizer-name-or-path",
            "test/tokenizer",
            "--quantization-hint",
            "none",
            "--configured-context-tokens",
            "8192",
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
    assert loaded["context_window_hint"]["configured_context_tokens"] == 8192
