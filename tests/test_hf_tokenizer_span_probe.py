import json
import os
from pathlib import Path
import subprocess
import sys


def test_hf_tokenizer_span_probe_default_run(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_tokenizer_span_probe.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_tokenizer_span_probe.py",
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
    assert stdout_summary["artifact_kind"] == "hf_tokenizer_span_probe"
    assert stdout_summary["no_model_loaded"] is True
    assert stdout_summary["no_tokenizer_loaded"] is True
    assert stdout_summary["no_gpu_inspection"] is True

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "relaystack.tokenizer_span_probe.v0.1"
    assert loaded["phase"] == "12-E"
    assert loaded["artifact_kind"] == "hf_tokenizer_span_probe"
    assert loaded["adapter_kind"] == "hf_prototype"
    assert loaded["runtime_target"] == "huggingface_transformers"
    assert len(loaded["spans"]) == 1
    span = loaded["spans"][0]
    assert span["token_start"] == 0
    assert span["token_end"] > 0
    assert span["token_span_is_estimated"] is True
    assert span["tokenizer_scoped"] is True
    assert span["lineage"]["engine_block_ref"] is None
    assert loaded["safety_scope"]["no_model_loading_required"] is True
    assert loaded["safety_scope"]["no_tokenizer_loading_required"] is True
    assert loaded["safety_scope"]["no_attention_connection"] is True
    assert loaded["summary"]["no_model_loaded"] is True
    assert loaded["summary"]["no_tokenizer_loaded"] is True
    assert json.loads(json.dumps(loaded)) == loaded


def test_hf_tokenizer_span_probe_cli_override(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_tokenizer_span_probe_override.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_tokenizer_span_probe.py",
            "--output",
            str(output_path),
            "--model-id",
            "test/model",
            "--tokenizer-name-or-path",
            "test/tokenizer",
            "--tokenizer-revision",
            "rev1",
            "--tokenizer-config-hash",
            "hash1",
            "--text",
            "alpha beta gamma",
            "--source-item-id",
            "mem:test",
            "--span-kind",
            "context_item",
            "--estimated-token-count",
            "42",
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
    assert loaded["tokenizer_ref"]["tokenizer_revision"] == "rev1"
    assert loaded["tokenizer_ref"]["tokenizer_config_hash"] == "hash1"
    assert loaded["input"]["text"] == "alpha beta gamma"
    assert loaded["input"]["source_item_id"] == "mem:test"
    span = loaded["spans"][0]
    assert span["span_kind"] == "context_item"
    assert span["token_end"] == 42
    assert span["estimated_token_count"] == 42


def test_hf_tokenizer_span_probe_rejects_non_positive_estimate(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_tokenizer_span_probe_invalid.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    for invalid_value in ("-1", "0"):
        if output_path.exists():
            output_path.unlink()
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_hf_tokenizer_span_probe.py",
                "--output",
                str(output_path),
                "--estimated-token-count",
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
        assert "estimated-token-count" in combined_output
        assert "positive" in combined_output or "non-positive" in combined_output
