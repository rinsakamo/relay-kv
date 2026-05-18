import json
import os
from pathlib import Path
import subprocess
import sys


def _env(
    repo_root: Path,
    *,
    extra_pythonpath: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = []
    if extra_pythonpath is not None:
        pythonpath_parts.append(str(extra_pythonpath))
    pythonpath_parts.append(str(repo_root))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if extra_env:
        env.update(extra_env)
    return env


def _run(
    repo_root: Path,
    *args: str,
    extra_pythonpath: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        env=_env(repo_root, extra_pythonpath=extra_pythonpath, extra_env=extra_env),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_happy_path_artifacts(tmp_path: Path, repo_root: Path) -> tuple[Path, Path, Path, Path]:
    adapter_path = tmp_path / "relaystack_adapter_capabilities.json"
    tokenizer_path = tmp_path / "relaystack_tokenizer_span_probe.json"
    engine_path = tmp_path / "relaystack_engine_metadata_probe.json"
    readiness_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    assert (
        _run(
            repo_root,
            "scripts/run_hf_adapter_capability_smoke.py",
            "--output",
            str(adapter_path),
        ).returncode
        == 0
    )
    assert (
        _run(
            repo_root,
            "scripts/run_hf_tokenizer_span_probe.py",
            "--output",
            str(tokenizer_path),
        ).returncode
        == 0
    )
    assert (
        _run(
            repo_root,
            "scripts/run_hf_engine_metadata_probe.py",
            "--output",
            str(engine_path),
        ).returncode
        == 0
    )
    assert (
        _run(
            repo_root,
            "scripts/run_hf_adapter_readiness_report.py",
            "--adapter-capabilities",
            str(adapter_path),
            "--tokenizer-span-probe",
            str(tokenizer_path),
            "--engine-metadata-probe",
            str(engine_path),
            "--output",
            str(readiness_path),
        ).returncode
        == 0
    )
    return adapter_path, tokenizer_path, engine_path, readiness_path


def _write_fake_transformers(fake_root: Path) -> Path:
    package_dir = fake_root / "transformers"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        """
import os


def _csv_ints(name, default):
    raw = os.environ.get(name, default)
    return [int(part) for part in raw.split(",") if part]


class FakeTokenizer:
    def __init__(self):
        self.vocab_size = int(os.environ.get("HF_FAKE_VOCAB_SIZE", "32000"))
        self.model_max_length = int(os.environ.get("HF_FAKE_MODEL_MAX_LENGTH", "32768"))
        self.special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>"}

    def encode(self, text, add_special_tokens=False):
        return _csv_ints("HF_FAKE_TOKEN_IDS", "11,22,33,44")


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if os.environ.get("HF_FAKE_TOKENIZER_MODE") == "fail":
            raise RuntimeError(os.environ.get("HF_FAKE_TOKENIZER_ERROR", "fake tokenizer load failure"))
        tokenizer = FakeTokenizer()
        tokenizer.__class__.__name__ = os.environ.get("HF_FAKE_TOKENIZER_CLASS", "FakeAutoTokenizer")
        return tokenizer


class FakeConfig:
    def __init__(self):
        self.model_type = os.environ.get("HF_FAKE_MODEL_TYPE", "qwen2")
        self.max_position_embeddings = int(os.environ.get("HF_FAKE_MAX_POSITION_EMBEDDINGS", "32768"))
        self.rope_scaling = None
        self.num_hidden_layers = int(os.environ.get("HF_FAKE_NUM_HIDDEN_LAYERS", "32"))
        self.num_attention_heads = int(os.environ.get("HF_FAKE_NUM_ATTENTION_HEADS", "32"))
        self.num_key_value_heads = int(os.environ.get("HF_FAKE_NUM_KEY_VALUE_HEADS", "8"))
        self.hidden_size = int(os.environ.get("HF_FAKE_HIDDEN_SIZE", "4096"))
        self.intermediate_size = int(os.environ.get("HF_FAKE_INTERMEDIATE_SIZE", "11008"))
        self.torch_dtype = os.environ.get("HF_FAKE_TORCH_DTYPE", "float16")
        self.quantization_config = {"quant_method": os.environ.get("HF_FAKE_QUANT_METHOD", "awq")}


class AutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if os.environ.get("HF_FAKE_CONFIG_MODE") == "fail":
            raise RuntimeError(os.environ.get("HF_FAKE_CONFIG_ERROR", "fake config load failure"))
        config = FakeConfig()
        config.__class__.__name__ = os.environ.get("HF_FAKE_CONFIG_CLASS", "FakeAutoConfig")
        return config
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return fake_root


def test_hf_tokenizer_config_probe_readiness_blocks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readiness_path = tmp_path / "readiness.json"
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"
    _write_json(
        readiness_path,
        {
            "summary": {"ok": False},
            "readiness": {
                "ready_for_real_tokenizer_probe": False,
                "ready_for_model_config_probe": False,
            },
        },
    )

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["summary"]["blocked_reason"] == "readiness summary.ok is not true"


def test_hf_tokenizer_config_probe_skip_loads_happy_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        "--skip-tokenizer-load",
        "--skip-config-load",
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True
    assert loaded["tokenizer_probe"]["skipped"] is True
    assert loaded["config_probe"]["skipped"] is True
    assert loaded["safety_scope"]["model_loaded"] is False
    assert loaded["safety_scope"]["no_gpu_inspection"] is True


def test_hf_tokenizer_config_probe_missing_readiness_fails_without_output(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readiness_path = tmp_path / "missing.json"
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 2
    assert not output_path.exists()


def test_hf_tokenizer_config_probe_fake_load_success(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"
    fake_root = _write_fake_transformers(tmp_path / "fake_py")

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        "--local-files-only",
        extra_pythonpath=fake_root,
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True
    assert loaded["tokenizer_probe"]["loaded"] is True
    assert loaded["config_probe"]["loaded"] is True
    assert loaded["tokenizer_probe"]["sample_token_count"] == 4
    assert loaded["consistency"]["config_attention_type_hint"] == "gqa"
    assert loaded["consistency"]["kv_head_group_count"] == 4


def test_hf_tokenizer_config_probe_model_id_override_mismatch_fails_in_skip_mode(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        "--skip-tokenizer-load",
        "--skip-config-load",
        "--model-id",
        "different/model",
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["readiness_model_ref_match"] is False
    assert loaded["summary"]["ready_for_next_metadata_step"] is False
    assert any("model_ref" in note for note in loaded["notes"])


def test_hf_tokenizer_config_probe_tokenizer_override_mismatch_fails_in_skip_mode(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        "--skip-tokenizer-load",
        "--skip-config-load",
        "--tokenizer-name-or-path",
        "different/tokenizer",
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["readiness_tokenizer_ref_match"] is False


def test_hf_tokenizer_config_probe_matching_overrides_still_pass(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        "--skip-tokenizer-load",
        "--skip-config-load",
        "--model-id",
        "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "--tokenizer-name-or-path",
        "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True


def test_hf_tokenizer_config_probe_fake_load_success_with_mismatch_still_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"
    fake_root = _write_fake_transformers(tmp_path / "fake_py")

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        "--local-files-only",
        "--model-id",
        "different/model",
        extra_pythonpath=fake_root,
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["tokenizer_probe"]["loaded"] is True
    assert loaded["config_probe"]["loaded"] is True
    assert loaded["consistency"]["readiness_model_ref_match"] is False


def test_hf_tokenizer_config_probe_fake_tokenizer_failure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"
    fake_root = _write_fake_transformers(tmp_path / "fake_py")

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        extra_pythonpath=fake_root,
        extra_env={"HF_FAKE_TOKENIZER_MODE": "fail"},
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["tokenizer_probe"]["loaded"] is False
    assert isinstance(loaded["tokenizer_probe"]["tokenizer_loading_error"], str)
    assert "traceback" not in loaded["tokenizer_probe"]["tokenizer_loading_error"].lower()


def test_hf_tokenizer_config_probe_invalid_config_head_metadata_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, _, readiness_path = _build_happy_path_artifacts(tmp_path, repo_root)
    output_path = tmp_path / "relaystack_hf_tokenizer_config_probe.json"
    fake_root = _write_fake_transformers(tmp_path / "fake_py")

    result = _run(
        repo_root,
        "scripts/run_hf_tokenizer_config_probe.py",
        "--readiness-report",
        str(readiness_path),
        "--output",
        str(output_path),
        extra_pythonpath=fake_root,
        extra_env={
            "HF_FAKE_NUM_ATTENTION_HEADS": "12",
            "HF_FAKE_NUM_KEY_VALUE_HEADS": "5",
        },
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["config_attention_type_hint"] == "unknown"
    assert loaded["consistency"]["kv_head_group_count"] is None
