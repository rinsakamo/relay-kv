import json
import os
from pathlib import Path
import subprocess
import sys


def _env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    return env


def _run(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        env=_env(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_happy_path_artifacts(tmp_path: Path, repo_root: Path) -> tuple[Path, Path, Path]:
    adapter_path = tmp_path / "relaystack_adapter_capabilities.json"
    tokenizer_path = tmp_path / "relaystack_tokenizer_span_probe.json"
    engine_path = tmp_path / "relaystack_engine_metadata_probe.json"

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
    return adapter_path, tokenizer_path, engine_path


def test_hf_adapter_readiness_report_happy_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    stdout_summary = json.loads(result.stdout)
    assert stdout_summary["ok"] is True
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True
    assert loaded["readiness"]["ready_for_next_metadata_step"] is True
    assert loaded["readiness"]["ready_for_real_tokenizer_probe"] is True
    assert loaded["readiness"]["ready_for_model_config_probe"] is True
    assert loaded["readiness"]["ready_for_materialization"] is False
    assert loaded["readiness"]["ready_for_apply"] is False
    assert loaded["summary"]["failed_check_count"] == 0
    assert json.loads(json.dumps(loaded)) == loaded


def test_hf_adapter_readiness_report_mismatched_model_id_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    engine_payload["model_ref"]["model_id"] = "other/model"
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert any("model_ref.model_id" in reason for reason in loaded["readiness"]["blocking_reasons"])


def test_hf_adapter_readiness_report_model_revision_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["model_ref"]["model_revision"] = "rev-a"
    tokenizer_payload["model_ref"]["model_revision"] = "rev-a"
    engine_payload["model_ref"]["model_revision"] = "rev-b"
    _write_json(adapter_path, adapter_payload)
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "model_ref_model_revision_consistency" in failed_names


def test_hf_adapter_readiness_report_tokenizer_null_model_revision_is_allowed(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["model_ref"]["model_revision"] = "rev-a"
    tokenizer_payload["model_ref"]["model_revision"] = None
    engine_payload["model_ref"]["model_revision"] = "rev-a"
    _write_json(adapter_path, adapter_payload)
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True
    assert loaded["readiness"]["ready_for_next_metadata_step"] is True


def test_hf_adapter_readiness_report_local_path_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["model_ref"]["local_path"] = "/tmp/model-a"
    tokenizer_payload["model_ref"]["local_path"] = None
    engine_payload["model_ref"]["local_path"] = "/tmp/model-b"
    _write_json(adapter_path, adapter_payload)
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "model_ref_local_path_consistency" in failed_names


def test_hf_adapter_readiness_report_local_path_omitted_is_allowed(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["model_ref"]["local_path"] = "/tmp/model-a"
    tokenizer_payload["model_ref"]["local_path"] = None
    engine_payload["model_ref"]["local_path"] = None
    _write_json(adapter_path, adapter_payload)
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True


def test_hf_adapter_readiness_report_adapter_name_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["adapter_name"] = "hf_local_prototype"
    engine_payload["adapter_name"] = "other_adapter"
    _write_json(adapter_path, adapter_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "adapter_name_consistency" in failed_names


def test_hf_adapter_readiness_report_adapter_name_omitted_is_allowed(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    del engine_payload["adapter_name"]
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True


def test_hf_adapter_readiness_report_tokenizer_model_revision_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["model_ref"]["model_revision"] = "rev-a"
    tokenizer_payload["model_ref"]["model_revision"] = "rev-b"
    engine_payload["model_ref"]["model_revision"] = "rev-a"
    _write_json(adapter_path, adapter_payload)
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "model_ref_model_revision_consistency" in failed_names


def test_hf_adapter_readiness_report_unsafe_apply_claim_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_payload["capabilities"]["supports_apply"] = True
    _write_json(adapter_path, adapter_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["readiness"]["ready_for_apply"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "adapter_supports_apply_false" in failed_names


def test_hf_adapter_readiness_report_adapter_missing_tokenizer_span_probe_support_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_payload["capabilities"]["supports_tokenizer_span_probe"] = False
    _write_json(adapter_path, adapter_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["readiness"]["ready_for_real_tokenizer_probe"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "adapter_supports_tokenizer_span_probe_true" in failed_names


def test_hf_adapter_readiness_report_adapter_missing_engine_metadata_probe_support_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_payload["capabilities"]["supports_engine_metadata_probe"] = False
    _write_json(adapter_path, adapter_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["readiness"]["ready_for_model_config_probe"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "adapter_supports_engine_metadata_probe_true" in failed_names


def test_hf_adapter_readiness_report_engine_missing_tokenizer_span_probe_support_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    engine_payload["capabilities_snapshot"]["supports_tokenizer_span_probe"] = False
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "engine_supports_tokenizer_span_probe_true" in failed_names


def test_hf_adapter_readiness_report_engine_missing_engine_metadata_probe_support_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    engine_payload["capabilities_snapshot"]["supports_engine_metadata_probe"] = False
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "engine_supports_engine_metadata_probe_true" in failed_names


def test_hf_adapter_readiness_report_missing_or_non_bool_probe_support_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    invalid_cases = (
        ("adapter_missing", "adapter", "supports_tokenizer_span_probe", None),
        ("adapter_non_bool", "adapter", "supports_tokenizer_span_probe", "yes"),
        ("engine_missing", "engine", "supports_tokenizer_span_probe", None),
        ("engine_non_bool", "engine", "supports_tokenizer_span_probe", "yes"),
    )

    for _, target_kind, field_name, value in invalid_cases:
        adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
        adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
        engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
        if target_kind == "adapter":
            target = adapter_payload["capabilities"]
            if value is None:
                del target[field_name]
            else:
                target[field_name] = value
        else:
            target = engine_payload["capabilities_snapshot"]
            if value is None:
                del target[field_name]
            else:
                target[field_name] = value
        _write_json(adapter_path, adapter_payload)
        _write_json(engine_path, engine_payload)
        output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"
        if output_path.exists():
            output_path.unlink()

        result = _run(
            repo_root,
            "scripts/run_hf_adapter_readiness_report.py",
            "--adapter-capabilities",
            str(adapter_path),
            "--tokenizer-span-probe",
            str(tokenizer_path),
            "--engine-metadata-probe",
            str(engine_path),
            "--output",
            str(output_path),
        )

        assert result.returncode == 1
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["ok"] is False
        assert "AttributeError" not in f"{result.stdout}\n{result.stderr}"


def test_hf_adapter_readiness_report_adapter_summary_ok_false_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_payload["summary"]["ok"] = False
    _write_json(adapter_path, adapter_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "adapter_capabilities_summary_ok_true" in failed_names


def test_hf_adapter_readiness_report_tokenizer_summary_ok_false_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_payload["summary"]["ok"] = False
    _write_json(tokenizer_path, tokenizer_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_span_probe_summary_ok_true" in failed_names


def test_hf_adapter_readiness_report_engine_summary_ok_false_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    engine_payload["summary"]["ok"] = False
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "engine_metadata_probe_summary_ok_true" in failed_names


def test_hf_adapter_readiness_report_missing_or_non_bool_summary_ok_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    invalid_cases = (
        ("adapter_missing", "adapter", None),
        ("adapter_non_bool", "adapter", "true"),
        ("tokenizer_missing", "tokenizer", None),
        ("tokenizer_non_bool", "tokenizer", "true"),
        ("engine_missing", "engine", None),
        ("engine_non_bool", "engine", "true"),
    )

    for _, target_kind, value in invalid_cases:
        adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(tmp_path, repo_root)
        payload_map = {
            "adapter": json.loads(adapter_path.read_text(encoding="utf-8")),
            "tokenizer": json.loads(tokenizer_path.read_text(encoding="utf-8")),
            "engine": json.loads(engine_path.read_text(encoding="utf-8")),
        }
        target_summary = payload_map[target_kind]["summary"]
        if value is None:
            del payload_map[target_kind]["summary"]
        else:
            target_summary["ok"] = value
        _write_json(adapter_path, payload_map["adapter"])
        _write_json(tokenizer_path, payload_map["tokenizer"])
        _write_json(engine_path, payload_map["engine"])
        output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"
        if output_path.exists():
            output_path.unlink()

        result = _run(
            repo_root,
            "scripts/run_hf_adapter_readiness_report.py",
            "--adapter-capabilities",
            str(adapter_path),
            "--tokenizer-span-probe",
            str(tokenizer_path),
            "--engine-metadata-probe",
            str(engine_path),
            "--output",
            str(output_path),
        )

        assert result.returncode == 1
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["ok"] is False
        assert "AttributeError" not in f"{result.stdout}\n{result.stderr}"


def test_hf_adapter_readiness_report_tokenizer_revision_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    adapter_payload = json.loads(adapter_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    adapter_payload["tokenizer_ref"]["tokenizer_revision"] = "rev-a"
    tokenizer_payload["tokenizer_ref"]["tokenizer_revision"] = "rev-b"
    engine_payload["tokenizer_ref"]["tokenizer_revision"] = "rev-a"
    _write_json(adapter_path, adapter_payload)
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_ref_tokenizer_revision_consistency" in failed_names


def test_hf_adapter_readiness_report_tokenizer_config_hash_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    engine_payload = json.loads(engine_path.read_text(encoding="utf-8"))
    tokenizer_payload["tokenizer_ref"]["tokenizer_config_hash"] = "hash-a"
    engine_payload["tokenizer_ref"]["tokenizer_config_hash"] = "hash-b"
    _write_json(tokenizer_path, tokenizer_payload)
    _write_json(engine_path, engine_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_ref_tokenizer_config_hash_consistency" in failed_names


def test_hf_adapter_readiness_report_span_embedded_tokenizer_ref_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_payload["spans"][0]["tokenizer_ref"]["tokenizer_name_or_path"] = "other/tokenizer"
    _write_json(tokenizer_path, tokenizer_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_span_tokenizer_name_or_path_consistency_0" in failed_names


def test_hf_adapter_readiness_report_span_embedded_tokenizer_revision_mismatch_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_payload["spans"][0]["tokenizer_ref"]["tokenizer_revision"] = "other-rev"
    _write_json(tokenizer_path, tokenizer_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_span_tokenizer_revision_consistency_0" in failed_names


def test_hf_adapter_readiness_report_span_missing_tokenizer_ref_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    del tokenizer_payload["spans"][0]["tokenizer_ref"]
    _write_json(tokenizer_path, tokenizer_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_span_tokenizer_ref_present_0" in failed_names


def test_hf_adapter_readiness_report_malformed_non_object_span_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_payload["spans"] = ["not-a-span"]
    _write_json(tokenizer_path, tokenizer_payload)
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(adapter_path),
        "--tokenizer-span-probe",
        str(tokenizer_path),
        "--engine-metadata-probe",
        str(engine_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    failed_names = {
        check["name"] for check in loaded["checks"] if not check["passed"]
    }
    assert "tokenizer_span_object_0" in failed_names
    assert "AttributeError" not in result.stderr


def test_hf_adapter_readiness_report_missing_input_fails_without_output(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"
    missing_path = tmp_path / "missing.json"

    result = _run(
        repo_root,
        "scripts/run_hf_adapter_readiness_report.py",
        "--adapter-capabilities",
        str(missing_path),
        "--tokenizer-span-probe",
        str(missing_path),
        "--engine-metadata-probe",
        str(missing_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 2
    assert not output_path.exists()


def test_hf_adapter_readiness_report_rejects_non_object_input_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"
    adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
        tmp_path,
        repo_root,
    )

    for target_path in (adapter_path, tokenizer_path, engine_path):
        _write_json(target_path, [])  # type: ignore[arg-type]
        result = _run(
            repo_root,
            "scripts/run_hf_adapter_readiness_report.py",
            "--adapter-capabilities",
            str(adapter_path),
            "--tokenizer-span-probe",
            str(tokenizer_path),
            "--engine-metadata-probe",
            str(engine_path),
            "--output",
            str(output_path),
        )

        assert result.returncode == 2
        assert not output_path.exists()
        combined_output = f"{result.stdout}\n{result.stderr}"
        assert "JSON object" in combined_output or "non-object" in combined_output
        assert "AttributeError" not in combined_output

        adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
            tmp_path,
            repo_root,
        )


def test_hf_adapter_readiness_report_rejects_malformed_nested_mappings(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    invalid_cases = (
        ("adapter_model_ref_bad", "adapter", ("model_ref",), "adapter_capabilities_model_ref_object"),
        ("adapter_tokenizer_ref_bad", "adapter", ("tokenizer_ref",), "adapter_capabilities_tokenizer_ref_object"),
        ("adapter_capabilities_bad", "adapter", ("capabilities",), "adapter_capabilities_capabilities_object"),
        ("tokenizer_span_lineage_bad", "tokenizer", ("spans", 0, "lineage"), "tokenizer_span_0_lineage_object"),
        ("engine_metadata_bad", "engine", ("engine_metadata",), "engine_metadata_probe_engine_metadata_object"),
        ("engine_capabilities_snapshot_bad", "engine", ("capabilities_snapshot",), "engine_metadata_probe_capabilities_snapshot_object"),
    )

    for _, target_kind, path_spec, expected_failed_name in invalid_cases:
        adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
            tmp_path,
            repo_root,
        )
        payload_map = {
            "adapter": json.loads(adapter_path.read_text(encoding="utf-8")),
            "tokenizer": json.loads(tokenizer_path.read_text(encoding="utf-8")),
            "engine": json.loads(engine_path.read_text(encoding="utf-8")),
        }
        target_payload = payload_map[target_kind]
        cursor = target_payload
        for key in path_spec[:-1]:
            cursor = cursor[key]
        cursor[path_spec[-1]] = "bad"
        _write_json(adapter_path, payload_map["adapter"])
        _write_json(tokenizer_path, payload_map["tokenizer"])
        _write_json(engine_path, payload_map["engine"])
        output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"
        if output_path.exists():
            output_path.unlink()

        result = _run(
            repo_root,
            "scripts/run_hf_adapter_readiness_report.py",
            "--adapter-capabilities",
            str(adapter_path),
            "--tokenizer-span-probe",
            str(tokenizer_path),
            "--engine-metadata-probe",
            str(engine_path),
            "--output",
            str(output_path),
        )

        assert result.returncode == 1
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["ok"] is False
        failed_names = {
            check["name"] for check in loaded["checks"] if not check["passed"]
        }
        assert expected_failed_name in failed_names
        assert "AttributeError" not in f"{result.stdout}\n{result.stderr}"


def test_hf_adapter_readiness_report_rejects_malformed_summary_mappings(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    invalid_cases = (
        (
            "adapter_summary_bad",
            "adapter",
            False,
            "adapter_capabilities_summary_object",
        ),
        (
            "tokenizer_summary_bad",
            "tokenizer",
            False,
            "tokenizer_span_probe_summary_object",
        ),
        (
            "engine_summary_bad",
            "engine",
            False,
            "engine_metadata_probe_summary_object",
        ),
    )

    for _, target_kind, remove_adapter_gpu_flag, expected_failed_name in invalid_cases:
        adapter_path, tokenizer_path, engine_path = _build_happy_path_artifacts(
            tmp_path,
            repo_root,
        )
        payload_map = {
            "adapter": json.loads(adapter_path.read_text(encoding="utf-8")),
            "tokenizer": json.loads(tokenizer_path.read_text(encoding="utf-8")),
            "engine": json.loads(engine_path.read_text(encoding="utf-8")),
        }
        if remove_adapter_gpu_flag:
            payload_map["adapter"]["safety_scope"].pop("no_gpu_inspection", None)
        payload_map[target_kind]["summary"] = "bad"
        _write_json(adapter_path, payload_map["adapter"])
        _write_json(tokenizer_path, payload_map["tokenizer"])
        _write_json(engine_path, payload_map["engine"])
        output_path = tmp_path / "relaystack_hf_adapter_readiness_report.json"
        if output_path.exists():
            output_path.unlink()

        result = _run(
            repo_root,
            "scripts/run_hf_adapter_readiness_report.py",
            "--adapter-capabilities",
            str(adapter_path),
            "--tokenizer-span-probe",
            str(tokenizer_path),
            "--engine-metadata-probe",
            str(engine_path),
            "--output",
            str(output_path),
        )

        assert result.returncode == 1
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["ok"] is False
        failed_names = {
            check["name"] for check in loaded["checks"] if not check["passed"]
        }
        assert expected_failed_name in failed_names
        assert "AttributeError" not in f"{result.stdout}\n{result.stderr}"
