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
