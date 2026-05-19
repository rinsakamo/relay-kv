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


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_happy_path_payloads(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "adapter": tmp_path / "relaystack_adapter_capabilities.json",
        "tokenizer": tmp_path / "relaystack_tokenizer_span_probe.json",
        "engine": tmp_path / "relaystack_engine_metadata_probe.json",
        "readiness": tmp_path / "relaystack_hf_adapter_readiness_report.json",
        "probe": tmp_path / "relaystack_hf_tokenizer_config_probe.json",
    }
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    tokenizer_name = model_id
    tokenizer_ref = {
        "tokenizer_name_or_path": tokenizer_name,
        "tokenizer_revision": None,
        "tokenizer_config_hash": None,
        "tokenizer_family": "hf_auto_tokenizer",
    }
    model_ref = {
        "model_id": model_id,
        "model_revision": None,
        "local_path": None,
    }
    _write_json(
        paths["adapter"],
        {
            "schema_version": "relaystack.adapter_capabilities.v0.1",
            "phase": "12-C",
            "adapter_kind": "hf_prototype",
            "adapter_name": "hf_local_prototype",
            "runtime_target": "huggingface_transformers",
            "model_ref": {
                **model_ref,
                "quantization_hint": "awq",
                "dtype_hint": None,
            },
            "tokenizer_ref": tokenizer_ref,
            "context_window_hint": {
                "max_model_context_tokens": None,
                "configured_context_tokens": None,
                "source": "user_override_or_unknown",
            },
            "capabilities": {
                "supports_tokenizer_span_probe": True,
                "supports_engine_metadata_probe": True,
                "supports_fullkv_reference": True,
                "supports_shadow_compare": False,
                "supports_materialization": False,
                "supports_apply": False,
                "supports_safe_degrade": False,
                "supports_context_reduction_request": True,
            },
            "safety_scope": {
                "dry_run_only": True,
                "no_model_loading_required": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
            },
            "summary": {
                "ok": True,
                "artifact_kind": "hf_adapter_capability_smoke",
                "no_model_loaded": True,
                "no_gpu_inspection": True,
                "supports_apply": False,
                "supports_materialization": False,
                "output_path": str(paths["adapter"]),
            },
        },
    )
    _write_json(
        paths["tokenizer"],
        {
            "schema_version": "relaystack.tokenizer_span_probe.v0.1",
            "phase": "12-E",
            "artifact_kind": "hf_tokenizer_span_probe",
            "adapter_kind": "hf_prototype",
            "runtime_target": "huggingface_transformers",
            "model_ref": model_ref,
            "tokenizer_ref": tokenizer_ref,
            "spans": [
                {
                    "source_item_id": "synthetic:item",
                    "span_kind": "prompt_text",
                    "token_start": 0,
                    "token_end": 4,
                    "estimated_token_count": 4,
                    "token_span_is_estimated": True,
                    "tokenizer_scoped": True,
                    "tokenizer_ref": tokenizer_ref,
                    "lineage": {
                        "relaymem_item_id": "synthetic:item",
                        "relayctx_source_item_id": "synthetic:item",
                        "logical_block_id": None,
                        "engine_block_ref": None,
                    },
                }
            ],
            "safety_scope": {
                "dry_run_only": True,
                "no_model_loading_required": True,
                "no_tokenizer_loading_required": True,
                "no_gpu_inspection": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
            },
            "summary": {
                "ok": True,
                "artifact_kind": "hf_tokenizer_span_probe",
                "no_model_loaded": True,
                "no_tokenizer_loaded": True,
                "no_gpu_inspection": True,
                "span_count": 1,
                "token_span_is_estimated": True,
                "tokenizer_scoped": True,
                "output_path": str(paths["tokenizer"]),
            },
        },
    )
    _write_json(
        paths["engine"],
        {
            "schema_version": "relaystack.engine_metadata_probe.v0.1",
            "phase": "12-F",
            "artifact_kind": "hf_engine_metadata_probe",
            "adapter_kind": "hf_prototype",
            "runtime_target": "huggingface_transformers",
            "adapter_name": "hf_local_prototype",
            "model_ref": {
                **model_ref,
                "quantization_hint": "awq",
                "dtype_hint": None,
            },
            "tokenizer_ref": tokenizer_ref,
            "context_window_hint": {
                "max_model_context_tokens": None,
                "configured_context_tokens": None,
                "source": "user_override_or_unknown",
            },
            "engine_metadata": {
                "device_target": "cuda",
                "config_source": "cli_or_unknown",
                "model_config_loaded": False,
                "tokenizer_loaded": False,
                "model_loaded": False,
                "gpu_inspected": False,
                "num_hidden_layers": None,
                "num_attention_heads": None,
                "num_key_value_heads": None,
                "head_dim": None,
                "hidden_size": None,
                "attention_type_hint": "unknown",
                "kv_head_group_count": None,
            },
            "capabilities_snapshot": {
                "supports_tokenizer_span_probe": True,
                "supports_engine_metadata_probe": True,
                "supports_fullkv_reference": True,
                "supports_shadow_compare": False,
                "supports_materialization": False,
                "supports_apply": False,
                "supports_safe_degrade": False,
                "supports_context_reduction_request": True,
            },
            "safety_scope": {
                "dry_run_only": True,
                "no_model_loading_required": True,
                "no_tokenizer_loading_required": True,
                "no_gpu_inspection": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
            },
            "summary": {
                "ok": True,
                "artifact_kind": "hf_engine_metadata_probe",
                "no_model_loaded": True,
                "no_tokenizer_loaded": True,
                "no_gpu_inspection": True,
                "supports_apply": False,
                "supports_materialization": False,
                "output_path": str(paths["engine"]),
            },
        },
    )
    _write_json(
        paths["readiness"],
        {
            "schema_version": "relaystack.hf_adapter_readiness_report.v0.1",
            "phase": "12-H",
            "artifact_kind": "hf_adapter_readiness_report",
            "input_refs": {
                "adapter_capabilities_path": str(paths["adapter"]),
                "tokenizer_span_probe_path": str(paths["tokenizer"]),
                "engine_metadata_probe_path": str(paths["engine"]),
            },
            "readiness": {
                "ready_for_next_metadata_step": True,
                "ready_for_real_tokenizer_probe": True,
                "ready_for_model_config_probe": True,
                "ready_for_materialization": False,
                "ready_for_apply": False,
                "blocking_reasons": [],
                "warning_reasons": [],
            },
            "safety_scope": {
                "dry_run_only": True,
                "no_model_loading_required": True,
                "no_tokenizer_loading_required": True,
                "no_gpu_inspection": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
            },
            "summary": {
                "ok": True,
                "artifact_kind": "hf_adapter_readiness_report",
                "check_count": 54,
                "passed_check_count": 54,
                "failed_check_count": 0,
                "ready_for_next_metadata_step": True,
                "ready_for_real_tokenizer_probe": True,
                "ready_for_model_config_probe": True,
                "ready_for_materialization": False,
                "ready_for_apply": False,
                "output_path": str(paths["readiness"]),
                "strict": False,
            },
        },
    )
    _write_json(
        paths["probe"],
        {
            "schema_version": "relaystack.hf_tokenizer_config_probe.v0.1",
            "phase": "12-I",
            "artifact_kind": "hf_tokenizer_config_probe",
            "input_refs": {
                "readiness_report_path": str(paths["readiness"]),
                "adapter_capabilities_path": str(paths["adapter"]),
                "tokenizer_span_probe_path": str(paths["tokenizer"]),
                "engine_metadata_probe_path": str(paths["engine"]),
            },
            "model_ref": model_ref,
            "tokenizer_ref": tokenizer_ref,
            "readiness_ref": {
                "path": str(paths["readiness"]),
                "ok": True,
                "ready_for_next_metadata_step": True,
                "ready_for_real_tokenizer_probe": True,
                "ready_for_model_config_probe": True,
            },
            "tokenizer_probe": {
                "attempted": False,
                "loaded": False,
                "skipped": True,
                "tokenizer_loading_error": None,
            },
            "config_probe": {
                "attempted": False,
                "loaded": False,
                "skipped": True,
                "config_loading_error": None,
            },
            "consistency": {
                "readiness_model_ref_match": True,
                "readiness_tokenizer_ref_match": True,
                "sample_span_token_count_available": True,
                "config_attention_type_hint": "unknown",
                "kv_head_group_count": None,
                "context_window_candidate": None,
            },
            "safety_scope": {
                "dry_run_only": True,
                "model_loaded": False,
                "tokenizer_loaded": False,
                "config_loaded": False,
                "no_model_loading": True,
                "no_gpu_inspection": True,
                "no_kv_materialization": True,
                "no_attention_connection": True,
                "no_scheduler_change": True,
                "no_runtime_apply": True,
                "local_files_only": True,
                "allow_network": False,
            },
            "summary": {
                "ok": True,
                "artifact_kind": "hf_tokenizer_config_probe",
                "tokenizer_loaded": False,
                "config_loaded": False,
                "ready_for_next_metadata_step": True,
                "ready_for_materialization": False,
                "ready_for_apply": False,
                "blocked_reason": None,
                "error_count": 0,
                "warning_count": 0,
                "output_path": str(paths["probe"]),
            },
            "notes": [],
        },
    )
    return paths


def test_hf_phase12_chain_acceptance_report_happy_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True
    assert loaded["acceptance"]["accepted"] is True
    assert loaded["acceptance"]["ready_for_next_metadata_step"] is True
    assert loaded["acceptance"]["ready_for_materialization"] is False
    assert loaded["acceptance"]["ready_for_apply"] is False
    assert loaded["consistency"]["readiness_input_refs_match"] is True
    assert json.loads(json.dumps(loaded)) == loaded


def test_hf_phase12_chain_acceptance_report_missing_artifact_blocks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    paths["probe"].unlink()
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 2
    assert not output_path.exists()


def test_hf_phase12_chain_acceptance_report_non_object_json_blocks_gracefully(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    _write_json(paths["adapter"], [])
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 2
    assert not output_path.exists()
    assert "JSON object" in result.stdout or "non-object" in result.stdout


def test_hf_phase12_chain_acceptance_report_model_ref_mismatch_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    probe_payload = json.loads(paths["probe"].read_text(encoding="utf-8"))
    probe_payload["model_ref"]["model_id"] = "other/model"
    _write_json(paths["probe"], probe_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["model_ref_consistent"] is False


def test_hf_phase12_chain_acceptance_report_tokenizer_ref_mismatch_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    probe_payload = json.loads(paths["probe"].read_text(encoding="utf-8"))
    probe_payload["tokenizer_ref"]["tokenizer_name_or_path"] = "other/tokenizer"
    _write_json(paths["probe"], probe_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["tokenizer_ref_consistent"] is False


def test_hf_phase12_chain_acceptance_report_readiness_not_ok_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    readiness_payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    readiness_payload["summary"]["ok"] = False
    _write_json(paths["readiness"], readiness_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["readiness_gate_ok"] is False


def test_hf_phase12_chain_acceptance_report_stale_readiness_adapter_path_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    readiness_payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    readiness_payload["input_refs"]["adapter_capabilities_path"] = str(tmp_path / "other_adapter.json")
    _write_json(paths["readiness"], readiness_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["acceptance"]["accepted"] is False
    assert loaded["consistency"]["readiness_input_refs_match"] is False
    assert any("input_refs.adapter_capabilities_path" in reason for reason in loaded["acceptance"]["blocking_reasons"])


def test_hf_phase12_chain_acceptance_report_stale_readiness_tokenizer_path_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    readiness_payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    readiness_payload["input_refs"]["tokenizer_span_probe_path"] = str(tmp_path / "other_tokenizer.json")
    _write_json(paths["readiness"], readiness_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["consistency"]["readiness_input_refs_match"] is False
    assert any("input_refs.tokenizer_span_probe_path" in reason for reason in loaded["acceptance"]["blocking_reasons"])


def test_hf_phase12_chain_acceptance_report_stale_readiness_engine_path_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    readiness_payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    readiness_payload["input_refs"]["engine_metadata_probe_path"] = str(tmp_path / "other_engine.json")
    _write_json(paths["readiness"], readiness_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["consistency"]["readiness_input_refs_match"] is False
    assert any("input_refs.engine_metadata_probe_path" in reason for reason in loaded["acceptance"]["blocking_reasons"])


def test_hf_phase12_chain_acceptance_report_missing_readiness_input_ref_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    readiness_payload = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    del readiness_payload["input_refs"]["adapter_capabilities_path"]
    _write_json(paths["readiness"], readiness_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["readiness_input_refs_match"] is False
    assert any("input_refs.adapter_capabilities_path" in reason for reason in loaded["acceptance"]["blocking_reasons"])


def test_hf_phase12_chain_acceptance_report_tokenizer_config_hard_failure_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    probe_payload = json.loads(paths["probe"].read_text(encoding="utf-8"))
    probe_payload["summary"]["ok"] = False
    probe_payload["summary"]["blocked_reason"] = "RuntimeError: tokenizer load failure"
    probe_payload["tokenizer_probe"] = {
        "attempted": True,
        "loaded": False,
        "skipped": False,
        "tokenizer_loading_error": "RuntimeError: tokenizer load failure",
    }
    probe_payload["config_probe"] = {
        "attempted": True,
        "loaded": False,
        "skipped": False,
        "config_loading_error": "RuntimeError: config load failure",
    }
    _write_json(paths["probe"], probe_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["consistency"]["tokenizer_config_probe_accepted"] is False


def test_hf_phase12_chain_acceptance_report_tokenizer_config_skip_unavailable_can_pass(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    probe_payload = json.loads(paths["probe"].read_text(encoding="utf-8"))
    probe_payload["summary"]["ok"] = False
    probe_payload["summary"]["blocked_reason"] = (
        "tokenizer/config metadata unavailable in local-files-only metadata mode"
    )
    probe_payload["tokenizer_probe"]["skipped"] = True
    probe_payload["config_probe"]["skipped"] = True
    _write_json(paths["probe"], probe_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is True
    assert loaded["consistency"]["tokenizer_config_probe_accepted"] is True


def test_hf_phase12_chain_acceptance_report_safety_scope_violation_blocks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = _build_happy_path_payloads(tmp_path)
    probe_payload = json.loads(paths["probe"].read_text(encoding="utf-8"))
    probe_payload["safety_scope"]["model_loaded"] = True
    _write_json(paths["probe"], probe_payload)
    output_path = tmp_path / "relaystack_hf_phase12_chain_acceptance_report.json"

    result = _run(
        repo_root,
        "scripts/run_hf_phase12_chain_acceptance_report.py",
        "--adapter-capabilities",
        str(paths["adapter"]),
        "--tokenizer-span-probe",
        str(paths["tokenizer"]),
        "--engine-metadata-probe",
        str(paths["engine"]),
        "--readiness-report",
        str(paths["readiness"]),
        "--tokenizer-config-probe",
        str(paths["probe"]),
        "--output",
        str(output_path),
    )

    assert result.returncode == 1
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["ok"] is False
    assert loaded["safety_scope"]["model_loaded"] is True
