import json
import os
from pathlib import Path
import subprocess
import sys


def test_pressure_shadow_quality_chain_generates_synthetic_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    output_dir = tmp_path / "chain"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_pressure_shadow_quality_chain.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    artifact_names = [
        "relaystack_dry_run.json",
        "hf_context_smoke_synthetic.json",
        "relaystack_hf_smoke_report.json",
        "synthetic_relaykv_pipeline_summary.json",
        "relaykv_pressure_shadow_quality_report.json",
        "chain_summary.json",
    ]
    for artifact_name in artifact_names:
        artifact_path = output_dir / artifact_name
        assert artifact_path.exists()
        with artifact_path.open(encoding="utf-8") as f:
            json.load(f)

    chain_summary = json.loads(
        (output_dir / "chain_summary.json").read_text(encoding="utf-8")
    )
    assert chain_summary["mode"] == "synthetic"
    assert chain_summary["artifacts"]["pressure_shadow_quality_report"].endswith(
        "relaykv_pressure_shadow_quality_report.json"
    )
    assert chain_summary["shadow_quality_test_recommended"] is True


def test_pressure_shadow_quality_chain_uses_existing_pipeline_json(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    output_dir = tmp_path / "chain"
    pipeline_path = tmp_path / "existing_pipeline.json"
    pipeline_path.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "seq_len_actual": 8192,
                "layer_idx": 14,
                "prompt_type": "structured",
                "candidate_k_len": 2048,
                "cold_k_len": 4096,
                "working_k_len": 3072,
                "full_k_len": 8192,
                "coverage_ratio": 0.5,
                "working_ratio": 0.375,
                "attention_compare": {
                    "mean_abs_diff": 0.005,
                    "max_abs_diff": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_pressure_shadow_quality_chain.py",
            "--output-dir",
            str(output_dir),
            "--relaykv-pipeline-json",
            str(pipeline_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    chain_summary = json.loads(
        (output_dir / "chain_summary.json").read_text(encoding="utf-8")
    )
    assert chain_summary["mode"] == "existing_pipeline_json"
    assert chain_summary["artifacts"]["relaykv_pipeline_summary"] == str(pipeline_path)


def test_pressure_shadow_quality_chain_runs_when_torch_and_transformers_are_blocked(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "chain"
    script = f"""
import builtins
import json
from pathlib import Path

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name in {{"torch", "transformers", "pynvml"}}:
        raise ModuleNotFoundError(f"blocked import: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

from scripts.run_relaykv_pressure_shadow_quality_chain import run_relaykv_pressure_shadow_quality_chain

summary = run_relaykv_pressure_shadow_quality_chain(output_dir=Path({str(output_dir)!r}))
print(summary["mode"])
print(summary["final_quality_status"])
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines[-2:] == ["synthetic", "recommended_quality_within_threshold"]


def test_pressure_shadow_quality_chain_resolves_relative_output_dir_from_caller_cwd(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    relative_output_dir = Path("relaykv_reltest")
    expected_output_dir = tmp_path / relative_output_dir

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/run_relaykv_pressure_shadow_quality_chain.py"),
            "--output-dir",
            str(relative_output_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    for artifact_name in [
        "chain_summary.json",
        "relaykv_pressure_shadow_quality_report.json",
    ]:
        artifact_path = expected_output_dir / artifact_name
        assert artifact_path.exists()
        with artifact_path.open(encoding="utf-8") as f:
            json.load(f)
