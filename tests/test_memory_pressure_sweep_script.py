import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_memory_pressure_sweep import run_memory_pressure_sweep


def test_memory_pressure_sweep_script_writes_expected_json(tmp_path: Path) -> None:
    output_path = tmp_path / "memory_pressure_sweep.json"

    payload = run_memory_pressure_sweep(output=output_path)

    assert output_path.exists()

    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert payload == loaded
    assert set(loaded.keys()) == {"metadata", "cases", "decisions", "summary"}
    assert loaded["metadata"]["script"] == "run_memory_pressure_sweep.py"
    assert loaded["metadata"]["schema_version"] == 1
    assert loaded["summary"]["total_decisions"] == len(loaded["decisions"])
    assert loaded["summary"]["state_counts"]

    case_names = {case["case_name"] for case in loaded["cases"]}
    assert "short_context" in case_names
    assert "fullkv_within_budget_by_bytes" in case_names
    assert "fullkv_within_budget_by_fallback_reason" in case_names
    assert "shadow_compare_not_ready" in case_names
    assert "routed_ready_under_pressure" in case_names

    assert loaded["summary"]["routed_ready_count"] >= 1
    assert loaded["summary"]["shadow_compare_not_ready_count"] >= 1
    assert loaded["summary"]["fallback_required_count"] >= 1
    assert loaded["summary"]["fullkv_within_budget_count"] >= 1

    assert json.loads(json.dumps(loaded)) == loaded


def test_memory_pressure_sweep_script_skips_short_context_for_zero_min_seq(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "memory_pressure_sweep_minseq0.json"

    payload = run_memory_pressure_sweep(
        output=output_path,
        min_seq_len_for_relaykv=0,
    )

    assert output_path.exists()
    case_names = {case["case_name"] for case in payload["cases"]}
    assert "short_context" not in case_names
    assert payload["summary"]["total_decisions"] == len(payload["decisions"])


def test_memory_pressure_sweep_script_handles_small_positive_stability_threshold(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "memory_pressure_sweep_stability005.json"

    payload = run_memory_pressure_sweep(
        output=output_path,
        min_selection_stability_ratio=0.05,
    )

    assert output_path.exists()
    case_by_name = {case["case_name"]: case for case in payload["cases"]}
    if "selection_unstable" in case_by_name:
        assert (
            case_by_name["selection_unstable"]["inputs"]["selection_stability_ratio"]
            >= 0.0
        )


def test_memory_pressure_sweep_script_skips_selection_unstable_for_zero_threshold(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "memory_pressure_sweep_stability0.json"

    payload = run_memory_pressure_sweep(
        output=output_path,
        min_selection_stability_ratio=0.0,
    )

    assert output_path.exists()
    case_names = {case["case_name"] for case in payload["cases"]}
    assert "selection_unstable" not in case_names
    assert payload["summary"]["total_decisions"] == len(payload["decisions"])


def test_memory_pressure_sweep_script_runs_when_torch_import_is_blocked(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "memory_pressure_sweep_no_torch.json"
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

from scripts.run_memory_pressure_sweep import run_memory_pressure_sweep

out = Path({str(output_path)!r})
run_memory_pressure_sweep(output=out)
print(out.exists())
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
    assert output_path.exists()
    assert result.stdout.strip().endswith("True")
