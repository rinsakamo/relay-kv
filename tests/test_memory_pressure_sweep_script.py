import json
from pathlib import Path

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
