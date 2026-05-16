import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import build_relaykv_pressure_shadow_quality_report


def _make_relaystack_hf_report_payload(
    *,
    recommended: bool,
    reason: str | None,
) -> dict:
    return {
        "relaykv_shadow_quality_test_recommended": recommended,
        "relaykv_shadow_quality_test_reason": reason,
        "relaykv_shadow_quality_test_inputs": {
            "any_oom": reason == "oom_observed",
            "first_failed_context_tokens": 8192 if recommended else None,
            "measured_vram_pressure_level": "oom_observed" if recommended else "low",
            "vram_reservation_status": "ok",
        },
    }


def _make_pipeline_payload(
    *,
    mean_abs_diff: float | None = 0.005,
    max_abs_diff: float | None = 0.05,
) -> dict:
    payload = {
        "seq_len_actual": 8192,
        "layer_idx": 14,
        "prompt_type": "structured",
        "candidate_k_len": 2048,
        "cold_k_len": 4096,
        "working_k_len": 3072,
        "full_k_len": 8192,
        "coverage_ratio": 0.5,
        "working_ratio": 0.375,
    }
    if mean_abs_diff is not None or max_abs_diff is not None:
        payload["attention_compare"] = {
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
        }
    return payload


def test_pressure_shadow_quality_report_within_threshold() -> None:
    report = build_relaykv_pressure_shadow_quality_report(
        relaystack_hf_report_payload=_make_relaystack_hf_report_payload(
            recommended=True,
            reason="oom_observed",
        ),
        relaykv_pipeline_payload=_make_pipeline_payload(),
    )

    assert report.shadow_quality_test_recommended is True
    assert report.pressure_reason == "oom_observed"
    assert report.quality_status == "recommended_quality_within_threshold"
    assert report.quality_summary is not None
    assert report.quality_summary["mean_abs_diff"] == 0.005


def test_pressure_shadow_quality_report_exceeds_threshold() -> None:
    report = build_relaykv_pressure_shadow_quality_report(
        relaystack_hf_report_payload=_make_relaystack_hf_report_payload(
            recommended=True,
            reason="context_length_failure_observed",
        ),
        relaykv_pipeline_payload=_make_pipeline_payload(
            mean_abs_diff=0.02,
            max_abs_diff=0.2,
        ),
    )

    assert report.shadow_quality_test_recommended is True
    assert report.quality_status == "recommended_quality_exceeds_threshold"


def test_pressure_shadow_quality_report_context_mismatch() -> None:
    report = build_relaykv_pressure_shadow_quality_report(
        relaystack_hf_report_payload=_make_relaystack_hf_report_payload(
            recommended=True,
            reason="context_length_failure_observed",
        ),
        relaykv_pipeline_payload={
            **_make_pipeline_payload(),
            "seq_len_actual": 1024,
        },
    )

    assert report.shadow_quality_test_recommended is True
    assert report.quality_status == "recommended_quality_context_mismatch"
    assert "recommended_quality_within_threshold" != report.quality_status
    assert any("expected=8192:observed=1024" in note for note in report.notes)


def test_pressure_shadow_quality_report_not_recommended() -> None:
    report = build_relaykv_pressure_shadow_quality_report(
        relaystack_hf_report_payload=_make_relaystack_hf_report_payload(
            recommended=False,
            reason=None,
        ),
        relaykv_pipeline_payload=_make_pipeline_payload(),
    )

    assert report.shadow_quality_test_recommended is False
    assert report.quality_status == "not_recommended"


def test_pressure_shadow_quality_report_unknown_when_quality_missing() -> None:
    report = build_relaykv_pressure_shadow_quality_report(
        relaystack_hf_report_payload=_make_relaystack_hf_report_payload(
            recommended=True,
            reason="relaykv_routed_ready",
        ),
        relaykv_pipeline_payload=_make_pipeline_payload(
            mean_abs_diff=None,
            max_abs_diff=None,
        ),
    )

    assert report.shadow_quality_test_recommended is True
    assert report.quality_status == "recommended_quality_unknown"
    assert report.quality_summary is None


def test_pressure_shadow_quality_report_script_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    relaystack_path = tmp_path / "relaystack_hf_report.json"
    pipeline_path = tmp_path / "relaykv_pipeline.json"
    out_path = tmp_path / "shadow_quality_report.json"

    relaystack_path.write_text(
        json.dumps(
            _make_relaystack_hf_report_payload(
                recommended=True,
                reason="oom_observed",
            )
        ),
        encoding="utf-8",
    )
    pipeline_path.write_text(
        json.dumps(_make_pipeline_payload()),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_pressure_shadow_quality_report.py",
            "--relaystack-hf-report-json",
            str(relaystack_path),
            "--relaykv-pipeline-json",
            str(pipeline_path),
            "--output",
            str(out_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["shadow_quality_test_recommended"] is True
    assert loaded["quality_status"] == "recommended_quality_within_threshold"
    assert json.loads(json.dumps(loaded)) == loaded
