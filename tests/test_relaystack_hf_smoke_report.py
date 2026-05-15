import json
import os
from pathlib import Path
import subprocess
import sys

from relaykv import build_relaystack_hf_smoke_report


def _make_relaystack_payload(
    *,
    final_routing_state: str = "relaymem_and_relaykv_ready",
    relaymem_apply_allowed: bool = True,
    relaykv_routing_allowed: bool = True,
    fallback_reason: str | None = None,
    blocking_reasons: list[str] | None = None,
    vram_status: str = "ok",
    available_working_kv_budget_mib: int = 2048,
    total_vram_mib: int = 12288,
) -> dict:
    return {
        "relaystack": {
            "final_routing_decision": {
                "state": final_routing_state,
                "relaymem_apply_allowed": relaymem_apply_allowed,
                "relaykv_routing_allowed": relaykv_routing_allowed,
                "fallback_reason": fallback_reason,
                "blocking_reasons": blocking_reasons or [],
            }
        },
        "relaykv": {
            "vram_reservation": {
                "total_vram_mib": total_vram_mib,
            },
            "vram_reservation_decision": {
                "status": vram_status,
                "available_working_kv_budget_mib": available_working_kv_budget_mib,
            },
        },
        "summary": {
            "available_working_kv_budget_mib": available_working_kv_budget_mib,
            "final_routing_state": final_routing_state,
        },
    }


def _make_hf_payload(
    *,
    load_ok: bool = True,
    rows: list[dict] | None = None,
) -> dict:
    return {
        "script": "hf_context_length_smoke.py",
        "model": "synthetic/model",
        "env": {
            "torch": "synthetic-torch",
            "cuda_version_torch": "synthetic-cuda",
            "transformers": "synthetic-transformers",
        },
        "load": {
            "ok": load_ok,
            "config": {
                "model_type": "synthetic",
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 8192,
            },
        }
        if load_ok
        else {
            "ok": False,
            "error": {"type": "RuntimeError", "message": "synthetic load failure"},
        },
        "results": rows or [],
    }


def test_build_relaystack_hf_smoke_report_all_lengths_ok() -> None:
    hf_payload = _make_hf_payload(
        rows=[
            {
                "target_context_tokens": 4096,
                "input_tokens": 4100,
                "ok": True,
                "elapsed_sec": 1.0,
                "new_tokens": 16,
                "tokens_per_sec_new": 16.0,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 5000.0,
                        "peak_reserved_mib": 6000.0,
                    }
                },
            },
            {
                "target_context_tokens": 8192,
                "input_tokens": 8200,
                "ok": True,
                "elapsed_sec": 2.0,
                "new_tokens": 16,
                "tokens_per_sec_new": 8.0,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 6500.0,
                        "peak_reserved_mib": 7000.0,
                    }
                },
            },
        ]
    )
    relaystack_payload = _make_relaystack_payload()

    report = build_relaystack_hf_smoke_report(
        hf_smoke_payload=hf_payload,
        relaystack_payload=relaystack_payload,
    )

    assert report.hf_load_ok is True
    assert report.max_ok_context_tokens == 8192
    assert report.first_failed_context_tokens is None
    assert report.any_oom is False
    assert report.measured_vram_pressure_level == "low"
    assert report.final_routing_state == "relaymem_and_relaykv_ready"
    assert report.hf_all_lengths_ok is True


def test_build_relaystack_hf_smoke_report_oom_at_longer_length() -> None:
    hf_payload = _make_hf_payload(
        rows=[
            {
                "target_context_tokens": 4096,
                "input_tokens": 4100,
                "ok": True,
                "elapsed_sec": 1.0,
                "new_tokens": 16,
                "tokens_per_sec_new": 16.0,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 7000.0,
                        "peak_reserved_mib": 7800.0,
                    }
                },
            },
            {
                "target_context_tokens": 8192,
                "input_tokens": 8200,
                "ok": False,
                "error": {
                    "type": "torch.cuda.OutOfMemoryError",
                    "message": "synthetic oom",
                },
                "after_error": {
                    "cuda": {
                        "peak_allocated_mib": 8100.0,
                        "peak_reserved_mib": 11080.0,
                    }
                },
            },
        ]
    )
    relaystack_payload = _make_relaystack_payload()

    report = build_relaystack_hf_smoke_report(
        hf_smoke_payload=hf_payload,
        relaystack_payload=relaystack_payload,
    )

    assert report.any_oom is True
    assert report.max_ok_context_tokens == 4096
    assert report.first_failed_context_tokens == 8192
    assert report.measured_vram_pressure_level == "oom_observed"
    assert "hf_oom_observed" in report.report_notes


def test_build_relaystack_hf_smoke_report_load_failed() -> None:
    hf_payload = _make_hf_payload(load_ok=False, rows=[])
    relaystack_payload = _make_relaystack_payload(
        final_routing_state="waiting_for_user_approval",
        relaymem_apply_allowed=False,
        relaykv_routing_allowed=False,
        blocking_reasons=["user_approval_required"],
    )

    report = build_relaystack_hf_smoke_report(
        hf_smoke_payload=hf_payload,
        relaystack_payload=relaystack_payload,
    )

    assert report.hf_load_ok is False
    assert report.context_rows == []
    assert "hf_load_failed" in report.report_notes
    assert "relaystack_waiting_for_user_approval" in report.report_notes


def test_build_relaystack_hf_smoke_report_load_time_oom() -> None:
    hf_payload = {
        "script": "hf_context_length_smoke.py",
        "model": "synthetic/model",
        "env": {
            "torch": "synthetic-torch",
            "cuda_version_torch": "synthetic-cuda",
            "transformers": "synthetic-transformers",
        },
        "load": {
            "ok": False,
            "error": {
                "type": "OutOfMemoryError",
                "message": "CUDA out of memory",
            },
            "cuda": {
                "peak_allocated_mib": 10000.0,
                "peak_reserved_mib": 11800.0,
            },
        },
        "results": [],
    }
    relaystack_payload = _make_relaystack_payload()

    report = build_relaystack_hf_smoke_report(
        hf_smoke_payload=hf_payload,
        relaystack_payload=relaystack_payload,
    )

    assert report.hf_load_ok is False
    assert report.any_oom is True
    assert report.measured_vram_pressure_level == "oom_observed"
    assert "hf_load_failed" in report.report_notes
    assert "hf_oom_observed" in report.report_notes
    assert report.max_peak_allocated_mib == 10000.0
    assert report.max_peak_reserved_mib == 11800.0


def test_build_relaystack_hf_smoke_report_relaymem_only_note() -> None:
    hf_payload = _make_hf_payload(
        rows=[
            {
                "target_context_tokens": 4096,
                "input_tokens": 4100,
                "ok": True,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 6000.0,
                        "peak_reserved_mib": 9000.0,
                    }
                },
            }
        ]
    )
    relaystack_payload = _make_relaystack_payload(
        final_routing_state="relaymem_only",
        relaymem_apply_allowed=True,
        relaykv_routing_allowed=False,
        vram_status="no_kv_budget",
    )

    report = build_relaystack_hf_smoke_report(
        hf_smoke_payload=hf_payload,
        relaystack_payload=relaystack_payload,
    )

    assert report.final_routing_state == "relaymem_only"
    assert report.relaykv_routing_allowed is False
    assert "relaymem_only_possible_without_relaykv" in report.report_notes
    assert "relaystack_no_kv_budget" in report.report_notes
    assert report.measured_vram_pressure_level == "moderate"


def test_run_relaystack_hf_smoke_report_script_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    hf_path = tmp_path / "hf_smoke.json"
    relaystack_path = tmp_path / "relaystack.json"
    out_path = tmp_path / "report.json"

    hf_payload = _make_hf_payload(
        rows=[
            {
                "target_context_tokens": 4096,
                "input_tokens": 4100,
                "ok": True,
                "elapsed_sec": 1.0,
                "new_tokens": 16,
                "tokens_per_sec_new": 16.0,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 6500.0,
                        "peak_reserved_mib": 7200.0,
                    }
                },
            },
            {
                "target_context_tokens": 8192,
                "input_tokens": 8200,
                "ok": True,
                "elapsed_sec": 2.0,
                "new_tokens": 16,
                "tokens_per_sec_new": 8.0,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 7600.0,
                        "peak_reserved_mib": 8600.0,
                    }
                },
            },
        ]
    )
    relaystack_payload = _make_relaystack_payload(
        final_routing_state="relaymem_only",
        relaymem_apply_allowed=True,
        relaykv_routing_allowed=False,
    )
    hf_path.write_text(json.dumps(hf_payload), encoding="utf-8")
    relaystack_path.write_text(json.dumps(relaystack_payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaystack_hf_smoke_report.py",
            "--hf-smoke-json",
            str(hf_path),
            "--relaystack-json",
            str(relaystack_path),
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
    assert out_path.exists()
    stdout_summary = json.loads(result.stdout)
    assert stdout_summary["max_ok_context_tokens"] == 8192
    assert stdout_summary["final_routing_state"] == "relaymem_only"

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["hf_load_ok"] is True
    assert loaded["max_ok_context_tokens"] == 8192
    assert loaded["final_routing_state"] == "relaymem_only"
    assert json.loads(json.dumps(loaded)) == loaded
