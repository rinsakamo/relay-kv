import json
import os
from pathlib import Path
import subprocess
import sys


def test_relaystack_dry_run_writes_expected_json(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_dry_run.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaystack_dry_run.py",
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
    assert stdout_summary["final_routing_state"] == "waiting_for_user_approval"

    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert set(loaded.keys()) == {
        "metadata",
        "runtime_policy",
        "relaymem",
        "relaykv",
        "relaystack",
        "user_gated_fallback",
        "summary",
    }
    assert loaded["metadata"]["no_model_loaded"] is True
    assert loaded["metadata"]["no_gpu_inspection"] is True
    assert "backend_capabilities" not in loaded["relaymem"]
    assert (
        "fast_recall_fallback_backend_capabilities" in loaded["relaymem"]
    )
    fallback_backend_capabilities = loaded["relaymem"][
        "fast_recall_fallback_backend_capabilities"
    ]
    assert fallback_backend_capabilities["backend_type"] == "fast_recall"
    assert fallback_backend_capabilities["runs_on_cpu"] is True
    assert fallback_backend_capabilities["requires_gpu"] is False
    assert fallback_backend_capabilities["uses_vram"] is False
    assert "context_assembly_plan" in loaded["relaymem"]
    assert "prompt_preview_plan" in loaded["relaymem"]
    assert "selected_items" in loaded["relaymem"]["context_assembly_plan"]
    assert "preview_items" in loaded["relaymem"]["prompt_preview_plan"]
    assert loaded["relaymem"]["prompt_preview_plan"]["retrieval_mode"] == "deep_recall"
    assert "vram_reservation_decision" in loaded["relaykv"]
    assert loaded["relaykv"]["relaykv_routing_allowed"] is True
    assert (
        "available_working_kv_budget_mib"
        in loaded["relaykv"]["vram_reservation_decision"]
    )
    assert loaded["relaykv"]["memory_pressure_decision"] is not None
    final_decision = loaded["relaystack"]["final_routing_decision"]
    assert final_decision["state"] == "waiting_for_user_approval"
    assert final_decision["relaymem_apply_allowed"] is False
    assert final_decision["approval_required"] is True
    assert "user_approval_required" in final_decision["blocking_reasons"]
    assert final_decision["proposed_retrieval_mode"] == "deep_recall"
    assert final_decision["fallback_if_denied"] == "fast_recall"
    assert loaded["user_gated_fallback"]["approval_required"] is True
    assert loaded["user_gated_fallback"]["proposed_retrieval_mode"] == "deep_recall"
    assert loaded["user_gated_fallback"]["fallback_if_denied"] == "fast_recall"
    assert (
        loaded["relaymem"]["fast_recall_fallback_backend_capabilities"] is not None
    )
    assert "deep" in loaded["user_gated_fallback"]["user_visible_message"].lower()
    assert "fast recall prepared" not in loaded["user_gated_fallback"][
        "user_visible_message"
    ].lower()
    assert loaded["relaymem"]["context_assembly_plan"]["approval_required"] is True
    assert loaded["relaymem"]["context_assembly_plan"]["approval_reason"] is not None
    assert loaded["relaymem"]["context_assembly_plan"]["user_visible_message"] is not None
    assert "deep" in loaded["relaymem"]["context_assembly_plan"][
        "user_visible_message"
    ].lower()
    assert (
        loaded["user_gated_fallback"]["approval_required"]
        == loaded["relaymem"]["prompt_preview_plan"]["approval_required"]
    )
    assert (
        loaded["user_gated_fallback"]["approval_reason"]
        == loaded["relaymem"]["prompt_preview_plan"]["approval_reason"]
    )
    assert (
        loaded["user_gated_fallback"]["fallback_if_denied"]
        == loaded["relaymem"]["prompt_preview_plan"]["fallback_if_denied"]
    )
    assert (
        loaded["user_gated_fallback"]["fallback_reason"]
        == loaded["relaymem"]["prompt_preview_plan"]["fallback_reason"]
    )
    assert (
        loaded["user_gated_fallback"]["user_visible_message"]
        == loaded["relaymem"]["prompt_preview_plan"]["user_visible_message"]
    )
    assert (
        loaded["user_gated_fallback"]["can_apply_without_user_approval"]
        == loaded["relaymem"]["prompt_preview_plan"][
            "can_apply_without_user_approval"
        ]
    )
    assert loaded["summary"]["preview_item_count"] == len(
        loaded["relaymem"]["prompt_preview_plan"]["preview_items"]
    )
    assert loaded["summary"]["prompt_preview_dropped_memory_count"] == len(
        loaded["relaymem"]["prompt_preview_plan"]["dropped_memory_ids"]
    )
    assert loaded["summary"]["prompt_preview_approval_required"] is True
    assert loaded["summary"]["prompt_preview_can_apply_without_user_approval"] is False
    assert loaded["summary"]["final_routing_state"] == "waiting_for_user_approval"
    assert loaded["summary"]["relaymem_apply_allowed"] is False
    assert loaded["summary"]["final_relaykv_routing_allowed"] is False
    assert "user_approval_required" in loaded["summary"]["final_blocking_reasons"]
    assert json.loads(json.dumps(loaded)) == loaded


def test_relaystack_dry_run_blocks_routing_when_no_kv_budget(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_dry_run_no_kv_budget.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaystack_dry_run.py",
            "--output",
            str(output_path),
            "--min-working-kv-budget-mib",
            "2048",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["relaykv"]["vram_reservation_decision"]["status"] in {
        "no_kv_budget",
        "over_budget",
    }
    assert loaded["relaykv"]["memory_pressure_decision"] is None
    assert loaded["relaykv"]["relaykv_routing_allowed"] is False
    final_decision = loaded["relaystack"]["final_routing_decision"]
    assert final_decision["state"] == "blocked_no_kv_budget"
    assert final_decision["relaykv_routing_allowed"] is False
    assert final_decision["relaymem_apply_allowed"] is False
    assert any(
        "vram_status" in reason or "no_kv_budget" in reason
        for reason in final_decision["blocking_reasons"]
    )
    assert (
        loaded["relaykv"]["memory_pressure_note"]
        == "skipped: vram reservation status is not ok"
    )
    assert "relaykv_routed_ready" not in json.dumps(loaded)


def test_relaystack_dry_run_no_kv_budget_preserves_relaymem_only_when_ungated(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaystack_dry_run_no_kv_relaymem_only.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaystack_dry_run.py",
            "--output",
            str(output_path),
            "--disable-approval-gate",
            "--min-working-kv-budget-mib",
            "2048",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["relaykv"]["vram_reservation_decision"]["status"] != "ok"
    assert (
        loaded["relaymem"]["prompt_preview_plan"]["can_apply_without_user_approval"]
        is True
    )
    final_decision = loaded["relaystack"]["final_routing_decision"]
    assert final_decision["state"] == "relaymem_only"
    assert final_decision["relaymem_apply_allowed"] is True
    assert final_decision["relaykv_routing_allowed"] is False
    assert final_decision["selected_retrieval_mode"] == "deep_recall"
    assert any(
        "vram_status" in reason or "no_kv_budget" in reason
        for reason in final_decision["blocking_reasons"]
    )


def test_relaystack_dry_run_disable_approval_gate(tmp_path: Path) -> None:
    output_path = tmp_path / "relaystack_dry_run_no_gate.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaystack_dry_run.py",
            "--output",
            str(output_path),
            "--disable-approval-gate",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["user_gated_fallback"]["approval_required"] is False
    assert loaded["user_gated_fallback"]["proposed_retrieval_mode"] is None
    assert loaded["relaymem"]["prompt_preview_plan"]["approval_required"] is False
    assert (
        loaded["relaymem"]["prompt_preview_plan"]["can_apply_without_user_approval"]
        is True
    )
    assert loaded["user_gated_fallback"]["can_apply_without_user_approval"] is True
    assert not loaded["user_gated_fallback"]["approval_reason"]
    assert loaded["user_gated_fallback"]["fallback_if_denied"] is None
    assert loaded["relaymem"]["fast_recall_fallback_backend_capabilities"] is None
    assert (
        loaded["relaymem"]["prompt_preview_plan"]["retrieval_mode"] == "deep_recall"
    )
    assert "deep" in loaded["user_gated_fallback"]["user_visible_message"].lower()
    assert "Apply these Fast Recall memories" not in loaded["user_gated_fallback"][
        "user_visible_message"
    ]
    assert "fast recall prepared" not in loaded["user_gated_fallback"][
        "user_visible_message"
    ].lower()
    final_decision = loaded["relaystack"]["final_routing_decision"]
    assert final_decision["approval_required"] is False
    assert final_decision["relaymem_apply_allowed"] is True
    assert final_decision["state"] in {
        "relaymem_and_relaykv_ready",
        "relaymem_only",
    }
    if final_decision["state"] == "relaymem_and_relaykv_ready":
        assert final_decision["relaykv_routing_allowed"] is True


def test_relaystack_dry_run_tight_budget_uses_prompt_preview_fallback(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaystack_dry_run_tight_budget.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaystack_dry_run.py",
            "--output",
            str(output_path),
            "--token-budget",
            "1",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    prompt_preview_plan = loaded["relaymem"]["prompt_preview_plan"]
    assert prompt_preview_plan["retrieval_mode"] == "deep_recall"
    assert prompt_preview_plan["fallback_reason"] == "token_budget_exceeded"
    assert prompt_preview_plan["can_apply_without_user_approval"] is False
    assert (
        prompt_preview_plan["dropped_memory_ids"]
        or prompt_preview_plan["preview_items"] == []
    )
    assert "Apply these Fast Recall memories" not in prompt_preview_plan[
        "user_visible_message"
    ]
    assert "deep" in prompt_preview_plan["user_visible_message"].lower()
    assert "budget" in prompt_preview_plan["user_visible_message"].lower() or "fallback" in prompt_preview_plan["user_visible_message"].lower()
    assert "fast recall" not in prompt_preview_plan["user_visible_message"].lower()
    assert loaded["user_gated_fallback"]["fallback_reason"] == "token_budget_exceeded"
    assert loaded["user_gated_fallback"]["can_apply_without_user_approval"] is False
    assert "deep" in loaded["user_gated_fallback"]["user_visible_message"].lower()
    assert "fast recall" not in loaded["user_gated_fallback"]["user_visible_message"].lower()
    assert loaded["summary"]["prompt_preview_fallback_reason"] == "token_budget_exceeded"
    final_decision = loaded["relaystack"]["final_routing_decision"]
    assert final_decision["state"] != "waiting_for_user_approval"
    assert final_decision["state"] != "relaymem_and_relaykv_ready"
    assert final_decision["state"] == "fallback_safe_baseline"
    assert final_decision["relaymem_apply_allowed"] is False
    assert final_decision["fallback_reason"] == "token_budget_exceeded"
    assert "token_budget_exceeded" in final_decision["blocking_reasons"]
    assert final_decision["blocking_reasons"] != ["user_approval_required"]
    assert (
        loaded["summary"]["final_fallback_reason"] == "token_budget_exceeded"
        or "token_budget_exceeded" in loaded["summary"]["final_blocking_reasons"]
    )


def test_relaystack_dry_run_runs_when_model_and_gpu_imports_are_blocked(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaystack_dry_run_blocked.json"
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name in {{"torch", "pynvml", "transformers", "requests", "httpx"}}:
        raise ModuleNotFoundError(f"blocked import: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

from scripts.run_relaystack_dry_run import run_relaystack_dry_run

out = Path({str(output_path)!r})
payload = run_relaystack_dry_run(output=out)
print(out.exists())
print(payload["metadata"]["no_model_loaded"])
print(payload["metadata"]["no_gpu_inspection"])
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
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines[-3:] == ["True", "True", "True"]
