import json
import os
from math import ceil
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayKVFixedBudgetConfig,
    build_relaykv_fixed_budget_working_set_decision,
)


def test_fixed_budget_working_set_default_ratios_fit_total() -> None:
    decision = build_relaykv_fixed_budget_working_set_decision(
        config=RelayKVFixedBudgetConfig(
            total_working_budget_tokens=2048,
            block_size=64,
        )
    )

    assert decision.decision_state == "dry_run_ready"
    assert decision.fallback_reason is None
    assert decision.estimated_working_tokens == 2048
    assert decision.class_budgets["recent"].budget_tokens == 1024
    assert decision.class_budgets["anchor"].budget_tokens == 204
    assert decision.class_budgets["retrieved"].budget_tokens == 820
    assert decision.selected_block_plan["recent_block_count"] == 16
    assert decision.selected_block_plan["anchor_block_count"] == ceil(204 / 64)
    assert decision.selected_block_plan["retrieved_block_count"] == ceil(820 / 64)


def test_fixed_budget_working_set_explicit_budgets_override_ratios() -> None:
    decision = build_relaykv_fixed_budget_working_set_decision(
        config=RelayKVFixedBudgetConfig(
            total_working_budget_tokens=2048,
            recent_budget_tokens=512,
            anchor_budget_tokens=128,
            retrieved_budget_tokens=256,
            block_size=64,
        )
    )

    assert decision.decision_state == "dry_run_ready"
    assert decision.class_budgets["recent"].budget_tokens == 512
    assert decision.class_budgets["anchor"].budget_tokens == 128
    assert decision.class_budgets["retrieved"].budget_tokens == 256
    assert decision.class_budgets["recent"].source == "explicit"
    assert decision.class_budgets["anchor"].source == "explicit"
    assert decision.class_budgets["retrieved"].source == "explicit"
    assert decision.estimated_working_tokens == 896
    assert "working_budget_tokens_left_unallocated" in decision.notes


def test_fixed_budget_working_set_invalid_when_explicit_budgets_exceed_total() -> None:
    decision = build_relaykv_fixed_budget_working_set_decision(
        config=RelayKVFixedBudgetConfig(
            total_working_budget_tokens=512,
            recent_budget_tokens=256,
            anchor_budget_tokens=256,
            retrieved_budget_tokens=256,
            block_size=64,
        )
    )

    assert decision.decision_state == "invalid_budget"
    assert (
        decision.fallback_reason
        == "explicit_class_budgets_exceed_total_working_budget_tokens"
    )


def test_fixed_budget_working_set_summary_is_json_safe_and_block_counts_match() -> None:
    decision = build_relaykv_fixed_budget_working_set_decision(
        config=RelayKVFixedBudgetConfig(
            total_working_budget_tokens=1536,
            transient_budget_tokens=128,
            block_size=64,
        )
    )

    summary = decision.summary()
    assert json.loads(json.dumps(summary)) == summary
    assert summary["dry_run_only"] is True
    for class_name in ("recent", "anchor", "retrieved", "transient"):
        class_budget = summary["class_budgets"][class_name]
        expected_blocks = (
            ceil(class_budget["budget_tokens"] / 64)
            if class_budget["budget_tokens"] > 0
            else 0
        )
        assert class_budget["budget_blocks"] == expected_blocks


def test_fixed_budget_working_set_script_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    output_path = tmp_path / "relaykv_fixed_budget_working_set.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_fixed_budget_working_set_dry_run.py",
            "--total-working-budget-tokens",
            "2048",
            "--block-size",
            "64",
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
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["decision_state"] == "dry_run_ready"
    assert loaded["estimated_working_tokens"] == 2048
    assert loaded["selected_block_plan"]["recent_block_count"] == 16
