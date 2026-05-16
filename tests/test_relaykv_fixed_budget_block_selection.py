import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayKVBlockCandidate,
    RelayKVFixedBudgetConfig,
    build_relaykv_fixed_budget_block_selection_decision,
    build_relaykv_fixed_budget_working_set_decision,
    build_synthetic_block_candidates,
)


def test_fixed_budget_block_selection_default_synthetic_is_ready() -> None:
    decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=build_relaykv_fixed_budget_working_set_decision(
            config=RelayKVFixedBudgetConfig(
                total_working_budget_tokens=2048,
                block_size=64,
            )
        ),
        candidates=build_synthetic_block_candidates(num_blocks=64, block_size=64),
        block_size=64,
    )

    assert decision.decision_state == "dry_run_ready"
    assert decision.materialized_working_tokens <= 2048
    assert decision.estimated_working_tokens <= 2048
    assert decision.selected_block_count_by_class["recent"] <= 16
    assert decision.selected_block_count_by_class["anchor"] <= 3
    assert decision.selected_block_count_by_class["retrieved"] <= 13


def test_fixed_budget_block_selection_has_no_duplicate_block_ids_across_classes() -> None:
    decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=build_relaykv_fixed_budget_working_set_decision(
            config=RelayKVFixedBudgetConfig(
                total_working_budget_tokens=2048,
                block_size=64,
            )
        ),
        candidates=build_synthetic_block_candidates(num_blocks=64, block_size=64),
        block_size=64,
    )

    selected_ids = [
        block_id
        for block_ids in decision.selected_block_ids_by_class.values()
        for block_id in block_ids
    ]
    assert len(selected_ids) == len(set(selected_ids))


def test_fixed_budget_block_selection_rejected_and_overflow_are_json_safe() -> None:
    decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=build_relaykv_fixed_budget_working_set_decision(
            config=RelayKVFixedBudgetConfig(
                total_working_budget_tokens=1024,
                block_size=64,
            )
        ),
        candidates=build_synthetic_block_candidates(num_blocks=32, block_size=64),
        block_size=64,
    )

    summary = decision.summary()
    assert json.loads(json.dumps(summary)) == summary
    assert isinstance(summary["rejected_block_ids"], list)
    assert isinstance(summary["overflow_block_ids"], list)
    assert isinstance(summary["rejection_reason_counts"], dict)


def test_fixed_budget_block_selection_tiny_budget_stays_within_total() -> None:
    fixed_budget_decision = build_relaykv_fixed_budget_working_set_decision(
        config=RelayKVFixedBudgetConfig(
            total_working_budget_tokens=192,
            block_size=64,
        )
    )
    decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=fixed_budget_decision,
        candidates=build_synthetic_block_candidates(num_blocks=16, block_size=64),
        block_size=64,
    )
    fixed_budget_summary = fixed_budget_decision.summary()

    assert decision.materialized_working_tokens <= 192
    assert (
        decision.selected_block_count_by_class["recent"]
        <= fixed_budget_summary["selected_block_plan"]["recent_block_count"]
    )
    assert (
        decision.selected_block_count_by_class["anchor"]
        <= fixed_budget_summary["selected_block_plan"]["anchor_block_count"]
    )
    assert (
        decision.selected_block_count_by_class["retrieved"]
        <= fixed_budget_summary["selected_block_plan"]["retrieved_block_count"]
    )


def test_fixed_budget_block_selection_candidates_json_path(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "token_start": 0,
                    "token_end": 64,
                    "score": 10.0,
                    "is_recent": False,
                    "is_anchor": True,
                    "is_retrieval_candidate": True,
                },
                {
                    "block_id": 1,
                    "token_start": 64,
                    "token_end": 128,
                    "score": 9.0,
                    "is_recent": False,
                    "is_anchor": False,
                    "is_retrieval_candidate": True,
                },
                {
                    "block_id": 2,
                    "token_start": 128,
                    "token_end": 192,
                    "score": 8.0,
                    "is_recent": True,
                    "is_anchor": False,
                    "is_retrieval_candidate": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    output_path = tmp_path / "block_selection.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_fixed_budget_block_selection_dry_run.py",
            "--total-working-budget-tokens",
            "192",
            "--block-size",
            "64",
            "--candidates-json",
            str(candidates_path),
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
    assert loaded["selected_block_count_by_class"]["recent"] <= 2


def test_fixed_budget_block_selection_script_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    output_path = tmp_path / "relaykv_fixed_budget_block_selection.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_fixed_budget_block_selection_dry_run.py",
            "--total-working-budget-tokens",
            "2048",
            "--block-size",
            "64",
            "--num-blocks",
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
    assert loaded["materialized_working_tokens"] <= 2048
    assert loaded["no_kv_materialization"] is True
