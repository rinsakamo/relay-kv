import json
import os
from pathlib import Path
import subprocess
import sys


def test_fixed_budget_selection_chain_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    input_path = tmp_path / "pipeline_scores.json"
    output_dir = tmp_path / "chain_output"
    input_path.write_text(
        json.dumps(
            {
                "block_scores": [
                    {
                        "block_id": 0,
                        "start": 0,
                        "end": 64,
                        "score": 0.95,
                        "layer_idx": 0,
                        "tier": "anchor",
                    },
                    {
                        "block_id": 1,
                        "start": 64,
                        "end": 128,
                        "score": 0.80,
                        "layer_idx": 0,
                    },
                    {
                        "block_id": 2,
                        "start": 128,
                        "end": 192,
                        "score": 0.70,
                        "layer_idx": 0,
                    },
                    {
                        "block_id": 3,
                        "start": 192,
                        "end": 256,
                        "score": 0.60,
                        "layer_idx": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_fixed_budget_selection_chain.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--total-working-budget-tokens",
            "512",
            "--block-size",
            "64",
            "--mark-recent-tail-blocks",
            "1",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    candidates = json.loads(
        (output_dir / "relaykv_candidates.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (
            output_dir / "relaykv_fixed_budget_block_selection.json"
        ).read_text(encoding="utf-8")
    )
    chain_summary = json.loads(
        (output_dir / "chain_summary.json").read_text(encoding="utf-8")
    )

    assert chain_summary["phase"] == "11-D"
    assert chain_summary["dry_run_only"] is True
    assert chain_summary["no_model_loading"] is True
    assert chain_summary["candidate_count"] == len(candidates)
    assert isinstance(chain_summary["candidate_artifact"], str)
    assert isinstance(chain_summary["selection_artifact"], str)
    assert set(chain_summary["selected_block_count_by_class"]) == {
        "recent",
        "anchor",
        "retrieved",
    }
    assert (
        chain_summary["materialized_working_tokens"]
        <= chain_summary["total_working_budget_tokens"]
    )
    assert selection["decision_state"] == chain_summary["selection_decision_state"]


def test_fixed_budget_selection_chain_rejects_top_scores_tail_marking(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    input_path = tmp_path / "top_scores.json"
    output_dir = tmp_path / "chain_output"
    input_path.write_text(
        json.dumps(
            {
                "top_scores": [
                    {"block_id": 0, "start": 0, "end": 64, "score": 0.95},
                    {"block_id": 2, "start": 128, "end": 192, "score": 0.70},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_fixed_budget_selection_chain.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--total-working-budget-tokens",
            "512",
            "--block-size",
            "64",
            "--mark-recent-tail-blocks",
            "1",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "top_scores" in result.stderr
    assert "filtered score-ranked subset" in result.stderr

