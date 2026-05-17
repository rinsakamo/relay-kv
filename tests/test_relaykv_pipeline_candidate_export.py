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
    export_pipeline_candidates_from_payload,
)


def test_export_pipeline_candidates_accepts_list_input() -> None:
    exported = export_pipeline_candidates_from_payload(
        [
            {
                "block_id": 0,
                "token_start": 0,
                "token_end": 64,
                "score": 1.0,
                "layer_id": 0,
                "is_anchor": True,
            },
            {
                "block_id": 1,
                "token_start": 64,
                "token_end": 128,
                "score": 0.5,
                "layer_id": 0,
            },
        ],
        block_size=64,
    )

    assert exported[0]["block_id"] == 0
    assert exported[0]["token_start"] == 0
    assert exported[0]["token_end"] == 64
    assert exported[0]["is_anchor"] is True
    assert exported[0]["source"] == "pipeline_scoring"
    assert exported[1]["is_retrieval_candidate"] is True


def test_export_pipeline_candidates_accepts_top_scores_pipeline_keys() -> None:
    exported = export_pipeline_candidates_from_payload(
        {
            "top_scores": [
                {
                    "block_id": 0,
                    "start": 0,
                    "end": 64,
                    "score": 0.95,
                    "layer_idx": 0,
                    "tier": "anchor",
                },
                {
                    "block_id": 2,
                    "start": 128,
                    "end": 192,
                    "selected_score": 0.70,
                    "layer_idx": 0,
                    "tier": "recent",
                },
            ]
        },
        block_size=64,
    )

    assert exported[0]["is_anchor"] is True
    assert exported[0]["layer_id"] == 0
    assert exported[1]["is_recent"] is True
    assert exported[1]["is_retrieval_candidate"] is False
    assert exported[1]["score"] == 0.70


def test_export_pipeline_candidates_rejects_tail_marking_for_top_scores() -> None:
    try:
        export_pipeline_candidates_from_payload(
            {
                "top_scores": [
                    {"block_id": 0, "start": 0, "end": 64, "score": 0.9},
                    {"block_id": 2, "start": 128, "end": 192, "score": 0.7},
                ]
            },
            block_size=64,
            mark_recent_tail_blocks=1,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("ValueError was not raised")

    assert "top_scores" in message
    assert "filtered score-ranked subset" in message
    assert "full block inventory" in message


def test_export_pipeline_candidates_recent_flag_defaults_to_non_retrieval() -> None:
    exported = export_pipeline_candidates_from_payload(
        [
            {
                "block_id": 3,
                "token_start": 192,
                "token_end": 256,
                "score": 0.4,
                "recent": True,
            }
        ],
        block_size=64,
    )

    assert exported[0]["is_recent"] is True
    assert exported[0]["is_retrieval_candidate"] is False


def test_export_pipeline_candidates_derives_block_id_and_marks_head_tail() -> None:
    exported = export_pipeline_candidates_from_payload(
        {
            "candidates": [
                {"start": 64, "end": 128, "importance_score": 0.4},
                {"start": 128, "end": 192, "importance_score": 0.3},
                {"start": 192, "end": 256, "importance_score": 0.2},
            ]
        },
        block_size=64,
        default_layer_id=7,
        mark_anchor_head_blocks=1,
        mark_recent_tail_blocks=1,
    )

    assert exported[0]["block_id"] == 1
    assert exported[0]["layer_id"] == 7
    assert exported[0]["is_anchor"] is True
    assert exported[-1]["is_recent"] is True
    assert exported[-1]["is_retrieval_candidate"] is False


def test_export_pipeline_candidates_block_scores_allow_tail_marking() -> None:
    exported = export_pipeline_candidates_from_payload(
        {
            "block_scores": [
                {"block_id": 0, "start": 0, "end": 64, "score": 0.4},
                {"block_id": 1, "start": 64, "end": 128, "score": 0.3},
                {"block_id": 2, "start": 128, "end": 192, "score": 0.2},
            ]
        },
        block_size=64,
        mark_recent_tail_blocks=1,
    )

    assert exported[-1]["block_id"] == 2
    assert exported[-1]["is_recent"] is True
    assert exported[-1]["is_retrieval_candidate"] is False


def test_export_pipeline_candidates_recent_explicit_retrieval_override_is_preserved() -> None:
    exported = export_pipeline_candidates_from_payload(
        [
            {
                "block_id": 4,
                "token_start": 256,
                "token_end": 320,
                "score": 0.6,
                "tier": "recent",
                "is_retrieval_candidate": True,
            }
        ],
        block_size=64,
    )

    assert exported[0]["is_recent"] is True
    assert exported[0]["is_retrieval_candidate"] is True


def test_export_pipeline_candidates_missing_span_is_readable() -> None:
    try:
        export_pipeline_candidates_from_payload(
            [{"block_id": 0, "score": 1.0}],
            block_size=64,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("ValueError was not raised")

    assert "row 0" in message
    assert "token_start" in message


def test_exported_pipeline_candidates_can_feed_fixed_budget_selection() -> None:
    exported = export_pipeline_candidates_from_payload(
        {
            "top_blocks": [
                {
                    "block_idx": 0,
                    "start": 0,
                    "end": 64,
                    "block_score": 0.9,
                    "layer_idx": 0,
                    "anchor": True,
                },
                {
                    "idx": 1,
                    "start": 64,
                    "end": 128,
                    "block_score": 0.8,
                    "layer_idx": 0,
                },
                {
                    "idx": 2,
                    "start": 128,
                    "end": 192,
                    "block_score": 0.7,
                    "layer_idx": 0,
                    "recent": True,
                    "is_retrieval_candidate": False,
                },
            ]
        },
        block_size=64,
    )
    candidates = [
        RelayKVBlockCandidate(
            block_id=candidate["block_id"],
            token_start=candidate["token_start"],
            token_end=candidate["token_end"],
            score=candidate["score"],
            is_recent=candidate["is_recent"],
            is_anchor=candidate["is_anchor"],
            is_retrieval_candidate=candidate["is_retrieval_candidate"],
            layer_id=candidate["layer_id"],
            kv_head_group=candidate["kv_head_group"],
        )
        for candidate in exported
    ]
    decision = build_relaykv_fixed_budget_block_selection_decision(
        fixed_budget_decision=build_relaykv_fixed_budget_working_set_decision(
            config=RelayKVFixedBudgetConfig(
                total_working_budget_tokens=512,
                block_size=64,
            )
        ),
        candidates=candidates,
        block_size=64,
    )

    assert decision.decision_state == "dry_run_ready"
    assert decision.selected_block_ids_by_class["anchor"] == [0]
    assert decision.selected_block_ids_by_class["recent"] == [2]


def test_export_pipeline_candidates_script_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    pipeline_scores_path = tmp_path / "pipeline_scores.json"
    candidates_path = tmp_path / "relaykv_candidates.json"
    selection_path = tmp_path / "relaykv_fixed_budget_from_pipeline_candidates.json"
    pipeline_scores_path.write_text(
        json.dumps(
            {
                "top_scores": [
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
                        "tier": "recent",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    export_result = subprocess.run(
        [
            sys.executable,
            "scripts/export_relaykv_pipeline_candidates.py",
            "--input",
            str(pipeline_scores_path),
            "--output",
            str(candidates_path),
            "--block-size",
            "64",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stderr

    selection_result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaykv_fixed_budget_block_selection_dry_run.py",
            "--total-working-budget-tokens",
            "512",
            "--block-size",
            "64",
            "--candidates-json",
            str(candidates_path),
            "--output",
            str(selection_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert selection_result.returncode == 0, selection_result.stderr

    exported = json.loads(candidates_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert exported[0]["token_start"] == 0
    assert selection["decision_state"] == "dry_run_ready"
    assert selection["selected_block_ids_by_class"]["anchor"] == [0]
