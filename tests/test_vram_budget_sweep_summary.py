import csv
from pathlib import Path

from scripts.summarize_vram_budget_demotion_sweep import (
    build_aggregate_counts,
    classify_case,
    make_case_row,
    make_markdown,
    write_csv,
)


def test_classify_budget_not_ok_error_case() -> None:
    case = {
        "error": "demotion dry-run requires ...",
        "fallback_reason": None,
        "demotion_target_resolution": {
            "fallback_reason": "vram_budget_not_ok",
        },
        "drop_block_ids": [],
    }
    assert classify_case(case) == "budget_not_ok"


def test_classify_fullkv_within_budget_case() -> None:
    case = {
        "error": None,
        "fallback_reason": "fullkv_within_budget",
        "drop_block_ids": [],
    }
    assert classify_case(case) == "fullkv_within_budget"


def test_classify_actual_demotion_case() -> None:
    case = {
        "error": None,
        "fallback_reason": None,
        "drop_block_ids": [1, 2],
    }
    assert classify_case(case) == "actual_demotion"


def test_classify_case_error() -> None:
    case = {
        "error": "unexpected failure",
        "fallback_reason": None,
        "demotion_target_resolution": {},
        "drop_block_ids": [],
    }
    assert classify_case(case) == "case_error"


def test_markdown_generation_includes_aggregate_counts_and_case_rows() -> None:
    case_rows = [
        make_case_row(
            {
                "case_name": "budget_8_concurrency_4",
                "global_working_kv_budget_mib": 8.0,
                "target_concurrent_requests": 4,
                "request_working_kv_budget_mib": 2.0,
                "derived_target_keep_blocks": 0,
                "keep_block_ids": [],
                "drop_block_ids": [],
                "working_ratio": None,
                "mean_abs_diff": None,
                "max_abs_diff": None,
                "fallback_reason": "vram_budget_not_ok",
                "error": "demotion dry-run requires ...",
                "demotion_target_resolution": {
                    "fallback_reason": "vram_budget_not_ok"
                },
            }
        ),
        make_case_row(
            {
                "case_name": "budget_32_concurrency_1",
                "global_working_kv_budget_mib": 32.0,
                "target_concurrent_requests": 1,
                "request_working_kv_budget_mib": 32.0,
                "derived_target_keep_blocks": 1,
                "keep_block_ids": [1],
                "drop_block_ids": [0],
                "working_ratio": 0.5,
                "mean_abs_diff": 0.125,
                "max_abs_diff": 0.25,
                "fallback_reason": None,
                "error": None,
            }
        ),
    ]
    aggregate_counts = build_aggregate_counts(case_rows)

    markdown = make_markdown(
        source_input_path=Path("results/processed/example.json"),
        summary={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "device": "cpu",
            "seq_len_actual": 256,
            "block_size": 128,
            "layer_idx": 0,
            "prompt_type": "structured",
            "total_blocks": 2,
        },
        aggregate_counts=aggregate_counts,
        case_rows=case_rows,
    )

    assert "# VRAM Budget Demotion Sweep Summary" in markdown
    assert "| total_cases | 2 |" in markdown
    assert "| budget_not_ok_cases | 1 |" in markdown
    assert "| actual_demotion_cases | 1 |" in markdown
    assert "budget_32_concurrency_1" in markdown


def test_csv_generation(tmp_path: Path) -> None:
    output_csv = tmp_path / "summary.csv"
    case_rows = [
        {
            "case_name": "budget_32_concurrency_1",
            "case_class": "actual_demotion",
            "global_working_kv_budget_mib": 32.0,
            "target_concurrent_requests": 1,
            "request_working_kv_budget_mib": 32.0,
            "derived_target_keep_blocks": 1,
            "kept_blocks_count": 1,
            "dropped_blocks_count": 1,
            "working_ratio": 0.5,
            "mean_abs_diff": 0.125,
            "max_abs_diff": 0.25,
            "fallback_reason": None,
            "error": None,
        }
    ]

    write_csv(output_csv, case_rows)

    with output_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["case_name"] == "budget_32_concurrency_1"
    assert rows[0]["case_class"] == "actual_demotion"
