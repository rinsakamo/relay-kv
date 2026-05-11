from scripts.run_vram_budget_demotion_sweep import (
    make_markdown_table,
    parse_float_list,
    parse_int_list,
)


def test_parse_float_list() -> None:
    assert parse_float_list("32, 64,128") == [32.0, 64.0, 128.0]


def test_parse_int_list() -> None:
    assert parse_int_list("1, 2,4") == [1, 2, 4]


def test_make_markdown_table_from_fake_cases() -> None:
    markdown = make_markdown_table(
        [
            {
                "case_name": "budget_32_concurrency_1",
                "global_working_kv_budget_mib": 32.0,
                "target_concurrent_requests": 1,
                "request_working_kv_budget_mib": 32.0,
                "derived_target_keep_blocks": 4,
                "keep_block_ids": [0, 1, 2, 3],
                "drop_block_ids": [4, 5],
                "working_ratio": 0.5,
                "mean_abs_diff": 0.125,
                "max_abs_diff": 0.25,
                "fallback_reason": None,
                "error": None,
            },
            {
                "case_name": "budget_32_concurrency_4",
                "global_working_kv_budget_mib": 32.0,
                "target_concurrent_requests": 4,
                "request_working_kv_budget_mib": 8.0,
                "derived_target_keep_blocks": 1,
                "keep_block_ids": [5],
                "drop_block_ids": [0, 1, 2, 3, 4],
                "working_ratio": 0.125,
                "mean_abs_diff": None,
                "max_abs_diff": None,
                "fallback_reason": "request_budget_smaller_than_one_block",
                "error": "demotion dry-run requires ...",
            },
        ]
    )

    assert "| case_name |" in markdown
    assert "budget_32_concurrency_1" in markdown
    assert "0.500000000" in markdown
    assert "request_budget_smaller_than_one_block" in markdown
    assert "demotion dry-run requires ..." in markdown
