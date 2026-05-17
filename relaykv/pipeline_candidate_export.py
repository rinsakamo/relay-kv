from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _get_alias_value(
    item: dict[str, Any],
    keys: tuple[str, ...],
) -> Any | None:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _require_alias_value(
    item: dict[str, Any],
    *,
    keys: tuple[str, ...],
    row_index: int,
    label: str,
) -> Any:
    value = _get_alias_value(item, keys)
    if value is None:
        joined = ", ".join(keys)
        raise ValueError(
            f"pipeline candidate row {row_index} is missing required field {label} "
            f"(accepted keys: {joined})"
        )
    return value


def _coerce_flag(item: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return bool(value)
    return False


def _resolve_input_rows(
    payload: Any,
    *,
    input_key: str,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if input_key != "auto":
            candidate_rows = payload.get(input_key)
            if not isinstance(candidate_rows, list):
                raise ValueError(
                    f"input_key={input_key!r} did not resolve to a list of rows"
                )
            rows = candidate_rows
        else:
            rows = None
            for key in ("top_scores", "block_scores", "candidates", "top_blocks"):
                candidate_rows = payload.get(key)
                if isinstance(candidate_rows, list):
                    rows = candidate_rows
                    break
            if rows is None:
                raise ValueError(
                    "input JSON must be a list or contain one of: "
                    "top_scores, block_scores, candidates, top_blocks"
                )
    else:
        raise ValueError(
            f"input JSON must be a list or object, got {type(payload).__name__}"
        )

    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"pipeline candidate row {row_index} must be an object, got "
                f"{type(row).__name__}"
            )
        normalized_rows.append(row)
    return normalized_rows


def export_pipeline_candidates_from_payload(
    payload: Any,
    *,
    block_size: int,
    input_key: str = "auto",
    default_layer_id: int | None = None,
    mark_recent_tail_blocks: int = 0,
    mark_anchor_head_blocks: int = 0,
) -> list[dict[str, Any]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if mark_recent_tail_blocks < 0:
        raise ValueError("mark_recent_tail_blocks must be >= 0")
    if mark_anchor_head_blocks < 0:
        raise ValueError("mark_anchor_head_blocks must be >= 0")

    rows = _resolve_input_rows(payload, input_key=input_key)
    exported: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        token_start = int(
            _require_alias_value(
                row,
                keys=("token_start", "start"),
                row_index=row_index,
                label="token_start",
            )
        )
        token_end = int(
            _require_alias_value(
                row,
                keys=("token_end", "end"),
                row_index=row_index,
                label="token_end",
            )
        )
        block_id_value = _get_alias_value(row, ("block_id", "block_idx", "idx"))
        if block_id_value is None:
            block_id_value = token_start // block_size

        score_value = _get_alias_value(
            row,
            ("score", "block_score", "importance_score", "selected_score", "retrieval_score"),
        )
        layer_id_value = _get_alias_value(row, ("layer_id", "layer_idx"))
        tier_value = row.get("tier")

        exported.append(
            {
                "block_id": int(block_id_value),
                "token_start": token_start,
                "token_end": token_end,
                "score": float(score_value) if score_value is not None else None,
                "is_recent": (
                    _coerce_flag(row, "is_recent", "recent")
                    or tier_value == "recent"
                ),
                "is_anchor": (
                    _coerce_flag(row, "is_anchor", "anchor")
                    or tier_value == "anchor"
                ),
                "is_retrieval_candidate": (
                    bool(row.get("is_retrieval_candidate"))
                    if row.get("is_retrieval_candidate") is not None
                    else True
                ),
                "layer_id": (
                    int(layer_id_value)
                    if layer_id_value is not None
                    else default_layer_id
                ),
                "kv_head_group": (
                    int(row["kv_head_group"])
                    if row.get("kv_head_group") is not None
                    else None
                ),
                "source": "pipeline_scoring",
            }
        )

    if mark_anchor_head_blocks > 0:
        for candidate in sorted(exported, key=lambda item: item["block_id"])[
            :mark_anchor_head_blocks
        ]:
            candidate["is_anchor"] = True

    if mark_recent_tail_blocks > 0:
        for candidate in sorted(exported, key=lambda item: item["block_id"])[
            -mark_recent_tail_blocks:
        ]:
            candidate["is_recent"] = True

    return exported


def export_pipeline_candidates_from_json_file(
    path: Path,
    *,
    block_size: int,
    input_key: str = "auto",
    default_layer_id: int | None = None,
    mark_recent_tail_blocks: int = 0,
    mark_anchor_head_blocks: int = 0,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return export_pipeline_candidates_from_payload(
        payload,
        block_size=block_size,
        input_key=input_key,
        default_layer_id=default_layer_id,
        mark_recent_tail_blocks=mark_recent_tail_blocks,
        mark_anchor_head_blocks=mark_anchor_head_blocks,
    )
