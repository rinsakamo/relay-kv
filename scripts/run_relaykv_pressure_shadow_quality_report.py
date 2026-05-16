#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import build_relaykv_pressure_shadow_quality_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relaystack-hf-report-json", type=Path, required=True)
    parser.add_argument("--relaykv-pipeline-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mean-abs-diff-threshold", type=float, default=0.01)
    parser.add_argument("--max-abs-diff-threshold", type=float, default=0.10)
    args = parser.parse_args()

    relaystack_hf_report_payload = json.loads(
        args.relaystack_hf_report_json.read_text(encoding="utf-8")
    )
    relaykv_pipeline_payload = json.loads(
        args.relaykv_pipeline_json.read_text(encoding="utf-8")
    )
    report = build_relaykv_pressure_shadow_quality_report(
        relaystack_hf_report_payload=relaystack_hf_report_payload,
        relaykv_pipeline_payload=relaykv_pipeline_payload,
        mean_abs_diff_threshold=args.mean_abs_diff_threshold,
        max_abs_diff_threshold=args.max_abs_diff_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.summary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "shadow_quality_test_recommended": (
                    report.shadow_quality_test_recommended
                ),
                "pressure_reason": report.pressure_reason,
                "quality_status": report.quality_status,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
