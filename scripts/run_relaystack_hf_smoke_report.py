#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import build_relaystack_hf_smoke_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-smoke-json", type=Path, required=True)
    parser.add_argument("--relaystack-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hf_smoke_payload = json.loads(args.hf_smoke_json.read_text(encoding="utf-8"))
    relaystack_payload = json.loads(args.relaystack_json.read_text(encoding="utf-8"))
    report = build_relaystack_hf_smoke_report(
        hf_smoke_payload=hf_smoke_payload,
        relaystack_payload=relaystack_payload,
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
                "hf_load_ok": report.hf_load_ok,
                "max_ok_context_tokens": report.max_ok_context_tokens,
                "first_failed_context_tokens": report.first_failed_context_tokens,
                "any_oom": report.any_oom,
                "final_routing_state": report.final_routing_state,
                "measured_vram_pressure_level": (
                    report.measured_vram_pressure_level
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
