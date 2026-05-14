#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv.vram_reservation import (
    RelayKVVramReservation,
    build_vram_budget_decision,
)


def run_vram_reservation_smoke(
    *,
    output: Path,
    total_vram_mib: int = 12288,
    model_weights_reserved_mib: int = 6144,
    tts_reserved_mib: int = 1536,
    asr_reserved_mib: int = 1024,
    avatar_reserved_mib: int = 512,
    safety_margin_mib: int = 1024,
    other_reserved_mib: int = 0,
    min_working_kv_budget_mib: int = 512,
) -> dict:
    reservation = RelayKVVramReservation(
        total_vram_mib=total_vram_mib,
        model_weights_reserved_mib=model_weights_reserved_mib,
        tts_reserved_mib=tts_reserved_mib,
        asr_reserved_mib=asr_reserved_mib,
        avatar_reserved_mib=avatar_reserved_mib,
        safety_margin_mib=safety_margin_mib,
        other_reserved_mib=other_reserved_mib,
    )
    decision = build_vram_budget_decision(
        reservation,
        min_working_kv_budget_mib=min_working_kv_budget_mib,
    )
    payload = {
        "scenario": {
            "script": "run_vram_reservation_smoke.py",
            "schema_version": 1,
            "notes": (
                "Offline VRAM reservation dry-run for a local multimodal stack. "
                "No GPU inspection and no model loading."
            ),
        },
        "reservation": reservation.summary(),
        "decision": decision.summary(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-vram-mib", type=int, default=12288)
    parser.add_argument("--model-weights-reserved-mib", type=int, default=6144)
    parser.add_argument("--tts-reserved-mib", type=int, default=1536)
    parser.add_argument("--asr-reserved-mib", type=int, default=1024)
    parser.add_argument("--avatar-reserved-mib", type=int, default=512)
    parser.add_argument("--safety-margin-mib", type=int, default=1024)
    parser.add_argument("--other-reserved-mib", type=int, default=0)
    parser.add_argument("--min-working-kv-budget-mib", type=int, default=512)
    args = parser.parse_args()

    payload = run_vram_reservation_smoke(
        output=args.output,
        total_vram_mib=args.total_vram_mib,
        model_weights_reserved_mib=args.model_weights_reserved_mib,
        tts_reserved_mib=args.tts_reserved_mib,
        asr_reserved_mib=args.asr_reserved_mib,
        avatar_reserved_mib=args.avatar_reserved_mib,
        safety_margin_mib=args.safety_margin_mib,
        other_reserved_mib=args.other_reserved_mib,
        min_working_kv_budget_mib=args.min_working_kv_budget_mib,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output),
                "available_working_kv_budget_mib": payload["decision"][
                    "available_working_kv_budget_mib"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
