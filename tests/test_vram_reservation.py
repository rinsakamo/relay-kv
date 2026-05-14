import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayKVVramReservation,
    RelayKVVramReservationBudgetDecision,
    RelayKVVramReservationStatus,
    build_vram_reservation_budget_decision,
)
from relaykv.vram_reservation import (
    RelayKVVramBudgetDecision,
    build_vram_budget_decision,
)


def test_vram_reservation_summary_is_json_friendly() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=12288,
        model_weights_reserved_mib=6144,
        tts_reserved_mib=1536,
        asr_reserved_mib=1024,
        avatar_reserved_mib=512,
        safety_margin_mib=1024,
    )

    assert json.loads(json.dumps(reservation.summary())) == reservation.summary()


def test_vram_reservation_totals_are_correct() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=12288,
        model_weights_reserved_mib=6144,
        tts_reserved_mib=1536,
        asr_reserved_mib=1024,
        avatar_reserved_mib=512,
        safety_margin_mib=1024,
        other_reserved_mib=0,
    )

    assert reservation.total_reserved_mib == 10240
    assert reservation.residual_vram_mib == 2048


def test_build_vram_reservation_budget_decision_ok_for_typical_12gb_budget() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=12288,
        model_weights_reserved_mib=6144,
        tts_reserved_mib=1536,
        asr_reserved_mib=1024,
        avatar_reserved_mib=512,
        safety_margin_mib=1024,
        other_reserved_mib=0,
    )

    decision = build_vram_budget_decision(
        reservation,
        min_working_kv_budget_mib=512,
    )

    assert decision.status == RelayKVVramReservationStatus.OK
    assert decision.available_working_kv_budget_mib == 2048
    assert decision.warning is None


def test_build_vram_reservation_budget_decision_over_budget() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=4096,
        model_weights_reserved_mib=3072,
        tts_reserved_mib=1024,
        asr_reserved_mib=512,
    )

    decision = build_vram_budget_decision(reservation)

    assert decision.status == RelayKVVramReservationStatus.OVER_BUDGET
    assert decision.available_working_kv_budget_mib == 0
    assert decision.warning == "reserved VRAM exceeds total VRAM"


def test_build_vram_reservation_budget_decision_no_kv_budget() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=8192,
        model_weights_reserved_mib=6144,
        tts_reserved_mib=512,
        asr_reserved_mib=512,
        avatar_reserved_mib=256,
        safety_margin_mib=256,
    )

    decision = build_vram_budget_decision(
        reservation,
        min_working_kv_budget_mib=512,
    )

    assert decision.status == RelayKVVramReservationStatus.NO_KV_BUDGET
    assert decision.available_working_kv_budget_mib == 512
    assert (
        decision.warning
        == "residual VRAM is at or below minimum working KV budget"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"total_vram_mib": 0}, "total_vram_mib"),
        ({"total_vram_mib": -1}, "total_vram_mib"),
        ({"model_weights_reserved_mib": -1}, "model_weights_reserved_mib"),
        ({"tts_reserved_mib": -1}, "tts_reserved_mib"),
        ({"asr_reserved_mib": -1}, "asr_reserved_mib"),
        ({"avatar_reserved_mib": -1}, "avatar_reserved_mib"),
        ({"safety_margin_mib": -1}, "safety_margin_mib"),
        ({"other_reserved_mib": -1}, "other_reserved_mib"),
    ],
)
def test_vram_reservation_rejects_invalid_values(kwargs: dict, match: str) -> None:
    base_kwargs = {
        "total_vram_mib": 12288,
        "model_weights_reserved_mib": 6144,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        RelayKVVramReservation(**base_kwargs)


def test_vram_reservation_decision_summary_serializes_status_as_string() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=12288,
        model_weights_reserved_mib=6144,
    )
    decision = build_vram_budget_decision(reservation)

    assert decision.summary()["status"] == "ok"
    assert decision.summary()["reservation"]["total_vram_mib"] == 12288


def test_top_level_relaykv_exports_stay_torch_free_for_vram_reservation() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import relaykv; "
            "assert 'torch' not in sys.modules; "
            "assert relaykv.RelayKVVramReservationStatus.OK.value == 'ok'; "
            "assert relaykv.RelayKVVramReservationBudgetDecision.__name__ == 'RelayKVVramBudgetDecision'; "
            "print('ok')"
        ),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


def test_top_level_aliases_point_to_vram_reservation_schema() -> None:
    reservation = RelayKVVramReservation(
        total_vram_mib=12288,
        model_weights_reserved_mib=6144,
    )

    decision = build_vram_reservation_budget_decision(reservation)

    assert isinstance(decision, RelayKVVramBudgetDecision)
    assert isinstance(decision, RelayKVVramReservationBudgetDecision)
