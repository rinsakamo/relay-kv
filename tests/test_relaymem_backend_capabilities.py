import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relaykv import (
    RelayMEMBackendCapabilities,
    build_default_fast_recall_backend_capabilities,
)


def test_default_fast_recall_backend_capabilities_are_cpu_first_and_vram_zero() -> None:
    capabilities = build_default_fast_recall_backend_capabilities()

    assert capabilities.backend_id == "relaymem.fast_recall.default"
    assert capabilities.backend_type == "fast_recall"
    assert capabilities.runs_on_cpu is True
    assert capabilities.requires_gpu is False
    assert capabilities.uses_vram is False
    assert capabilities.local_first is True
    assert capabilities.remote_allowed is False


def test_backend_capabilities_summary_is_json_safe() -> None:
    summary = build_default_fast_recall_backend_capabilities().summary()

    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize(("backend_id", "backend_type", "match"), [("", "fast_recall", "backend_id"), ("relaymem.fast_recall.default", "", "backend_type")])
def test_backend_capabilities_reject_empty_required_fields(
    backend_id: str,
    backend_type: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        RelayMEMBackendCapabilities(
            backend_id=backend_id,
            backend_type=backend_type,
        )


def test_import_from_relaykv_with_backend_capabilities_stays_torch_free() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import relaykv; "
            "assert 'torch' not in sys.modules; "
            "caps = relaykv.build_default_fast_recall_backend_capabilities(); "
            "assert caps.uses_vram is False; "
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
