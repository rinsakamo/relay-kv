import json
import os
from pathlib import Path
import subprocess
import sys


def test_vram_reservation_smoke_writes_expected_json(tmp_path: Path) -> None:
    output_path = tmp_path / "vram_reservation_smoke.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_vram_reservation_smoke.py",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    with output_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert set(loaded.keys()) == {"scenario", "reservation", "decision"}
    assert loaded["decision"]["available_working_kv_budget_mib"] == 2048
    assert loaded["decision"]["status"] == "ok"


def test_vram_reservation_smoke_runs_when_gpu_related_imports_are_blocked(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "vram_reservation_smoke_blocked.json"
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name in {{"torch", "pynvml", "subprocess"}} and name != "subprocess":
        raise ModuleNotFoundError(f"blocked import: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

from scripts.run_vram_reservation_smoke import run_vram_reservation_smoke

out = Path({str(output_path)!r})
payload = run_vram_reservation_smoke(output=out)
print(out.exists())
print(payload["decision"]["status"])
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines[-2:] == ["True", "ok"]
