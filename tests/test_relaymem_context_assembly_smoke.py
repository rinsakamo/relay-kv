import json
import os
from pathlib import Path
import subprocess
import sys


def test_relaymem_context_assembly_smoke_writes_expected_json(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaymem_context_assembly_smoke.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaymem_context_assembly_smoke.py",
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

    assert set(loaded.keys()) == {"metadata", "retrieval_results", "plan", "summary"}
    assert loaded["metadata"]["script"] == "run_relaymem_context_assembly_smoke.py"
    assert loaded["metadata"]["external_backend_called"] is False
    assert isinstance(loaded["retrieval_results"], list)
    assert isinstance(loaded["plan"]["selected_items"], list)
    assert isinstance(loaded["plan"]["dropped_memory_ids"], list)
    assert loaded["plan"]["dropped_memory_ids"] == []

    total_selected_tokens = sum(
        item["estimated_tokens"] for item in loaded["plan"]["selected_items"]
    )
    assert total_selected_tokens == loaded["plan"]["total_estimated_tokens"]
    assert loaded["plan"]["total_estimated_tokens"] <= loaded["plan"]["token_budget"]

    selected_memory_sources = {
        item["memory_source"] for item in loaded["plan"]["selected_items"]
    }
    selected_backends = {
        result_item["retrieval_backend"]
        for result_item in loaded["retrieval_results"]
        if result_item["memory_id"]
        in {item["memory_id"] for item in loaded["plan"]["selected_items"]}
    }
    assert "rag_chunk" in selected_memory_sources
    assert "evidence_chain" in selected_backends

    retrieval_backends = {
        result_item["retrieval_backend"]
        for result_item in loaded["retrieval_results"]
    }
    assert "evidence_chain" in retrieval_backends
    assert loaded["summary"]["evidence_chain_present"] is True
    assert json.loads(json.dumps(loaded)) == loaded


def test_relaymem_context_assembly_smoke_low_budget_keeps_dropped_memory_ids(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaymem_context_assembly_smoke_low_budget.json"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_relaymem_context_assembly_smoke.py",
            "--output",
            str(output_path),
            "--token-budget",
            "100",
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

    assert loaded["plan"]["token_budget"] == 100
    assert loaded["plan"]["dropped_memory_ids"]
    assert loaded["plan"]["total_estimated_tokens"] <= loaded["plan"]["token_budget"]


def test_relaymem_context_assembly_smoke_runs_when_common_backend_imports_are_blocked(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "relaymem_context_assembly_smoke_blocked.json"
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name in {{"requests", "httpx", "faiss", "neo4j", "whoosh"}}:
        raise ModuleNotFoundError(f"blocked import: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

from scripts.run_relaymem_context_assembly_smoke import run_relaymem_context_assembly_smoke

out = Path({str(output_path)!r})
payload = run_relaymem_context_assembly_smoke(output=out, token_budget=100)
print(out.exists())
print(payload["metadata"]["external_backend_called"])
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
    assert stdout_lines[-2:] == ["True", "False"]
