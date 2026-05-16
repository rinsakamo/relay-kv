#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_synthetic_hf_context_smoke_payload() -> dict[str, Any]:
    return {
        "script": "hf_context_length_smoke.py",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "env": {
            "torch": "synthetic-torch",
            "cuda_version_torch": "synthetic-cuda",
            "transformers": "synthetic-transformers",
        },
        "load": {
            "ok": True,
            "config": {
                "model_type": "synthetic",
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 8192,
            },
        },
        "results": [
            {
                "target_context_tokens": 4096,
                "input_tokens": 4100,
                "ok": True,
                "elapsed_sec": 1.0,
                "new_tokens": 16,
                "tokens_per_sec_new": 16.0,
                "after": {
                    "cuda": {
                        "peak_allocated_mib": 7000.0,
                        "peak_reserved_mib": 7800.0,
                    }
                },
            },
            {
                "target_context_tokens": 8192,
                "input_tokens": 8200,
                "ok": False,
                "error": {
                    "type": "torch.cuda.OutOfMemoryError",
                    "message": "synthetic oom",
                },
                "after_error": {
                    "cuda": {
                        "peak_allocated_mib": 8100.0,
                        "peak_reserved_mib": 11080.0,
                    }
                },
            },
        ],
    }


def build_synthetic_relaykv_pipeline_payload() -> dict[str, Any]:
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "seq_len_actual": 8192,
        "layer_idx": 14,
        "prompt_type": "structured",
        "candidate_k_len": 2048,
        "cold_k_len": 4096,
        "working_k_len": 3072,
        "full_k_len": 8192,
        "coverage_ratio": 0.5,
        "working_ratio": 0.375,
        "attention_compare": {
            "mean_abs_diff": 0.005,
            "max_abs_diff": 0.05,
        },
    }


def _run_step(
    *,
    repo_root: Path,
    command: list[str],
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return {
        "command": command,
        "stdout": completed.stdout.strip(),
    }


def run_relaykv_pressure_shadow_quality_chain(
    *,
    output_dir: Path,
    relaykv_pipeline_json: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    relaystack_dry_run_path = output_dir / "relaystack_dry_run.json"
    hf_context_smoke_path = output_dir / "hf_context_smoke_synthetic.json"
    relaystack_hf_smoke_report_path = output_dir / "relaystack_hf_smoke_report.json"
    synthetic_pipeline_path = output_dir / "synthetic_relaykv_pipeline_summary.json"
    pressure_shadow_quality_report_path = (
        output_dir / "relaykv_pressure_shadow_quality_report.json"
    )
    chain_summary_path = output_dir / "chain_summary.json"

    steps: list[dict[str, Any]] = []

    relaystack_result = _run_step(
        repo_root=repo_root,
        command=[
            sys.executable,
            "scripts/run_relaystack_dry_run.py",
            "--output",
            str(relaystack_dry_run_path),
        ],
    )
    steps.append(
        {
            "step": "relaystack_dry_run",
            "command": relaystack_result["command"],
            "output": str(relaystack_dry_run_path),
        }
    )

    hf_context_smoke_payload = build_synthetic_hf_context_smoke_payload()
    hf_context_smoke_path.write_text(
        json.dumps(hf_context_smoke_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    steps.append(
        {
            "step": "hf_context_smoke_synthetic",
            "command": ["write_synthetic_hf_context_smoke_payload"],
            "output": str(hf_context_smoke_path),
        }
    )

    relaystack_hf_result = _run_step(
        repo_root=repo_root,
        command=[
            sys.executable,
            "scripts/run_relaystack_hf_smoke_report.py",
            "--hf-smoke-json",
            str(hf_context_smoke_path),
            "--relaystack-json",
            str(relaystack_dry_run_path),
            "--output",
            str(relaystack_hf_smoke_report_path),
        ],
    )
    steps.append(
        {
            "step": "relaystack_hf_smoke_report",
            "command": relaystack_hf_result["command"],
            "inputs": {
                "hf_smoke_json": str(hf_context_smoke_path),
                "relaystack_json": str(relaystack_dry_run_path),
            },
            "output": str(relaystack_hf_smoke_report_path),
        }
    )

    mode = "synthetic"
    if relaykv_pipeline_json is None:
        synthetic_pipeline_payload = build_synthetic_relaykv_pipeline_payload()
        synthetic_pipeline_path.write_text(
            json.dumps(synthetic_pipeline_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pipeline_input_path = synthetic_pipeline_path
        steps.append(
            {
                "step": "relaykv_pipeline_summary_synthetic",
                "command": ["write_synthetic_relaykv_pipeline_payload"],
                "output": str(pipeline_input_path),
            }
        )
    else:
        pipeline_input_path = relaykv_pipeline_json
        mode = "existing_pipeline_json"
        steps.append(
            {
                "step": "relaykv_pipeline_summary_existing",
                "command": ["use_existing_relaykv_pipeline_json"],
                "input": str(pipeline_input_path),
            }
        )

    shadow_quality_result = _run_step(
        repo_root=repo_root,
        command=[
            sys.executable,
            "scripts/run_relaykv_pressure_shadow_quality_report.py",
            "--relaystack-hf-report-json",
            str(relaystack_hf_smoke_report_path),
            "--relaykv-pipeline-json",
            str(pipeline_input_path),
            "--output",
            str(pressure_shadow_quality_report_path),
        ],
    )
    steps.append(
        {
            "step": "relaykv_pressure_shadow_quality_report",
            "command": shadow_quality_result["command"],
            "inputs": {
                "relaystack_hf_report_json": str(relaystack_hf_smoke_report_path),
                "relaykv_pipeline_json": str(pipeline_input_path),
            },
            "output": str(pressure_shadow_quality_report_path),
        }
    )

    final_report = json.loads(
        pressure_shadow_quality_report_path.read_text(encoding="utf-8")
    )
    chain_summary = {
        "script": "run_relaykv_pressure_shadow_quality_chain.py",
        "mode": mode,
        "artifacts": {
            "relaystack_dry_run": str(relaystack_dry_run_path),
            "hf_context_smoke": str(hf_context_smoke_path),
            "relaystack_hf_smoke_report": str(relaystack_hf_smoke_report_path),
            "relaykv_pipeline_summary": str(pipeline_input_path),
            "pressure_shadow_quality_report": str(
                pressure_shadow_quality_report_path
            ),
        },
        "steps": steps,
        "final_quality_status": final_report.get("quality_status"),
        "shadow_quality_test_recommended": final_report.get(
            "shadow_quality_test_recommended"
        ),
        "pressure_reason": final_report.get("pressure_reason"),
    }
    chain_summary_path.write_text(
        json.dumps(chain_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chain_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relaykv-pipeline-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_relaykv_pressure_shadow_quality_chain(
        output_dir=args.output_dir,
        relaykv_pipeline_json=args.relaykv_pipeline_json,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.output_dir / "relaykv_pressure_shadow_quality_report.json"),
                "chain_summary": str(args.output_dir / "chain_summary.json"),
                "mode": summary["mode"],
                "final_quality_status": summary["final_quality_status"],
                "shadow_quality_test_recommended": summary[
                    "shadow_quality_test_recommended"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
