"""Small contract checks for the opt-in process/thread engineering experiment."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "benchmarks/benchmark_parallel.py"
    spec = importlib.util.spec_from_file_location("spathi_parallel_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scientific_fingerprint_ignores_only_fit_timing() -> None:
    module = _module()
    execution = module.prepare_case("smoke", 123)
    result, evidence = module.fit_task(execution.tasks[0], execution.context)
    changed_time = replace(result, stat=replace(result.stat, fit_seconds=123.0))
    changed_seed = replace(
        result, stat=replace(result.stat, random_seed=result.stat.random_seed + 1)
    )
    original = module.result_fingerprint([(result, evidence)])
    assert module.result_fingerprint([(changed_time, {"pid": 999})]) == original
    assert module.result_fingerprint([(changed_seed, evidence)]) != original


@pytest.mark.parametrize(
    "changed",
    ["src/spathi/core.py", "benchmarks/benchmark_parallel.py", "benchmarks/benchmark_scaling.py"],
)
def test_source_identity_rejects_changes_to_code_and_measurement_helper(
    tmp_path: Path,
    changed: str,
) -> None:
    module = _module()
    for relative in (
        "src/spathi/core.py",
        "benchmarks/benchmark_parallel.py",
        "benchmarks/benchmark_scaling.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original = True\n")
    identity = module.source_identity(tmp_path)
    module.require_source_identity(identity, tmp_path)
    (tmp_path / changed).write_text("original = False\n")
    with pytest.raises(RuntimeError, match="source changed during measurement"):
        module.require_source_identity(identity, tmp_path)


def test_smoke_cli_compares_backends_and_reuses_process_workers(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "benchmarks/benchmark_parallel.py"
    output = tmp_path / "parallel"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "smoke",
            "--threads",
            "2",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["equivalence"] == {"smoke": True}
    assert {run["backend"] for run in report["runs"]} == {
        "sequential",
        "threading",
        "loky",
        "loky_bounded",
    }
    for run in report["runs"]:
        assert run["result"]["source_sha256"] == report["source_sha256"]
        first, second = run["result"]["measurements"]
        assert first["scientific_sha256"] == second["scientific_sha256"]
        assert first["pool_state"] == "first"
        assert second["pool_state"] == "reused"
        assert set(second["worker_pids"]) & set(first["worker_pids"])
        assert first["status_counts"] == second["status_counts"] == {"trained": 8}
        assert first["sampled_peak_process_tree_rss_bytes"] > 0
        assert second["wall_seconds"] > 0
    assert (output / "smoke-sequential.profile.txt").is_file()
