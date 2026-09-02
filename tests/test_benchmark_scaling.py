"""Fast structural tests for the opt-in scaling benchmark."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_scaling.py"
    spec = importlib.util.spec_from_file_location("spathi_benchmark_scaling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark() -> ModuleType:
    return _load_benchmark_module()


def _fixed_provenance(benchmark: ModuleType) -> object:
    return benchmark.BenchmarkProvenance(
        spathi_version="0.1.0.dev0",
        implementation_sha256="a" * 64,
        python_version="3.11.0",
        platform="TestOS-1.0",
        machine="test-machine",
        logical_cpus=4,
        dependency_versions_json='{"numpy":"1.0"}',
    )


def test_benchmark_defaults_exclude_checkpoint_and_report(benchmark: ModuleType) -> None:
    args = benchmark.parse_args([])

    assert args.checkpoint is False
    assert args.report is False
    assert args.warmups == 1
    assert args.targets is None

    enabled = benchmark.parse_args(["--checkpoint", "--report"])
    assert enabled.checkpoint is True
    assert enabled.report is True


def test_benchmark_builds_explicit_target_and_overhead_flags(
    benchmark: ModuleType, tmp_path: Path
) -> None:
    dataset = benchmark.Dataset(
        expression=tmp_path / "expression.tsv",
        tf_list=tmp_path / "tf-list.txt",
        groups=tmp_path / "groups.tsv",
        gene_names=("TF", "G"),
    )
    target_list = tmp_path / "targets.txt"

    command = benchmark.build_command(
        dataset=dataset,
        target_list=target_list,
        output_dir=tmp_path / "output",
        threads=2,
        n_components=3,
        n_estimators=5,
        seed=17,
        checkpoint=False,
        report=True,
    )

    assert command[0:3] == [benchmark.sys.executable, "-m", "spathi"]
    assert command[command.index("--target-list") + 1] == str(target_list)
    assert "--no-checkpoint" in command
    assert "--report" in command
    assert "--no-progress" in command


def test_balanced_order_is_reproducible_and_rotates_every_case(
    benchmark: ModuleType,
) -> None:
    cases = [
        benchmark.BenchmarkCase(threads=value, target_count=10, target_list=None)
        for value in (1, 2, 4)
    ]

    first = benchmark.balanced_orders(cases, rounds=3, seed=99)
    second = benchmark.balanced_orders(cases, rounds=3, seed=99)

    assert first == second
    assert all(set(order) == set(cases) for order in first)
    for position in range(len(cases)):
        assert {order[position] for order in first} == set(cases)


def test_target_subsets_are_nested_and_reproducible(benchmark: ModuleType, tmp_path: Path) -> None:
    gene_names = tuple(f"G{index}" for index in range(8))

    paths = benchmark.create_target_lists(
        tmp_path,
        gene_names=gene_names,
        target_counts=[3, 6],
        seed=41,
    )

    smaller = paths[3].read_text(encoding="utf-8").splitlines()
    larger = paths[6].read_text(encoding="utf-8").splitlines()
    assert len(smaller) == 3
    assert len(larger) == 6
    assert set(smaller) < set(larger)
    assert smaller == sorted(smaller, key=gene_names.index)


def test_provenance_identifies_code_platform_and_dependencies(benchmark: ModuleType) -> None:
    provenance = benchmark.collect_provenance()

    assert provenance.spathi_version == "0.1.0.dev0"
    assert len(provenance.implementation_sha256) == 64
    assert set(provenance.implementation_sha256) <= set("0123456789abcdef")
    assert provenance.python_version
    assert provenance.platform
    assert provenance.machine
    assert provenance.logical_cpus >= 1
    dependency_versions = json.loads(provenance.dependency_versions_json)
    assert set(dependency_versions) == set(benchmark._DEPENDENCY_DISTRIBUTIONS)
    assert all(isinstance(value, str) and value for value in dependency_versions.values())


def test_csv_recorder_flushes_and_fsyncs_every_completed_row(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "benchmark-results.csv"
    mirror = io.StringIO()
    fsync_calls: list[int] = []
    real_fsync = benchmark.os.fsync

    def tracking_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(benchmark.os, "fsync", tracking_fsync)
    with result_path.open("x+", encoding="utf-8", newline="") as durable_output:
        recorder = benchmark.CsvRecorder(
            durable_output=durable_output,
            mirror_output=mirror,
            provenance=_fixed_provenance(benchmark),
        )
        recorder.write_header()
        durable_output.seek(0)
        assert len(durable_output.read().splitlines()) == 1
        durable_output.seek(0, 2)

        recorder.write_row({"run_index": 1, "status": "success"})
        durable_output.seek(0)
        first_snapshot = durable_output.read()
        assert len(first_snapshot.splitlines()) == 2
        durable_output.seek(0, 2)

        recorder.write_row({"run_index": 2, "status": "failed", "error": "boom"})
        durable_output.seek(0)
        final_snapshot = durable_output.read()

    assert len(fsync_calls) == 3
    assert mirror.getvalue() == final_snapshot
    rows = list(csv.DictReader(io.StringIO(final_snapshot)))
    assert [row["run_index"] for row in rows] == ["1", "2"]
    assert rows[0]["implementation_sha256"] == "a" * 64
    assert rows[0]["dependency_versions_json"] == '{"numpy":"1.0"}'


def test_interruption_preserves_incremental_csv_and_logs(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def measure_then_interrupt(
        command: object,
        *,
        sample_interval_seconds: float,
        show_output: bool,
        stdout_log_path: Path,
        stderr_log_path: Path,
    ) -> object:
        del command, sample_interval_seconds, show_output
        nonlocal calls
        calls += 1
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text(f"stdout-{calls}\n", encoding="utf-8")
        stderr_log_path.write_text(f"stderr-{calls}\n", encoding="utf-8")
        if calls == 2:
            raise KeyboardInterrupt
        return benchmark.ProcessMeasurement(
            wall_seconds=1.25,
            peak_rss_bytes=2048,
            exit_code=0,
            status="success",
        )

    monkeypatch.setattr(benchmark, "collect_provenance", lambda: _fixed_provenance(benchmark))
    monkeypatch.setattr(benchmark, "measure_command", measure_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark.main(
            [
                "--cells",
                "4",
                "--genes",
                "3",
                "--tfs",
                "1",
                "--groups",
                "2",
                "--n-estimators",
                "1",
                "--n-components",
                "1",
                "--threads",
                "1",
                "2",
                "--warmups",
                "0",
                "--repeats",
                "1",
                "--work-dir",
                str(tmp_path),
            ]
        )

    benchmark_root = next(tmp_path.glob("spathi-benchmark-*"))
    rows = list(csv.DictReader((benchmark_root / "benchmark-results.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert len(list((benchmark_root / "logs").glob("*.log"))) == 4
    captured = capsys.readouterr()
    assert len(list(csv.DictReader(io.StringIO(captured.out)))) == 1
    assert "Retained benchmark workspace" in captured.err
    assert "Incremental benchmark CSV" in captured.err


def test_failed_run_preserves_csv_and_diagnostic_logs(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_measurement(
        command: object,
        *,
        sample_interval_seconds: float,
        show_output: bool,
        stdout_log_path: Path,
        stderr_log_path: Path,
    ) -> object:
        del command, sample_interval_seconds, show_output
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text("partial output\n", encoding="utf-8")
        stderr_log_path.write_text("failure details\n", encoding="utf-8")
        return benchmark.ProcessMeasurement(
            wall_seconds=0.5,
            peak_rss_bytes=1024,
            exit_code=7,
            status="failed",
            error="process exited with status 7",
        )

    monkeypatch.setattr(benchmark, "collect_provenance", lambda: _fixed_provenance(benchmark))
    monkeypatch.setattr(benchmark, "measure_command", fail_measurement)
    exit_code = benchmark.main(
        [
            "--cells",
            "4",
            "--genes",
            "3",
            "--tfs",
            "1",
            "--groups",
            "2",
            "--n-estimators",
            "1",
            "--n-components",
            "1",
            "--threads",
            "1",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--work-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    benchmark_root = next(tmp_path.glob("spathi-benchmark-*"))
    rows = list(csv.DictReader((benchmark_root / "benchmark-results.csv").open(encoding="utf-8")))
    assert [(row["status"], row["exit_code"]) for row in rows] == [("failed", "7")]
    assert any(
        "failure details" in path.read_text(encoding="utf-8")
        for path in (benchmark_root / "logs").glob("*.stderr.log")
    )


@pytest.mark.parametrize(("exit_code", "status"), [(0, "success"), (7, "failed")])
def test_process_measurement_records_peak_rss_and_exit_state(
    benchmark: ModuleType, tmp_path: Path, exit_code: int, status: str
) -> None:
    measurement = benchmark.measure_command(
        [
            sys.executable,
            "-c",
            f"payload = bytearray(1_000_000); raise SystemExit({exit_code})",
        ],
        sample_interval_seconds=0.001,
        show_output=False,
        stdout_log_path=tmp_path / "stdout.log",
        stderr_log_path=tmp_path / "stderr.log",
    )

    assert measurement.exit_code == exit_code
    assert measurement.status == status
    assert measurement.wall_seconds > 0
    assert measurement.peak_rss_bytes > 0
    assert (tmp_path / "stdout.log").is_file()
    assert (tmp_path / "stderr.log").is_file()


def test_process_output_is_persisted_while_streaming_to_stderr(
    benchmark: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stdout_log = tmp_path / "logs" / "case.stdout.log"
    stderr_log = tmp_path / "logs" / "case.stderr.log"

    measurement = benchmark.measure_command(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-token'); print('stderr-token', file=sys.stderr)",
        ],
        sample_interval_seconds=0.001,
        show_output=True,
        stdout_log_path=stdout_log,
        stderr_log_path=stderr_log,
    )

    assert measurement.status == "success"
    assert stdout_log.read_text(encoding="utf-8") == "stdout-token\n"
    assert stderr_log.read_text(encoding="utf-8") == "stderr-token\n"
    streamed = capsys.readouterr().err
    assert "stdout-token" in streamed
    assert "stderr-token" in streamed


def test_sampling_interruption_terminates_and_reaps_the_process_tree(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    psutil = pytest.importorskip("psutil")
    marker = tmp_path / "pids.txt"
    child_program = """
import os
from pathlib import Path
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(f"{os.getpid()}\\n{child.pid}\\n", encoding="utf-8")
time.sleep(60)
"""
    original_rss = benchmark._process_tree_rss

    def interrupt_after_tree_started(process: object) -> int:
        if marker.exists():
            raise KeyboardInterrupt
        return original_rss(process)

    monkeypatch.setattr(benchmark, "_process_tree_rss", interrupt_after_tree_started)
    with pytest.raises(KeyboardInterrupt):
        benchmark.measure_command(
            [sys.executable, "-c", child_program, str(marker)],
            sample_interval_seconds=0.005,
            show_output=False,
            stdout_log_path=tmp_path / "cancelled.stdout.log",
            stderr_log_path=tmp_path / "cancelled.stderr.log",
        )

    pids = [int(value) for value in marker.read_text(encoding="utf-8").splitlines()]
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        active = []
        for pid in pids:
            try:
                active.append(psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                active.append(False)
        if not any(active):
            break
        time.sleep(0.02)
    assert not any(active), f"benchmark left active descendant processes: {pids}"
