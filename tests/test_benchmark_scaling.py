"""Fast structural tests for the opt-in scaling benchmark."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
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
        benchmark_sha256="b" * 64,
        python_version="3.11.0",
        platform="TestOS-1.0",
        machine="test-machine",
        logical_cpus=4,
        dependency_versions_json='{"numpy":"1.0"}',
    )


def _run_metadata_document() -> dict[str, object]:
    return {
        "status": "complete",
        "input_dimensions": {
            "cells": 4,
            "genes": 3,
            "targets": 3,
            "transcription_factors": 1,
            "groups": 2,
        },
        "models": {
            "requested": 6,
            "completed": 6,
            "trained": 6,
            "preflight_skipped": 0,
            "fit_or_importance_failures": 0,
            "trained_with_positive_edges": 5,
            "trained_without_positive_edges": 1,
            "positive_edges": 5,
            "reused_from_checkpoint": 0,
            "processed_this_attempt": 6,
        },
        "parallelism": {
            "threads_effective": 1,
            "threads_available": 4,
            "inference_thread_budget": 1,
            "maximum_concurrent_model_fits": 1,
            "memory_concurrent_model_cap": 1,
            "memory_available_bytes_at_planning": 8_000_000_000,
            "memory_usable_bytes_at_planning": 6_000_000_000,
            "memory_usable_fraction": 0.75,
            "memory_reserved_for_batch_bytes": 1_000_000,
            "backend": "sequential",
            "parallel_level": "sequential",
            "persistent_worker_pool": False,
        },
        "effective_parameters": {
            "effective_n_components": 1,
            "maximum_informative_n_components": 3,
            "pca_svd_solver_resolution": "full",
            "bandwidth": {
                "method": "median-positive-distance",
                "value": 1.25,
                "automatic_reference_value": 1.25,
                "automatic_scale": 1.0,
                "positive_distance_count": 8,
                "fallback_reason": None,
            },
            "tree_target_dtype": "float32",
            "tree_predictor_dtype": "float32",
            "bootstrap_effective": False,
            "targets_per_batch": 3,
            "targets_per_batch_without_memory_limit": 3,
            "target_groups_per_batch": 2,
            "target_groups_per_batch_without_memory_limit": 2,
            "cell_centroid_distance_storage": "memory",
            "cell_centroid_distances_computed": True,
            "distance_storage_reason": "fits-in-memory",
            "centroid_distance_memory_available_bytes_at_planning": 8_000_000_000,
            "centroid_distance_memory_usable_bytes_at_planning": 6_000_000_000,
            "distance_memory_available_bytes_at_planning": 8_000_000_000,
            "distance_memory_usable_bytes_at_planning": 6_000_000_000,
        },
        "phase_times_seconds": {
            "input_validation": 0.01,
            "distance_representation": 0.02,
            "centroids_and_distances": 0.03,
            "bandwidth_selection": 0.04,
            "inference_preparation": 0.05,
            "weighting_and_diagnostics": 0.06,
            "model_inference": 0.07,
            "artifact_writing": 0.08,
            "report": 0.0,
            "total": 0.36,
        },
    }


def _write_run_metadata_for_command(command: object) -> Path:
    assert isinstance(command, list)
    output_dir = Path(command[command.index("--output-dir") + 1])
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(_run_metadata_document()), encoding="utf-8")
    return metadata_path


def test_benchmark_defaults_exclude_checkpoint_and_report(benchmark: ModuleType) -> None:
    args = benchmark.parse_args([])

    assert args.checkpoint is False
    assert args.report is False
    assert args.warmups == 1
    assert args.targets is None
    assert args.case_timeout_seconds == 3_600.0
    assert args.threads == [1, 2, "auto"]

    enabled = benchmark.parse_args(["--checkpoint", "--report"])
    assert enabled.checkpoint is True
    assert enabled.report is True


def test_benchmark_accepts_public_thread_budgets(
    benchmark: ModuleType,
) -> None:
    assert benchmark.parse_args(["--threads", "auto", "4"]).threads == ["auto", 4]


def test_sigterm_uses_the_interrupt_cleanup_path(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_handler = object()
    installed_handlers: list[object] = []
    monkeypatch.setattr(benchmark.signal, "getsignal", lambda _signal: previous_handler)
    monkeypatch.setattr(
        benchmark.signal,
        "signal",
        lambda _signal, handler: installed_handlers.append(handler),
    )

    with benchmark._sigterm_as_keyboard_interrupt():
        with pytest.raises(KeyboardInterrupt):
            installed_handlers[-1](benchmark.signal.SIGTERM, None)

    assert installed_handlers[-1] is previous_handler


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_case_timeout_must_be_positive_and_finite(benchmark: ModuleType, value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--case-timeout-seconds", value])


@pytest.mark.parametrize("value", ["-1", str(2**32)])
def test_benchmark_seed_must_fit_sklearn_uint32(benchmark: ModuleType, value: str) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--seed", value])


def test_benchmark_builds_explicit_target_and_overhead_flags(
    benchmark: ModuleType, tmp_path: Path
) -> None:
    dataset = benchmark.Dataset(
        expression=tmp_path / "expression.tsv",
        tf_list=tmp_path / "tf-list.txt",
        groups=tmp_path / "groups.tsv",
        gene_names=("TF", "G"),
        n_groups=2,
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
    assert command[command.index("--weight-mode") + 1] == "cell-distance-group-anchored"
    assert command[command.index("--distance-metric") + 1] == "cosine"
    assert command[command.index("--bandwidth") + 1] == "auto"
    assert command[command.index("--bandwidth-scale") + 1] == "1.0"
    assert command[command.index("--group-size-correction") + 1] == "cap-to-target"


def test_benchmark_builds_the_required_single_group_weighting_configuration(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    dataset = benchmark.Dataset(
        expression=tmp_path / "expression.tsv",
        tf_list=tmp_path / "tf-list.txt",
        groups=tmp_path / "groups.tsv",
        gene_names=("TF", "G"),
        n_groups=1,
    )

    command = benchmark.build_command(
        dataset=dataset,
        target_list=None,
        output_dir=tmp_path / "output",
        threads=1,
        n_components=1,
        n_estimators=1,
        seed=17,
        checkpoint=False,
        report=False,
    )

    assert command[command.index("--weight-mode") + 1] == "cell-distance"
    assert command[command.index("--distance-metric") + 1] == "euclidean"
    assert command[command.index("--group-size-correction") + 1] == "none"


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


def test_synthetic_dataset_is_blockwise_deterministic_and_well_formed(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "_SYNTHETIC_GENE_BLOCK_SIZE", 2)
    first = benchmark.create_dataset(
        tmp_path / "first",
        n_cells=9,
        n_genes=7,
        n_tfs=2,
        n_groups=3,
        seed=41,
    )
    second = benchmark.create_dataset(
        tmp_path / "second",
        n_cells=9,
        n_genes=7,
        n_tfs=2,
        n_groups=3,
        seed=41,
    )

    assert first.expression.read_bytes() == second.expression.read_bytes()
    assert first.tf_list.read_bytes() == second.tf_list.read_bytes()
    assert first.groups.read_bytes() == second.groups.read_bytes()
    expression = benchmark.pd.read_csv(first.expression, sep="\t", index_col=0)
    assert expression.shape == (7, 9)
    assert tuple(map(str, expression.index)) == first.gene_names
    assert (expression.to_numpy() >= 0.0).all()


def test_provenance_identifies_code_platform_and_dependencies(benchmark: ModuleType) -> None:
    provenance = benchmark.collect_provenance()

    assert provenance.spathi_version == "0.1.0.dev0"
    assert len(provenance.implementation_sha256) == 64
    assert set(provenance.implementation_sha256) <= set("0123456789abcdef")
    assert len(provenance.benchmark_sha256) == 64
    assert set(provenance.benchmark_sha256) <= set("0123456789abcdef")
    assert provenance.python_version
    assert provenance.platform
    assert provenance.machine
    assert provenance.logical_cpus >= 1
    dependency_versions = json.loads(provenance.dependency_versions_json)
    assert set(dependency_versions) == set(benchmark._DEPENDENCY_DISTRIBUTIONS)
    assert all(isinstance(value, str) and value for value in dependency_versions.values())


def test_storage_measurement_reports_file_count_and_disk_bytes(
    benchmark: ModuleType, tmp_path: Path
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "first.bin").write_bytes(b"a" * 1_000)
    (nested / "second.bin").write_bytes(b"b" * 2_000)

    measurement = benchmark.measure_storage([tmp_path])

    assert measurement.file_count == 2
    assert measurement.logical_bytes == 3_000
    assert measurement.allocated_bytes >= measurement.logical_bytes


def test_peak_run_storage_includes_private_staging_left_by_timeout(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "requested-output"
    staging_dir = tmp_path / ".requested-output.staging-test"
    program = """
from pathlib import Path
import sys
import time

directory = Path(sys.argv[1])
directory.mkdir()
(directory / "partial.bin").write_bytes(b"x" * 1_000_000)
time.sleep(60)
"""

    measurement = benchmark.measure_command(
        [sys.executable, "-c", program, str(staging_dir)],
        sample_interval_seconds=0.005,
        timeout_seconds=0.15,
        show_output=False,
        stdout_log_path=tmp_path / "storage.stdout.log",
        stderr_log_path=tmp_path / "storage.stderr.log",
        run_storage_probe=lambda: benchmark.measure_run_storage(output_dir),
    )

    assert measurement.status == "timeout"
    assert measurement.peak_run_logical_bytes >= 1_000_000
    assert measurement.peak_run_allocated_bytes >= 1_000_000
    assert measurement.peak_run_file_count >= 1
    retained = benchmark.measure_run_storage(output_dir)
    assert retained.logical_bytes == 1_000_000
    assert not output_dir.exists()


def test_run_metadata_is_flattened_into_explicit_csv_fields(
    benchmark: ModuleType, tmp_path: Path
) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(json.dumps(_run_metadata_document()), encoding="utf-8")

    measurement = benchmark.extract_run_metadata(metadata_path)

    assert measurement.error == ""
    assert measurement.csv_fields["run_metadata_status"] == "complete"
    assert measurement.csv_fields["actual_cells"] == 4
    assert measurement.csv_fields["models_requested"] == 6
    assert measurement.csv_fields["models_trained_with_positive_edges"] == 5
    assert measurement.csv_fields["models_trained_without_positive_edges"] == 1
    assert measurement.csv_fields["positive_edges"] == 5
    assert measurement.csv_fields["threads_effective"] == 1
    assert measurement.csv_fields["effective_n_components"] == 1
    assert measurement.csv_fields["bandwidth_automatic_reference_value"] == 1.25
    assert measurement.csv_fields["bandwidth_automatic_scale"] == 1.0
    assert measurement.csv_fields["phase_model_inference_seconds"] == 0.07
    assert measurement.csv_fields["phase_total_seconds"] == 0.36
    assert set(measurement.csv_fields) == set(benchmark._METADATA_PATHS)


def test_invalid_run_metadata_returns_a_diagnostic_instead_of_raising(
    benchmark: ModuleType, tmp_path: Path
) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text('{"status":"complete"}', encoding="utf-8")

    measurement = benchmark.extract_run_metadata(metadata_path)

    assert measurement.csv_fields == {}
    assert "invalid metadata file" in measurement.error
    assert "input_dimensions.cells" in measurement.error


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
    assert rows[0]["benchmark_sha256"] == "b" * 64
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
        timeout_seconds: float,
        show_output: bool,
        stdout_log_path: Path,
        stderr_log_path: Path,
        run_storage_probe: object,
    ) -> object:
        del sample_interval_seconds, timeout_seconds, show_output, run_storage_probe
        nonlocal calls
        calls += 1
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text(f"stdout-{calls}\n", encoding="utf-8")
        stderr_log_path.write_text(f"stderr-{calls}\n", encoding="utf-8")
        if calls == 2:
            raise KeyboardInterrupt
        _write_run_metadata_for_command(command)
        return benchmark.ProcessMeasurement(
            wall_seconds=1.25,
            peak_rss_bytes=2048,
            sampled_cpu_user_seconds=1.0,
            sampled_cpu_system_seconds=0.25,
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
                "--targets",
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
    assert rows[0]["sampled_cpu_user_seconds"] == "1.000000"
    assert rows[0]["sampled_cpu_system_seconds"] == "0.250000"
    assert rows[0]["run_metadata_status"] == "complete"
    assert rows[0]["models_requested"] == "6"
    assert rows[0]["published_output_file_count"] == "1"
    assert rows[0]["resource_sample_ms"] == "20.0"
    assert rows[0]["case_timeout_seconds"] == "3600.0"
    data_dir = benchmark_root / "data"
    assert (
        rows[0]["expression_sha256"]
        == hashlib.sha256((data_dir / "expression.tsv").read_bytes()).hexdigest()
    )
    assert (
        rows[0]["tf_list_sha256"]
        == hashlib.sha256((data_dir / "tf_list.txt").read_bytes()).hexdigest()
    )
    assert (
        rows[0]["groups_sha256"]
        == hashlib.sha256((data_dir / "groups.tsv").read_bytes()).hexdigest()
    )
    assert (
        rows[0]["target_list_sha256"]
        == hashlib.sha256((data_dir / "targets-00002.txt").read_bytes()).hexdigest()
    )
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
        timeout_seconds: float,
        show_output: bool,
        stdout_log_path: Path,
        stderr_log_path: Path,
        run_storage_probe: object,
    ) -> object:
        del command, sample_interval_seconds, timeout_seconds, show_output, run_storage_probe
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text("partial output\n", encoding="utf-8")
        stderr_log_path.write_text("failure details\n", encoding="utf-8")
        return benchmark.ProcessMeasurement(
            wall_seconds=0.5,
            peak_rss_bytes=1024,
            sampled_cpu_user_seconds=0.3,
            sampled_cpu_system_seconds=0.1,
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
    assert rows[0]["target_list_sha256"] == ""
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
        timeout_seconds=5.0,
        show_output=False,
        stdout_log_path=tmp_path / "stdout.log",
        stderr_log_path=tmp_path / "stderr.log",
    )

    assert measurement.exit_code == exit_code
    assert measurement.status == status
    assert measurement.wall_seconds > 0
    assert measurement.peak_rss_bytes > 0
    assert measurement.sampled_cpu_user_seconds >= 0
    assert measurement.sampled_cpu_system_seconds >= 0
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
        timeout_seconds=5.0,
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


def test_process_tree_cpu_includes_descendant_work(benchmark: ModuleType, tmp_path: Path) -> None:
    child_program = """
import time

deadline = time.process_time() + 0.15
value = 1
while time.process_time() < deadline:
    value = (value * 17 + 3) % 1_000_003
"""
    parent_program = """
import subprocess
import sys

subprocess.run([sys.executable, "-c", sys.argv[1]], check=True)
"""

    measurement = benchmark.measure_command(
        [sys.executable, "-c", parent_program, child_program],
        sample_interval_seconds=0.002,
        timeout_seconds=5.0,
        show_output=False,
        stdout_log_path=tmp_path / "cpu.stdout.log",
        stderr_log_path=tmp_path / "cpu.stderr.log",
    )

    assert measurement.status == "success"
    assert measurement.sampled_cpu_user_seconds >= 0.08
    assert measurement.sampled_cpu_system_seconds >= 0


def test_case_timeout_terminates_process_tree(benchmark: ModuleType, tmp_path: Path) -> None:
    psutil = pytest.importorskip("psutil")
    marker = tmp_path / "timeout-pids.txt"
    parent_program = """
import os
from pathlib import Path
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(f"{os.getpid()}\\n{child.pid}\\n", encoding="utf-8")
time.sleep(60)
"""

    measurement = benchmark.measure_command(
        [sys.executable, "-c", parent_program, str(marker)],
        sample_interval_seconds=0.005,
        timeout_seconds=0.1,
        show_output=False,
        stdout_log_path=tmp_path / "timeout.stdout.log",
        stderr_log_path=tmp_path / "timeout.stderr.log",
    )

    assert measurement.status == "timeout"
    assert measurement.wall_seconds < 3.0
    assert "0.1 seconds" in measurement.error
    pids = [int(value) for value in marker.read_text(encoding="utf-8").splitlines()]
    assert all(
        not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        for pid in pids
    )


def test_slow_storage_probe_cannot_hide_or_inflate_a_timeout(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    def slow_probe() -> object:
        time.sleep(0.1)
        return benchmark.StorageMeasurement(
            logical_bytes=0,
            allocated_bytes=0,
            file_count=0,
        )

    measurement = benchmark.measure_command(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        sample_interval_seconds=0.001,
        timeout_seconds=0.01,
        show_output=False,
        stdout_log_path=tmp_path / "slow-probe.stdout.log",
        stderr_log_path=tmp_path / "slow-probe.stderr.log",
        run_storage_probe=slow_probe,
    )

    assert measurement.status == "timeout"
    assert measurement.wall_seconds >= 0.01
    assert measurement.wall_seconds < 0.5


def test_cpu_bound_storage_probe_cannot_hide_a_timeout(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    def cpu_bound_probe() -> object:
        # Deliberately contend for the GIL past both the child duration and deadline.
        probe_started = time.perf_counter()
        while time.perf_counter() - probe_started < 0.2:
            pass
        return benchmark.StorageMeasurement(
            logical_bytes=0,
            allocated_bytes=0,
            file_count=0,
        )

    measurement = benchmark.measure_command(
        [sys.executable, "-c", "import time; time.sleep(0.08)"],
        sample_interval_seconds=0.001,
        timeout_seconds=0.05,
        show_output=False,
        stdout_log_path=tmp_path / "cpu-bound-probe.stdout.log",
        stderr_log_path=tmp_path / "cpu-bound-probe.stderr.log",
        run_storage_probe=cpu_bound_probe,
    )

    assert measurement.status == "timeout"
    assert measurement.wall_seconds >= 0.05
    assert measurement.wall_seconds < 0.5


@pytest.mark.parametrize(
    ("child_seconds", "expected_status"),
    [(0.02, "success"), (0.15, "timeout")],
)
def test_deadline_is_independent_of_a_cpu_bound_probe(
    benchmark: ModuleType,
    tmp_path: Path,
    child_seconds: float,
    expected_status: str,
) -> None:
    probe_calls = 0

    def cpu_bound_probe() -> object:
        nonlocal probe_calls
        probe_calls += 1
        probe_started = time.perf_counter()
        while time.perf_counter() - probe_started < 0.2:
            pass
        return benchmark.StorageMeasurement(
            logical_bytes=0,
            allocated_bytes=0,
            file_count=0,
        )

    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    try:
        measurement = benchmark.measure_command(
            [sys.executable, "-c", f"import time; time.sleep({child_seconds!r})"],
            sample_interval_seconds=0.001,
            timeout_seconds=0.1,
            show_output=False,
            stdout_log_path=tmp_path / "on-time.stdout.log",
            stderr_log_path=tmp_path / "on-time.stderr.log",
            run_storage_probe=cpu_bound_probe,
        )
    finally:
        sys.setswitchinterval(previous_switch_interval)

    assert probe_calls >= 1
    assert measurement.status == expected_status
    if expected_status == "success":
        assert measurement.exit_code == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group regression")
def test_timeout_reaps_orphan_created_by_sigterm_handler(
    benchmark: ModuleType,
    tmp_path: Path,
) -> None:
    psutil = pytest.importorskip("psutil")
    late_pid_path = tmp_path / "late-child.pid"
    parent_program = """
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def handle_sigterm(_signal_number, _frame):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
    os._exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
while True:
    time.sleep(1)
"""

    measurement = benchmark.measure_command(
        [sys.executable, "-c", parent_program, str(late_pid_path)],
        sample_interval_seconds=0.005,
        timeout_seconds=0.15,
        show_output=False,
        stdout_log_path=tmp_path / "late-child.stdout.log",
        stderr_log_path=tmp_path / "late-child.stderr.log",
    )

    late_pid = int(late_pid_path.read_text(encoding="utf-8"))
    try:
        try:
            late_child_active = psutil.Process(late_pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            late_child_active = False
        assert measurement.status == "timeout"
        assert not late_child_active
    finally:
        try:
            late_child = psutil.Process(late_pid)
            if late_child.status() != psutil.STATUS_ZOMBIE:
                late_child.kill()
        except psutil.NoSuchProcess:
            pass


def test_incomplete_cleanup_is_reported_as_an_error_status(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminate_process_tree = benchmark._terminate_process_tree

    def cleanup_then_report_failure(*args: object, **kwargs: object) -> bool:
        terminate_process_tree(*args, **kwargs)
        raise RuntimeError("simulated active survivor")

    monkeypatch.setattr(benchmark, "_terminate_process_tree", cleanup_then_report_failure)
    measurement = benchmark.measure_command(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        sample_interval_seconds=0.005,
        timeout_seconds=0.05,
        show_output=False,
        stdout_log_path=tmp_path / "cleanup-error.stdout.log",
        stderr_log_path=tmp_path / "cleanup-error.stderr.log",
    )

    assert measurement.status == "cleanup_error"
    assert "simulated active survivor" in measurement.error


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
    original_sample = benchmark.ProcessTreeUsageSampler.sample

    def interrupt_after_tree_started(sampler: object, process: object) -> int:
        if marker.exists():
            raise KeyboardInterrupt
        return original_sample(sampler, process)

    monkeypatch.setattr(benchmark.ProcessTreeUsageSampler, "sample", interrupt_after_tree_started)
    with pytest.raises(KeyboardInterrupt):
        benchmark.measure_command(
            [sys.executable, "-c", child_program, str(marker)],
            sample_interval_seconds=0.005,
            timeout_seconds=5.0,
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
