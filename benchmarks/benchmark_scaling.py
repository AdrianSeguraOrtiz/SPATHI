#!/usr/bin/env python3
"""Benchmark end-to-end SPATHI scaling on one reproducible synthetic dataset.

The benchmark lives outside the test suite. Every measured process includes CLI
startup, validation, inference, and artifact writing. Results are written as CSV to
stdout; diagnostics and optional SPATHI output are written to stderr.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import psutil


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    """Paths and identifiers belonging to one generated benchmark dataset."""

    expression: Path
    tf_list: Path
    groups: Path
    gene_names: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkCase:
    """One thread-budget and target-universe combination."""

    threads: int
    target_count: int
    target_list: Path | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessMeasurement:
    """Observed resource use and exit state for one child process."""

    wall_seconds: float
    peak_rss_bytes: int
    exit_code: int | None
    status: str
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkProvenance:
    """Versioned implementation and execution environment for every result row."""

    spathi_version: str
    implementation_sha256: str
    python_version: str
    platform: str
    machine: str
    logical_cpus: int
    dependency_versions_json: str

    def as_csv_fields(self) -> dict[str, str | int]:
        """Return provenance using the canonical CSV column names."""

        return {
            "spathi_version": self.spathi_version,
            "implementation_sha256": self.implementation_sha256,
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "logical_cpus": self.logical_cpus,
            "dependency_versions_json": self.dependency_versions_json,
        }


class CsvRecorder:
    """Write every completed row to durable storage before mirroring it to stdout."""

    def __init__(
        self,
        *,
        durable_output: IO[str],
        mirror_output: IO[str],
        provenance: BenchmarkProvenance,
    ) -> None:
        self._durable_output = durable_output
        self._mirror_output = mirror_output
        self._provenance = provenance.as_csv_fields()
        self._durable_writer = csv.DictWriter(durable_output, fieldnames=CSV_FIELDS)
        self._mirror_writer = csv.DictWriter(mirror_output, fieldnames=CSV_FIELDS)

    def write_header(self) -> None:
        """Persist and expose the CSV schema before benchmark execution starts."""

        self._durable_writer.writeheader()
        self._sync_durable_output()
        self._mirror_writer.writeheader()
        self._mirror_output.flush()

    def write_row(self, row: dict[str, object]) -> None:
        """Persist, fsync, and expose one completed benchmark run immediately."""

        complete_row = {**row, **self._provenance}
        self._durable_writer.writerow(complete_row)
        self._sync_durable_output()
        self._mirror_writer.writerow(complete_row)
        self._mirror_output.flush()

    def _sync_durable_output(self) -> None:
        self._durable_output.flush()
        os.fsync(self._durable_output.fileno())


_DEPENDENCY_DISTRIBUTIONS = (
    "joblib",
    "numpy",
    "pandas",
    "plotly",
    "psutil",
    "rich",
    "rich-argparse",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
)


CSV_FIELDS = (
    "run_index",
    "run_type",
    "round",
    "position",
    "threads",
    "wall_seconds",
    "peak_rss_bytes",
    "status",
    "exit_code",
    "error",
    "cells",
    "genes",
    "targets",
    "target_list",
    "tfs",
    "groups",
    "n_estimators",
    "n_components",
    "checkpoint",
    "report",
    "seed",
    "spathi_version",
    "implementation_sha256",
    "python_version",
    "platform",
    "machine",
    "logical_cpus",
    "dependency_versions_json",
)


def collect_provenance() -> BenchmarkProvenance:
    """Capture the exact SPATHI implementation and relevant execution environment."""

    from spathi._version import __version__
    from spathi.checkpoint import implementation_fingerprint

    dependency_versions: dict[str, str] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            dependency_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            dependency_versions[distribution] = "not-installed"

    return BenchmarkProvenance(
        spathi_version=__version__,
        implementation_sha256=implementation_fingerprint(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        logical_cpus=os.cpu_count() or 1,
        dependency_versions_json=json.dumps(
            dependency_versions,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=240, help="number of synthetic cells")
    parser.add_argument("--genes", type=int, default=80, help="number of expression genes")
    parser.add_argument("--tfs", type=int, default=12, help="number of candidate TF genes")
    parser.add_argument("--groups", type=int, default=4, help="number of cell groups")
    parser.add_argument("--n-estimators", type=int, default=25, help="trees per target model")
    parser.add_argument("--n-components", type=int, default=20, help="requested PCA components")
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        metavar="N",
        help=(
            "target-subset sizes to benchmark; each size gets a reproducible --target-list; "
            "omit to infer every expression gene without a target-list"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, -1],
        help="SPATHI thread budgets to compare (default: 1 2 -1)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="warm-up rounds reported separately from measurements",
    )
    parser.add_argument("--repeats", type=int, default=2, help="measured rounds")
    parser.add_argument("--seed", type=int, default=1729, help="dataset, schedule, and SPATHI seed")
    parser.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include per-model checkpointing in the measured SPATHI runs",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include interactive report generation in the measured SPATHI runs",
    )
    parser.add_argument(
        "--rss-sample-ms",
        type=float,
        default=20.0,
        metavar="MILLISECONDS",
        help="interval between process-tree RSS samples",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="parent directory for temporary inputs and outputs",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="retain generated inputs, logs, and run directories for inspection",
    )
    parser.add_argument(
        "--show-spathi-output",
        action="store_true",
        help="stream SPATHI stdout and logging to benchmark stderr",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate benchmark controls."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cells < args.groups:
        parser.error("--cells must be at least --groups")
    if args.genes < 2:
        parser.error("--genes must be at least 2")
    if not 1 <= args.tfs < args.genes:
        parser.error("--tfs must be positive and smaller than --genes")
    if args.groups < 2:
        parser.error("--groups must be at least 2")
    if args.n_estimators < 1 or args.n_components < 1 or args.repeats < 1:
        parser.error("--n-estimators, --n-components, and --repeats must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if not np.isfinite(args.rss_sample_ms) or args.rss_sample_ms <= 0:
        parser.error("--rss-sample-ms must be a positive finite number")
    if any(value == 0 or value < -1 for value in args.threads):
        parser.error("each --threads value must be -1 or a positive integer")
    if len(set(args.threads)) != len(args.threads):
        parser.error("--threads must not contain duplicate budgets")
    if args.targets is not None:
        if any(value < 1 or value > args.genes for value in args.targets):
            parser.error("each --targets value must be between 1 and --genes")
        if len(set(args.targets)) != len(args.targets):
            parser.error("--targets must not contain duplicate sizes")
    return args


def create_dataset(
    destination: Path,
    *,
    n_cells: int,
    n_genes: int,
    n_tfs: int,
    n_groups: int,
    seed: int,
) -> Dataset:
    """Create a non-negative expression matrix with reproducible group structure."""

    rng = np.random.default_rng(seed)
    cell_names = [f"cell_{index:05d}" for index in range(n_cells)]
    group_index = np.arange(n_cells, dtype=np.int64) % n_groups
    rng.shuffle(group_index)
    group_names = [f"population_{index + 1}" for index in range(n_groups)]

    group_tf_means = rng.uniform(0.4, 2.4, size=(n_groups, n_tfs))
    tf_values = group_tf_means[group_index] + rng.normal(0.0, 0.15, size=(n_cells, n_tfs))
    tf_values = np.clip(tf_values, 0.0, None)

    n_non_tfs = n_genes - n_tfs
    coefficients = rng.uniform(0.0, 1.0, size=(n_non_tfs, n_tfs))
    coefficients /= coefficients.sum(axis=1, keepdims=True)
    group_offsets = rng.uniform(0.0, 0.6, size=(n_groups, n_non_tfs))
    target_values = tf_values @ coefficients.T
    target_values += group_offsets[group_index]
    target_values += rng.normal(0.0, 0.08, size=target_values.shape)
    target_values = np.clip(target_values, 0.0, None)

    gene_names = [f"TF_{index:03d}" for index in range(n_tfs)]
    gene_names.extend(f"GENE_{index:05d}" for index in range(n_non_tfs))
    cells_by_genes = np.concatenate((tf_values, target_values), axis=1)

    data_dir = destination / "data"
    data_dir.mkdir(parents=True)
    expression_path = data_dir / "expression.tsv"
    tf_path = data_dir / "tf_list.txt"
    groups_path = data_dir / "groups.tsv"

    pd.DataFrame(cells_by_genes.T, index=gene_names, columns=cell_names).to_csv(
        expression_path,
        sep="\t",
        index_label="gene",
        float_format="%.8g",
    )
    tf_path.write_text("".join(f"{name}\n" for name in gene_names[:n_tfs]), encoding="utf-8")
    pd.DataFrame(
        {
            "cell": cell_names,
            "cluster": [group_names[index] for index in group_index],
        }
    ).to_csv(groups_path, sep="\t", index=False)
    return Dataset(
        expression=expression_path,
        tf_list=tf_path,
        groups=groups_path,
        gene_names=tuple(gene_names),
    )


def create_target_lists(
    destination: Path,
    *,
    gene_names: Sequence[str],
    target_counts: Sequence[int],
    seed: int,
) -> dict[int, Path]:
    """Write deterministic nested target subsets while preserving expression order."""

    target_dir = destination / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed ^ 0x5A17_4E7)
    permutation = rng.permutation(len(gene_names))
    paths: dict[int, Path] = {}
    for target_count in target_counts:
        selected = sorted(int(index) for index in permutation[:target_count])
        path = target_dir / f"targets-{target_count:05d}.txt"
        path.write_text(
            "".join(f"{gene_names[index]}\n" for index in selected),
            encoding="utf-8",
        )
        paths[target_count] = path
    return paths


def balanced_orders(
    cases: Sequence[BenchmarkCase],
    *,
    rounds: int,
    seed: int,
) -> list[list[BenchmarkCase]]:
    """Return reproducible randomized cyclic orders without a fixed first case.

    Within every complete block of ``len(cases)`` rounds, every case occupies every
    execution position once. A fresh seeded permutation is used for each block.
    """

    if not cases:
        raise ValueError("at least one benchmark case is required")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")

    rng = random.Random(seed)
    orders: list[list[BenchmarkCase]] = []
    while len(orders) < rounds:
        base = list(cases)
        rng.shuffle(base)
        block_length = min(len(base), rounds - len(orders))
        for offset in range(block_length):
            orders.append(base[offset:] + base[:offset])
    return orders


def build_command(
    *,
    dataset: Dataset,
    target_list: Path | None,
    output_dir: Path,
    threads: int,
    n_components: int,
    n_estimators: int,
    seed: int,
    checkpoint: bool,
    report: bool,
) -> list[str]:
    """Build one explicit SPATHI CLI invocation."""

    command = [
        sys.executable,
        "-m",
        "spathi",
        "infer",
        "--expression",
        str(dataset.expression),
        "--tf-list",
        str(dataset.tf_list),
        "--groups",
        str(dataset.groups),
        "--output-dir",
        str(output_dir),
        "--weight-mode",
        "cell-distance-group-anchored",
        "--distance-space",
        "pca",
        "--n-components",
        str(n_components),
        "--tree-method",
        "extra-trees",
        "--n-estimators",
        str(n_estimators),
        "--random-seed",
        str(seed),
        "--threads",
        str(threads),
        "--checkpoint" if checkpoint else "--no-checkpoint",
        "--report" if report else "--no-report",
        "--no-progress",
        "--log-level",
        "WARNING",
    ]
    if target_list is not None:
        command.extend(("--target-list", str(target_list)))
    return command


def _process_tree_rss(process: psutil.Process) -> int:
    """Return current resident bytes for a process and all accessible descendants."""

    import psutil

    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        pass

    rss = 0
    seen_pids: set[int] = set()
    for member in processes:
        if member.pid in seen_pids:
            continue
        seen_pids.add(member.pid)
        try:
            rss += member.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return rss


def _terminate_process_tree(process: psutil.Popen, *, timeout_seconds: float = 3.0) -> None:
    """Terminate and reap a benchmark child plus every descendant it created."""

    import psutil

    try:
        descendants = process.children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        descendants = []
    members = [*descendants, process]

    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - exercised on Windows
        for member in reversed(members):
            with suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                member.terminate()

    _, alive = psutil.wait_procs(members, timeout=timeout_seconds)
    if alive:
        if os.name == "posix":
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        for member in reversed(alive):
            with suppress(psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                member.kill()
        psutil.wait_procs(alive, timeout=timeout_seconds)

    with suppress(psutil.NoSuchProcess, psutil.TimeoutExpired, subprocess.TimeoutExpired):
        process.wait(timeout=timeout_seconds)


def _drain_output(
    stream: IO[str],
    destination: IO[str],
    *,
    echo: bool,
    echo_lock: threading.Lock,
    errors: list[BaseException],
) -> None:
    """Persist one child stream and optionally tee it to benchmark stderr."""

    try:
        for line in iter(stream.readline, ""):
            destination.write(line)
            destination.flush()
            if echo:
                with echo_lock:
                    sys.stderr.write(line)
                    sys.stderr.flush()
    except BaseException as exc:  # pragma: no cover - defensive I/O failure path
        errors.append(exc)
    finally:
        with suppress(OSError):
            stream.close()


def measure_command(
    command: Sequence[str],
    *,
    sample_interval_seconds: float,
    show_output: bool,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> ProcessMeasurement:
    """Execute, log, and sample peak resident memory across a process tree."""

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - exercised by installation failures
        raise RuntimeError(
            "benchmarking peak RSS requires the development dependency 'psutil'; "
            "install SPATHI with `pip install -e '.[dev]'`"
        ) from exc

    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (
        stdout_log_path.open("w", encoding="utf-8") as stdout_log,
        stderr_log_path.open("w", encoding="utf-8") as stderr_log,
    ):
        try:
            process = psutil.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            return ProcessMeasurement(
                wall_seconds=time.perf_counter() - started,
                peak_rss_bytes=0,
                exit_code=None,
                status="launch_error",
                error=str(exc),
            )

        assert process.stdout is not None and process.stderr is not None
        output_errors: list[BaseException] = []
        echo_lock = threading.Lock()
        output_threads = [
            threading.Thread(
                target=_drain_output,
                kwargs={
                    "stream": process.stdout,
                    "destination": stdout_log,
                    "echo": show_output,
                    "echo_lock": echo_lock,
                    "errors": output_errors,
                },
                name="spathi-benchmark-stdout",
            ),
            threading.Thread(
                target=_drain_output,
                kwargs={
                    "stream": process.stderr,
                    "destination": stderr_log,
                    "echo": show_output,
                    "echo_lock": echo_lock,
                    "errors": output_errors,
                },
                name="spathi-benchmark-stderr",
            ),
        ]
        for output_thread in output_threads:
            output_thread.start()

        peak_rss_bytes = 0
        try:
            while True:
                peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss(process))
                exit_code = process.poll()
                if exit_code is not None:
                    peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss(process))
                    break
                time.sleep(sample_interval_seconds)
        except BaseException:
            _terminate_process_tree(process)
            for output_thread in output_threads:
                output_thread.join(timeout=3.0)
            raise

        for output_thread in output_threads:
            output_thread.join(timeout=3.0)
        if any(output_thread.is_alive() for output_thread in output_threads):
            _terminate_process_tree(process)
            with suppress(OSError):
                process.stdout.close()
            with suppress(OSError):
                process.stderr.close()
            for output_thread in output_threads:
                output_thread.join(timeout=3.0)
            output_errors.append(RuntimeError("child output streams did not close after exit"))

    wall_seconds = time.perf_counter() - started
    if output_errors:
        status = "output_error"
        error = "; ".join(str(exc) for exc in output_errors)
    else:
        status = "success" if exit_code == 0 else "failed"
        error = "" if exit_code == 0 else f"process exited with status {exit_code}"

    if exit_code != 0 and not show_output:
        for log_path in (stdout_log_path, stderr_log_path):
            captured = log_path.read_text(encoding="utf-8")
            if captured:
                print(captured, file=sys.stderr, end="" if captured.endswith("\n") else "\n")

    return ProcessMeasurement(
        wall_seconds=wall_seconds,
        peak_rss_bytes=peak_rss_bytes,
        exit_code=exit_code,
        status=status,
        error=error,
    )


def _run_rows(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    dataset: Dataset,
    orders: Sequence[Sequence[BenchmarkCase]],
    run_type: str,
    first_run_index: int,
    recorder: CsvRecorder,
) -> tuple[int, bool]:
    """Execute scheduled rounds, recording each result before starting the next."""

    run_index = first_run_index
    had_failures = False
    for round_index, order in enumerate(orders, start=1):
        for position, case in enumerate(order, start=1):
            run_index += 1
            thread_label = "all" if case.threads == -1 else str(case.threads)
            output_dir = benchmark_root / (
                f"{run_type}-{round_index:03d}-{position:03d}-"
                f"threads-{thread_label}-targets-{case.target_count}"
            )
            log_stem = output_dir.name
            stdout_log_path = benchmark_root / "logs" / f"{log_stem}.stdout.log"
            stderr_log_path = benchmark_root / "logs" / f"{log_stem}.stderr.log"
            command = build_command(
                dataset=dataset,
                target_list=case.target_list,
                output_dir=output_dir,
                threads=case.threads,
                n_components=args.n_components,
                n_estimators=args.n_estimators,
                seed=args.seed,
                checkpoint=args.checkpoint,
                report=args.report,
            )
            measurement = measure_command(
                command,
                sample_interval_seconds=args.rss_sample_ms / 1_000.0,
                show_output=args.show_spathi_output,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
            )
            recorder.write_row(
                {
                    "run_index": run_index,
                    "run_type": run_type,
                    "round": round_index,
                    "position": position,
                    "threads": case.threads,
                    "wall_seconds": f"{measurement.wall_seconds:.6f}",
                    "peak_rss_bytes": measurement.peak_rss_bytes,
                    "status": measurement.status,
                    "exit_code": "" if measurement.exit_code is None else measurement.exit_code,
                    "error": measurement.error,
                    "cells": args.cells,
                    "genes": args.genes,
                    "targets": case.target_count,
                    "target_list": case.target_list is not None,
                    "tfs": args.tfs,
                    "groups": args.groups,
                    "n_estimators": args.n_estimators,
                    "n_components": args.n_components,
                    "checkpoint": args.checkpoint,
                    "report": args.report,
                    "seed": args.seed,
                }
            )
            had_failures = had_failures or measurement.status != "success"
            print(
                f"{run_type} round={round_index} position={position} "
                f"threads={case.threads} targets={case.target_count}: "
                f"{measurement.status}, {measurement.wall_seconds:.3f} s, "
                f"peak RSS={measurement.peak_rss_bytes / (1024**2):.1f} MiB",
                file=sys.stderr,
            )
    return run_index, had_failures


def main(argv: Sequence[str] | None = None) -> int:
    """Generate data and incrementally persist randomized balanced benchmark runs."""

    args = parse_args(argv)
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = Path(tempfile.mkdtemp(prefix="spathi-benchmark-", dir=args.work_dir))
    results_path = benchmark_root / "benchmark-results.csv"
    print(f"Benchmark workspace: {benchmark_root}", file=sys.stderr)

    completed = False
    had_failures = False
    try:
        with results_path.open("x", encoding="utf-8", newline="") as durable_output:
            recorder = CsvRecorder(
                durable_output=durable_output,
                mirror_output=sys.stdout,
                provenance=collect_provenance(),
            )
            recorder.write_header()
            dataset = create_dataset(
                benchmark_root,
                n_cells=args.cells,
                n_genes=args.genes,
                n_tfs=args.tfs,
                n_groups=args.groups,
                seed=args.seed,
            )
            if args.targets is None:
                target_paths: dict[int, Path] = {}
                target_counts = [args.genes]
            else:
                target_counts = args.targets
                target_paths = create_target_lists(
                    benchmark_root,
                    gene_names=dataset.gene_names,
                    target_counts=target_counts,
                    seed=args.seed,
                )

            cases = [
                BenchmarkCase(
                    threads=thread_budget,
                    target_count=target_count,
                    target_list=target_paths.get(target_count),
                )
                for target_count in target_counts
                for thread_budget in args.threads
            ]
            warmup_orders = balanced_orders(cases, rounds=args.warmups, seed=args.seed ^ 0xA11CE)
            measured_orders = balanced_orders(cases, rounds=args.repeats, seed=args.seed ^ 0xB3C4)

            next_run_index, warmup_failed = _run_rows(
                args=args,
                benchmark_root=benchmark_root,
                dataset=dataset,
                orders=warmup_orders,
                run_type="warmup",
                first_run_index=0,
                recorder=recorder,
            )
            _, measured_failed = _run_rows(
                args=args,
                benchmark_root=benchmark_root,
                dataset=dataset,
                orders=measured_orders,
                run_type="measurement",
                first_run_index=next_run_index,
                recorder=recorder,
            )
            had_failures = warmup_failed or measured_failed
        completed = True
        return int(had_failures)
    finally:
        retain_workspace = args.keep_work_dir or not completed or had_failures
        if retain_workspace:
            print(f"Retained benchmark workspace: {benchmark_root}", file=sys.stderr)
            if results_path.is_file():
                print(f"Incremental benchmark CSV: {results_path}", file=sys.stderr)
        else:
            shutil.rmtree(benchmark_root)


if __name__ == "__main__":
    raise SystemExit(main())
