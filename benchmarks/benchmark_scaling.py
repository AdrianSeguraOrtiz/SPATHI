#!/usr/bin/env python3
"""Benchmark end-to-end SPATHI scaling on one reproducible synthetic dataset.

The benchmark lives outside the test suite. Every measured process includes CLI
startup, validation, inference, and artifact writing. Results are written as CSV to
stdout and an fsynced workspace file; diagnostics and optional SPATHI output go to
stderr.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import psutil


MAX_RANDOM_SEED = (1 << 32) - 1
ThreadBudget = int | Literal["auto"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    """Paths and identifiers belonging to one generated benchmark dataset."""

    expression: Path
    tf_list: Path
    groups: Path
    gene_names: tuple[str, ...]
    n_groups: int


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkCase:
    """One thread-budget and target-universe combination."""

    threads: ThreadBudget
    target_count: int
    target_list: Path | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessMeasurement:
    """Observed resource use and exit state for one child process."""

    wall_seconds: float
    peak_rss_bytes: int
    sampled_cpu_user_seconds: float
    sampled_cpu_system_seconds: float
    exit_code: int | None
    status: str
    peak_run_logical_bytes: int = 0
    peak_run_allocated_bytes: int = 0
    peak_run_file_count: int = 0
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class StorageMeasurement:
    """Logical and physically allocated storage occupied by regular files."""

    logical_bytes: int
    allocated_bytes: int
    file_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MetadataMeasurement:
    """Flattened measurements read from one published SPATHI run metadata file."""

    csv_fields: dict[str, object]
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkProvenance:
    """Pinned implementation and execution environment for every result row."""

    spathi_version: str
    implementation_sha256: str
    benchmark_sha256: str
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
            "benchmark_sha256": self.benchmark_sha256,
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

_SYNTHETIC_GENE_BLOCK_SIZE = 64
_SYNTHETIC_REGULATORS_PER_TARGET = 4


@contextmanager
def _sigterm_as_keyboard_interrupt() -> Iterator[None]:
    """Give external termination the same child-cleanup path as Ctrl-C."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt(_signal_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


CSV_FIELDS = (
    "run_index",
    "run_type",
    "round",
    "position",
    "threads",
    "wall_seconds",
    "peak_rss_bytes",
    "sampled_cpu_user_seconds",
    "sampled_cpu_system_seconds",
    "status",
    "exit_code",
    "error",
    "cells",
    "genes",
    "targets",
    "target_list",
    "tfs",
    "groups",
    "weight_mode",
    "distance_metric",
    "group_size_correction",
    "n_estimators",
    "n_components",
    "checkpoint",
    "report",
    "show_spathi_output",
    "seed",
    "resource_sample_ms",
    "case_timeout_seconds",
    "expression_sha256",
    "tf_list_sha256",
    "groups_sha256",
    "target_list_sha256",
    "input_logical_bytes",
    "input_allocated_bytes",
    "input_file_count",
    "peak_run_logical_bytes",
    "peak_run_allocated_bytes",
    "peak_run_file_count",
    "published_output_logical_bytes",
    "published_output_allocated_bytes",
    "published_output_file_count",
    "retained_run_logical_bytes",
    "retained_run_allocated_bytes",
    "retained_run_file_count",
    "run_metadata_status",
    "actual_cells",
    "actual_genes",
    "actual_targets",
    "actual_tfs",
    "actual_groups",
    "models_requested",
    "models_completed",
    "models_trained",
    "models_preflight_skipped",
    "models_fit_or_importance_failures",
    "models_trained_with_positive_edges",
    "models_trained_without_positive_edges",
    "positive_edges",
    "models_reused_from_checkpoint",
    "models_processed_this_attempt",
    "threads_effective",
    "threads_available",
    "inference_thread_budget",
    "maximum_concurrent_model_fits",
    "memory_concurrent_model_cap",
    "memory_available_bytes_at_planning",
    "memory_usable_bytes_at_planning",
    "memory_usable_fraction",
    "memory_reserved_for_batch_bytes",
    "parallel_backend",
    "parallel_level",
    "persistent_worker_pool",
    "effective_n_components",
    "maximum_informative_n_components",
    "pca_svd_solver_resolution",
    "bandwidth_method",
    "bandwidth_value",
    "bandwidth_automatic_reference_value",
    "bandwidth_automatic_scale",
    "bandwidth_positive_distance_count",
    "bandwidth_fallback_reason",
    "tree_target_dtype",
    "tree_predictor_dtype",
    "bootstrap_effective",
    "targets_per_batch",
    "targets_per_batch_without_memory_limit",
    "target_groups_per_batch",
    "target_groups_per_batch_without_memory_limit",
    "cell_centroid_distance_storage",
    "cell_centroid_distances_computed",
    "distance_storage_reason",
    "centroid_distance_memory_available_bytes_at_planning",
    "centroid_distance_memory_usable_bytes_at_planning",
    "distance_memory_available_bytes_at_planning",
    "distance_memory_usable_bytes_at_planning",
    "phase_input_validation_seconds",
    "phase_distance_representation_seconds",
    "phase_centroids_and_distances_seconds",
    "phase_bandwidth_selection_seconds",
    "phase_inference_preparation_seconds",
    "phase_weighting_and_diagnostics_seconds",
    "phase_model_inference_seconds",
    "phase_artifact_writing_seconds",
    "phase_report_seconds",
    "phase_total_seconds",
    "spathi_version",
    "implementation_sha256",
    "benchmark_sha256",
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
        benchmark_sha256=_file_sha256(Path(__file__).resolve()),
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


def _file_sha256(path: Path) -> str:
    """Hash one file without loading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        type=_thread_budget,
        nargs="+",
        default=[1, 2, "auto"],
        help="SPATHI thread budgets to compare (default: 1 2 auto)",
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
        "--resource-sample-ms",
        type=float,
        default=20.0,
        metavar="MILLISECONDS",
        help="interval between process-tree RSS and CPU samples",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=3_600.0,
        metavar="SECONDS",
        help="maximum wall time for each SPATHI child process (default: 3600)",
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
    if args.groups < 1:
        parser.error("--groups must be at least 1")
    if args.n_estimators < 1 or args.n_components < 1 or args.repeats < 1:
        parser.error("--n-estimators, --n-components, and --repeats must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if not 0 <= args.seed <= MAX_RANDOM_SEED:
        parser.error(f"--seed must be between 0 and {MAX_RANDOM_SEED}")
    if not np.isfinite(args.resource_sample_ms) or args.resource_sample_ms <= 0:
        parser.error("--resource-sample-ms must be a positive finite number")
    if not np.isfinite(args.case_timeout_seconds) or args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be a positive finite number")
    if len(set(args.threads)) != len(args.threads):
        parser.error("--threads must not contain duplicate budgets")
    if args.targets is not None:
        if any(value < 1 or value > args.genes for value in args.targets):
            parser.error("each --targets value must be between 1 and --genes")
        if len(set(args.targets)) != len(args.targets):
            parser.error("--targets must not contain duplicate sizes")
    return args


def _thread_budget(value: str) -> ThreadBudget:
    """Parse the same public thread budget accepted by ``spathi infer``."""

    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def create_dataset(
    destination: Path,
    *,
    n_cells: int,
    n_genes: int,
    n_tfs: int,
    n_groups: int,
    seed: int,
) -> Dataset:
    """Create a non-negative expression matrix with reproducible group structure.

    The expression TSV is necessarily dense because that is SPATHI's public input
    contract. Generation itself is bounded, however: target genes are synthesized and
    appended in fixed-size blocks instead of materializing a cells-by-genes array or a
    dense genes-by-TF coefficient matrix in the benchmark process. This distinction is
    important for the large-scale profiles, where the implementation under test must
    bear the dense-input cost rather than the data generator failing first.
    """

    rng = np.random.default_rng(seed)
    cell_names = [f"cell_{index:05d}" for index in range(n_cells)]
    group_index = np.arange(n_cells, dtype=np.int64) % n_groups
    rng.shuffle(group_index)
    group_names = [f"population_{index + 1}" for index in range(n_groups)]

    group_tf_means = rng.uniform(0.4, 2.4, size=(n_groups, n_tfs))
    tf_values = group_tf_means[group_index] + rng.normal(0.0, 0.15, size=(n_cells, n_tfs))
    tf_values = np.clip(tf_values, 0.0, None)

    n_non_tfs = n_genes - n_tfs
    gene_names = [f"TF_{index:03d}" for index in range(n_tfs)]
    gene_names.extend(f"GENE_{index:05d}" for index in range(n_non_tfs))

    data_dir = destination / "data"
    data_dir.mkdir(parents=True)
    expression_path = data_dir / "expression.tsv"
    tf_path = data_dir / "tf_list.txt"
    groups_path = data_dir / "groups.tsv"

    pd.DataFrame(
        tf_values.T,
        index=gene_names[:n_tfs],
        columns=cell_names,
    ).to_csv(
        expression_path,
        sep="\t",
        index_label="gene",
        float_format="%.8g",
        mode="x",
    )
    regulators_per_target = min(_SYNTHETIC_REGULATORS_PER_TARGET, n_tfs)
    for start in range(0, n_non_tfs, _SYNTHETIC_GENE_BLOCK_SIZE):
        stop = min(start + _SYNTHETIC_GENE_BLOCK_SIZE, n_non_tfs)
        block_size = stop - start
        regulator_positions = rng.integers(
            0,
            n_tfs,
            size=(block_size, regulators_per_target),
        )
        coefficients = rng.uniform(
            0.0,
            1.0,
            size=(block_size, regulators_per_target),
        )
        coefficients /= coefficients.sum(axis=1, keepdims=True)
        target_values = np.sum(
            tf_values[:, regulator_positions] * coefficients[np.newaxis, :, :],
            axis=2,
        )
        group_offsets = rng.uniform(0.0, 0.6, size=(n_groups, block_size))
        target_values += group_offsets[group_index]
        target_values += rng.normal(0.0, 0.08, size=target_values.shape)
        np.clip(target_values, 0.0, None, out=target_values)
        pd.DataFrame(
            target_values.T,
            index=gene_names[n_tfs + start : n_tfs + stop],
            columns=cell_names,
        ).to_csv(
            expression_path,
            sep="\t",
            header=False,
            float_format="%.8g",
            mode="a",
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
        n_groups=n_groups,
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
    threads: ThreadBudget,
    n_components: int,
    n_estimators: int,
    seed: int,
    checkpoint: bool,
    report: bool,
) -> list[str]:
    """Build one explicit SPATHI CLI invocation."""

    single_group = dataset.n_groups == 1
    weight_mode = "cell-distance" if single_group else "cell-distance-group-anchored"
    distance_metric = "euclidean" if single_group else "cosine"
    group_size_correction = "none" if single_group else "cap-to-target"

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
        weight_mode,
        "--distance-space",
        "pca",
        "--distance-metric",
        distance_metric,
        "--bandwidth",
        "auto",
        "--bandwidth-scale",
        "1.0",
        "--group-size-correction",
        group_size_correction,
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


class ProcessTreeUsageSampler:
    """Accumulate RSS and CPU observations for a process and its descendants."""

    def __init__(self) -> None:
        self._cpu_seconds_by_process: dict[tuple[int, float], tuple[float, float]] = {}
        self._processes_by_identity: dict[tuple[int, float], psutil.Process] = {}
        self._lock = threading.Lock()

    @property
    def sampled_cpu_user_seconds(self) -> float:
        """Return the sampled lower bound for process-tree user CPU seconds."""

        with self._lock:
            return sum(value[0] for value in self._cpu_seconds_by_process.values())

    @property
    def sampled_cpu_system_seconds(self) -> float:
        """Return the sampled lower bound for process-tree system CPU seconds."""

        with self._lock:
            return sum(value[1] for value in self._cpu_seconds_by_process.values())

    def tracked_processes(self) -> tuple[psutil.Process, ...]:
        """Return processes observed while they still belonged to the run tree."""

        with self._lock:
            return tuple(self._processes_by_identity.values())

    def sample(self, process: psutil.Process) -> int:
        """Observe current tree RSS and retain each process's latest cumulative CPU time."""

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
                with member.oneshot():
                    identity = (member.pid, member.create_time())
                    rss += member.memory_info().rss
                    cpu_times = member.cpu_times()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            with self._lock:
                previous_user, previous_system = self._cpu_seconds_by_process.get(
                    identity, (0.0, 0.0)
                )
                self._cpu_seconds_by_process[identity] = (
                    max(previous_user, float(cpu_times.user)),
                    max(previous_system, float(cpu_times.system)),
                )
                self._processes_by_identity[identity] = member
        return rss


def measure_storage(paths: Sequence[Path]) -> StorageMeasurement:
    """Measure regular files reachable from explicit files or directory roots."""

    files: list[Path] = []
    for path in paths:
        try:
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        except OSError:
            continue

    logical_bytes = 0
    allocated_bytes = 0
    for path in files:
        try:
            file_stat = path.stat()
        except OSError:
            continue
        logical_bytes += file_stat.st_size
        blocks = getattr(file_stat, "st_blocks", None)
        allocated_bytes += file_stat.st_size if blocks is None else blocks * 512
    return StorageMeasurement(
        logical_bytes=logical_bytes,
        allocated_bytes=allocated_bytes,
        file_count=len(files),
    )


def measure_run_storage(output_dir: Path) -> StorageMeasurement:
    """Measure published or private artifacts owned by one exact SPATHI run.

    Abrupt timeout termination can leave the atomic staging directory and, when
    enabled, the resumable checkpoint beside the requested output. Include those
    exact name-derived paths without charging unrelated benchmark files to the run.
    """

    parent = output_dir.parent
    owned_paths = [output_dir, parent / f".{output_dir.name}.checkpoint"]
    if parent.is_dir():
        staging_prefix = f".{output_dir.name}.staging-"
        owned_paths.extend(
            path for path in parent.iterdir() if path.name.startswith(staging_prefix)
        )
    return measure_storage(owned_paths)


_METADATA_PATHS: dict[str, tuple[str, ...]] = {
    "run_metadata_status": ("status",),
    "actual_cells": ("input_dimensions", "cells"),
    "actual_genes": ("input_dimensions", "genes"),
    "actual_targets": ("input_dimensions", "targets"),
    "actual_tfs": ("input_dimensions", "transcription_factors"),
    "actual_groups": ("input_dimensions", "groups"),
    "models_requested": ("models", "requested"),
    "models_completed": ("models", "completed"),
    "models_trained": ("models", "trained"),
    "models_preflight_skipped": ("models", "preflight_skipped"),
    "models_fit_or_importance_failures": ("models", "fit_or_importance_failures"),
    "models_trained_with_positive_edges": ("models", "trained_with_positive_edges"),
    "models_trained_without_positive_edges": ("models", "trained_without_positive_edges"),
    "positive_edges": ("models", "positive_edges"),
    "models_reused_from_checkpoint": ("models", "reused_from_checkpoint"),
    "models_processed_this_attempt": ("models", "processed_this_attempt"),
    "threads_effective": ("parallelism", "threads_effective"),
    "threads_available": ("parallelism", "threads_available"),
    "inference_thread_budget": ("parallelism", "inference_thread_budget"),
    "maximum_concurrent_model_fits": ("parallelism", "maximum_concurrent_model_fits"),
    "memory_concurrent_model_cap": ("parallelism", "memory_concurrent_model_cap"),
    "memory_available_bytes_at_planning": (
        "parallelism",
        "memory_available_bytes_at_planning",
    ),
    "memory_usable_bytes_at_planning": ("parallelism", "memory_usable_bytes_at_planning"),
    "memory_usable_fraction": ("parallelism", "memory_usable_fraction"),
    "memory_reserved_for_batch_bytes": ("parallelism", "memory_reserved_for_batch_bytes"),
    "parallel_backend": ("parallelism", "backend"),
    "parallel_level": ("parallelism", "parallel_level"),
    "persistent_worker_pool": ("parallelism", "persistent_worker_pool"),
    "effective_n_components": ("effective_parameters", "effective_n_components"),
    "maximum_informative_n_components": (
        "effective_parameters",
        "maximum_informative_n_components",
    ),
    "pca_svd_solver_resolution": (
        "effective_parameters",
        "pca_svd_solver_resolution",
    ),
    "bandwidth_method": ("effective_parameters", "bandwidth", "method"),
    "bandwidth_value": ("effective_parameters", "bandwidth", "value"),
    "bandwidth_automatic_reference_value": (
        "effective_parameters",
        "bandwidth",
        "automatic_reference_value",
    ),
    "bandwidth_automatic_scale": (
        "effective_parameters",
        "bandwidth",
        "automatic_scale",
    ),
    "bandwidth_positive_distance_count": (
        "effective_parameters",
        "bandwidth",
        "positive_distance_count",
    ),
    "bandwidth_fallback_reason": (
        "effective_parameters",
        "bandwidth",
        "fallback_reason",
    ),
    "tree_target_dtype": ("effective_parameters", "tree_target_dtype"),
    "tree_predictor_dtype": ("effective_parameters", "tree_predictor_dtype"),
    "bootstrap_effective": ("effective_parameters", "bootstrap_effective"),
    "targets_per_batch": ("effective_parameters", "targets_per_batch"),
    "targets_per_batch_without_memory_limit": (
        "effective_parameters",
        "targets_per_batch_without_memory_limit",
    ),
    "target_groups_per_batch": ("effective_parameters", "target_groups_per_batch"),
    "target_groups_per_batch_without_memory_limit": (
        "effective_parameters",
        "target_groups_per_batch_without_memory_limit",
    ),
    "cell_centroid_distance_storage": (
        "effective_parameters",
        "cell_centroid_distance_storage",
    ),
    "cell_centroid_distances_computed": (
        "effective_parameters",
        "cell_centroid_distances_computed",
    ),
    "distance_storage_reason": ("effective_parameters", "distance_storage_reason"),
    "centroid_distance_memory_available_bytes_at_planning": (
        "effective_parameters",
        "centroid_distance_memory_available_bytes_at_planning",
    ),
    "centroid_distance_memory_usable_bytes_at_planning": (
        "effective_parameters",
        "centroid_distance_memory_usable_bytes_at_planning",
    ),
    "distance_memory_available_bytes_at_planning": (
        "effective_parameters",
        "distance_memory_available_bytes_at_planning",
    ),
    "distance_memory_usable_bytes_at_planning": (
        "effective_parameters",
        "distance_memory_usable_bytes_at_planning",
    ),
    "phase_input_validation_seconds": ("phase_times_seconds", "input_validation"),
    "phase_distance_representation_seconds": (
        "phase_times_seconds",
        "distance_representation",
    ),
    "phase_centroids_and_distances_seconds": (
        "phase_times_seconds",
        "centroids_and_distances",
    ),
    "phase_bandwidth_selection_seconds": ("phase_times_seconds", "bandwidth_selection"),
    "phase_inference_preparation_seconds": (
        "phase_times_seconds",
        "inference_preparation",
    ),
    "phase_weighting_and_diagnostics_seconds": (
        "phase_times_seconds",
        "weighting_and_diagnostics",
    ),
    "phase_model_inference_seconds": ("phase_times_seconds", "model_inference"),
    "phase_artifact_writing_seconds": ("phase_times_seconds", "artifact_writing"),
    "phase_report_seconds": ("phase_times_seconds", "report"),
    "phase_total_seconds": ("phase_times_seconds", "total"),
}


def extract_run_metadata(path: Path) -> MetadataMeasurement:
    """Read the current run metadata contract and flatten selected scaling fields."""

    if not path.is_file():
        return MetadataMeasurement(csv_fields={}, error=f"missing metadata file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError("top-level JSON value must be an object")
        csv_fields: dict[str, object] = {}
        for csv_field, metadata_path in _METADATA_PATHS.items():
            value: object = document
            for key in metadata_path:
                if not isinstance(value, dict) or key not in value:
                    raise KeyError(".".join(metadata_path))
                value = value[key]
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"{'.'.join(metadata_path)} must be a scalar or null")
            csv_fields[csv_field] = "" if value is None else value
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return MetadataMeasurement(csv_fields={}, error=f"invalid metadata file {path}: {exc}")
    return MetadataMeasurement(csv_fields=csv_fields)


def _terminate_process_tree(
    process: psutil.Popen,
    *,
    process_group_id: int | None = None,
    tracked_processes: Sequence[psutil.Process] = (),
    timeout_seconds: float = 3.0,
) -> bool:
    """Stop and verify a run tree, including members created while handling SIGTERM.

    Returns whether an active process needed termination. On POSIX every discovery
    pass includes the dedicated process group as well as previously observed tree
    members. This catches a child forked by a SIGTERM handler after the first group
    signal. Any process which remains active after SIGKILL is a hard supervision
    error rather than a silently leaked benchmark process.
    """

    import psutil

    known: dict[tuple[int, float], psutil.Process] = {}

    def remember(member: psutil.Process) -> None:
        try:
            identity = (member.pid, member.create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        except psutil.AccessDenied:
            identity = (member.pid, -1.0)
        known[identity] = member

    remember(process)
    for member in tracked_processes:
        remember(member)

    if os.name == "posix":
        group_id = process.pid if process_group_id is None else process_group_id
        if group_id == os.getpgrp():  # pragma: no cover - defensive invariant
            raise RuntimeError("refusing to signal the benchmark supervisor process group")
    else:  # pragma: no cover - exercised on Windows
        group_id = None

    def is_active(member: psutil.Process) -> bool:
        try:
            return member.is_running() and member.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except psutil.AccessDenied:
            return True

    def discover_active() -> dict[tuple[int, float], psutil.Process]:
        for parent in tuple(known.values()):
            try:
                for descendant in parent.children(recursive=True):
                    remember(descendant)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        if group_id is not None:
            for candidate in psutil.process_iter():
                if candidate.pid == os.getpid():
                    continue
                try:
                    if os.getpgid(candidate.pid) == group_id:
                        remember(candidate)
                except (ProcessLookupError, PermissionError):
                    continue
        return {identity: member for identity, member in known.items() if is_active(member)}

    active = discover_active()
    needed_termination = bool(active)
    if not active:
        with suppress(psutil.NoSuchProcess, psutil.TimeoutExpired, subprocess.TimeoutExpired):
            process.wait(timeout=0)
        return False

    term_signalled: set[tuple[int, float]] = set()
    if group_id is not None:
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:  # pragma: no cover - same-user child invariant
            raise RuntimeError(f"could not terminate process group {group_id}: {exc}") from exc
        for identity, member in active.items():
            try:
                if os.getpgid(member.pid) == group_id:
                    term_signalled.add(identity)
            except (ProcessLookupError, PermissionError):
                pass

    grace_deadline = time.monotonic() + timeout_seconds
    while active and time.monotonic() < grace_deadline:
        for identity, member in reversed(tuple(active.items())):
            if identity in term_signalled:
                continue
            try:
                member.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            term_signalled.add(identity)
        psutil.wait_procs(
            list(active.values()),
            timeout=min(0.02, max(0.0, grace_deadline - time.monotonic())),
        )
        active = discover_active()

    if active:
        if group_id is not None:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:  # pragma: no cover - same-user child invariant
                raise RuntimeError(f"could not kill process group {group_id}: {exc}") from exc
        kill_deadline = time.monotonic() + timeout_seconds
        while active and time.monotonic() < kill_deadline:
            for member in reversed(tuple(active.values())):
                try:
                    member.kill()
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
            psutil.wait_procs(
                list(active.values()),
                timeout=min(0.02, max(0.0, kill_deadline - time.monotonic())),
            )
            active = discover_active()

    with suppress(psutil.NoSuchProcess, psutil.TimeoutExpired, subprocess.TimeoutExpired):
        process.wait(timeout=0)
    active = discover_active()
    if active:
        survivors = ", ".join(str(member.pid) for member in active.values())
        raise RuntimeError(f"incomplete benchmark process cleanup; active PIDs: {survivors}")
    return needed_termination


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


_DEADLINE_WATCHDOG_CODE = r"""
import math
import os
import select
import sys
import time

sys.stdout.write("READY\n")
sys.stdout.flush()
request = sys.stdin.readline().split()
if len(request) != 3:
    raise SystemExit(2)
target_pid = int(request[0])
deadline = float(request[1])
supervisor_pid = int(request[2])
try:
    pidfd = os.pidfd_open(target_pid)
except BaseException as exc:
    sys.stdout.write(f"ERROR {type(exc).__name__}: {exc}\n")
    sys.stdout.flush()
    raise SystemExit(1)

try:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    while True:
        remaining = deadline - time.monotonic()
        timeout_ms = max(0, math.ceil(min(remaining, 0.1) * 1_000))
        if poller.poll(timeout_ms):
            outcome = "EXITED"
            break
        if time.monotonic() >= deadline:
            outcome = "TIMEOUT"
            break
        if os.getppid() != supervisor_pid:
            raise RuntimeError("benchmark supervisor disappeared")
finally:
    os.close(pidfd)

sys.stdout.write(outcome + "\n")
sys.stdout.flush()
"""


def _start_deadline_watchdog() -> subprocess.Popen[str] | None:
    """Start an independent Linux deadline observer before launching the target."""

    if not (sys.platform.startswith("linux") and hasattr(os, "pidfd_open")):
        return None
    watchdog = subprocess.Popen(
        [sys.executable, "-c", _DEADLINE_WATCHDOG_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert watchdog.stdout is not None
    if watchdog.stdout.readline() != "READY\n":
        _, stderr = watchdog.communicate(timeout=3.0)
        raise RuntimeError(f"could not start deadline watchdog: {stderr.strip()}")
    return watchdog


def _arm_deadline_watchdog(
    watchdog: subprocess.Popen[str] | None,
    *,
    target_pid: int,
    deadline: float,
) -> None:
    """Give a ready watchdog the exact target identity and absolute deadline."""

    if watchdog is None:
        return
    assert watchdog.stdin is not None
    watchdog.stdin.write(f"{target_pid} {deadline!r} {os.getpid()}\n")
    watchdog.stdin.flush()
    watchdog.stdin.close()


def _deadline_watchdog_outcome(watchdog: subprocess.Popen[str]) -> str | None:
    """Return a completed watchdog result, raising for a broken observer."""

    if watchdog.poll() is None:
        return None
    assert watchdog.stdout is not None and watchdog.stderr is not None
    outcome = watchdog.stdout.readline().strip()
    stderr = watchdog.stderr.read().strip()
    if outcome not in {"EXITED", "TIMEOUT"}:
        detail = outcome or stderr or f"exit status {watchdog.returncode}"
        raise RuntimeError(f"deadline watchdog failed: {detail}")
    return outcome


def _stop_deadline_watchdog(watchdog: subprocess.Popen[str] | None) -> None:
    """Reap the auxiliary observer on cancellation or launch failure."""

    if watchdog is None:
        return
    if watchdog.poll() is None:
        watchdog.terminate()
        try:
            watchdog.wait(timeout=1.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive fallback
            watchdog.kill()
            watchdog.wait(timeout=1.0)
    for stream in (watchdog.stdin, watchdog.stdout, watchdog.stderr):
        if stream is not None:
            with suppress(OSError):
                stream.close()


def measure_command(
    command: Sequence[str],
    *,
    sample_interval_seconds: float,
    timeout_seconds: float,
    show_output: bool,
    stdout_log_path: Path,
    stderr_log_path: Path,
    run_storage_probe: Callable[[], StorageMeasurement] | None = None,
) -> ProcessMeasurement:
    """Execute, log, and sample memory and CPU across a bounded process tree run."""

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - exercised by installation failures
        raise RuntimeError(
            "benchmarking peak RSS requires the development dependency 'psutil'; "
            "install SPATHI with `pip install -e '.[dev]'`"
        ) from exc

    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_log_path.open("w", encoding="utf-8") as stdout_log,
        stderr_log_path.open("w", encoding="utf-8") as stderr_log,
    ):
        watchdog = _start_deadline_watchdog()
        started = time.monotonic()
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
            _stop_deadline_watchdog(watchdog)
            return ProcessMeasurement(
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=0,
                sampled_cpu_user_seconds=0.0,
                sampled_cpu_system_seconds=0.0,
                exit_code=None,
                status="launch_error",
                error=str(exc),
            )

        deadline = started + timeout_seconds
        try:
            _arm_deadline_watchdog(
                watchdog,
                target_pid=process.pid,
                deadline=deadline,
            )
        except BaseException:
            with suppress(BaseException):
                _terminate_process_tree(
                    process,
                    process_group_id=process.pid if os.name == "posix" else None,
                )
            _stop_deadline_watchdog(watchdog)
            raise

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

        usage_sampler = ProcessTreeUsageSampler()
        peak_rss_bytes = 0
        peak_run_logical_bytes = 0
        peak_run_allocated_bytes = 0
        peak_run_file_count = 0
        sampling_stop = threading.Event()
        sampling_failed = threading.Event()
        sampling_errors: list[BaseException] = []

        def sample_once(*, include_processes: bool = True) -> None:
            nonlocal peak_rss_bytes
            nonlocal peak_run_allocated_bytes
            nonlocal peak_run_file_count
            nonlocal peak_run_logical_bytes
            if include_processes:
                peak_rss_bytes = max(peak_rss_bytes, usage_sampler.sample(process))
            if run_storage_probe is not None:
                run_storage = run_storage_probe()
                peak_run_logical_bytes = max(peak_run_logical_bytes, run_storage.logical_bytes)
                peak_run_allocated_bytes = max(
                    peak_run_allocated_bytes, run_storage.allocated_bytes
                )
                peak_run_file_count = max(peak_run_file_count, run_storage.file_count)

        def sample_resources() -> None:
            try:
                while True:
                    sample_once()
                    if sampling_stop.is_set():
                        break
                    if sampling_stop.wait(sample_interval_seconds):
                        sample_once(include_processes=False)
                        break
            except BaseException as exc:
                sampling_errors.append(exc)
                sampling_failed.set()

        sampling_thread = threading.Thread(
            target=sample_resources,
            name="spathi-benchmark-resource-sampler",
            daemon=True,
        )
        timed_out = False
        exit_code: int | None = None
        child_finished_at = time.monotonic()
        cleanup_errors: list[BaseException] = []
        pending_exception: BaseException | None = None
        sampling_started = False

        try:
            # Retain an observation for commands shorter than one scheduling quantum;
            # the potentially expensive storage probe always stays in its worker.
            peak_rss_bytes = usage_sampler.sample(process)
            sampling_thread.start()
            sampling_started = True
            while True:
                if sampling_failed.is_set():
                    raise sampling_errors[0]
                if watchdog is not None:
                    if exit_code is None:
                        try:
                            exit_code = process.wait(timeout=0.02)
                        except (psutil.TimeoutExpired, subprocess.TimeoutExpired):
                            pass
                    else:
                        with suppress(subprocess.TimeoutExpired):
                            watchdog.wait(timeout=0.02)
                    watchdog_outcome = _deadline_watchdog_outcome(watchdog)
                    if watchdog_outcome is None:
                        continue
                    timed_out = watchdog_outcome == "TIMEOUT"
                    if not timed_out and exit_code is None:
                        exit_code = process.wait(timeout=0.1)
                    break

                remaining = deadline - time.monotonic()
                try:
                    exit_code = process.wait(timeout=max(0.0, min(0.05, remaining)))
                except (psutil.TimeoutExpired, subprocess.TimeoutExpired):
                    if sampling_failed.is_set():
                        raise sampling_errors[0] from None
                    if time.monotonic() < deadline:
                        continue
                    timed_out = True
                    break
                else:
                    break

            if timed_out:
                try:
                    _terminate_process_tree(
                        process,
                        process_group_id=process.pid if os.name == "posix" else None,
                        tracked_processes=usage_sampler.tracked_processes(),
                    )
                except RuntimeError as exc:
                    cleanup_errors.append(exc)
                exit_code = process.poll()
                child_finished_at = time.monotonic()
            else:
                leaked_tree = False
                try:
                    leaked_tree = _terminate_process_tree(
                        process,
                        process_group_id=process.pid if os.name == "posix" else None,
                        tracked_processes=usage_sampler.tracked_processes(),
                    )
                except RuntimeError as exc:
                    cleanup_errors.append(exc)
                if leaked_tree:
                    cleanup_errors.append(
                        RuntimeError("process exited while active descendants remained")
                    )
                child_finished_at = time.monotonic()
        except BaseException as exc:
            pending_exception = exc
            try:
                _terminate_process_tree(
                    process,
                    process_group_id=process.pid if os.name == "posix" else None,
                    tracked_processes=usage_sampler.tracked_processes(),
                )
            except RuntimeError as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
            child_finished_at = time.monotonic()
        finally:
            sampling_stop.set()
            if sampling_started:
                sampling_thread.join(timeout=3.0)
                if sampling_thread.is_alive():
                    sampling_errors.append(RuntimeError("resource sampler did not stop"))
            _stop_deadline_watchdog(watchdog)

        if pending_exception is None and sampling_errors:
            pending_exception = sampling_errors[0]

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

        if pending_exception is not None:
            if cleanup_errors:
                details = "; ".join(str(exc) for exc in cleanup_errors)
                raise RuntimeError(f"benchmark cleanup failed: {details}") from pending_exception
            raise pending_exception

    wall_seconds = child_finished_at - started
    if cleanup_errors:
        status = "cleanup_error"
        prefix = f"exceeded case timeout of {timeout_seconds:g} seconds; " if timed_out else ""
        error = prefix + "; ".join(str(exc) for exc in cleanup_errors)
        if output_errors:
            error += "; " + "; ".join(str(exc) for exc in output_errors)
    elif timed_out:
        status = "timeout"
        error = f"exceeded case timeout of {timeout_seconds:g} seconds"
        if output_errors:
            error += "; " + "; ".join(str(exc) for exc in output_errors)
    elif output_errors:
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
        sampled_cpu_user_seconds=usage_sampler.sampled_cpu_user_seconds,
        sampled_cpu_system_seconds=usage_sampler.sampled_cpu_system_seconds,
        exit_code=exit_code,
        status=status,
        peak_run_logical_bytes=peak_run_logical_bytes,
        peak_run_allocated_bytes=peak_run_allocated_bytes,
        peak_run_file_count=peak_run_file_count,
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
    dataset_sha256: Mapping[str, str],
    target_list_sha256: Mapping[Path, str],
) -> tuple[int, bool]:
    """Execute scheduled rounds, recording each result before starting the next."""

    run_index = first_run_index
    had_failures = False
    for round_index, order in enumerate(orders, start=1):
        for position, case in enumerate(order, start=1):
            run_index += 1
            thread_label = str(case.threads)
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
                sample_interval_seconds=args.resource_sample_ms / 1_000.0,
                timeout_seconds=args.case_timeout_seconds,
                show_output=args.show_spathi_output,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                run_storage_probe=lambda path=output_dir: measure_run_storage(path),
            )
            input_paths = [dataset.expression, dataset.tf_list, dataset.groups]
            if case.target_list is not None:
                input_paths.append(case.target_list)
            input_storage = measure_storage(input_paths)
            published_output_storage = measure_storage([output_dir])
            retained_run_storage = measure_run_storage(output_dir)
            metadata_path = output_dir / "run_metadata.json"
            metadata = extract_run_metadata(metadata_path)
            status = measurement.status
            errors = [measurement.error] if measurement.error else []
            if metadata.error and (measurement.status == "success" or metadata_path.exists()):
                errors.append(metadata.error)
                if measurement.status == "success":
                    status = "metadata_error"
            elif (
                measurement.status == "success"
                and metadata.csv_fields["run_metadata_status"] != "complete"
            ):
                status = "metadata_error"
                errors.append(
                    "successful process published run_metadata.json with status "
                    f"{metadata.csv_fields['run_metadata_status']!r}"
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
                    "sampled_cpu_user_seconds": (f"{measurement.sampled_cpu_user_seconds:.6f}"),
                    "sampled_cpu_system_seconds": (f"{measurement.sampled_cpu_system_seconds:.6f}"),
                    "status": status,
                    "exit_code": "" if measurement.exit_code is None else measurement.exit_code,
                    "error": "; ".join(errors),
                    "cells": args.cells,
                    "genes": args.genes,
                    "targets": case.target_count,
                    "target_list": case.target_list is not None,
                    "tfs": args.tfs,
                    "groups": args.groups,
                    "weight_mode": (
                        "cell-distance" if dataset.n_groups == 1 else "cell-distance-group-anchored"
                    ),
                    "distance_metric": "euclidean" if dataset.n_groups == 1 else "cosine",
                    "group_size_correction": ("none" if dataset.n_groups == 1 else "cap-to-target"),
                    "n_estimators": args.n_estimators,
                    "n_components": args.n_components,
                    "checkpoint": args.checkpoint,
                    "report": args.report,
                    "show_spathi_output": args.show_spathi_output,
                    "seed": args.seed,
                    "resource_sample_ms": args.resource_sample_ms,
                    "case_timeout_seconds": args.case_timeout_seconds,
                    **dataset_sha256,
                    "target_list_sha256": (
                        "" if case.target_list is None else target_list_sha256[case.target_list]
                    ),
                    "input_logical_bytes": input_storage.logical_bytes,
                    "input_allocated_bytes": input_storage.allocated_bytes,
                    "input_file_count": input_storage.file_count,
                    "peak_run_logical_bytes": measurement.peak_run_logical_bytes,
                    "peak_run_allocated_bytes": measurement.peak_run_allocated_bytes,
                    "peak_run_file_count": measurement.peak_run_file_count,
                    "published_output_logical_bytes": published_output_storage.logical_bytes,
                    "published_output_allocated_bytes": (published_output_storage.allocated_bytes),
                    "published_output_file_count": published_output_storage.file_count,
                    "retained_run_logical_bytes": retained_run_storage.logical_bytes,
                    "retained_run_allocated_bytes": retained_run_storage.allocated_bytes,
                    "retained_run_file_count": retained_run_storage.file_count,
                    **metadata.csv_fields,
                }
            )
            had_failures = had_failures or status != "success"
            print(
                f"{run_type} round={round_index} position={position} "
                f"threads={case.threads} targets={case.target_count}: "
                f"{status}, wall={measurement.wall_seconds:.3f} s, "
                "sampled CPU lower bound="
                f"{measurement.sampled_cpu_user_seconds + measurement.sampled_cpu_system_seconds:.3f} s, "
                f"peak RSS={measurement.peak_rss_bytes / (1024**2):.1f} MiB, "
                f"published output={published_output_storage.logical_bytes / (1024**2):.1f} MiB, "
                f"peak run storage={measurement.peak_run_logical_bytes / (1024**2):.1f} MiB",
                file=sys.stderr,
            )
    return run_index, had_failures


def _run_benchmark(args: argparse.Namespace) -> int:
    """Generate data and incrementally persist randomized balanced benchmark runs."""

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
            dataset_sha256 = {
                "expression_sha256": _file_sha256(dataset.expression),
                "tf_list_sha256": _file_sha256(dataset.tf_list),
                "groups_sha256": _file_sha256(dataset.groups),
            }
            target_list_sha256 = {path: _file_sha256(path) for path in target_paths.values()}
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
                dataset_sha256=dataset_sha256,
                target_list_sha256=target_list_sha256,
            )
            _, measured_failed = _run_rows(
                args=args,
                benchmark_root=benchmark_root,
                dataset=dataset,
                orders=measured_orders,
                run_type="measurement",
                first_run_index=next_run_index,
                recorder=recorder,
                dataset_sha256=dataset_sha256,
                target_list_sha256=target_list_sha256,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Validate controls and run with graceful SIGTERM process-tree cleanup."""

    args = parse_args(argv)
    with _sigterm_as_keyboard_interrupt():
        return _run_benchmark(args)


if __name__ == "__main__":
    try:
        _exit_code = main()
    except KeyboardInterrupt:
        print("Interrupted; benchmark workspace and completed rows were retained", file=sys.stderr)
        _exit_code = 130
    raise SystemExit(_exit_code)
