#!/usr/bin/env python3
"""Benchmark two SPATHI implementations on local prepared-data slices.

The harness has two deliberately separate inputs:

* a strict experiment profile describes target/tree/thread scaling; and
* a local dataset manifest points to prepared SPATHI inputs by hash.

Raw expression values are never copied into the suite.  Both implementation
snapshots receive the same immutable input paths, target lists, parameters, and
random seeds.  The harness measures process-tree wall time, sampled CPU, peak RSS,
and transient/final storage, then compares the published scientific tables with an
explicit canonical tabular contract.  It tests computational equivalence; it does
not estimate biological accuracy.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

from joblib import cpu_count as joblib_cpu_count

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence


MAX_RANDOM_SEED = (1 << 32) - 1


def available_cpu_count() -> int:
    """Return the logical CPU capacity available to this standalone runner.

    Joblib combines host capacity, process affinity, and common container CPU
    quotas. The standard-library affinity/cpu count remains a defensive fallback
    for unusual environments in which joblib cannot resolve the capacity.
    """

    try:
        count = int(joblib_cpu_count(only_physical_cores=False))
    except (NotImplementedError, TypeError, ValueError):
        if hasattr(os, "sched_getaffinity"):
            count = len(os.sched_getaffinity(0))
        else:
            count = int(os.cpu_count() or 1)
    return max(1, count)


def _load_scaling_helpers() -> ModuleType:
    """Load the sibling standalone benchmark under direct and imported execution."""

    module_name = "_spathi_benchmark_scaling_helpers"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name("benchmark_scaling.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scaling benchmark helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SCALING_HELPERS = _load_scaling_helpers()
extract_run_metadata = _SCALING_HELPERS.extract_run_metadata
measure_command = _SCALING_HELPERS.measure_command
measure_run_storage = _SCALING_HELPERS.measure_run_storage
measure_storage = _SCALING_HELPERS.measure_storage


BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIR.parent
BUILTIN_PROFILE_DIR = BENCHMARK_DIR / "profiles" / "v1" / "cll-equivalence"
PROFILE_SCHEMA_VERSION = 1
DATASET_MANIFEST_SCHEMA_VERSION = 1
SUITE_SCHEMA_VERSION = 1
TARGET_MANIFEST_SCHEMA_VERSION = 1
PREPARE_MANIFEST_SCHEMA_VERSION = 1
COMPARISON_DETAILS_SCHEMA_VERSION = 1
COMMAND_SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_FIELDS = {
    "schema_version",
    "name",
    "description",
    "limitations",
    "minimum_datasets",
    "maximum_datasets",
    "allow_identical_implementations",
    "scientific_parameters",
    "defaults",
    "comparison",
    "cases",
}
_RUN_FIELDS = {
    "target_counts",
    "n_estimators",
    "threads",
    "n_components",
    "warmups",
    "repeats",
    "seed",
    "checkpoint",
    "report",
    "resource_sample_ms",
    "run_timeout_seconds",
}
_CASE_FIELDS = {"id", "description", "dataset_ids", *_RUN_FIELDS}
_COMPARISON_FIELDS = {"absolute_tolerance", "relative_tolerance"}
_SCIENTIFIC_PARAMETER_FIELDS = {
    "single_group_weight_mode",
    "multi_group_weight_mode",
    "distance_space",
    "distance_standardization",
    "pca_svd_solver",
    "single_group_distance_metric",
    "multi_group_distance_metric",
    "kernel",
    "bandwidth",
    "single_group_size_correction",
    "multi_group_size_correction",
    "tree_method",
    "max_features",
    "min_samples_leaf",
    "max_depth",
    "bootstrap",
    "adaptive_trees",
    "adaptive_min_estimators",
    "adaptive_tree_step",
    "adaptive_tolerance",
    "adaptive_patience",
    "target_eligibility",
    "min_target_detected_cells",
    "min_target_detected_fraction",
    "min_target_weighted_detected_fraction",
    "min_target_weighted_detected_ess",
}
_DATASET_MANIFEST_FIELDS = {"schema_version", "description", "datasets"}
_DATASET_FIELDS = {
    "id",
    "description",
    "analysis_unit",
    "dimensions",
    "expression",
    "groups",
    "tf_list",
    "centroid_weights",
    "prepare_manifest",
}
_DIMENSION_FIELDS = {"cells", "genes", "transcription_factors", "groups"}
_FILE_FIELDS = {"path", "sha256", "size_bytes"}
_TARGET_MANIFEST_FIELDS = {
    "schema_version",
    "profile_sha256",
    "dataset_manifest_sha256",
    "selected_dataset_ids",
    "entries",
}
_COMPARISON_DETAILS_FIELDS = {
    "schema_version",
    "equivalent",
    "error",
    "artifacts",
    "interpretation",
}
_SUITE_MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "status",
    "execution_attempt",
    "history",
    "started_at_utc",
    "completed_at_utc",
    "profile",
    "runner",
    "resource_measurement_helper",
    "target_manifest",
    "environment",
    "dataset_manifest",
    "implementations",
    "implementation_identity",
    "artifacts",
    "keep_successful_outputs",
    "verify_inputs",
    "schedule",
    "resource_measurement",
    "limitations",
    "runs_completed",
    "comparisons_completed",
    "failures",
}
_SUITE_MANIFEST_FIELDS = {
    *_SUITE_MANIFEST_REQUIRED_FIELDS,
    "disk_preflight",
    "resume_disk_preflight",
    "error",
}
_COMPARISON_INTERPRETATION = (
    "Computational output equivalence under the declared tolerances; this is not a "
    "biological-accuracy assessment."
)


class ContractError(ValueError):
    """Raised when a profile, manifest, or output violates its contract."""


TargetBudget = int | Literal["all"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FileReference:
    """One locally stored file pinned by exact size and SHA-256."""

    path_text: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetDimensions:
    cells: int
    genes: int
    transcription_factors: int
    groups: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalDataset:
    """One independently prepared patient/type analysis unit."""

    id: str
    description: str
    analysis_unit: str
    dimensions: DatasetDimensions
    expression: FileReference
    groups: FileReference
    tf_list: FileReference
    centroid_weights: FileReference | None
    prepare_manifest: FileReference


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetManifest:
    description: str
    datasets: tuple[LocalDataset, ...]
    source_path: Path
    source_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ScientificParameters:
    """Complete scientific configuration shared by reference and candidate."""

    single_group_weight_mode: str
    multi_group_weight_mode: str
    distance_space: str
    distance_standardization: str
    pca_svd_solver: str
    single_group_distance_metric: str
    multi_group_distance_metric: str
    kernel: str
    bandwidth: str | float
    single_group_size_correction: str
    multi_group_size_correction: str
    tree_method: str
    max_features: str | int | float
    min_samples_leaf: int
    max_depth: int | None
    bootstrap: bool
    adaptive_trees: bool
    adaptive_min_estimators: int
    adaptive_tree_step: int
    adaptive_tolerance: float
    adaptive_patience: int
    target_eligibility: str
    min_target_detected_cells: int
    min_target_detected_fraction: float
    min_target_weighted_detected_fraction: float
    min_target_weighted_detected_ess: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentCase:
    id: str
    description: str
    dataset_ids: tuple[str, ...]
    target_counts: tuple[TargetBudget, ...]
    n_estimators: tuple[int, ...]
    threads: tuple[int, ...]
    n_components: int
    warmups: int
    repeats: int
    seed: int
    checkpoint: bool
    report: bool
    resource_sample_ms: float
    run_timeout_seconds: float


@dataclass(frozen=True, slots=True, kw_only=True)
class EquivalenceProfile:
    name: str
    description: str
    limitations: tuple[str, ...]
    minimum_datasets: int
    maximum_datasets: int | None
    allow_identical_implementations: bool
    scientific_parameters: ScientificParameters
    absolute_tolerance: float
    relative_tolerance: float
    cases: tuple[ExperimentCase, ...]
    source_path: Path
    source_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkConfiguration:
    """One concrete, schedulable reference/candidate workload."""

    index: int
    target_scope: str
    target_count: int
    target_list: FileReference | None
    n_estimators: int
    threads: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Implementation:
    role: str
    source_path: Path
    snapshot_parent: Path
    sha256: str


@dataclass(slots=True, kw_only=True)
class ResumeState:
    """Validated, mutable execution state reconstructed from one suite snapshot."""

    profile: EquivalenceProfile
    data_manifest: DatasetManifest
    selected_dataset_ids: tuple[str, ...]
    implementations: dict[str, Implementation]
    manifest: dict[str, Any]
    run_rows: list[dict[str, object]]
    comparison_rows: list[dict[str, object]]
    target_lists: dict[tuple[str, int], dict[int, FileReference]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduledPair:
    """One deterministic reference/candidate pair in the immutable suite schedule."""

    comparison_id: str
    run_type: str
    round_index: int
    configuration_position: int
    case: ExperimentCase
    dataset: LocalDataset
    configuration: BenchmarkConfiguration
    role_order: tuple[str, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class TableRule:
    filename: str
    delimiter: str
    numeric_columns: frozenset[str]
    ignored_columns: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactComparison:
    artifact: str
    equivalent: bool
    reference_sha256: str
    candidate_sha256: str
    rows_compared: int
    numeric_values_compared: int
    max_absolute_difference: float
    max_relative_difference: float
    mismatch_count: int
    first_mismatches: tuple[str, ...]
    comparison_mode: str = "semantic-tolerance"
    error: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputComparison:
    equivalent: bool
    artifacts: tuple[ArtifactComparison, ...]
    error: str = ""


_TABLE_RULES = (
    TableRule(filename="network.csv", delimiter=",", numeric_columns=frozenset({"score"})),
    TableRule(filename="centroids.tsv", delimiter="\t", numeric_columns=frozenset({"centroid"})),
    TableRule(
        filename="group_affinities.tsv",
        delimiter="\t",
        numeric_columns=frozenset({"centroid_distance", "base_affinity", "group_size_factor"}),
    ),
    TableRule(
        filename="group_distances.tsv",
        delimiter="\t",
        numeric_columns=frozenset({"centroid_distance"}),
    ),
    TableRule(
        filename="weight_diagnostics.tsv",
        delimiter="\t",
        numeric_columns=frozenset(
            {
                "n_cells",
                "n_target_cells",
                "total_weight",
                "target_weight",
                "external_weight",
                "target_mass_percent",
                "external_mass_percent",
                "min_weight",
                "max_weight",
                "mean_weight",
                "median_weight",
                "positive_cell_count",
                "effective_sample_size",
                "source_weight",
                "source_mass_percent",
            }
        ),
    ),
    TableRule(filename="skipped_targets.tsv", delimiter="\t", numeric_columns=frozenset()),
    TableRule(
        filename="cell_weights.tsv.gz",
        delimiter="\t",
        numeric_columns=frozenset({"distance", "base_weight", "group_size_factor", "final_weight"}),
    ),
    TableRule(
        filename="cell_embedding.tsv.gz",
        delimiter="\t",
        numeric_columns=frozenset({"PC1", "PC2", "PC3"}),
    ),
    TableRule(
        filename="centroid_weights.tsv.gz",
        delimiter="\t",
        numeric_columns=frozenset({"centroid_weight", "normalized_centroid_weight"}),
    ),
    TableRule(
        filename="centroid_weight_diagnostics.tsv",
        delimiter="\t",
        numeric_columns=frozenset(
            {
                "n_cells",
                "weight_sum",
                "min_weight",
                "median_weight",
                "max_weight",
                "effective_sample_size",
            }
        ),
    ),
    TableRule(
        filename="pca_explained_variance.tsv",
        delimiter="\t",
        numeric_columns=frozenset(
            {"explained_variance_ratio", "cumulative_explained_variance_ratio"}
        ),
    ),
    TableRule(
        filename="model_diagnostics.tsv.gz",
        delimiter="\t",
        numeric_columns=frozenset(
            {
                "random_seed",
                "n_samples",
                "n_positive_weight_samples",
                "weight_sum",
                "n_predictors_input",
                "n_predictors_used",
                "n_edges",
                "importance_sum",
                "target_detected_cells",
                "target_detected_fraction",
                "target_weighted_detected_fraction",
                "target_weighted_detected_ess",
                "n_estimators_fitted",
                "convergence_delta",
                "convergence_checks",
            }
        ),
        ignored_columns=frozenset({"fit_seconds"}),
    ),
    TableRule(
        filename="target_eligibility.tsv.gz",
        delimiter="\t",
        numeric_columns=frozenset(
            {
                "detected_cells",
                "detected_fraction",
                "expression_min",
                "expression_max",
                "required_detected_cells",
            }
        ),
    ),
)

_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "network.csv": ("source", "target", "score", "sign", "evidence", "context"),
    "centroids.tsv": ("group", "dimension", "centroid"),
    "group_affinities.tsv": (
        "target_group",
        "source_group",
        "centroid_distance",
        "base_affinity",
        "group_size_factor",
    ),
    "group_distances.tsv": ("target_group", "source_group", "centroid_distance"),
    "weight_diagnostics.tsv": (
        "target_group",
        "n_cells",
        "n_target_cells",
        "total_weight",
        "target_weight",
        "external_weight",
        "target_mass_percent",
        "external_mass_percent",
        "min_weight",
        "max_weight",
        "mean_weight",
        "median_weight",
        "positive_cell_count",
        "effective_sample_size",
        "warnings_json",
        "source_group",
        "source_is_target",
        "source_weight",
        "source_mass_percent",
    ),
    "skipped_targets.tsv": ("target_group", "target", "reason", "detail"),
    "cell_weights.tsv.gz": (
        "target_group",
        "cell",
        "cell_group",
        "distance",
        "base_weight",
        "group_size_factor",
        "final_weight",
    ),
    "cell_embedding.tsv.gz": ("cell", "group", "PC1", "PC2", "PC3"),
    "centroid_weights.tsv.gz": (
        "cell",
        "group",
        "centroid_weight",
        "normalized_centroid_weight",
    ),
    "centroid_weight_diagnostics.tsv": (
        "group",
        "n_cells",
        "weight_sum",
        "min_weight",
        "median_weight",
        "max_weight",
        "effective_sample_size",
    ),
    "pca_explained_variance.tsv": (
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    ),
    "model_diagnostics.tsv.gz": (
        "target_group",
        "target",
        "status",
        "random_seed",
        "n_samples",
        "n_positive_weight_samples",
        "weight_sum",
        "n_predictors_input",
        "n_predictors_used",
        "discarded_predictors_json",
        "constant_predictors_json",
        "n_edges",
        "importance_sum",
        "fit_seconds",
        "target_detected_cells",
        "target_detected_fraction",
        "target_weighted_detected_fraction",
        "target_weighted_detected_ess",
        "n_estimators_fitted",
        "adaptive_converged",
        "convergence_delta",
        "convergence_checks",
        "message",
    ),
    "target_eligibility.tsv.gz": (
        "target",
        "mode",
        "eligible",
        "detected_cells",
        "detected_fraction",
        "expression_min",
        "expression_max",
        "required_detected_cells",
        "reason",
    ),
}


RUN_FIELDS = (
    "run_id",
    "attempt",
    "run_type",
    "round",
    "configuration_position",
    "execution_position",
    "profile",
    "profile_sha256",
    "dataset_manifest_sha256",
    "case_id",
    "dataset_id",
    "analysis_unit",
    "implementation_role",
    "implementation_sha256",
    "spathi_version",
    "dependency_versions_json",
    "target_scope",
    "target_count",
    "n_estimators",
    "threads",
    "n_components",
    "seed",
    "checkpoint",
    "resumed_from_checkpoint",
    "performance_eligible",
    "report",
    "wall_seconds",
    "peak_rss_bytes",
    "sampled_cpu_user_seconds",
    "sampled_cpu_system_seconds",
    "status",
    "exit_code",
    "error",
    "input_logical_bytes",
    "input_allocated_bytes",
    "input_file_count",
    "peak_run_logical_bytes",
    "peak_run_allocated_bytes",
    "peak_run_file_count",
    "published_output_logical_bytes",
    "published_output_allocated_bytes",
    "published_output_file_count",
    "expression_sha256",
    "groups_sha256",
    "tf_list_sha256",
    "centroid_weights_sha256",
    "target_list_sha256",
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
)
_METADATA_RUN_FIELDS = tuple(
    field for field in RUN_FIELDS if field in _SCALING_HELPERS._METADATA_PATHS
)

COMPARISON_FIELDS = (
    "comparison_id",
    "attempt",
    "run_type",
    "round",
    "configuration_position",
    "profile",
    "case_id",
    "dataset_id",
    "target_scope",
    "target_count",
    "n_estimators",
    "threads",
    "n_components",
    "reference_run_id",
    "reference_run_attempt",
    "candidate_run_id",
    "candidate_run_attempt",
    "status",
    "equivalent",
    "performance_eligible",
    "artifact_count",
    "mismatched_artifact_count",
    "max_absolute_difference",
    "max_relative_difference",
    "reference_wall_seconds",
    "candidate_wall_seconds",
    "wall_speedup_reference_over_candidate",
    "reference_peak_rss_bytes",
    "candidate_peak_rss_bytes",
    "candidate_over_reference_peak_rss",
    "reference_sampled_cpu_seconds",
    "candidate_sampled_cpu_seconds",
    "candidate_over_reference_sampled_cpu",
    "reference_peak_run_logical_bytes",
    "candidate_peak_run_logical_bytes",
    "candidate_over_reference_peak_run_logical_bytes",
    "reference_peak_run_allocated_bytes",
    "candidate_peak_run_allocated_bytes",
    "candidate_over_reference_peak_run_allocated_bytes",
    "reference_output_logical_bytes",
    "candidate_output_logical_bytes",
    "candidate_over_reference_output_logical_bytes",
    "reference_output_allocated_bytes",
    "candidate_output_allocated_bytes",
    "candidate_over_reference_output_allocated_bytes",
    "reference_model_seconds",
    "candidate_model_seconds",
    "model_speedup_reference_over_candidate",
    "details_json",
    "error",
)

_PAIRED_RATIO_SUMMARY_SUFFIXES = (
    "median",
    "q1",
    "q3",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
)
_PAIRED_RATIO_PREFIXES = (
    "paired_wall_speedup_reference_over_candidate",
    "paired_model_speedup_reference_over_candidate",
    "paired_candidate_over_reference_peak_rss",
    "paired_candidate_over_reference_sampled_cpu",
    "paired_candidate_over_reference_peak_run_logical_bytes",
    "paired_candidate_over_reference_peak_run_allocated_bytes",
)

SCALING_FIELDS = (
    "profile",
    "case_id",
    "dataset_id",
    "target_scope",
    "target_count",
    "n_estimators",
    "threads",
    "n_components",
    "expected_models",
    "measured_pairs",
    "equivalent_pairs",
    "performance_eligible_pairs",
    "reference_median_wall_seconds",
    "candidate_median_wall_seconds",
    "reference_median_model_seconds",
    "candidate_median_model_seconds",
    "reference_models_per_model_second",
    "candidate_models_per_model_second",
    "reference_max_peak_rss_bytes",
    "candidate_max_peak_rss_bytes",
    "reference_median_sampled_cpu_seconds",
    "candidate_median_sampled_cpu_seconds",
    "reference_max_peak_run_logical_bytes",
    "candidate_max_peak_run_logical_bytes",
    "reference_max_peak_run_allocated_bytes",
    "candidate_max_peak_run_allocated_bytes",
    "reference_max_output_logical_bytes",
    "candidate_max_output_logical_bytes",
    "reference_max_output_allocated_bytes",
    "candidate_max_output_allocated_bytes",
    *(
        f"{prefix}_{suffix}"
        for prefix in _PAIRED_RATIO_PREFIXES
        for suffix in _PAIRED_RATIO_SUMMARY_SUFFIXES
    ),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{location} must be a JSON object with string keys")
    return value


def _check_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    location: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ContractError(f"{location} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ContractError(f"{location} is missing fields: {', '.join(missing)}")


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, *, location: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{location} must be at least {minimum}")
    return value


def _finite_number(
    value: Any,
    *,
    location: str,
    minimum: float = 0.0,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{location} must be a number")
    result = float(value)
    valid = result > minimum if strict else result >= minimum
    if not math.isfinite(result) or not valid:
        comparator = "greater than" if strict else "at least"
        raise ContractError(f"{location} must be finite and {comparator} {minimum:g}")
    return result


def _boolean(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{location} must be a boolean")
    return value


def _choice(value: Any, *, location: str, choices: set[str]) -> str:
    result = _string(value, location=location)
    if result not in choices:
        expected = ", ".join(sorted(repr(choice) for choice in choices))
        raise ContractError(f"{location} must be one of: {expected}")
    return result


def _string_tuple(value: Any, *, location: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ContractError(f"{location} must be a JSON array of strings")
    result = tuple(
        _string(item, location=f"{location}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ContractError(f"{location} must not contain duplicates")
    return result


def _integer_tuple(
    value: Any,
    *,
    location: str,
    minimum: int = 1,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{location} must be a non-empty JSON array")
    result = tuple(
        _integer(item, location=f"{location}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ContractError(f"{location} must not contain duplicates")
    return result


def _target_budget_tuple(value: Any, *, location: str) -> tuple[TargetBudget, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{location} must be a non-empty JSON array")
    result: list[TargetBudget] = []
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        if item == "all":
            result.append("all")
        else:
            result.append(_integer(item, location=item_location, minimum=1))
    if len(result) != len(set(result)):
        raise ContractError(f"{location} must not contain duplicates")
    return tuple(result)


def _max_features(value: Any, *, location: str) -> str | int | float:
    if isinstance(value, str):
        return _choice(value, location=location, choices={"sqrt", "log2"})
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{location} must be 'sqrt', 'log2', an integer, or a fraction")
    if isinstance(value, int):
        return _integer(value, location=location, minimum=1)
    result = _finite_number(value, location=location, strict=True)
    if result > 1:
        raise ContractError(f"{location} fractions must not exceed 1")
    return result


def _unit_fraction(value: Any, *, location: str) -> float:
    result = _finite_number(value, location=location, strict=True)
    if result > 1:
        raise ContractError(f"{location} must not exceed 1")
    return result


def _parse_scientific_parameters(raw: Any) -> ScientificParameters:
    location = "scientific_parameters"
    value = _json_object(raw, location=location)
    _check_fields(
        value,
        allowed=_SCIENTIFIC_PARAMETER_FIELDS,
        required=_SCIENTIFIC_PARAMETER_FIELDS,
        location=location,
    )
    bandwidth_value = value["bandwidth"]
    bandwidth: str | float
    if bandwidth_value == "auto":
        bandwidth = "auto"
    else:
        bandwidth = _finite_number(
            bandwidth_value,
            location=f"{location}.bandwidth",
            strict=True,
        )
    max_depth_value = value["max_depth"]
    max_depth = (
        None
        if max_depth_value is None
        else _integer(max_depth_value, location=f"{location}.max_depth", minimum=1)
    )
    return ScientificParameters(
        single_group_weight_mode=_choice(
            value["single_group_weight_mode"],
            location=f"{location}.single_group_weight_mode",
            choices={"cell-distance"},
        ),
        multi_group_weight_mode=_choice(
            value["multi_group_weight_mode"],
            location=f"{location}.multi_group_weight_mode",
            choices={"cell-distance-group-anchored", "group-distance"},
        ),
        distance_space=_choice(
            value["distance_space"],
            location=f"{location}.distance_space",
            choices={"pca", "expression"},
        ),
        distance_standardization=_choice(
            value["distance_standardization"],
            location=f"{location}.distance_standardization",
            choices={"none", "standard"},
        ),
        pca_svd_solver=_choice(
            value["pca_svd_solver"],
            location=f"{location}.pca_svd_solver",
            choices={"auto", "randomized", "full"},
        ),
        single_group_distance_metric=_choice(
            value["single_group_distance_metric"],
            location=f"{location}.single_group_distance_metric",
            choices={"euclidean"},
        ),
        multi_group_distance_metric=_choice(
            value["multi_group_distance_metric"],
            location=f"{location}.multi_group_distance_metric",
            choices={"euclidean", "cosine"},
        ),
        kernel=_choice(
            value["kernel"],
            location=f"{location}.kernel",
            choices={"gaussian", "exponential"},
        ),
        bandwidth=bandwidth,
        single_group_size_correction=_choice(
            value["single_group_size_correction"],
            location=f"{location}.single_group_size_correction",
            choices={"none"},
        ),
        multi_group_size_correction=_choice(
            value["multi_group_size_correction"],
            location=f"{location}.multi_group_size_correction",
            choices={"none", "cap-to-target"},
        ),
        tree_method=_choice(
            value["tree_method"],
            location=f"{location}.tree_method",
            choices={"extra-trees", "random-forest"},
        ),
        max_features=_max_features(value["max_features"], location=f"{location}.max_features"),
        min_samples_leaf=_integer(
            value["min_samples_leaf"],
            location=f"{location}.min_samples_leaf",
            minimum=1,
        ),
        max_depth=max_depth,
        bootstrap=_boolean(value["bootstrap"], location=f"{location}.bootstrap"),
        adaptive_trees=_boolean(value["adaptive_trees"], location=f"{location}.adaptive_trees"),
        adaptive_min_estimators=_integer(
            value["adaptive_min_estimators"],
            location=f"{location}.adaptive_min_estimators",
            minimum=1,
        ),
        adaptive_tree_step=_integer(
            value["adaptive_tree_step"],
            location=f"{location}.adaptive_tree_step",
            minimum=1,
        ),
        adaptive_tolerance=_unit_fraction(
            value["adaptive_tolerance"],
            location=f"{location}.adaptive_tolerance",
        ),
        adaptive_patience=_integer(
            value["adaptive_patience"],
            location=f"{location}.adaptive_patience",
            minimum=1,
        ),
        target_eligibility=_choice(
            value["target_eligibility"],
            location=f"{location}.target_eligibility",
            choices={"all", "automatic"},
        ),
        min_target_detected_cells=_integer(
            value["min_target_detected_cells"],
            location=f"{location}.min_target_detected_cells",
            minimum=1,
        ),
        min_target_detected_fraction=_unit_fraction(
            value["min_target_detected_fraction"],
            location=f"{location}.min_target_detected_fraction",
        ),
        min_target_weighted_detected_fraction=_unit_fraction(
            value["min_target_weighted_detected_fraction"],
            location=f"{location}.min_target_weighted_detected_fraction",
        ),
        min_target_weighted_detected_ess=_finite_number(
            value["min_target_weighted_detected_ess"],
            location=f"{location}.min_target_weighted_detected_ess",
            strict=True,
        ),
    )


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, object]]:
    """Read one durable suite table with an exact schema and complete rows."""

    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(fields):
                raise ContractError(
                    f"CSV schema mismatch for {path}: expected {list(fields)}, "
                    f"observed {reader.fieldnames}"
                )
            rows: list[dict[str, object]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ContractError(f"malformed CSV row {row_number}: {path}")
                rows.append(dict(row))
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(f"cannot read suite CSV {path}: {exc}") from exc


def _restore_run_row_types(row: dict[str, object], *, location: str) -> dict[str, object]:
    for field in (
        "attempt",
        "round",
        "configuration_position",
        "execution_position",
        "target_count",
        "n_estimators",
        "threads",
        "n_components",
    ):
        row[field] = _integer_text(row[field], location=f"{location}.{field}", minimum=1)
    row["seed"] = _integer_text(row["seed"], location=f"{location}.seed", minimum=0)
    for field in ("checkpoint", "resumed_from_checkpoint", "performance_eligible", "report"):
        row[field] = _boolean_text(row[field], location=f"{location}.{field}")
    if row["run_type"] not in {"warmup", "measurement"}:
        raise ContractError(f"{location}.run_type is invalid")
    if row["status"] not in {
        "success",
        "failed",
        "timeout",
        "cleanup_error",
        "metadata_error",
    }:
        raise ContractError(f"{location}.status is invalid")
    for field in (
        "wall_seconds",
        "sampled_cpu_user_seconds",
        "sampled_cpu_system_seconds",
    ):
        row[field] = _number_text(row[field], location=f"{location}.{field}")
    for field in (
        "peak_rss_bytes",
        "input_logical_bytes",
        "input_allocated_bytes",
        "input_file_count",
        "peak_run_logical_bytes",
        "peak_run_allocated_bytes",
        "peak_run_file_count",
        "published_output_logical_bytes",
        "published_output_allocated_bytes",
        "published_output_file_count",
    ):
        row[field] = _integer_text(row[field], location=f"{location}.{field}", minimum=0)
    for field in (
        "profile_sha256",
        "dataset_manifest_sha256",
        "implementation_sha256",
        "expression_sha256",
        "groups_sha256",
        "tf_list_sha256",
    ):
        value = row[field]
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ContractError(f"{location}.{field} must be a lowercase SHA-256 digest")
    reused = row["models_reused_from_checkpoint"]
    if reused != "":
        reused = _integer_text(
            reused, location=f"{location}.models_reused_from_checkpoint", minimum=0
        )
        row["models_reused_from_checkpoint"] = reused
    if row["performance_eligible"] is True and (
        row["status"] != "success"
        or row["resumed_from_checkpoint"] is True
        or reused not in {"", 0}
    ):
        raise ContractError(f"{location}.performance_eligible contradicts the run provenance")
    return row


def _restore_comparison_row_types(row: dict[str, object], *, location: str) -> dict[str, object]:
    for field in (
        "attempt",
        "round",
        "configuration_position",
        "target_count",
        "n_estimators",
        "threads",
        "n_components",
        "reference_run_attempt",
        "candidate_run_attempt",
    ):
        row[field] = _integer_text(row[field], location=f"{location}.{field}", minimum=1)
    value = row["equivalent"]
    row["equivalent"] = _boolean_text(value, location=f"{location}.equivalent")
    row["performance_eligible"] = _boolean_text(
        row["performance_eligible"], location=f"{location}.performance_eligible"
    )
    if row["run_type"] not in {"warmup", "measurement"}:
        raise ContractError(f"{location}.run_type is invalid")
    if row["status"] not in {"compared", "not-compared"}:
        raise ContractError(f"{location}.status is invalid")
    for field in (
        "reference_wall_seconds",
        "candidate_wall_seconds",
        "reference_sampled_cpu_seconds",
        "candidate_sampled_cpu_seconds",
    ):
        row[field] = _number_text(row[field], location=f"{location}.{field}")
    for field in (
        "reference_peak_rss_bytes",
        "candidate_peak_rss_bytes",
        "reference_peak_run_logical_bytes",
        "candidate_peak_run_logical_bytes",
        "reference_peak_run_allocated_bytes",
        "candidate_peak_run_allocated_bytes",
        "reference_output_logical_bytes",
        "candidate_output_logical_bytes",
        "reference_output_allocated_bytes",
        "candidate_output_allocated_bytes",
    ):
        row[field] = _number_text(row[field], location=f"{location}.{field}")
    if row["performance_eligible"] is True and (
        row["status"] != "compared"
        or row["equivalent"] is not True
        or row["reference_run_attempt"] != row["candidate_run_attempt"]
    ):
        raise ContractError(f"{location}.performance_eligible contradicts pair provenance")
    return row


def _integer_text(value: object, *, location: str, minimum: int) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise ContractError(f"{location} must be a base-10 integer")
    result = int(value)
    if result < minimum:
        raise ContractError(f"{location} must be at least {minimum}")
    return result


def _boolean_text(value: object, *, location: str) -> bool:
    if value not in {"True", "False"}:
        raise ContractError(f"{location} must be True or False")
    return value == "True"


def _number_text(value: object, *, location: str) -> float:
    if not isinstance(value, str):
        raise ContractError(f"{location} must be a number")
    try:
        result = float(value)
    except ValueError as exc:
        raise ContractError(f"{location} must be a number") from exc
    if not math.isfinite(result) or result < 0:
        raise ContractError(f"{location} must be finite and non-negative")
    return result


def _resolve_profile_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if path.parent != Path(".") or path.suffix:
        raise ContractError(f"profile file does not exist: {path}")
    builtin = BUILTIN_PROFILE_DIR / f"{path.name}.json"
    if not builtin.is_file():
        available = ", ".join(
            sorted(path.stem for path in BUILTIN_PROFILE_DIR.glob("cll-equivalence-*.json"))
        )
        raise ContractError(f"unknown equivalence profile {path.name!r}; available: {available}")
    return builtin.resolve()


def _parse_experiment_case(
    raw: Any,
    *,
    defaults: Mapping[str, Any],
    index: int,
) -> ExperimentCase:
    location = f"cases[{index}]"
    value = _json_object(raw, location=location)
    _check_fields(
        value,
        allowed=_CASE_FIELDS,
        required={"id", "description", "dataset_ids"},
        location=location,
    )
    merged = {**defaults, **value}
    case_id = _string(merged["id"], location=f"{location}.id")
    if not _ID_PATTERN.fullmatch(case_id):
        raise ContractError(f"{location}.id must be a lowercase dash-separated identifier")
    dataset_ids = _string_tuple(
        merged["dataset_ids"], location=f"{location}.dataset_ids", allow_empty=True
    )
    if any(not _ID_PATTERN.fullmatch(dataset_id) for dataset_id in dataset_ids):
        raise ContractError(
            f"{location}.dataset_ids must contain lowercase dash-separated identifiers"
        )
    target_counts = _target_budget_tuple(
        merged["target_counts"], location=f"{location}.target_counts"
    )
    n_estimators = _integer_tuple(merged["n_estimators"], location=f"{location}.n_estimators")
    threads = _integer_tuple(merged["threads"], location=f"{location}.threads")
    n_components = _integer(merged["n_components"], location=f"{location}.n_components", minimum=3)
    warmups = _integer(merged["warmups"], location=f"{location}.warmups", minimum=0)
    repeats = _integer(merged["repeats"], location=f"{location}.repeats", minimum=1)
    seed = _integer(merged["seed"], location=f"{location}.seed", minimum=0)
    if seed > MAX_RANDOM_SEED:
        raise ContractError(f"{location}.seed must not exceed {MAX_RANDOM_SEED}")
    return ExperimentCase(
        id=case_id,
        description=_string(merged["description"], location=f"{location}.description"),
        dataset_ids=dataset_ids,
        target_counts=target_counts,
        n_estimators=n_estimators,
        threads=threads,
        n_components=n_components,
        warmups=warmups,
        repeats=repeats,
        seed=seed,
        checkpoint=_boolean(merged["checkpoint"], location=f"{location}.checkpoint"),
        report=_boolean(merged["report"], location=f"{location}.report"),
        resource_sample_ms=_finite_number(
            merged["resource_sample_ms"],
            location=f"{location}.resource_sample_ms",
            strict=True,
        ),
        run_timeout_seconds=_finite_number(
            merged["run_timeout_seconds"],
            location=f"{location}.run_timeout_seconds",
            strict=True,
        ),
    )


def load_profile(value: str | Path) -> EquivalenceProfile:
    """Load and strictly validate one equivalence profile."""

    path = _resolve_profile_path(value)
    try:
        source_bytes = path.read_bytes()
        document = json.loads(source_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read profile {path}: {exc}") from exc
    root = _json_object(document, location="profile")
    _check_fields(root, allowed=_PROFILE_FIELDS, required=_PROFILE_FIELDS, location="profile")
    version = _integer(root["schema_version"], location="schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        raise ContractError(
            f"profile schema_version must be {PROFILE_SCHEMA_VERSION}; received {version}"
        )
    name = _string(root["name"], location="name")
    if not _ID_PATTERN.fullmatch(name):
        raise ContractError("name must be a lowercase dash-separated identifier")
    defaults = _json_object(root["defaults"], location="defaults")
    _check_fields(defaults, allowed=_RUN_FIELDS, required=_RUN_FIELDS, location="defaults")
    scientific_parameters = _parse_scientific_parameters(root["scientific_parameters"])
    comparison = _json_object(root["comparison"], location="comparison")
    _check_fields(
        comparison,
        allowed=_COMPARISON_FIELDS,
        required=_COMPARISON_FIELDS,
        location="comparison",
    )
    raw_cases = root["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("cases must be a non-empty JSON array")
    cases = tuple(
        _parse_experiment_case(raw, defaults=defaults, index=index)
        for index, raw in enumerate(raw_cases)
    )
    ids = tuple(case.id for case in cases)
    if len(ids) != len(set(ids)):
        raise ContractError("case ids must be unique")
    if scientific_parameters.adaptive_trees:
        invalid_estimators = sorted(
            {
                n_estimators
                for case in cases
                for n_estimators in case.n_estimators
                if n_estimators <= scientific_parameters.adaptive_min_estimators
            }
        )
        if invalid_estimators:
            raise ContractError(
                "adaptive_min_estimators must be smaller than every n_estimators budget; "
                f"invalid values: {invalid_estimators}"
            )
    signatures: dict[tuple[object, ...], str] = {}
    for case in cases:
        values = asdict(case)
        signature = tuple(
            value for key, value in values.items() if key not in {"id", "description"}
        )
        previous = signatures.get(signature)
        if previous is not None:
            raise ContractError(f"cases {previous!r} and {case.id!r} are computational duplicates")
        signatures[signature] = case.id
    minimum_datasets = _integer(root["minimum_datasets"], location="minimum_datasets", minimum=0)
    maximum_datasets = (
        None
        if root["maximum_datasets"] is None
        else _integer(root["maximum_datasets"], location="maximum_datasets", minimum=1)
    )
    if maximum_datasets is not None and maximum_datasets < minimum_datasets:
        raise ContractError("maximum_datasets must be greater than or equal to minimum_datasets")
    return EquivalenceProfile(
        name=name,
        description=_string(root["description"], location="description"),
        limitations=_string_tuple(root["limitations"], location="limitations", allow_empty=True),
        minimum_datasets=minimum_datasets,
        maximum_datasets=maximum_datasets,
        allow_identical_implementations=_boolean(
            root["allow_identical_implementations"],
            location="allow_identical_implementations",
        ),
        scientific_parameters=scientific_parameters,
        absolute_tolerance=_finite_number(
            comparison["absolute_tolerance"], location="comparison.absolute_tolerance"
        ),
        relative_tolerance=_finite_number(
            comparison["relative_tolerance"], location="comparison.relative_tolerance"
        ),
        cases=cases,
        source_path=path,
        source_bytes=source_bytes,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def _parse_file_reference(
    raw: Any,
    *,
    location: str,
    manifest_path: Path,
) -> FileReference:
    value = _json_object(raw, location=location)
    _check_fields(value, allowed=_FILE_FIELDS, required=_FILE_FIELDS, location=location)
    path_text = _string(value["path"], location=f"{location}.path")
    supplied = Path(path_text).expanduser()
    path = supplied if supplied.is_absolute() else manifest_path.parent / supplied
    sha256 = _string(value["sha256"], location=f"{location}.sha256")
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise ContractError(f"{location}.sha256 must be a lowercase SHA-256 digest")
    return FileReference(
        path_text=path_text,
        path=path.resolve(),
        sha256=sha256,
        size_bytes=_integer(value["size_bytes"], location=f"{location}.size_bytes", minimum=0),
    )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load local prepared-data references without reading expression values."""

    source_path = Path(path).expanduser().resolve()
    try:
        source_bytes = source_path.read_bytes()
        document = json.loads(source_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read dataset manifest {source_path}: {exc}") from exc
    root = _json_object(document, location="dataset manifest")
    _check_fields(
        root,
        allowed=_DATASET_MANIFEST_FIELDS,
        required=_DATASET_MANIFEST_FIELDS,
        location="dataset manifest",
    )
    version = _integer(root["schema_version"], location="schema_version")
    if version != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ContractError(
            "dataset manifest schema_version must be "
            f"{DATASET_MANIFEST_SCHEMA_VERSION}; received {version}"
        )
    raw_datasets = root["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ContractError("datasets must be a non-empty JSON array")
    datasets: list[LocalDataset] = []
    for index, raw_dataset in enumerate(raw_datasets):
        location = f"datasets[{index}]"
        value = _json_object(raw_dataset, location=location)
        _check_fields(value, allowed=_DATASET_FIELDS, required=_DATASET_FIELDS, location=location)
        dataset_id = _string(value["id"], location=f"{location}.id")
        if not _ID_PATTERN.fullmatch(dataset_id):
            raise ContractError(f"{location}.id must be a lowercase dash-separated identifier")
        dimensions_value = _json_object(value["dimensions"], location=f"{location}.dimensions")
        _check_fields(
            dimensions_value,
            allowed=_DIMENSION_FIELDS,
            required=_DIMENSION_FIELDS,
            location=f"{location}.dimensions",
        )
        dimensions = DatasetDimensions(
            cells=_integer(
                dimensions_value["cells"], location=f"{location}.dimensions.cells", minimum=1
            ),
            genes=_integer(
                dimensions_value["genes"], location=f"{location}.dimensions.genes", minimum=2
            ),
            transcription_factors=_integer(
                dimensions_value["transcription_factors"],
                location=f"{location}.dimensions.transcription_factors",
                minimum=1,
            ),
            groups=_integer(
                dimensions_value["groups"], location=f"{location}.dimensions.groups", minimum=1
            ),
        )
        if dimensions.transcription_factors >= dimensions.genes:
            raise ContractError(f"{location} must contain fewer TFs than genes")
        if dimensions.groups > dimensions.cells:
            raise ContractError(f"{location} must not contain more groups than cells")
        datasets.append(
            LocalDataset(
                id=dataset_id,
                description=_string(value["description"], location=f"{location}.description"),
                analysis_unit=_string(value["analysis_unit"], location=f"{location}.analysis_unit"),
                dimensions=dimensions,
                expression=_parse_file_reference(
                    value["expression"],
                    location=f"{location}.expression",
                    manifest_path=source_path,
                ),
                groups=_parse_file_reference(
                    value["groups"], location=f"{location}.groups", manifest_path=source_path
                ),
                tf_list=_parse_file_reference(
                    value["tf_list"], location=f"{location}.tf_list", manifest_path=source_path
                ),
                centroid_weights=(
                    None
                    if value["centroid_weights"] is None
                    else _parse_file_reference(
                        value["centroid_weights"],
                        location=f"{location}.centroid_weights",
                        manifest_path=source_path,
                    )
                ),
                prepare_manifest=_parse_file_reference(
                    value["prepare_manifest"],
                    location=f"{location}.prepare_manifest",
                    manifest_path=source_path,
                ),
            )
        )
    ids = tuple(dataset.id for dataset in datasets)
    if len(ids) != len(set(ids)):
        raise ContractError("dataset ids must be unique")
    return DatasetManifest(
        description=_string(root["description"], location="description"),
        datasets=tuple(datasets),
        source_path=source_path,
        source_bytes=source_bytes,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def verify_dataset(dataset: LocalDataset) -> None:
    """Verify every local input before target selection or timed execution."""

    references = [dataset.expression, dataset.groups, dataset.tf_list, dataset.prepare_manifest]
    if dataset.centroid_weights is not None:
        references.append(dataset.centroid_weights)
    for reference in references:
        if not reference.path.is_file():
            raise ContractError(f"dataset {dataset.id!r} file does not exist: {reference.path}")
        observed_size = reference.path.stat().st_size
        if observed_size != reference.size_bytes:
            raise ContractError(
                f"dataset {dataset.id!r} size mismatch for {reference.path}: expected "
                f"{reference.size_bytes}, observed {observed_size}"
            )
        observed_hash = _sha256(reference.path)
        if observed_hash != reference.sha256:
            raise ContractError(
                f"dataset {dataset.id!r} hash mismatch for {reference.path}: expected "
                f"{reference.sha256}, observed {observed_hash}"
            )


def _resolved_dataset_manifest_document(manifest: DatasetManifest) -> dict[str, object]:
    """Make a replayable snapshot whose local paths keep their original meaning."""

    def file_document(reference: FileReference) -> dict[str, object]:
        return {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "size_bytes": reference.size_bytes,
        }

    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "description": manifest.description,
        "datasets": [
            {
                "id": dataset.id,
                "description": dataset.description,
                "analysis_unit": dataset.analysis_unit,
                "dimensions": asdict(dataset.dimensions),
                "expression": file_document(dataset.expression),
                "groups": file_document(dataset.groups),
                "tf_list": file_document(dataset.tf_list),
                "centroid_weights": (
                    None
                    if dataset.centroid_weights is None
                    else file_document(dataset.centroid_weights)
                ),
                "prepare_manifest": file_document(dataset.prepare_manifest),
            }
            for dataset in manifest.datasets
        ],
    }


def _safe_id(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ContractError(f"cannot derive an identifier from {value!r}")
    return result


def _manifest_file_entry(path: Path, *, relative_to: Path) -> dict[str, object]:
    try:
        path_text = os.path.relpath(path, start=relative_to)
    except ValueError:  # pragma: no cover - different Windows drives
        path_text = str(path)
    return {"path": path_text, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def create_dataset_manifest(
    labelled_prepare_manifests: Sequence[tuple[str, Path]],
    *,
    output_path: Path,
) -> dict[str, object]:
    """Create a local benchmark manifest from ordinary ``spathi prepare`` outputs."""

    datasets: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for label, supplied_path in labelled_prepare_manifests:
        prepare_path = supplied_path.expanduser().resolve()
        try:
            prepare_document = json.loads(prepare_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read prepare manifest {prepare_path}: {exc}") from exc
        if not isinstance(prepare_document, dict):
            raise ContractError(f"prepare manifest must contain an object: {prepare_path}")
        prepare_schema_version = _integer(
            prepare_document.get("schema_version"),
            location=f"prepare manifest {prepare_path}.schema_version",
        )
        if prepare_schema_version != PREPARE_MANIFEST_SCHEMA_VERSION:
            raise ContractError(
                "prepare manifest schema_version must be "
                f"{PREPARE_MANIFEST_SCHEMA_VERSION}; received {prepare_schema_version}"
            )
        units = prepare_document.get("analysis_units")
        if not isinstance(units, list):
            raise ContractError(f"prepare manifest lacks analysis_units: {prepare_path}")
        prepare_reference = _manifest_file_entry(prepare_path, relative_to=output_path.parent)
        for unit_index, raw_unit in enumerate(units):
            if not isinstance(raw_unit, dict) or raw_unit.get("status") != "prepared":
                continue
            analysis_unit = raw_unit.get("analysis_unit")
            outputs = raw_unit.get("outputs")
            if not isinstance(analysis_unit, str) or not isinstance(outputs, dict):
                raise ContractError(
                    f"invalid prepared unit {unit_index} in prepare manifest {prepare_path}"
                )
            dataset_id = f"{_safe_id(label)}-{_safe_id(analysis_unit)}"
            if dataset_id in seen_ids:
                raise ContractError(f"duplicate generated dataset id: {dataset_id}")
            seen_ids.add(dataset_id)
            output_entries: dict[str, dict[str, object] | None] = {}
            for key in ("expression", "groups", "tf_list", "centroid_weights"):
                entry = outputs.get(key)
                if entry is None:
                    output_entries[key] = None
                    continue
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    raise ContractError(
                        f"invalid outputs.{key} for prepared unit {analysis_unit!r}"
                    )
                local_path = (prepare_path.parent / entry["path"]).resolve()
                if not local_path.is_file():
                    raise ContractError(f"prepared output does not exist: {local_path}")
                actual = _manifest_file_entry(local_path, relative_to=output_path.parent)
                expected_hash = entry.get("sha256")
                expected_size = entry.get("size_bytes")
                if actual["sha256"] != expected_hash or actual["size_bytes"] != expected_size:
                    raise ContractError(
                        f"prepared output no longer matches {prepare_path}: {local_path}"
                    )
                output_entries[key] = actual
            if any(output_entries[key] is None for key in ("expression", "groups", "tf_list")):
                raise ContractError(f"prepared unit {analysis_unit!r} lacks required outputs")
            dimensions = {
                "cells": raw_unit.get("n_cells"),
                "genes": raw_unit.get("n_genes"),
                "transcription_factors": raw_unit.get("n_transcription_factors"),
                "groups": raw_unit.get("n_groups"),
            }
            for key, value in dimensions.items():
                _integer(value, location=f"{dataset_id}.dimensions.{key}", minimum=1)
            datasets.append(
                {
                    "id": dataset_id,
                    "description": f"Prepared analysis unit {analysis_unit} from {label}.",
                    "analysis_unit": analysis_unit,
                    "dimensions": dimensions,
                    "expression": output_entries["expression"],
                    "groups": output_entries["groups"],
                    "tf_list": output_entries["tf_list"],
                    "centroid_weights": output_entries["centroid_weights"],
                    "prepare_manifest": prepare_reference,
                }
            )
    if not datasets:
        raise ContractError("the supplied prepare manifests contain no prepared analysis units")
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "description": "Local prepared datasets for SPATHI equivalence benchmarking.",
        "datasets": datasets,
    }


def _resolve_package_source(path: Path) -> Path:
    supplied = path.expanduser().resolve()
    candidates = (supplied / "src" / "spathi", supplied / "spathi", supplied)
    for candidate in candidates:
        if (candidate / "__init__.py").is_file() and (candidate / "__main__.py").is_file():
            return candidate
    raise ContractError(
        f"implementation source must be a repository, src directory, or spathi package: {supplied}"
    )


def _snapshot_package(source: Path, destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ),
        key=lambda item: item.relative_to(source).as_posix(),
    )
    if not any(path.suffix == ".py" for path in files):
        raise ContractError(f"SPATHI package has no Python files: {source}")
    for source_path in files:
        relative_path = source_path.relative_to(source)
        content = source_path.read_bytes()
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        digest.update(relative_path.as_posix().encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _package_digest(source: Path) -> str:
    """Hash the exact package-tree contract used by implementation snapshots."""

    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ),
        key=lambda item: item.relative_to(source).as_posix(),
    )
    if not any(path.suffix == ".py" for path in files):
        raise ContractError(f"SPATHI package has no Python files: {source}")
    for source_path in files:
        relative_path = source_path.relative_to(source)
        digest.update(relative_path.as_posix().encode())
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_implementation_identity(
    profile: EquivalenceProfile,
    hashes: Mapping[str, str],
    *,
    allow_identical_implementations: bool,
) -> bool:
    identical = hashes["reference"] == hashes["candidate"]
    if (
        identical
        and not profile.allow_identical_implementations
        and not allow_identical_implementations
    ):
        raise ContractError(
            "reference and candidate implementation hashes are identical; pass "
            "--allow-identical-implementations only for an intentional harness self-check"
        )
    return identical


def snapshot_implementation(role: str, source: Path, *, output_root: Path) -> Implementation:
    package_source = _resolve_package_source(source)
    snapshot_parent = output_root / "implementations" / role
    snapshot_parent.mkdir(parents=True, exist_ok=False)
    digest = _snapshot_package(package_source, snapshot_parent / "spathi")
    return Implementation(
        role=role,
        source_path=package_source,
        snapshot_parent=snapshot_parent,
        sha256=digest,
    )


def _read_expression_genes(path: Path, *, expected_count: int) -> tuple[str, ...]:
    """Read only the first field of every dense expression row once per dataset."""

    genes: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        header = stream.readline()
        if not header or "\t" not in header:
            raise ContractError(f"expression file lacks a tabular header: {path}")
        for line_number, line in enumerate(stream, start=2):
            gene, separator, _ = line.partition("\t")
            if not separator or not gene:
                raise ContractError(f"invalid expression row {line_number}: {path}")
            if gene in seen:
                raise ContractError(f"duplicate expression gene {gene!r}: {path}")
            seen.add(gene)
            genes.append(gene)
    if len(genes) != expected_count:
        raise ContractError(
            f"expression gene count mismatch for {path}: expected {expected_count}, "
            f"observed {len(genes)}"
        )
    return tuple(genes)


def create_target_lists(
    dataset: LocalDataset,
    *,
    target_counts: Iterable[int],
    seed: int,
    destination: Path,
) -> dict[int, FileReference]:
    """Create deterministic nested target slices while retaining expression order."""

    counts = tuple(sorted(set(target_counts)))
    if not counts or counts[0] < 1 or counts[-1] > dataset.dimensions.genes:
        raise ContractError(
            f"target counts for {dataset.id!r} must be within 1..{dataset.dimensions.genes}"
        )
    genes = _read_expression_genes(dataset.expression.path, expected_count=dataset.dimensions.genes)
    ranked = sorted(
        range(len(genes)),
        key=lambda index: hashlib.sha256(f"{seed}\0{genes[index]}".encode()).digest(),
    )
    destination.mkdir(parents=True, exist_ok=True)
    references: dict[int, FileReference] = {}
    for count in counts:
        selected = set(ranked[:count])
        path = destination / f"targets-{count:05d}.txt"
        content = "".join(f"{gene}\n" for index, gene in enumerate(genes) if index in selected)
        path.write_text(content, encoding="utf-8")
        references[count] = FileReference(
            path_text=str(path),
            path=path,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
        )
    return references


def _required_target_counts(
    profile: EquivalenceProfile,
    selected: Sequence[LocalDataset],
) -> dict[tuple[str, int], set[int]]:
    """Return every deterministic target slice required by the selected schedule."""

    required: dict[tuple[str, int], set[int]] = {}
    for case in profile.cases:
        for dataset in _case_datasets(case, selected):
            counts = {
                int(target_count) for target_count in case.target_counts if target_count != "all"
            }
            if counts:
                required.setdefault(
                    (dataset.id, _target_selection_seed(case, dataset)), set()
                ).update(counts)
    return required


def _target_manifest_document(
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    selected: Sequence[LocalDataset],
    target_lists: Mapping[tuple[str, int], Mapping[int, FileReference]],
    *,
    suite_root: Path,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for (dataset_id, selection_seed), references in sorted(target_lists.items()):
        for target_count, reference in sorted(references.items()):
            entries.append(
                {
                    "dataset_id": dataset_id,
                    "selection_seed": selection_seed,
                    "target_count": target_count,
                    "path": str(reference.path.relative_to(suite_root)),
                    "sha256": reference.sha256,
                    "size_bytes": reference.size_bytes,
                }
            )
    return {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "profile_sha256": profile.sha256,
        "dataset_manifest_sha256": data_manifest.sha256,
        "selected_dataset_ids": [dataset.id for dataset in selected],
        "entries": entries,
    }


def _create_target_manifest(
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    selected: Sequence[LocalDataset],
    *,
    suite_root: Path,
) -> dict[tuple[str, int], dict[int, FileReference]]:
    """Create target slices once and persist their exact replay identity."""

    required = _required_target_counts(profile, selected)
    by_id = {dataset.id: dataset for dataset in selected}
    target_lists = {
        key: create_target_lists(
            by_id[key[0]],
            target_counts=counts,
            seed=key[1],
            destination=suite_root / "targets" / key[0] / f"seed-{key[1]}",
        )
        for key, counts in sorted(required.items())
    }
    _atomic_json(
        suite_root / "targets-manifest.json",
        _target_manifest_document(
            profile,
            data_manifest,
            selected,
            target_lists,
            suite_root=suite_root,
        ),
    )
    return target_lists


def _load_target_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    selected: Sequence[LocalDataset],
) -> dict[tuple[str, int], dict[int, FileReference]]:
    """Load hashed slices without rescanning the potentially huge expression TSV."""

    metadata = _manifest_object(manifest, "target_manifest")
    path = _verify_snapshotted_file(root, metadata, location="manifest.target_manifest")
    document = _load_json_object(path, location="target manifest")
    _check_fields(
        document,
        allowed=_TARGET_MANIFEST_FIELDS,
        required=_TARGET_MANIFEST_FIELDS,
        location="target manifest",
    )
    expected_header = {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "profile_sha256": profile.sha256,
        "dataset_manifest_sha256": data_manifest.sha256,
        "selected_dataset_ids": [dataset.id for dataset in selected],
    }
    mismatches = {
        key: (document.get(key), expected)
        for key, expected in expected_header.items()
        if document.get(key) != expected
    }
    if mismatches:
        raise ContractError(f"target manifest identity mismatch: {mismatches}")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ContractError("target manifest entries must be a list")
    expected = {
        (dataset_id, seed, target_count)
        for (dataset_id, seed), counts in _required_target_counts(profile, selected).items()
        for target_count in counts
    }
    observed: set[tuple[str, int, int]] = set()
    target_lists: dict[tuple[str, int], dict[int, FileReference]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _json_object(raw_entry, location=f"target manifest entries[{index}]")
        _check_fields(
            entry,
            allowed={
                "dataset_id",
                "selection_seed",
                "target_count",
                "path",
                "sha256",
                "size_bytes",
            },
            required={
                "dataset_id",
                "selection_seed",
                "target_count",
                "path",
                "sha256",
                "size_bytes",
            },
            location=f"target manifest entries[{index}]",
        )
        dataset_id = _string(entry["dataset_id"], location=f"target entries[{index}].dataset_id")
        selection_seed = _integer(
            entry["selection_seed"], location=f"target entries[{index}].selection_seed", minimum=0
        )
        target_count = _integer(
            entry["target_count"], location=f"target entries[{index}].target_count", minimum=1
        )
        identity = (dataset_id, selection_seed, target_count)
        if identity in observed:
            raise ContractError(f"duplicate target manifest entry: {identity}")
        observed.add(identity)
        target_path = _suite_member(root, entry["path"], location=f"target entries[{index}].path")
        expected_hash = _string(entry["sha256"], location=f"target entries[{index}].sha256")
        expected_size = _integer(
            entry["size_bytes"], location=f"target entries[{index}].size_bytes", minimum=0
        )
        if not target_path.is_file():
            raise ContractError(f"missing snapshotted target list: {target_path}")
        observed_size = target_path.stat().st_size
        observed_hash = _sha256(target_path)
        if observed_size != expected_size or observed_hash != expected_hash:
            raise ContractError(
                f"target-list snapshot mismatch for {target_path}: expected "
                f"size/hash {expected_size}/{expected_hash}, observed "
                f"{observed_size}/{observed_hash}"
            )
        target_lists.setdefault((dataset_id, selection_seed), {})[target_count] = FileReference(
            path_text=str(target_path),
            path=target_path,
            sha256=observed_hash,
            size_bytes=observed_size,
        )
    if observed != expected:
        raise ContractError(
            "target manifest does not match the immutable schedule: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return target_lists


_BOOTSTRAP = (
    "import runpy,sys; sys.dont_write_bytecode=True; "
    "sys.path.insert(0,sys.argv.pop(1)); "
    "runpy.run_module('spathi',run_name='__main__')"
)


def build_infer_command(
    implementation: Implementation,
    dataset: LocalDataset,
    *,
    target_list: Path | None,
    output_dir: Path,
    scientific_parameters: ScientificParameters,
    n_estimators: int,
    n_components: int,
    threads: int,
    seed: int,
    checkpoint: bool,
    resume: bool,
    report: bool,
) -> list[str]:
    """Build identical explicit infer arguments for either implementation snapshot."""

    if resume and not checkpoint:
        raise ContractError("a child run can resume only when checkpointing is enabled")
    single_group = dataset.dimensions.groups == 1
    weight_mode = (
        scientific_parameters.single_group_weight_mode
        if single_group
        else scientific_parameters.multi_group_weight_mode
    )
    distance_metric = (
        scientific_parameters.single_group_distance_metric
        if single_group
        else scientific_parameters.multi_group_distance_metric
    )
    group_size_correction = (
        scientific_parameters.single_group_size_correction
        if single_group
        else scientific_parameters.multi_group_size_correction
    )
    command = [
        sys.executable,
        "-c",
        _BOOTSTRAP,
        str(implementation.snapshot_parent),
        "infer",
        "--expression",
        str(dataset.expression.path),
        "--tf-list",
        str(dataset.tf_list.path),
        "--groups",
        str(dataset.groups.path),
        "--output-dir",
        str(output_dir),
        "--weight-mode",
        weight_mode,
        "--distance-space",
        scientific_parameters.distance_space,
        "--distance-standardization",
        scientific_parameters.distance_standardization,
        "--pca-svd-solver",
        scientific_parameters.pca_svd_solver,
        "--distance-metric",
        distance_metric,
        "--kernel",
        scientific_parameters.kernel,
        "--bandwidth",
        str(scientific_parameters.bandwidth),
        "--group-size-correction",
        group_size_correction,
        "--n-components",
        str(n_components),
        "--tree-method",
        scientific_parameters.tree_method,
        "--n-estimators",
        str(n_estimators),
        "--max-features",
        str(scientific_parameters.max_features),
        "--min-samples-leaf",
        str(scientific_parameters.min_samples_leaf),
        "--bootstrap" if scientific_parameters.bootstrap else "--no-bootstrap",
        "--adaptive-trees" if scientific_parameters.adaptive_trees else "--no-adaptive-trees",
        "--adaptive-min-estimators",
        str(scientific_parameters.adaptive_min_estimators),
        "--adaptive-tree-step",
        str(scientific_parameters.adaptive_tree_step),
        "--adaptive-tolerance",
        str(scientific_parameters.adaptive_tolerance),
        "--adaptive-patience",
        str(scientific_parameters.adaptive_patience),
        "--target-eligibility",
        scientific_parameters.target_eligibility,
        "--min-target-detected-cells",
        str(scientific_parameters.min_target_detected_cells),
        "--min-target-detected-fraction",
        str(scientific_parameters.min_target_detected_fraction),
        "--min-target-weighted-detected-fraction",
        str(scientific_parameters.min_target_weighted_detected_fraction),
        "--min-target-weighted-detected-ess",
        str(scientific_parameters.min_target_weighted_detected_ess),
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
    if resume:
        command.append("--resume")
    if scientific_parameters.max_depth is not None:
        command.extend(("--max-depth", str(scientific_parameters.max_depth)))
    if target_list is not None:
        command.extend(("--target-list", str(target_list)))
    if dataset.centroid_weights is not None:
        command.extend(("--centroid-weights", str(dataset.centroid_weights.path)))
    return command


def _open_table(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _relative_difference(reference: float, candidate: float) -> float:
    denominator = max(abs(reference), abs(candidate))
    if denominator == 0:
        return 0.0
    return abs(reference - candidate) / denominator


def compare_table(
    reference_path: Path,
    candidate_path: Path,
    *,
    rule: TableRule,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> ArtifactComparison:
    """Compare one canonical ordered table while ignoring declared volatile fields."""

    reference_hash = _sha256(reference_path) if reference_path.is_file() else ""
    candidate_hash = _sha256(candidate_path) if candidate_path.is_file() else ""
    if not reference_path.is_file() or not candidate_path.is_file():
        missing = [str(path) for path in (reference_path, candidate_path) if not path.is_file()]
        return ArtifactComparison(
            artifact=rule.filename,
            equivalent=False,
            reference_sha256=reference_hash,
            candidate_sha256=candidate_hash,
            rows_compared=0,
            numeric_values_compared=0,
            max_absolute_difference=0.0,
            max_relative_difference=0.0,
            mismatch_count=1,
            first_mismatches=("missing file: " + ", ".join(missing),),
            comparison_mode="unavailable",
            error="required artifact is missing",
        )

    if reference_hash == candidate_hash:
        try:
            with _open_table(reference_path) as stream:
                observed_columns = next(csv.reader(stream, delimiter=rule.delimiter), None)
            expected_columns = list(_TABLE_COLUMNS[rule.filename])
            if observed_columns != expected_columns:
                raise ContractError(
                    "byte-identical artifact violates the canonical schema: "
                    f"expected={expected_columns}, observed={observed_columns}"
                )
        except (OSError, UnicodeDecodeError, csv.Error, ContractError) as exc:
            return ArtifactComparison(
                artifact=rule.filename,
                equivalent=False,
                reference_sha256=reference_hash,
                candidate_sha256=candidate_hash,
                rows_compared=0,
                numeric_values_compared=0,
                max_absolute_difference=0.0,
                max_relative_difference=0.0,
                mismatch_count=1,
                first_mismatches=(),
                comparison_mode="byte-identical-invalid-schema",
                error=str(exc),
            )
        return ArtifactComparison(
            artifact=rule.filename,
            equivalent=True,
            reference_sha256=reference_hash,
            candidate_sha256=candidate_hash,
            rows_compared=0,
            numeric_values_compared=0,
            max_absolute_difference=0.0,
            max_relative_difference=0.0,
            mismatch_count=0,
            first_mismatches=(),
            comparison_mode="byte-identical",
        )

    mismatches: list[str] = []
    mismatch_count = 0
    rows_compared = 0
    numeric_values = 0
    max_absolute = 0.0
    max_relative = 0.0
    try:
        with (
            _open_table(reference_path) as reference_stream,
            _open_table(candidate_path) as candidate_stream,
        ):
            reference_reader = csv.DictReader(reference_stream, delimiter=rule.delimiter)
            candidate_reader = csv.DictReader(candidate_stream, delimiter=rule.delimiter)
            if reference_reader.fieldnames is None or candidate_reader.fieldnames is None:
                raise ContractError("missing table header")
            expected_columns = list(_TABLE_COLUMNS[rule.filename])
            if reference_reader.fieldnames != expected_columns:
                raise ContractError(
                    f"reference header violates the canonical schema: "
                    f"expected={expected_columns}, observed={reference_reader.fieldnames}"
                )
            if candidate_reader.fieldnames != expected_columns:
                raise ContractError(
                    f"candidate header violates the canonical schema: "
                    f"expected={expected_columns}, observed={candidate_reader.fieldnames}"
                )
            unknown_numeric = sorted(rule.numeric_columns - set(reference_reader.fieldnames))
            unknown_ignored = sorted(rule.ignored_columns - set(reference_reader.fieldnames))
            if unknown_numeric or unknown_ignored:
                raise ContractError(
                    f"comparison schema does not match header; numeric={unknown_numeric}, "
                    f"ignored={unknown_ignored}"
                )
            for row_number, pair in enumerate(
                zip_longest(reference_reader, candidate_reader), start=2
            ):
                reference_row, candidate_row = pair
                if reference_row is None or candidate_row is None:
                    mismatch_count += 1
                    if len(mismatches) < 10:
                        mismatches.append(f"row-count mismatch at row {row_number}")
                    break
                if (
                    None in reference_row
                    or None in candidate_row
                    or any(value is None for value in reference_row.values())
                    or any(value is None for value in candidate_row.values())
                ):
                    raise ContractError(f"malformed tabular row {row_number}")
                rows_compared += 1
                for field in reference_reader.fieldnames:
                    if field in rule.ignored_columns:
                        continue
                    reference_value = reference_row[field]
                    candidate_value = candidate_row[field]
                    if field not in rule.numeric_columns:
                        if reference_value != candidate_value:
                            mismatch_count += 1
                            if len(mismatches) < 10:
                                mismatches.append(
                                    f"row {row_number}, {field}: "
                                    f"{reference_value!r} != {candidate_value!r}"
                                )
                        continue
                    if not reference_value or not candidate_value:
                        if reference_value != candidate_value:
                            mismatch_count += 1
                            if len(mismatches) < 10:
                                mismatches.append(
                                    f"row {row_number}, {field}: "
                                    f"{reference_value!r} != {candidate_value!r}"
                                )
                        continue
                    numeric_values += 1
                    reference_number = float(reference_value)
                    candidate_number = float(candidate_value)
                    if not math.isfinite(reference_number) or not math.isfinite(candidate_number):
                        equal = False
                        absolute_difference = math.inf
                        relative_difference = math.inf
                    else:
                        absolute_difference = abs(reference_number - candidate_number)
                        relative_difference = _relative_difference(
                            reference_number, candidate_number
                        )
                        equal = math.isclose(
                            reference_number,
                            candidate_number,
                            rel_tol=relative_tolerance,
                            abs_tol=absolute_tolerance,
                        )
                    max_absolute = max(max_absolute, absolute_difference)
                    max_relative = max(max_relative, relative_difference)
                    if not equal:
                        mismatch_count += 1
                        if len(mismatches) < 10:
                            mismatches.append(
                                f"row {row_number}, {field}: {reference_value} != {candidate_value}"
                            )
    except (OSError, UnicodeDecodeError, csv.Error, ValueError, ContractError) as exc:
        return ArtifactComparison(
            artifact=rule.filename,
            equivalent=False,
            reference_sha256=reference_hash,
            candidate_sha256=candidate_hash,
            rows_compared=rows_compared,
            numeric_values_compared=numeric_values,
            max_absolute_difference=max_absolute,
            max_relative_difference=max_relative,
            mismatch_count=max(1, mismatch_count),
            first_mismatches=tuple(mismatches),
            comparison_mode="semantic-tolerance",
            error=str(exc),
        )
    return ArtifactComparison(
        artifact=rule.filename,
        equivalent=mismatch_count == 0,
        reference_sha256=reference_hash,
        candidate_sha256=candidate_hash,
        rows_compared=rows_compared,
        numeric_values_compared=numeric_values,
        max_absolute_difference=max_absolute,
        max_relative_difference=max_relative,
        mismatch_count=mismatch_count,
        first_mismatches=tuple(mismatches),
        comparison_mode="semantic-tolerance",
    )


def _normalized_parameters(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ContractError("parameters.json must contain a JSON object")
    normalized = dict(document)
    normalized.pop("output_dir", None)
    return normalized


def compare_outputs(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> OutputComparison:
    """Compare all scientific SPATHI tables and normalized requested parameters."""

    comparisons = tuple(
        compare_table(
            reference_dir / rule.filename,
            candidate_dir / rule.filename,
            rule=rule,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for rule in _TABLE_RULES
    )
    parameter_error = ""
    try:
        if _normalized_parameters(reference_dir / "parameters.json") != _normalized_parameters(
            candidate_dir / "parameters.json"
        ):
            parameter_error = "normalized parameters.json values differ"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        parameter_error = f"cannot compare parameters.json: {exc}"
    equivalent = not parameter_error and all(item.equivalent for item in comparisons)
    return OutputComparison(equivalent=equivalent, artifacts=comparisons, error=parameter_error)


def _ratio(numerator: float, denominator: float) -> float | str:
    if denominator == 0:
        return ""
    return numerator / denominator


def _metadata_value(metadata: Mapping[str, object], key: str) -> object:
    value = metadata.get(key, "")
    return "" if value is None else value


def _audit_run_metadata(
    path: Path,
    *,
    dataset: LocalDataset,
    target_list: FileReference | None,
    target_count: int,
    n_estimators: int,
    n_components: int,
    threads: int,
    seed: int,
    checkpoint: bool,
    resume: bool,
    report: bool,
    scientific_parameters: ScientificParameters,
) -> tuple[dict[str, str], str]:
    """Audit provenance and requested controls beyond the flattened timing fields."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ContractError("run_metadata.json must contain an object")
        dependencies = document.get("dependency_versions")
        inputs = document.get("inputs")
        requested = document.get("requested_parameters")
        checkpoint_metadata = document.get("checkpoint")
        report_metadata = document.get("report")
        input_dimensions = document.get("input_dimensions")
        if not all(
            isinstance(value, dict)
            for value in (
                dependencies,
                inputs,
                requested,
                checkpoint_metadata,
                report_metadata,
                input_dimensions,
            )
        ):
            raise ContractError("run metadata lacks provenance/control objects")
        expected_inputs = {
            "expression": dataset.expression.sha256,
            "groups": dataset.groups.sha256,
            "tf_list": dataset.tf_list.sha256,
        }
        if target_list is not None:
            expected_inputs["target_list"] = target_list.sha256
        if dataset.centroid_weights is not None:
            expected_inputs["centroid_weights"] = dataset.centroid_weights.sha256
        input_mismatches = {
            key: (inputs.get(key), expected)
            for key, expected in expected_inputs.items()
            if not isinstance(inputs.get(key), dict) or inputs[key].get("sha256") != expected
        }
        for optional_input, expected in (
            ("target_list", target_list),
            ("centroid_weights", dataset.centroid_weights),
        ):
            if expected is None and optional_input in inputs:
                input_mismatches[optional_input] = (inputs.get(optional_input), "absent")
        expected_dimensions = {
            "cells": dataset.dimensions.cells,
            "genes": dataset.dimensions.genes,
            "transcription_factors": dataset.dimensions.transcription_factors,
            "groups": dataset.dimensions.groups,
            "targets": target_count,
        }
        dimension_mismatches = {
            key: (input_dimensions.get(key), expected)
            for key, expected in expected_dimensions.items()
            if input_dimensions.get(key) != expected
        }
        single_group = dataset.dimensions.groups == 1
        expected_requested = {
            "n_estimators": n_estimators,
            "n_components": n_components,
            "threads": threads,
            "random_seed": seed,
            "tree_method": scientific_parameters.tree_method,
            "weight_mode": (
                scientific_parameters.single_group_weight_mode
                if single_group
                else scientific_parameters.multi_group_weight_mode
            ),
            "distance_space": scientific_parameters.distance_space,
            "distance_standardization": scientific_parameters.distance_standardization,
            "pca_svd_solver": scientific_parameters.pca_svd_solver,
            "distance_metric": (
                scientific_parameters.single_group_distance_metric
                if single_group
                else scientific_parameters.multi_group_distance_metric
            ),
            "kernel": scientific_parameters.kernel,
            "bandwidth": scientific_parameters.bandwidth,
            "group_size_correction": (
                scientific_parameters.single_group_size_correction
                if single_group
                else scientific_parameters.multi_group_size_correction
            ),
            "max_features": scientific_parameters.max_features,
            "min_samples_leaf": scientific_parameters.min_samples_leaf,
            "max_depth": scientific_parameters.max_depth,
            "bootstrap": scientific_parameters.bootstrap,
            "adaptive_trees": scientific_parameters.adaptive_trees,
            "adaptive_min_estimators": scientific_parameters.adaptive_min_estimators,
            "adaptive_tree_step": scientific_parameters.adaptive_tree_step,
            "adaptive_tolerance": scientific_parameters.adaptive_tolerance,
            "adaptive_patience": scientific_parameters.adaptive_patience,
            "target_eligibility": scientific_parameters.target_eligibility,
            "min_target_detected_cells": scientific_parameters.min_target_detected_cells,
            "min_target_detected_fraction": (scientific_parameters.min_target_detected_fraction),
            "min_target_weighted_detected_fraction": (
                scientific_parameters.min_target_weighted_detected_fraction
            ),
            "min_target_weighted_detected_ess": (
                scientific_parameters.min_target_weighted_detected_ess
            ),
            "report": report,
        }
        requested_mismatches = {
            key: (requested.get(key), expected)
            for key, expected in expected_requested.items()
            if requested.get(key) != expected
        }
        operational_mismatches: dict[str, tuple[object, object]] = {}
        if checkpoint_metadata.get("enabled") != checkpoint:
            operational_mismatches["checkpoint.enabled"] = (
                checkpoint_metadata.get("enabled"),
                checkpoint,
            )
        if checkpoint_metadata.get("resumed") != resume:
            operational_mismatches["checkpoint.resumed"] = (
                checkpoint_metadata.get("resumed"),
                resume,
            )
        if report_metadata.get("requested") != report:
            operational_mismatches["report.requested"] = (
                report_metadata.get("requested"),
                report,
            )
        mismatches = {
            **{f"inputs.{key}": value for key, value in input_mismatches.items()},
            **{f"input_dimensions.{key}": value for key, value in dimension_mismatches.items()},
            **{f"requested_parameters.{key}": value for key, value in requested_mismatches.items()},
            **operational_mismatches,
        }
        if mismatches:
            return {}, f"run metadata provenance/control mismatch: {mismatches}"
        version = document.get("spathi_version")
        if not isinstance(version, str) or not version:
            raise ContractError("run metadata lacks spathi_version")
        return {
            "spathi_version": version,
            "dependency_versions_json": json.dumps(
                dependencies,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }, ""
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        return {}, str(exc)


def _run_row(
    *,
    run_id: str,
    attempt: int,
    run_type: str,
    round_index: int,
    configuration_position: int,
    execution_position: int,
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    case: ExperimentCase,
    dataset: LocalDataset,
    implementation: Implementation,
    target_scope: str,
    target_count: int,
    target_list: FileReference | None,
    n_estimators: int,
    threads: int,
    resumed_from_checkpoint: bool,
    measurement: Any,
    output_dir: Path,
    input_storage: Any,
) -> dict[str, object]:
    published = measure_storage((output_dir,)) if output_dir.is_dir() else measure_storage(())
    metadata_measurement = extract_run_metadata(output_dir / "run_metadata.json")
    metadata = metadata_measurement.csv_fields
    provenance, provenance_error = _audit_run_metadata(
        output_dir / "run_metadata.json",
        dataset=dataset,
        target_list=target_list,
        target_count=target_count,
        n_estimators=n_estimators,
        n_components=case.n_components,
        threads=threads,
        seed=case.seed,
        checkpoint=case.checkpoint,
        resume=resumed_from_checkpoint,
        report=case.report,
        scientific_parameters=profile.scientific_parameters,
    )
    error = measurement.error
    status = measurement.status
    metadata_error = "; ".join(
        value for value in (metadata_measurement.error, provenance_error) if value
    )
    if measurement.status == "success" and metadata_error:
        status = "metadata_error"
        error = metadata_error
    row = {
        "run_id": run_id,
        "attempt": attempt,
        "run_type": run_type,
        "round": round_index,
        "configuration_position": configuration_position,
        "execution_position": execution_position,
        "profile": profile.name,
        "profile_sha256": profile.sha256,
        "dataset_manifest_sha256": data_manifest.sha256,
        "case_id": case.id,
        "dataset_id": dataset.id,
        "analysis_unit": dataset.analysis_unit,
        "implementation_role": implementation.role,
        "implementation_sha256": implementation.sha256,
        "spathi_version": provenance.get("spathi_version", ""),
        "dependency_versions_json": provenance.get("dependency_versions_json", ""),
        "target_scope": target_scope,
        "target_count": target_count,
        "n_estimators": n_estimators,
        "threads": threads,
        "n_components": case.n_components,
        "seed": case.seed,
        "checkpoint": case.checkpoint,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "performance_eligible": not resumed_from_checkpoint,
        "report": case.report,
        "wall_seconds": measurement.wall_seconds,
        "peak_rss_bytes": measurement.peak_rss_bytes,
        "sampled_cpu_user_seconds": measurement.sampled_cpu_user_seconds,
        "sampled_cpu_system_seconds": measurement.sampled_cpu_system_seconds,
        "status": status,
        "exit_code": "" if measurement.exit_code is None else measurement.exit_code,
        "error": error,
        "input_logical_bytes": input_storage.logical_bytes,
        "input_allocated_bytes": input_storage.allocated_bytes,
        "input_file_count": input_storage.file_count,
        "peak_run_logical_bytes": measurement.peak_run_logical_bytes,
        "peak_run_allocated_bytes": measurement.peak_run_allocated_bytes,
        "peak_run_file_count": measurement.peak_run_file_count,
        "published_output_logical_bytes": published.logical_bytes,
        "published_output_allocated_bytes": published.allocated_bytes,
        "published_output_file_count": published.file_count,
        "expression_sha256": dataset.expression.sha256,
        "groups_sha256": dataset.groups.sha256,
        "tf_list_sha256": dataset.tf_list.sha256,
        "centroid_weights_sha256": (
            "" if dataset.centroid_weights is None else dataset.centroid_weights.sha256
        ),
        "target_list_sha256": "" if target_list is None else target_list.sha256,
        **{key: _metadata_value(metadata, key) for key in _METADATA_RUN_FIELDS},
    }
    if status == "success":
        expected_metadata = {
            "run_metadata_status": "complete",
            "actual_cells": dataset.dimensions.cells,
            "actual_genes": dataset.dimensions.genes,
            "actual_targets": target_count,
            "actual_tfs": dataset.dimensions.transcription_factors,
            "actual_groups": dataset.dimensions.groups,
            "models_requested": dataset.dimensions.groups * target_count,
            "models_completed": dataset.dimensions.groups * target_count,
        }
        mismatches = {
            key: (row.get(key, ""), expected)
            for key, expected in expected_metadata.items()
            if row.get(key, "") != expected
        }
        if mismatches:
            row["status"] = "metadata_error"
            row["error"] = f"run metadata does not match the scheduled input: {mismatches}"
    reused_models = row.get("models_reused_from_checkpoint", "")
    row["performance_eligible"] = bool(
        row["status"] == "success" and not resumed_from_checkpoint and reused_models in {"", 0}
    )
    return row


def _comparison_row(
    *,
    comparison_id: str,
    attempt: int,
    run_type: str,
    round_index: int,
    configuration_position: int,
    profile: EquivalenceProfile,
    case: ExperimentCase,
    dataset: LocalDataset,
    target_scope: str,
    target_count: int,
    n_estimators: int,
    threads: int,
    reference_row: Mapping[str, object],
    candidate_row: Mapping[str, object],
    comparison: OutputComparison | None,
    details_path: Path | None,
) -> dict[str, object]:
    both_successful = reference_row["status"] == candidate_row["status"] == "success"
    status = "compared" if both_successful and comparison is not None else "not-compared"
    equivalent = bool(comparison is not None and comparison.equivalent)
    artifact_mismatches = (
        [] if comparison is None else [item for item in comparison.artifacts if not item.equivalent]
    )
    reference_wall = float(reference_row["wall_seconds"])
    candidate_wall = float(candidate_row["wall_seconds"])
    reference_rss = float(reference_row["peak_rss_bytes"])
    candidate_rss = float(candidate_row["peak_rss_bytes"])
    reference_cpu = float(reference_row["sampled_cpu_user_seconds"]) + float(
        reference_row["sampled_cpu_system_seconds"]
    )
    candidate_cpu = float(candidate_row["sampled_cpu_user_seconds"]) + float(
        candidate_row["sampled_cpu_system_seconds"]
    )
    reference_peak_logical = float(reference_row["peak_run_logical_bytes"])
    candidate_peak_logical = float(candidate_row["peak_run_logical_bytes"])
    reference_peak_allocated = float(reference_row["peak_run_allocated_bytes"])
    candidate_peak_allocated = float(candidate_row["peak_run_allocated_bytes"])
    reference_output = float(reference_row["published_output_logical_bytes"])
    candidate_output = float(candidate_row["published_output_logical_bytes"])
    reference_output_allocated = float(reference_row["published_output_allocated_bytes"])
    candidate_output_allocated = float(candidate_row["published_output_allocated_bytes"])
    reference_model = reference_row["phase_model_inference_seconds"]
    candidate_model = candidate_row["phase_model_inference_seconds"]
    error_parts = [
        str(row["error"])
        for row in (reference_row, candidate_row)
        if row["status"] != "success" and row["error"]
    ]
    if comparison is not None and comparison.error:
        error_parts.append(comparison.error)
    performance_comparable = bool(
        both_successful
        and comparison is not None
        and comparison.equivalent
        and reference_row["performance_eligible"] is True
        and candidate_row["performance_eligible"] is True
        and reference_row["attempt"] == candidate_row["attempt"]
    )

    def comparable_ratio(numerator: float, denominator: float) -> float | str:
        return _ratio(numerator, denominator) if performance_comparable else ""

    return {
        "comparison_id": comparison_id,
        "attempt": attempt,
        "run_type": run_type,
        "round": round_index,
        "configuration_position": configuration_position,
        "profile": profile.name,
        "case_id": case.id,
        "dataset_id": dataset.id,
        "target_scope": target_scope,
        "target_count": target_count,
        "n_estimators": n_estimators,
        "threads": threads,
        "n_components": case.n_components,
        "reference_run_id": reference_row["run_id"],
        "reference_run_attempt": reference_row["attempt"],
        "candidate_run_id": candidate_row["run_id"],
        "candidate_run_attempt": candidate_row["attempt"],
        "status": status,
        "equivalent": equivalent,
        "performance_eligible": performance_comparable,
        "artifact_count": 0 if comparison is None else len(comparison.artifacts),
        "mismatched_artifact_count": len(artifact_mismatches),
        "max_absolute_difference": (
            ""
            if comparison is None
            else max((item.max_absolute_difference for item in comparison.artifacts), default=0.0)
        ),
        "max_relative_difference": (
            ""
            if comparison is None
            else max((item.max_relative_difference for item in comparison.artifacts), default=0.0)
        ),
        "reference_wall_seconds": reference_wall,
        "candidate_wall_seconds": candidate_wall,
        "wall_speedup_reference_over_candidate": comparable_ratio(reference_wall, candidate_wall),
        "reference_peak_rss_bytes": reference_rss,
        "candidate_peak_rss_bytes": candidate_rss,
        "candidate_over_reference_peak_rss": comparable_ratio(candidate_rss, reference_rss),
        "reference_sampled_cpu_seconds": reference_cpu,
        "candidate_sampled_cpu_seconds": candidate_cpu,
        "candidate_over_reference_sampled_cpu": comparable_ratio(candidate_cpu, reference_cpu),
        "reference_peak_run_logical_bytes": reference_peak_logical,
        "candidate_peak_run_logical_bytes": candidate_peak_logical,
        "candidate_over_reference_peak_run_logical_bytes": comparable_ratio(
            candidate_peak_logical, reference_peak_logical
        ),
        "reference_peak_run_allocated_bytes": reference_peak_allocated,
        "candidate_peak_run_allocated_bytes": candidate_peak_allocated,
        "candidate_over_reference_peak_run_allocated_bytes": comparable_ratio(
            candidate_peak_allocated, reference_peak_allocated
        ),
        "reference_output_logical_bytes": reference_output,
        "candidate_output_logical_bytes": candidate_output,
        "candidate_over_reference_output_logical_bytes": comparable_ratio(
            candidate_output, reference_output
        ),
        "reference_output_allocated_bytes": reference_output_allocated,
        "candidate_output_allocated_bytes": candidate_output_allocated,
        "candidate_over_reference_output_allocated_bytes": comparable_ratio(
            candidate_output_allocated, reference_output_allocated
        ),
        "reference_model_seconds": reference_model,
        "candidate_model_seconds": candidate_model,
        "model_speedup_reference_over_candidate": (
            comparable_ratio(float(reference_model), float(candidate_model))
            if performance_comparable and reference_model != "" and candidate_model != ""
            else ""
        ),
        "details_json": "" if details_path is None else str(details_path),
        "error": "; ".join(error_parts),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return an inclusive linearly interpolated percentile of finite values."""

    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_ratio_summary(
    prefix: str,
    values: Sequence[float],
    *,
    identity: Sequence[object],
) -> dict[str, float | str]:
    """Summarize paired ratios with spread and a deterministic descriptive bootstrap."""

    fields = {f"{prefix}_{suffix}": "" for suffix in _PAIRED_RATIO_SUMMARY_SUFFIXES}
    if not values:
        return fields
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) != len(values):
        return fields
    observed_median = statistics.median(finite)
    if len(finite) == 1:
        bootstrap_low = observed_median
        bootstrap_high = observed_median
    else:
        seed_payload = json.dumps(
            [prefix, *map(str, identity)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big"))
        bootstrap_medians = [
            statistics.median(rng.choices(finite, k=len(finite))) for _ in range(5000)
        ]
        bootstrap_low = _percentile(bootstrap_medians, 0.025)
        bootstrap_high = _percentile(bootstrap_medians, 0.975)
    return {
        f"{prefix}_median": observed_median,
        f"{prefix}_q1": _percentile(finite, 0.25),
        f"{prefix}_q3": _percentile(finite, 0.75),
        f"{prefix}_bootstrap_ci95_low": bootstrap_low,
        f"{prefix}_bootstrap_ci95_high": bootstrap_high,
    }


def _scaling_rows(
    comparisons: Sequence[Mapping[str, object]],
    datasets: Mapping[str, LocalDataset],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in comparisons:
        if row["run_type"] != "measurement":
            continue
        key = tuple(
            row[field]
            for field in (
                "profile",
                "case_id",
                "dataset_id",
                "target_scope",
                "target_count",
                "n_estimators",
                "threads",
                "n_components",
            )
        )
        grouped.setdefault(key, []).append(row)

    result: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        (
            profile,
            case_id,
            dataset_id,
            target_scope,
            target_count,
            estimators,
            threads,
            components,
        ) = key
        expected_models = datasets[str(dataset_id)].dimensions.groups * int(target_count)
        # Runtime and memory speedups are meaningful only after the scientific
        # artifacts pass equivalence.  An inequivalent candidate may simply have
        # skipped work, so including it would reward a changed computation.
        equivalent = [
            row for row in rows if row["status"] == "compared" and row["equivalent"] is True
        ]
        compared = [row for row in equivalent if row["performance_eligible"] is True]

        def numeric_values(
            field: str,
            current_rows: tuple[Mapping[str, object], ...] = tuple(compared),
        ) -> list[float]:
            return [float(row[field]) for row in current_rows if row[field] != ""]

        reference_walls = numeric_values("reference_wall_seconds")
        candidate_walls = numeric_values("candidate_wall_seconds")
        reference_models = numeric_values("reference_model_seconds")
        candidate_models = numeric_values("candidate_model_seconds")
        reference_cpu = numeric_values("reference_sampled_cpu_seconds")
        candidate_cpu = numeric_values("candidate_sampled_cpu_seconds")

        def median_or_empty(values: Sequence[float]) -> float | str:
            return "" if not values else statistics.median(values)

        def max_or_empty(field: str) -> float | str:
            values = numeric_values(field)
            return "" if not values else max(values)

        row_summary: dict[str, object] = {
            "profile": profile,
            "case_id": case_id,
            "dataset_id": dataset_id,
            "target_scope": target_scope,
            "target_count": target_count,
            "n_estimators": estimators,
            "threads": threads,
            "n_components": components,
            "expected_models": expected_models,
            "measured_pairs": len(rows),
            "equivalent_pairs": len(equivalent),
            "performance_eligible_pairs": len(compared),
            "reference_median_wall_seconds": median_or_empty(reference_walls),
            "candidate_median_wall_seconds": median_or_empty(candidate_walls),
            "reference_median_model_seconds": median_or_empty(reference_models),
            "candidate_median_model_seconds": median_or_empty(candidate_models),
            "reference_models_per_model_second": (
                ""
                if not reference_models
                else _ratio(expected_models, statistics.median(reference_models))
            ),
            "candidate_models_per_model_second": (
                ""
                if not candidate_models
                else _ratio(expected_models, statistics.median(candidate_models))
            ),
            "reference_max_peak_rss_bytes": max_or_empty("reference_peak_rss_bytes"),
            "candidate_max_peak_rss_bytes": max_or_empty("candidate_peak_rss_bytes"),
            "reference_median_sampled_cpu_seconds": median_or_empty(reference_cpu),
            "candidate_median_sampled_cpu_seconds": median_or_empty(candidate_cpu),
            "reference_max_peak_run_logical_bytes": max_or_empty(
                "reference_peak_run_logical_bytes"
            ),
            "candidate_max_peak_run_logical_bytes": max_or_empty(
                "candidate_peak_run_logical_bytes"
            ),
            "reference_max_peak_run_allocated_bytes": max_or_empty(
                "reference_peak_run_allocated_bytes"
            ),
            "candidate_max_peak_run_allocated_bytes": max_or_empty(
                "candidate_peak_run_allocated_bytes"
            ),
            "reference_max_output_logical_bytes": max_or_empty("reference_output_logical_bytes"),
            "candidate_max_output_logical_bytes": max_or_empty("candidate_output_logical_bytes"),
            "reference_max_output_allocated_bytes": max_or_empty(
                "reference_output_allocated_bytes"
            ),
            "candidate_max_output_allocated_bytes": max_or_empty(
                "candidate_output_allocated_bytes"
            ),
        }
        ratio_fields = {
            "paired_wall_speedup_reference_over_candidate": (
                "wall_speedup_reference_over_candidate"
            ),
            "paired_model_speedup_reference_over_candidate": (
                "model_speedup_reference_over_candidate"
            ),
            "paired_candidate_over_reference_peak_rss": ("candidate_over_reference_peak_rss"),
            "paired_candidate_over_reference_sampled_cpu": ("candidate_over_reference_sampled_cpu"),
            "paired_candidate_over_reference_peak_run_logical_bytes": (
                "candidate_over_reference_peak_run_logical_bytes"
            ),
            "paired_candidate_over_reference_peak_run_allocated_bytes": (
                "candidate_over_reference_peak_run_allocated_bytes"
            ),
        }
        for prefix, field in ratio_fields.items():
            row_summary.update(
                _paired_ratio_summary(
                    prefix,
                    numeric_values(field),
                    identity=key,
                )
            )
        result.append(row_summary)
    return result


def _select_datasets(
    profile: EquivalenceProfile,
    manifest: DatasetManifest,
    requested_ids: Sequence[str],
) -> tuple[LocalDataset, ...]:
    by_id = {dataset.id: dataset for dataset in manifest.datasets}
    if profile.minimum_datasets > 0 and not requested_ids:
        raise ContractError(
            f"profile {profile.name!r} requires at least one explicit --dataset selection"
        )
    if len(requested_ids) != len(set(requested_ids)):
        raise ContractError("--dataset selections must not contain duplicates")
    requested = set(requested_ids)
    unknown_requested = sorted(requested - set(by_id))
    if unknown_requested:
        raise ContractError(f"unknown --dataset ids: {', '.join(unknown_requested)}")
    profile_ids = {dataset_id for case in profile.cases for dataset_id in case.dataset_ids}
    unknown_profile = sorted(profile_ids - set(by_id))
    if unknown_profile:
        raise ContractError(f"profile references unknown dataset ids: {', '.join(unknown_profile)}")
    relevant_ids = requested or set(by_id)
    selected = tuple(dataset for dataset in manifest.datasets if dataset.id in relevant_ids)
    if not selected:
        raise ContractError("no datasets selected")
    if len(selected) < profile.minimum_datasets:
        raise ContractError(
            f"profile {profile.name!r} requires at least {profile.minimum_datasets} "
            f"selected dataset(s); observed {len(selected)}"
        )
    if profile.maximum_datasets is not None and len(selected) > profile.maximum_datasets:
        raise ContractError(
            f"profile {profile.name!r} accepts at most {profile.maximum_datasets} "
            f"selected dataset(s); observed {len(selected)}"
        )
    return selected


def _case_datasets(
    case: ExperimentCase,
    selected: Sequence[LocalDataset],
) -> tuple[LocalDataset, ...]:
    allowed = set(case.dataset_ids)
    return tuple(dataset for dataset in selected if not allowed or dataset.id in allowed)


def _effective_target_count(target_budget: TargetBudget, dataset: LocalDataset) -> int:
    return dataset.dimensions.genes if target_budget == "all" else target_budget


def _target_scope(target_budget: TargetBudget) -> str:
    return "all-expression-genes" if target_budget == "all" else "deterministic-slice"


def _balanced_configuration_orders(
    configurations: Sequence[BenchmarkConfiguration],
    rounds: int,
    *,
    seed: int,
) -> tuple[tuple[BenchmarkConfiguration, ...], ...]:
    """Return deterministic shuffled rotations so size never fixes run position."""

    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not configurations or rounds == 0:
        return ()
    rng = random.Random(seed)
    orders: list[tuple[BenchmarkConfiguration, ...]] = []
    while len(orders) < rounds:
        base = list(configurations)
        rng.shuffle(base)
        block_length = min(len(base), rounds - len(orders))
        for offset in range(block_length):
            orders.append(tuple(base[offset:] + base[:offset]))
    return tuple(orders)


def _configuration_order_seed(
    profile: EquivalenceProfile,
    case: ExperimentCase,
    dataset: LocalDataset,
    *,
    run_type: str,
) -> int:
    payload = f"{profile.sha256}\0{case.seed}\0{case.id}\0{dataset.id}\0{run_type}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _estimated_output_bytes(
    profile: EquivalenceProfile,
    selected: Sequence[LocalDataset],
) -> int:
    total = 1024 * 1024 * 1024
    for case in profile.cases:
        for dataset in _case_datasets(case, selected):
            paired_rounds = 2 * (case.warmups + case.repeats)
            for target_budget in case.target_counts:
                targets = _effective_target_count(target_budget, dataset)
                models = dataset.dimensions.groups * targets
                maximum_edges = models * dataset.dimensions.transcription_factors
                tabular_bytes = (
                    maximum_edges * 112
                    + models * 768
                    + dataset.dimensions.groups * dataset.dimensions.cells * 144
                    + dataset.dimensions.cells * 192
                    + 16 * 1024 * 1024
                )
                checkpoint_multiplier = 2 if case.checkpoint else 1
                report_bytes = 32 * 1024 * 1024 if case.report else 0
                per_run = tabular_bytes * checkpoint_multiplier + report_bytes
                total += len(case.n_estimators) * len(case.threads) * paired_rounds * per_run
    return total


def _configuration_seed(case: ExperimentCase, dataset: LocalDataset) -> int:
    digest = hashlib.sha256(f"{case.seed}\0{case.id}\0{dataset.id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _target_selection_seed(case: ExperimentCase, dataset: LocalDataset) -> int:
    """Keep target samples stable across profile cases sharing a declared seed."""

    digest = hashlib.sha256(f"{case.seed}\0{dataset.id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _role_order(seed: int, *, round_index: int, configuration_index: int) -> tuple[str, str]:
    roles = ["reference", "candidate"]
    rng = random.Random(seed + configuration_index)
    rng.shuffle(roles)
    if round_index % 2 == 0:
        roles.reverse()
    return roles[0], roles[1]


def _build_schedule(
    profile: EquivalenceProfile,
    selected: Sequence[LocalDataset],
    target_lists: Mapping[tuple[str, int], Mapping[int, FileReference]],
) -> tuple[ScheduledPair, ...]:
    """Materialize the deterministic schedule used to validate durable journals."""

    schedule: list[ScheduledPair] = []
    configuration_index = 0
    for case in profile.cases:
        for dataset in _case_datasets(case, selected):
            configurations: list[BenchmarkConfiguration] = []
            for target_budget in case.target_counts:
                target_count = _effective_target_count(target_budget, dataset)
                target_list = (
                    None
                    if target_budget == "all"
                    else target_lists[(dataset.id, _target_selection_seed(case, dataset))][
                        target_budget
                    ]
                )
                for n_estimators in case.n_estimators:
                    for threads in case.threads:
                        configuration_index += 1
                        configurations.append(
                            BenchmarkConfiguration(
                                index=configuration_index,
                                target_scope=_target_scope(target_budget),
                                target_count=target_count,
                                target_list=target_list,
                                n_estimators=n_estimators,
                                threads=threads,
                            )
                        )
            for run_type, rounds in (("warmup", case.warmups), ("measurement", case.repeats)):
                orders = _balanced_configuration_orders(
                    configurations,
                    rounds,
                    seed=_configuration_order_seed(
                        profile,
                        case,
                        dataset,
                        run_type=run_type,
                    ),
                )
                for round_index, order in enumerate(orders, start=1):
                    for configuration_position, configuration in enumerate(order, start=1):
                        target_label = (
                            "all"
                            if configuration.target_scope == "all-expression-genes"
                            else str(configuration.target_count)
                        )
                        comparison_id = (
                            f"{case.id}--{dataset.id}--t{target_label}--"
                            f"k{configuration.n_estimators}--j{configuration.threads}--"
                            f"{run_type}-{round_index:02d}"
                        )
                        schedule.append(
                            ScheduledPair(
                                comparison_id=comparison_id,
                                run_type=run_type,
                                round_index=round_index,
                                configuration_position=configuration_position,
                                case=case,
                                dataset=dataset,
                                configuration=configuration,
                                role_order=_role_order(
                                    _configuration_seed(case, dataset),
                                    round_index=round_index,
                                    configuration_index=configuration.index,
                                ),
                            )
                        )
    return tuple(schedule)


def _scheduled_run_id(pair: ScheduledPair, role: str) -> str:
    return f"{pair.comparison_id}--{role}"


def _scheduled_run_identity(
    pair: ScheduledPair,
    role: str,
    *,
    position: int,
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    implementations: Mapping[str, Implementation],
) -> dict[str, object]:
    configuration = pair.configuration
    return {
        "run_id": _scheduled_run_id(pair, role),
        "run_type": pair.run_type,
        "round": pair.round_index,
        "configuration_position": pair.configuration_position,
        "execution_position": position,
        "profile": profile.name,
        "profile_sha256": profile.sha256,
        "dataset_manifest_sha256": data_manifest.sha256,
        "case_id": pair.case.id,
        "dataset_id": pair.dataset.id,
        "analysis_unit": pair.dataset.analysis_unit,
        "implementation_role": role,
        "implementation_sha256": implementations[role].sha256,
        "target_scope": configuration.target_scope,
        "target_count": configuration.target_count,
        "n_estimators": configuration.n_estimators,
        "threads": configuration.threads,
        "n_components": pair.case.n_components,
        "seed": pair.case.seed,
        "checkpoint": pair.case.checkpoint,
        "report": pair.case.report,
    }


def _scheduled_comparison_identity(
    pair: ScheduledPair,
    *,
    profile: EquivalenceProfile,
) -> dict[str, object]:
    configuration = pair.configuration
    return {
        "comparison_id": pair.comparison_id,
        "run_type": pair.run_type,
        "round": pair.round_index,
        "configuration_position": pair.configuration_position,
        "profile": profile.name,
        "case_id": pair.case.id,
        "dataset_id": pair.dataset.id,
        "target_scope": configuration.target_scope,
        "target_count": configuration.target_count,
        "n_estimators": configuration.n_estimators,
        "threads": configuration.threads,
        "n_components": pair.case.n_components,
        "reference_run_id": _scheduled_run_id(pair, "reference"),
        "candidate_run_id": _scheduled_run_id(pair, "candidate"),
    }


def _validate_journal_prefixes(
    schedule: Sequence[ScheduledPair],
    run_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    *,
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    implementations: Mapping[str, Implementation],
) -> None:
    expected_runs: list[dict[str, object]] = []
    for pair in schedule:
        for position, role in enumerate(pair.role_order, start=1):
            expected_runs.append(
                _scheduled_run_identity(
                    pair,
                    role,
                    position=position,
                    profile=profile,
                    data_manifest=data_manifest,
                    implementations=implementations,
                )
            )
    if len(run_rows) > len(expected_runs):
        raise ContractError("runs.csv contains more rows than the immutable schedule")
    for index, (row, expected) in enumerate(zip(run_rows, expected_runs, strict=False)):
        mismatches = {
            field: (row.get(field), value)
            for field, value in expected.items()
            if row.get(field) != value
        }
        if mismatches:
            raise ContractError(
                f"runs.csv is not the exact schedule prefix at row {index}: {mismatches}"
            )

    expected_comparisons = [
        _scheduled_comparison_identity(pair, profile=profile) for pair in schedule
    ]
    if len(comparison_rows) > len(expected_comparisons):
        raise ContractError("comparisons.csv contains more rows than the immutable schedule")
    for index, (row, expected) in enumerate(
        zip(comparison_rows, expected_comparisons, strict=False)
    ):
        mismatches = {
            field: (row.get(field), value)
            for field, value in expected.items()
            if row.get(field) != value
        }
        if mismatches:
            raise ContractError(
                f"comparisons.csv is not the exact schedule prefix at row {index}: {mismatches}"
            )
    if len(comparison_rows) > len(run_rows) // 2:
        raise ContractError("comparisons.csv closes pairs that runs.csv has not completed")
    runs_by_id = {str(row["run_id"]): row for row in run_rows}
    for index, row in enumerate(comparison_rows):
        reference = runs_by_id.get(str(row["reference_run_id"]))
        candidate = runs_by_id.get(str(row["candidate_run_id"]))
        if reference is None or candidate is None:
            raise ContractError(f"comparisons[{index}] references a missing run row")
        if (
            row["reference_run_attempt"] != reference["attempt"]
            or row["candidate_run_attempt"] != candidate["attempt"]
        ):
            raise ContractError(f"comparisons[{index}] run attempts do not match runs.csv")


def _comparison_details_document(comparison: OutputComparison) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for item in comparison.artifacts:
        document = asdict(item)
        for field in ("max_absolute_difference", "max_relative_difference"):
            value = document[field]
            if isinstance(value, float) and not math.isfinite(value):
                document[field] = str(value)
        artifacts.append(document)
    return {
        "schema_version": COMPARISON_DETAILS_SCHEMA_VERSION,
        "equivalent": comparison.equivalent,
        "error": comparison.error,
        "artifacts": artifacts,
        "interpretation": _COMPARISON_INTERPRETATION,
    }


def _load_comparison_details(path: Path) -> OutputComparison:
    document = _load_json_object(path, location="comparison details")
    _check_fields(
        document,
        allowed=_COMPARISON_DETAILS_FIELDS,
        required=_COMPARISON_DETAILS_FIELDS,
        location="comparison details",
    )
    schema_version = _integer(
        document["schema_version"],
        location="comparison details.schema_version",
    )
    if schema_version != COMPARISON_DETAILS_SCHEMA_VERSION:
        raise ContractError(
            "comparison details schema_version must be "
            f"{COMPARISON_DETAILS_SCHEMA_VERSION}; received {schema_version}"
        )
    if document["interpretation"] != _COMPARISON_INTERPRETATION:
        raise ContractError(f"comparison details interpretation is invalid: {path}")
    artifacts_value = document.get("artifacts")
    if not isinstance(artifacts_value, list):
        raise ContractError(f"comparison details artifacts must be a list: {path}")
    artifacts: list[ArtifactComparison] = []
    expected_fields = {field.name for field in ArtifactComparison.__dataclass_fields__.values()}
    for index, raw_artifact in enumerate(artifacts_value):
        artifact = _json_object(raw_artifact, location=f"comparison details artifacts[{index}]")
        if set(artifact) != expected_fields:
            raise ContractError(f"comparison details artifact schema mismatch: {path}")
        for field in ("max_absolute_difference", "max_relative_difference"):
            if artifact[field] in {"inf", "-inf", "nan"}:
                artifact[field] = float(str(artifact[field]))
        artifacts.append(ArtifactComparison(**artifact))
    equivalent = document.get("equivalent")
    error = document.get("error")
    if not isinstance(equivalent, bool) or not isinstance(error, str):
        raise ContractError(f"invalid comparison details header: {path}")
    result = OutputComparison(equivalent=equivalent, artifacts=tuple(artifacts), error=error)
    if result.equivalent != all(item.equivalent for item in result.artifacts) or (
        result.error and result.equivalent
    ):
        raise ContractError(f"inconsistent comparison details: {path}")
    return result


def _suite_member(root: Path, value: object, *, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} must be a non-empty relative path")
    supplied = Path(value)
    if supplied.is_absolute():
        raise ContractError(f"{location} must be relative to the suite")
    resolved = (root / supplied).resolve()
    if resolved != root and root not in resolved.parents:
        raise ContractError(f"{location} escapes the suite directory")
    return resolved


def _load_json_object(path: Path, *, location: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {location} {path}: {exc}") from exc
    return _json_object(value, location=location)


def _manifest_object(manifest: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _json_object(manifest.get(key), location=f"manifest.{key}")


def _verify_snapshotted_file(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    location: str,
) -> Path:
    path = _suite_member(root, metadata.get("snapshot_path"), location=f"{location}.snapshot_path")
    expected_hash = _string(metadata.get("sha256"), location=f"{location}.sha256")
    if not _SHA256_PATTERN.fullmatch(expected_hash):
        raise ContractError(f"{location}.sha256 must be a lowercase SHA-256 digest")
    if not path.is_file():
        raise ContractError(f"missing snapshotted file: {path}")
    observed_hash = _sha256(path)
    if observed_hash != expected_hash:
        raise ContractError(
            f"snapshot hash mismatch for {path}: expected {expected_hash}, observed {observed_hash}"
        )
    return path


def _load_resume_state(output_dir: Path) -> ResumeState:
    """Reconstruct and validate an interrupted suite exclusively from its snapshots."""

    root = output_dir.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"resume suite directory does not exist: {root}")
    manifest = _load_json_object(root / "manifest.json", location="suite manifest")
    _check_fields(
        manifest,
        allowed=_SUITE_MANIFEST_FIELDS,
        required=_SUITE_MANIFEST_REQUIRED_FIELDS,
        location="suite manifest",
    )
    schema_version = _integer(manifest["schema_version"], location="manifest.schema_version")
    if schema_version != SUITE_SCHEMA_VERSION:
        raise ContractError(
            f"suite schema_version must be {SUITE_SCHEMA_VERSION}; received {schema_version}"
        )
    _integer(
        manifest["execution_attempt"],
        location="manifest.execution_attempt",
        minimum=1,
    )
    _boolean(
        manifest["keep_successful_outputs"],
        location="manifest.keep_successful_outputs",
    )
    _boolean(manifest["verify_inputs"], location="manifest.verify_inputs")

    profile_metadata = _manifest_object(manifest, "profile")
    profile_path = _verify_snapshotted_file(
        root,
        profile_metadata,
        location="manifest.profile",
    )
    profile = load_profile(profile_path)
    if profile.sha256 != profile_metadata.get("sha256"):
        raise ContractError("loaded profile hash does not match the suite manifest")

    _verify_snapshotted_file(
        root,
        _manifest_object(manifest, "runner"),
        location="manifest.runner",
    )
    _verify_snapshotted_file(
        root,
        _manifest_object(manifest, "resource_measurement_helper"),
        location="manifest.resource_measurement_helper",
    )

    dataset_metadata = _manifest_object(manifest, "dataset_manifest")
    resolved_manifest_path = _verify_snapshotted_file(
        root,
        {
            "snapshot_path": dataset_metadata.get("snapshot_path"),
            "sha256": dataset_metadata.get("resolved_snapshot_sha256"),
        },
        location="manifest.dataset_manifest.resolved",
    )
    source_manifest_path = _verify_snapshotted_file(
        root,
        {
            "snapshot_path": dataset_metadata.get("source_snapshot_path"),
            "sha256": dataset_metadata.get("sha256"),
        },
        location="manifest.dataset_manifest.source",
    )
    resolved_manifest = load_dataset_manifest(resolved_manifest_path)
    source_bytes = source_manifest_path.read_bytes()
    data_manifest = DatasetManifest(
        description=resolved_manifest.description,
        datasets=resolved_manifest.datasets,
        source_path=source_manifest_path,
        source_bytes=source_bytes,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    selected_value = dataset_metadata.get("selected_dataset_ids")
    selected_ids = _string_tuple(
        selected_value,
        location="manifest.dataset_manifest.selected_dataset_ids",
        allow_empty=False,
    )
    selected = _select_datasets(profile, data_manifest, selected_ids)

    raw_implementations = _manifest_object(manifest, "implementations")
    if set(raw_implementations) != {"reference", "candidate"}:
        raise ContractError("manifest.implementations must contain reference and candidate")
    implementations: dict[str, Implementation] = {}
    for role in ("reference", "candidate"):
        metadata = _json_object(
            raw_implementations[role], location=f"manifest.implementations.{role}"
        )
        snapshot_parent = _suite_member(
            root,
            metadata.get("snapshot_path"),
            location=f"manifest.implementations.{role}.snapshot_path",
        )
        package_path = _resolve_package_source(snapshot_parent)
        observed_hash = _package_digest(package_path)
        expected_hash = _string(
            metadata.get("sha256"), location=f"manifest.implementations.{role}.sha256"
        )
        if observed_hash != expected_hash:
            raise ContractError(
                f"implementation snapshot hash mismatch for {role}: expected "
                f"{expected_hash}, observed {observed_hash}"
            )
        implementations[role] = Implementation(
            role=role,
            source_path=package_path,
            snapshot_parent=snapshot_parent,
            sha256=observed_hash,
        )

    identity = _manifest_object(manifest, "implementation_identity")
    identical = implementations["reference"].sha256 == implementations["candidate"].sha256
    if identity.get("identical") is not identical:
        raise ContractError("manifest implementation identity does not match its snapshots")
    if identical and not (
        identity.get("profile_allows_identical") is True
        or identity.get("explicitly_allowed") is True
    ):
        raise ContractError("identical implementation snapshots were not authorized by the suite")

    target_lists = _load_target_manifest(root, manifest, profile, data_manifest, selected)

    artifacts = _manifest_object(manifest, "artifacts")
    runs_path = _suite_member(root, artifacts.get("runs"), location="manifest.artifacts.runs")
    comparisons_path = _suite_member(
        root,
        artifacts.get("comparisons"),
        location="manifest.artifacts.comparisons",
    )
    run_rows = [
        _restore_run_row_types(row, location=f"runs[{index}]")
        for index, row in enumerate(_read_csv(runs_path, RUN_FIELDS))
    ]
    comparison_rows = [
        _restore_comparison_row_types(row, location=f"comparisons[{index}]")
        for index, row in enumerate(_read_csv(comparisons_path, COMPARISON_FIELDS))
    ]
    run_keys = [row["run_id"] for row in run_rows]
    if len(run_keys) != len(set(run_keys)):
        raise ContractError("runs.csv contains duplicate run_id rows")
    comparison_ids = [row["comparison_id"] for row in comparison_rows]
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ContractError("comparisons.csv contains duplicate comparison ids")
    selected_set = set(selected_ids)
    for index, row in enumerate(run_rows):
        role = row.get("implementation_role")
        if role not in implementations:
            raise ContractError(f"runs[{index}] has an unknown implementation role")
        expected = {
            "profile": profile.name,
            "profile_sha256": profile.sha256,
            "dataset_manifest_sha256": data_manifest.sha256,
            "implementation_sha256": implementations[str(role)].sha256,
        }
        mismatches = {
            key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value
        }
        if row.get("dataset_id") not in selected_set:
            mismatches["dataset_id"] = (row.get("dataset_id"), sorted(selected_set))
        if mismatches:
            raise ContractError(f"runs[{index}] does not belong to this suite: {mismatches}")
    for index, row in enumerate(comparison_rows):
        if row.get("profile") != profile.name or row.get("dataset_id") not in selected_set:
            raise ContractError(f"comparisons[{index}] does not belong to this suite")

    return ResumeState(
        profile=profile,
        data_manifest=data_manifest,
        selected_dataset_ids=selected_ids,
        implementations=implementations,
        manifest=manifest,
        run_rows=run_rows,
        comparison_rows=comparison_rows,
        target_lists=target_lists,
    )


def run_benchmark(
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    *,
    reference_source: Path,
    candidate_source: Path,
    output_dir: Path,
    requested_datasets: Sequence[str] = (),
    keep_outputs: bool = False,
    verify_inputs: bool = True,
    allow_identical_implementations: bool = False,
) -> int:
    """Execute a complete interruption-auditable reference/candidate suite."""

    selected = _select_datasets(profile, data_manifest, requested_datasets)
    reference_source = _resolve_package_source(reference_source)
    candidate_source = _resolve_package_source(candidate_source)
    source_hashes = {
        "reference": _package_digest(reference_source),
        "candidate": _package_digest(candidate_source),
    }
    identical_implementations = _validate_implementation_identity(
        profile,
        source_hashes,
        allow_identical_implementations=allow_identical_implementations,
    )
    scheduled_dataset_ids: set[str] = set()
    for case in profile.cases:
        for dataset in _case_datasets(case, selected):
            scheduled_dataset_ids.add(dataset.id)
            for target_budget in case.target_counts:
                if target_budget != "all" and target_budget > dataset.dimensions.genes:
                    raise ContractError(
                        f"case {case.id!r} target count {target_budget} exceeds dataset "
                        f"{dataset.id!r} genes ({dataset.dimensions.genes})"
                    )
    if not scheduled_dataset_ids:
        raise ContractError("the selected datasets do not match any profile case")
    unscheduled = sorted({dataset.id for dataset in selected} - scheduled_dataset_ids)
    if unscheduled:
        raise ContractError(
            "selected datasets are not referenced by any profile case: " + ", ".join(unscheduled)
        )
    maximum_requested_threads = max(thread for case in profile.cases for thread in case.threads)
    available_threads = available_cpu_count()
    if maximum_requested_threads > available_threads:
        raise ContractError(
            f"profile requires {maximum_requested_threads} distinct threads but this host "
            f"exposes {available_threads}"
        )
    if output_dir.exists():
        raise ContractError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "implementations").mkdir()
    (output_dir / "runs").mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "commands").mkdir()
    (output_dir / "targets").mkdir()
    (output_dir / "comparison-details").mkdir()
    (output_dir / "profile.json").write_bytes(profile.source_bytes)
    (output_dir / "dataset-manifest.source.json").write_bytes(data_manifest.source_bytes)
    runner_path = Path(__file__).resolve()
    runner_snapshot_path = output_dir / runner_path.name
    runner_snapshot_path.write_bytes(runner_path.read_bytes())
    scaling_helper_path = Path(_SCALING_HELPERS.__file__).resolve()
    scaling_helper_snapshot_path = output_dir / "benchmark_scaling.py"
    scaling_helper_snapshot_path.write_bytes(scaling_helper_path.read_bytes())
    _atomic_json(
        output_dir / "dataset-manifest.json",
        _resolved_dataset_manifest_document(data_manifest),
    )
    resolved_dataset_manifest_sha256 = _sha256(output_dir / "dataset-manifest.json")
    implementations = {
        role: snapshot_implementation(role, source, output_root=output_dir)
        for role, source in (
            ("reference", reference_source),
            ("candidate", candidate_source),
        )
    }
    snapshot_hashes = {role: item.sha256 for role, item in implementations.items()}
    if snapshot_hashes != source_hashes:
        raise ContractError("an implementation source changed while its snapshot was being created")
    target_lists = _create_target_manifest(
        profile,
        data_manifest,
        selected,
        suite_root=output_dir,
    )
    target_manifest_path = output_dir / "targets-manifest.json"
    manifest_path = output_dir / "manifest.json"
    runs_path = output_dir / "runs.csv"
    comparisons_path = output_dir / "comparisons.csv"
    scaling_path = output_dir / "scaling.csv"
    manifest: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "status": "running",
        "execution_attempt": 1,
        "history": [{"attempt": 1, "event": "started", "at_utc": _utc_now()}],
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "profile": {
            "name": profile.name,
            "source_path": str(profile.source_path),
            "snapshot_path": "profile.json",
            "sha256": profile.sha256,
            "absolute_tolerance": profile.absolute_tolerance,
            "relative_tolerance": profile.relative_tolerance,
            "scientific_parameters": asdict(profile.scientific_parameters),
        },
        "runner": {
            "path": str(runner_path),
            "snapshot_path": runner_snapshot_path.name,
            "sha256": _sha256(runner_snapshot_path),
        },
        "resource_measurement_helper": {
            "path": str(scaling_helper_path),
            "snapshot_path": scaling_helper_snapshot_path.name,
            "sha256": _sha256(scaling_helper_snapshot_path),
        },
        "target_manifest": {
            "snapshot_path": target_manifest_path.name,
            "sha256": _sha256(target_manifest_path),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count() or 1,
            "available_cpus": available_threads,
        },
        "dataset_manifest": {
            "source_path": str(data_manifest.source_path),
            "snapshot_path": "dataset-manifest.json",
            "source_snapshot_path": "dataset-manifest.source.json",
            "sha256": data_manifest.sha256,
            "resolved_snapshot_sha256": resolved_dataset_manifest_sha256,
            "selected_dataset_ids": [dataset.id for dataset in selected],
        },
        "implementations": {
            role: {
                "source_path": str(implementation.source_path),
                "snapshot_path": str(implementation.snapshot_parent.relative_to(output_dir)),
                "sha256": implementation.sha256,
            }
            for role, implementation in implementations.items()
        },
        "implementation_identity": {
            "identical": identical_implementations,
            "profile_allows_identical": profile.allow_identical_implementations,
            "explicitly_allowed": allow_identical_implementations,
        },
        "artifacts": {
            "runs": "runs.csv",
            "comparisons": "comparisons.csv",
            "scaling": "scaling.csv",
            "comparison_details": "comparison-details",
            "commands": "commands",
        },
        "keep_successful_outputs": keep_outputs,
        "verify_inputs": verify_inputs,
        "schedule": {
            "configuration_order": "deterministic-shuffled-rotations-within-case-dataset",
            "implementation_order": "deterministic-paired-counterbalance",
        },
        "resource_measurement": {
            "rss": "sampled process-tree RSS lower bound",
            "cpu": "sampled cumulative process-tree CPU lower bound",
            "storage": "sampled transient and exact final file storage",
        },
        "limitations": list(profile.limitations),
        "runs_completed": 0,
        "comparisons_completed": 0,
        "failures": [],
    }
    run_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    _write_csv(runs_path, RUN_FIELDS, run_rows)
    _write_csv(comparisons_path, COMPARISON_FIELDS, comparison_rows)
    _write_csv(scaling_path, SCALING_FIELDS, [])
    _atomic_json(manifest_path, manifest)

    state = ResumeState(
        profile=profile,
        data_manifest=data_manifest,
        selected_dataset_ids=tuple(dataset.id for dataset in selected),
        implementations=implementations,
        manifest=manifest,
        run_rows=[],
        comparison_rows=[],
        target_lists=target_lists,
    )
    with _suite_lock(output_dir):
        result = _execute_pending_suite(
            state,
            output_dir=output_dir,
            begin_resume_attempt=False,
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(output_dir),
                "runs": str(runs_path),
                "comparisons": str(comparisons_path),
                "scaling": str(scaling_path),
            },
            indent=2,
        )
    )
    return result


def _checkpoint_directory(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.checkpoint"


@contextmanager
def _suite_lock(output_dir: Path) -> Iterator[None]:
    """Hold an exclusive non-blocking lock for every stateful resume operation."""

    lock_path = output_dir / ".suite.lock"
    try:
        stream = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot open suite lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(
                f"suite is already locked by another process: {output_dir}"
            ) from exc
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _manifest_failures(
    run_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    failures = [
        str(row["comparison_id"])
        for row in comparison_rows
        if row["status"] != "compared" or row["equivalent"] is not True
    ]
    compared_run_ids = {
        str(row[field])
        for row in comparison_rows
        for field in ("reference_run_id", "candidate_run_id")
    }
    failures.extend(
        str(row["run_id"])
        for row in run_rows
        if row["status"] != "success" and row["run_id"] not in compared_run_ids
    )
    return failures


def _persist_suite_state(
    output_dir: Path,
    manifest: dict[str, Any],
    run_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    datasets: Mapping[str, LocalDataset],
) -> None:
    """Persist journals/derivatives in dependency order and rebuild manifest counters."""

    _write_csv(output_dir / "runs.csv", RUN_FIELDS, run_rows)
    _write_csv(output_dir / "comparisons.csv", COMPARISON_FIELDS, comparison_rows)
    _write_csv(output_dir / "scaling.csv", SCALING_FIELDS, _scaling_rows(comparison_rows, datasets))
    manifest["runs_completed"] = len(run_rows)
    manifest["comparisons_completed"] = len(comparison_rows)
    manifest["failures"] = _manifest_failures(run_rows, comparison_rows)
    _atomic_json(output_dir / "manifest.json", manifest)


def _execute_pending_suite(
    state: ResumeState,
    *,
    output_dir: Path,
    begin_resume_attempt: bool,
) -> int:
    """Execute the unclosed suffix of one initialized, verified suite."""

    profile = state.profile
    data_manifest = state.data_manifest
    selected = _select_datasets(profile, data_manifest, state.selected_dataset_ids)
    datasets_by_id = {dataset.id: dataset for dataset in selected}
    implementations = state.implementations
    manifest = state.manifest
    run_rows = state.run_rows
    comparison_rows = state.comparison_rows
    schedule = _build_schedule(profile, selected, state.target_lists)
    _validate_journal_prefixes(
        schedule,
        run_rows,
        comparison_rows,
        profile=profile,
        data_manifest=data_manifest,
        implementations=implementations,
    )

    if manifest.get("status") == "complete":
        if len(comparison_rows) != len(schedule) or _manifest_failures(run_rows, comparison_rows):
            raise ContractError("suite claims successful completion before the schedule is closed")
        return 0
    if manifest.get("status") == "complete_with_failures":
        if not _manifest_failures(run_rows, comparison_rows):
            raise ContractError("suite claims failures but its authoritative journals contain none")
        return 1

    history = manifest.get("history")
    if not isinstance(history, list):
        raise ContractError("manifest.history must be a list")
    current_attempt = _integer(
        manifest["execution_attempt"],
        location="manifest.execution_attempt",
        minimum=1,
    )
    if begin_resume_attempt:
        attempt = current_attempt + 1
        history.append({"attempt": attempt, "event": "started", "at_utc": _utc_now()})
        manifest["execution_attempt"] = attempt
        manifest["status"] = "running"
        manifest["completed_at_utc"] = None
        manifest.pop("error", None)
        _persist_suite_state(output_dir, manifest, run_rows, comparison_rows, datasets_by_id)
    else:
        attempt = current_attempt

    comparison_index = len(comparison_rows)
    existing_run_rows = {str(row["run_id"]): row for row in run_rows}
    if any(row["status"] != "success" for row in run_rows):
        manifest["status"] = "complete_with_failures"
        manifest["completed_at_utc"] = _utc_now()
        history.append({"attempt": attempt, "event": "terminal-failure", "at_utc": _utc_now()})
        _persist_suite_state(output_dir, manifest, run_rows, comparison_rows, datasets_by_id)
        return 1

    try:
        if manifest["verify_inputs"]:
            for dataset in selected:
                verify_dataset(dataset)
        full_estimate = _estimated_output_bytes(profile, selected)
        fixed_margin = 1024 * 1024 * 1024
        total_runs = 2 * len(schedule)
        pending_runs = max(0, total_runs - len(run_rows))
        free_bytes = shutil.disk_usage(output_dir).free
        if begin_resume_attempt:
            pending_estimate = fixed_margin + math.ceil(
                max(0, full_estimate - fixed_margin) * pending_runs / max(1, total_runs)
            )
            manifest["resume_disk_preflight"] = {
                "attempt": attempt,
                "free_bytes": free_bytes,
                "conservative_pending_estimated_bytes": pending_estimate,
            }
        else:
            pending_estimate = full_estimate
            manifest["disk_preflight"] = {
                "free_bytes": free_bytes,
                "conservative_estimated_bytes": pending_estimate,
            }
        if free_bytes < pending_estimate:
            raise ContractError(
                f"estimated pending suite storage {pending_estimate} exceeds available "
                f"{free_bytes} bytes"
            )
        _atomic_json(output_dir / "manifest.json", manifest)

        for pair_index, pair in enumerate(schedule):
            if pair_index < comparison_index:
                if not manifest["keep_successful_outputs"]:
                    for role in ("reference", "candidate"):
                        output_path = output_dir / "runs" / _scheduled_run_id(pair, role)
                        if output_path.is_dir():
                            shutil.rmtree(output_path)
                continue

            configuration = pair.configuration
            input_paths = [
                pair.dataset.expression.path,
                pair.dataset.groups.path,
                pair.dataset.tf_list.path,
            ]
            if pair.dataset.centroid_weights is not None:
                input_paths.append(pair.dataset.centroid_weights.path)
            if configuration.target_list is not None:
                input_paths.append(configuration.target_list.path)
            input_storage = measure_storage(input_paths)
            paired_rows: dict[str, dict[str, object]] = {}
            paired_outputs: dict[str, Path] = {}
            for position, role in enumerate(pair.role_order, start=1):
                run_id = _scheduled_run_id(pair, role)
                output_path = output_dir / "runs" / run_id
                paired_outputs[role] = output_path
                if run_id in existing_run_rows:
                    paired_rows[role] = existing_run_rows[run_id]
                    continue
                if output_path.exists():
                    raise ContractError(
                        f"published output exists without its authoritative run row: {output_path}"
                    )
                checkpoint_path = _checkpoint_directory(output_path)
                resumed = pair.case.checkpoint and checkpoint_path.is_dir()
                if checkpoint_path.exists() and not resumed:
                    raise ContractError(
                        f"unexpected checkpoint state for {run_id}: {checkpoint_path}"
                    )
                command = build_infer_command(
                    implementations[role],
                    pair.dataset,
                    target_list=(
                        None
                        if configuration.target_list is None
                        else configuration.target_list.path
                    ),
                    output_dir=output_path,
                    scientific_parameters=profile.scientific_parameters,
                    n_estimators=configuration.n_estimators,
                    n_components=pair.case.n_components,
                    threads=configuration.threads,
                    seed=pair.case.seed,
                    checkpoint=pair.case.checkpoint,
                    resume=resumed,
                    report=pair.case.report,
                )
                _atomic_json(
                    output_dir / "commands" / f"{run_id}.attempt-{attempt}.json",
                    {
                        "schema_version": COMMAND_SCHEMA_VERSION,
                        "run_id": run_id,
                        "attempt": attempt,
                        "implementation_role": role,
                        "implementation_sha256": implementations[role].sha256,
                        "resumed_from_checkpoint": resumed,
                        "configuration_position": pair.configuration_position,
                        "target_scope": configuration.target_scope,
                        "target_count": configuration.target_count,
                        "command": command,
                    },
                )
                print(f"Running {run_id}", file=sys.stderr)
                measurement = measure_command(
                    command,
                    sample_interval_seconds=pair.case.resource_sample_ms / 1000,
                    timeout_seconds=pair.case.run_timeout_seconds,
                    show_output=False,
                    stdout_log_path=(
                        output_dir / "logs" / f"{run_id}.attempt-{attempt}.stdout.log"
                    ),
                    stderr_log_path=(
                        output_dir / "logs" / f"{run_id}.attempt-{attempt}.stderr.log"
                    ),
                    run_storage_probe=lambda path=output_path: measure_run_storage(path),
                )
                row = _run_row(
                    run_id=run_id,
                    attempt=attempt,
                    run_type=pair.run_type,
                    round_index=pair.round_index,
                    configuration_position=pair.configuration_position,
                    execution_position=position,
                    profile=profile,
                    data_manifest=data_manifest,
                    case=pair.case,
                    dataset=pair.dataset,
                    implementation=implementations[role],
                    target_scope=configuration.target_scope,
                    target_count=configuration.target_count,
                    target_list=configuration.target_list,
                    n_estimators=configuration.n_estimators,
                    threads=configuration.threads,
                    resumed_from_checkpoint=resumed,
                    measurement=measurement,
                    output_dir=output_path,
                    input_storage=input_storage,
                )
                run_rows.append(row)
                existing_run_rows[run_id] = row
                paired_rows[role] = row
                _persist_suite_state(
                    output_dir, manifest, run_rows, comparison_rows, datasets_by_id
                )
                if row["status"] != "success":
                    manifest["status"] = "complete_with_failures"
                    manifest["completed_at_utc"] = _utc_now()
                    history.append(
                        {"attempt": attempt, "event": "terminal-failure", "at_utc": _utc_now()}
                    )
                    _persist_suite_state(
                        output_dir, manifest, run_rows, comparison_rows, datasets_by_id
                    )
                    return 1

            if set(paired_rows) != {"reference", "candidate"}:
                raise ContractError(f"incomplete pair state for {pair.comparison_id}")
            details_path = output_dir / "comparison-details" / f"{pair.comparison_id}.json"
            if details_path.is_file():
                output_comparison = _load_comparison_details(details_path)
            else:
                if not all(path.is_dir() for path in paired_outputs.values()):
                    raise ContractError(
                        f"successful run rows lack outputs needed to close {pair.comparison_id}"
                    )
                output_comparison = compare_outputs(
                    paired_outputs["reference"],
                    paired_outputs["candidate"],
                    absolute_tolerance=profile.absolute_tolerance,
                    relative_tolerance=profile.relative_tolerance,
                )
                _atomic_json(details_path, _comparison_details_document(output_comparison))
            comparison_row = _comparison_row(
                comparison_id=pair.comparison_id,
                attempt=attempt,
                run_type=pair.run_type,
                round_index=pair.round_index,
                configuration_position=pair.configuration_position,
                profile=profile,
                case=pair.case,
                dataset=pair.dataset,
                target_scope=configuration.target_scope,
                target_count=configuration.target_count,
                n_estimators=configuration.n_estimators,
                threads=configuration.threads,
                reference_row=paired_rows["reference"],
                candidate_row=paired_rows["candidate"],
                comparison=output_comparison,
                details_path=details_path.relative_to(output_dir),
            )
            comparison_rows.append(comparison_row)
            comparison_index += 1
            _persist_suite_state(output_dir, manifest, run_rows, comparison_rows, datasets_by_id)
            if (
                comparison_row["status"] == "compared"
                and comparison_row["equivalent"] is True
                and not manifest["keep_successful_outputs"]
            ):
                for output_path in paired_outputs.values():
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_at_utc"] = _utc_now()
        history.append({"attempt": attempt, "event": "interrupted", "at_utc": _utc_now()})
        _persist_suite_state(output_dir, manifest, run_rows, comparison_rows, datasets_by_id)
        raise
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        history.append({"attempt": attempt, "event": "failed", "at_utc": _utc_now()})
        _persist_suite_state(output_dir, manifest, run_rows, comparison_rows, datasets_by_id)
        raise

    failures = _manifest_failures(run_rows, comparison_rows)
    manifest["status"] = "complete_with_failures" if failures else "complete"
    manifest["completed_at_utc"] = _utc_now()
    history.append({"attempt": attempt, "event": "completed", "at_utc": _utc_now()})
    _persist_suite_state(output_dir, manifest, run_rows, comparison_rows, datasets_by_id)
    return int(bool(failures))


def resume_benchmark(output_dir: Path) -> int:
    """Resume an interrupted suite from verified in-suite snapshots and journals."""

    root = output_dir.expanduser().resolve()
    with _suite_lock(root):
        state = _load_resume_state(root)
        return _execute_pending_suite(
            state,
            output_dir=root,
            begin_resume_attempt=True,
        )


def _delegate_resume_to_snapshot(output_dir: Path) -> int | None:
    """Delegate CLI resume to the exact runner copied into the suite."""

    root = output_dir.expanduser().resolve()
    manifest = _load_json_object(root / "manifest.json", location="suite manifest")
    runner = _verify_snapshotted_file(
        root,
        _manifest_object(manifest, "runner"),
        location="manifest.runner",
    )
    if runner.resolve() == Path(__file__).resolve():
        return None
    completed = subprocess.run(
        [sys.executable, str(runner), "resume", "--output-dir", str(root)],
        check=False,
    )
    return completed.returncode


def _parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/prepare_manifest.json")
    safe_label = _safe_id(label)
    return safe_label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest",
        help="create a local hashed dataset manifest from SPATHI prepare manifests",
    )
    manifest.add_argument(
        "--prepare-manifest",
        action="append",
        type=_parse_labelled_path,
        required=True,
        metavar="LABEL=PATH",
    )
    manifest.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run reference/candidate equivalence benchmarks")
    run.add_argument("--profile", default="cll-equivalence-smoke")
    run.add_argument("--dataset-manifest", type=Path, required=True)
    run.add_argument("--reference-source", type=Path, required=True)
    run.add_argument("--candidate-source", type=Path, default=REPOSITORY_ROOT)
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="local manifest dataset id to include; repeat, or omit for all profile datasets",
    )
    run.add_argument(
        "--keep-outputs",
        action="store_true",
        help="retain successful reference/candidate run directories after comparison",
    )
    run.add_argument(
        "--allow-identical-implementations",
        action="store_true",
        help=(
            "allow a deliberate self-comparison when the profile normally requires distinct "
            "implementation hashes"
        ),
    )
    run.add_argument(
        "--skip-input-verification",
        action="store_true",
        help="skip expensive file rehashing; recorded in the suite manifest",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate contracts and print the selected execution matrix without running",
    )
    resume = subparsers.add_parser(
        "resume",
        help="resume an interrupted suite from its verified immutable snapshots",
    )
    resume.add_argument("--output-dir", type=Path, required=True)
    return parser


def _dry_run_document(
    profile: EquivalenceProfile,
    data_manifest: DatasetManifest,
    datasets: Sequence[LocalDataset],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for case in profile.cases:
        for dataset in _case_datasets(case, datasets):
            for target_budget in case.target_counts:
                target_count = _effective_target_count(target_budget, dataset)
                if target_count > dataset.dimensions.genes:
                    raise ContractError(
                        f"case {case.id!r} target count {target_count} exceeds dataset "
                        f"{dataset.id!r} genes ({dataset.dimensions.genes})"
                    )
                for estimators in case.n_estimators:
                    for threads in case.threads:
                        entries.append(
                            {
                                "case_id": case.id,
                                "dataset_id": dataset.id,
                                "target_scope": _target_scope(target_budget),
                                "target_count": target_count,
                                "n_estimators": estimators,
                                "threads": threads,
                                "n_components": case.n_components,
                                "warmup_pairs": case.warmups,
                                "measurement_pairs": case.repeats,
                                "run_timeout_seconds": case.run_timeout_seconds,
                                "processes": 2 * (case.warmups + case.repeats),
                            }
                        )
    return {
        "mode": "dry-run",
        "profile": profile.name,
        "profile_sha256": profile.sha256,
        "dataset_manifest_sha256": data_manifest.sha256,
        "selected_dataset_ids": [dataset.id for dataset in datasets],
        "scientific_parameters": asdict(profile.scientific_parameters),
        "comparison": {
            "absolute_tolerance": profile.absolute_tolerance,
            "relative_tolerance": profile.relative_tolerance,
            "meaning": "computational equivalence, not biological accuracy",
        },
        "matrix": entries,
        "total_processes": sum(int(entry["processes"]) for entry in entries),
        "sequential_timeout_upper_bound_seconds": sum(
            int(entry["processes"]) * float(entry["run_timeout_seconds"]) for entry in entries
        ),
        "conservative_estimated_bytes": _estimated_output_bytes(profile, datasets),
        "maximum_child_timeout_seconds": max(case.run_timeout_seconds for case in profile.cases),
        "schedule": {
            "configuration_order": "deterministic-shuffled-rotations-within-case-dataset",
            "implementation_order": "deterministic-paired-counterbalance",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            output_path = args.output.expanduser().resolve()
            if output_path.exists():
                parser.error(f"--output already exists: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            document = create_dataset_manifest(
                args.prepare_manifest,
                output_path=output_path,
            )
            _atomic_json(output_path, document)
            validated = load_dataset_manifest(output_path)
            print(
                json.dumps(
                    {
                        "status": "created",
                        "path": str(output_path),
                        "sha256": validated.sha256,
                        "dataset_ids": [dataset.id for dataset in validated.datasets],
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "resume":
            delegated = _delegate_resume_to_snapshot(args.output_dir)
            if delegated is not None:
                return delegated
            return resume_benchmark(args.output_dir)

        profile = load_profile(args.profile)
        data_manifest = load_dataset_manifest(args.dataset_manifest)
        selected = _select_datasets(profile, data_manifest, args.dataset)
        reference_source = _resolve_package_source(args.reference_source)
        candidate_source = _resolve_package_source(args.candidate_source)
        if args.dry_run:
            implementation_hashes = {
                "reference": _package_digest(reference_source),
                "candidate": _package_digest(candidate_source),
            }
            identical = _validate_implementation_identity(
                profile,
                implementation_hashes,
                allow_identical_implementations=args.allow_identical_implementations,
            )
            document = _dry_run_document(profile, data_manifest, selected)
            document["implementations"] = {
                "reference_source": str(reference_source),
                "candidate_source": str(candidate_source),
                "reference_sha256": implementation_hashes["reference"],
                "candidate_sha256": implementation_hashes["candidate"],
                "identical": identical,
            }
            print(json.dumps(document, indent=2))
            return 0
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else (
                BENCHMARK_DIR / "results" / "equivalence" / f"{profile.name}-{timestamp}"
            ).resolve()
        )
        return run_benchmark(
            profile,
            data_manifest,
            reference_source=reference_source,
            candidate_source=candidate_source,
            output_dir=output_dir,
            requested_datasets=args.dataset,
            keep_outputs=args.keep_outputs,
            verify_inputs=not args.skip_input_verification,
            allow_identical_implementations=args.allow_identical_implementations,
        )
    except ContractError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
