"""Private scientific workflow orchestrator used by :mod:`spathi.core`."""

from __future__ import annotations

import logging
import platform
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
from tempfile import TemporaryFile
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast
from weakref import finalize

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

from spathi._version import __version__
from spathi.centroids import compute_centroids
from spathi.checkpoint import ModelCheckpoint
from spathi.config import SpathiConfig
from spathi.diagnostics import compute_weight_diagnostics
from spathi.distances import (
    DEFAULT_WORKING_MEMORY_MIB,
    compute_cell_to_centroid_distances,
    compute_centroid_distances,
)
from spathi.inference import (
    FATAL_MODEL_STATUSES,
    TRAINED_MODEL_STATUSES,
    ModelResult,
    PreparedInference,
    prepare_inference,
)
from spathi.io import InputData, InputFingerprint
from spathi.kernels import resolve_bandwidth_for_mode
from spathi.outputs import (
    IncrementalRunWriter,
    write_json,
    write_tsv,
    write_tsv_gzip,
    write_tsv_records,
)
from spathi.parallel import (
    PersistentTaskExecutor,
    available_cpu_count,
    resolve_thread_budget,
)
from spathi.progress import (
    ProgressCallback,
    ProgressPhase,
    SpathiProgressEvent,
    emit_progress,
)
from spathi.representation import RepresentationResult, compute_distance_representation
from spathi.resources import (
    MemoryPlan,
    available_memory_bytes,
    estimate_model_memory_bytes,
    plan_model_memory,
)
from spathi.weighting import (
    compute_weights,
    iter_group_affinity_records,
    prepare_weighting_context,
)

if TYPE_CHECKING:
    from spathi._report import InteractiveReportBuilder, ReportEmbedding

LOGGER = logging.getLogger(__name__)
_DISTANCE_MEMMAP_THRESHOLD_BYTES = 512 * 1024**2
_TARGET_BATCH_MIN_MODELS = 32
_TARGET_BATCH_MAX_MODELS = 256
_CENTROID_COLUMNS = ("group", "dimension", "centroid")
_GROUP_DISTANCE_COLUMNS = ("target_group", "source_group", "centroid_distance")
_GROUP_AFFINITY_COLUMNS = (
    "target_group",
    "source_group",
    "centroid_distance",
    "base_affinity",
    "group_size_factor",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowSummary:
    """Counts and warnings produced by one completed private workflow."""

    n_edges: int
    total_models: int
    trained_models: int
    skipped_target_records: int
    failed_models: int
    resumed_models: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _BatchMemoryPlan:
    """Joint batch-size and model-concurrency plan for future allocations."""

    desired_group_batch_size: int
    desired_target_batch_size: int
    group_batch_size: int
    target_batch_size: int
    active_groups_per_inference_batch: int
    models_per_inference_batch: int
    concurrent_fits: int
    reserved_bytes: int
    model_plan: MemoryPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class _CellDistanceMemoryPlan:
    """Bound the next cell-to-centroid allocation against current headroom."""

    storage: str
    expected_output_bytes: int
    working_memory_mib: float | None
    working_memory_bytes: int
    available_bytes: int | None
    usable_bytes: int | None
    storage_reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _CentroidDistanceMemoryPlan:
    """Bound centroid and group-distance allocations before materializing them."""

    estimated_persistent_bytes: int
    working_memory_mib: float
    working_memory_bytes: int
    available_bytes: int | None
    usable_bytes: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _RepresentationSummary:
    """Scalar representation metadata retained after its numeric array is released."""

    distance_space: str
    standardization: str
    pca_svd_solver: str
    pca_svd_solver_resolution: str | None
    effective_n_components: int | None
    maximum_informative_n_components: int | None
    explained_variance_ratio: tuple[float, ...] | None
    pca_degenerate: bool
    pca_degeneracy_reason: str | None
    displayed_components: tuple[str, ...]

    @classmethod
    def from_result(cls, representation: RepresentationResult) -> _RepresentationSummary:
        displayed_components = (
            representation.dimension_names[: min(3, representation.values.shape[1])]
            if representation.distance_space == "pca"
            else ()
        )
        return cls(
            distance_space=representation.distance_space,
            standardization=representation.standardization,
            pca_svd_solver=representation.pca_svd_solver,
            pca_svd_solver_resolution=representation.pca_svd_solver_resolution,
            effective_n_components=representation.effective_n_components,
            maximum_informative_n_components=representation.maximum_informative_n_components,
            explained_variance_ratio=representation.explained_variance_ratio,
            pca_degenerate=representation.pca_degenerate,
            pca_degeneracy_reason=representation.pca_degeneracy_reason,
            displayed_components=displayed_components,
        )


@dataclass(slots=True)
class RunInputs:
    """Own validated inputs and relinquish the full expression table on demand."""

    expression: pd.DataFrame | None
    transcription_factors: tuple[str, ...]
    targets: tuple[str, ...]
    groups: pd.Series
    input_fingerprints: Mapping[str, InputFingerprint]

    @classmethod
    def from_input_data(cls, inputs: InputData) -> RunInputs:
        return cls(
            expression=inputs.expression,
            transcription_factors=inputs.transcription_factors,
            targets=inputs.targets,
            groups=inputs.groups,
            input_fingerprints=inputs.input_fingerprints,
        )

    def take_expression(self) -> pd.DataFrame:
        expression = self.expression
        if expression is None:
            raise RuntimeError("validated expression has already been consumed")
        self.expression = None
        return expression


def scientific_dependency_versions() -> dict[str, str]:
    """Return versions capable of changing scientific checkpoint results."""

    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "joblib": "joblib",
        "threadpoolctl": "threadpoolctl",
    }
    resolved: dict[str, str] = {"python": platform.python_version(), "spathi": __version__}
    for label, distribution in packages.items():
        try:
            resolved[label] = version(distribution)
        except PackageNotFoundError:
            resolved[label] = "unknown"
    return resolved


def dependency_versions(*, include_report: bool) -> dict[str, str]:
    """Return provenance for runtime components exercised by the current run."""

    resolved = scientific_dependency_versions()
    if include_report:
        try:
            resolved["plotly"] = version("plotly")
        except PackageNotFoundError:
            resolved["plotly"] = "unknown"
    return resolved


def _report_run_parameters(config: SpathiConfig) -> dict[str, Any]:
    """Return report-safe settings without embedding local input or output paths."""

    path_fields = {"expression", "tf_list", "groups", "target_list", "output_dir"}
    parameters = {key: value for key, value in config.to_dict().items() if key not in path_fields}
    parameters["target_selection"] = (
        "all-expression-genes" if config.target_list is None else "explicit-list"
    )
    return parameters


def _plan_cell_distance_memory(
    *,
    n_cells: int,
    n_groups: int,
    n_dimensions: int,
    compute_distances: bool,
    distance_metric: str,
) -> _CellDistanceMemoryPlan:
    """Choose heap or disk-backed distances from live system/cgroup headroom.

    The fixed size threshold remains useful for predictable disk-backed storage,
    but it is not a memory-safety boundary: a smaller matrix can still exceed the
    memory currently available to this process. The plan therefore reserves 30%
    of detected headroom and also adapts the transient pairwise-distance chunk.
    """

    expected_output_bytes = n_cells * n_groups * np.dtype(np.float64).itemsize
    if not compute_distances:
        return _CellDistanceMemoryPlan(
            storage="not-computed",
            expected_output_bytes=0,
            working_memory_mib=None,
            working_memory_bytes=0,
            available_bytes=None,
            usable_bytes=None,
            storage_reason="weight-mode-does-not-require-cell-distances",
        )

    available_bytes = available_memory_bytes()
    usable_bytes = None if available_bytes is None else int(available_bytes * 0.7)
    # The numerically stable Euclidean path retains one representation-difference
    # row and one output row while the latter is validated. The 16-byte factor is
    # a conservative allowance for validation temporaries.
    minimum_working_bytes = max(
        1,
        n_dimensions * 16,
        n_dimensions * np.dtype(np.float64).itemsize + n_groups * 24,
    )
    if distance_metric == "cosine":
        normalized_centroid_bytes = n_groups * n_dimensions * np.dtype(np.float64).itemsize
        minimum_working_bytes = normalized_centroid_bytes + max(
            n_dimensions * 24,
            n_dimensions * np.dtype(np.float64).itemsize + n_groups * 24,
        )
    size_requires_mapping = expected_output_bytes >= _DISTANCE_MEMMAP_THRESHOLD_BYTES
    memory_requires_mapping = (
        usable_bytes is not None and expected_output_bytes + minimum_working_bytes > usable_bytes
    )
    if size_requires_mapping:
        storage = "temporary-memory-map"
        storage_reason = "size-threshold"
    elif memory_requires_mapping:
        storage = "temporary-memory-map"
        storage_reason = "available-memory"
    else:
        storage = "memory"
        storage_reason = "fits-memory-plan"

    working_budget = usable_bytes
    if working_budget is not None and storage == "memory":
        working_budget -= expected_output_bytes
    if working_budget is not None and working_budget < minimum_working_bytes:
        raise MemoryError(
            "Insufficient available memory for one cell-to-centroid distance chunk: "
            f"usable={usable_bytes}, output={expected_output_bytes}, "
            f"minimum_chunk={minimum_working_bytes} bytes"
        )

    default_working_bytes = int(DEFAULT_WORKING_MEMORY_MIB * 1024**2)
    working_memory_bytes = (
        default_working_bytes
        if working_budget is None
        else min(default_working_bytes, working_budget)
    )
    # Guard against sub-byte rounding when converting the exact plan to MiB.
    working_memory_bytes = max(minimum_working_bytes, working_memory_bytes)
    return _CellDistanceMemoryPlan(
        storage=storage,
        expected_output_bytes=expected_output_bytes,
        working_memory_mib=working_memory_bytes / 1024**2,
        working_memory_bytes=working_memory_bytes,
        available_bytes=available_bytes,
        usable_bytes=usable_bytes,
        storage_reason=storage_reason,
    )


def _plan_centroid_distance_memory(
    *,
    n_groups: int,
    n_dimensions: int,
) -> _CentroidDistanceMemoryPlan:
    """Reserve centroid, group-distance, and bounded validation storage."""

    float_bytes = np.dtype(np.float64).itemsize
    centroid_bytes = n_groups * n_dimensions * float_bytes
    group_distance_bytes = n_groups * n_groups * float_bytes
    # Cosine preprocessing may retain one normalized centroid copy alongside
    # the authoritative centroid table. Euclidean paths use less.
    estimated_persistent_bytes = 2 * centroid_bytes + group_distance_bytes
    minimum_working_bytes = max(
        1,
        n_dimensions * 16,
        n_dimensions * float_bytes + n_groups * 24,
    )
    available_bytes = available_memory_bytes()
    usable_bytes = None if available_bytes is None else int(available_bytes * 0.7)
    working_budget = None if usable_bytes is None else usable_bytes - estimated_persistent_bytes
    if working_budget is not None and working_budget < minimum_working_bytes:
        raise MemoryError(
            "Insufficient available memory for centroids and group distances: "
            f"usable={usable_bytes}, persistent={estimated_persistent_bytes}, "
            f"minimum_chunk={minimum_working_bytes} bytes"
        )
    default_working_bytes = int(DEFAULT_WORKING_MEMORY_MIB * 1024**2)
    working_memory_bytes = (
        default_working_bytes
        if working_budget is None
        else min(default_working_bytes, working_budget)
    )
    working_memory_bytes = max(minimum_working_bytes, working_memory_bytes)
    return _CentroidDistanceMemoryPlan(
        estimated_persistent_bytes=estimated_persistent_bytes,
        working_memory_mib=working_memory_bytes / 1024**2,
        working_memory_bytes=working_memory_bytes,
        available_bytes=available_bytes,
        usable_bytes=usable_bytes,
    )


def _iter_group_distance_records(
    distances: pd.DataFrame, group_ids: list[str]
) -> Iterator[dict[str, str | float]]:
    values = distances.loc[group_ids, group_ids].to_numpy(dtype=np.float64, copy=False)
    for target_index, target_group in enumerate(group_ids):
        for source_index, source_group in enumerate(group_ids):
            yield {
                "target_group": target_group,
                "source_group": source_group,
                "centroid_distance": float(values[target_index, source_index]),
            }


def _iter_centroid_records(
    centroids: pd.DataFrame, group_ids: list[str]
) -> Iterator[dict[str, str | float]]:
    """Yield a name-collision-free long representation of group centroids."""

    dimensions = tuple(map(str, centroids.columns))
    values = centroids.loc[group_ids].to_numpy(dtype=np.float64, copy=False)
    for group_index, group in enumerate(group_ids):
        for dimension_index, dimension in enumerate(dimensions):
            yield {
                "group": group,
                "dimension": dimension,
                "centroid": float(values[group_index, dimension_index]),
            }


def _close_distance_backing(mapping: Any, temporary_file: Any) -> None:
    """Close both layers of one disposable disk-backed distance allocation."""

    try:
        # The matrix has already been consumed and is never a published artifact;
        # forcing a complete synchronous flush here would add avoidable I/O.
        mapping.close()
    finally:
        temporary_file.close()


def _estimated_memory_bytes(
    *,
    expression_bytes: int,
    representation_bytes: int,
    centroids: pd.DataFrame,
    cell_distances: pd.DataFrame | None,
    group_distances: pd.DataFrame,
    tree_target_bytes: int,
    additional_tree_target_bytes: int,
    predictor_bytes: int,
    cell_distance_storage: str,
) -> dict[str, int]:
    cell_distance_logical = (
        0
        if cell_distances is None
        else int(cell_distances.shape[0] * cell_distances.shape[1] * np.dtype(np.float64).itemsize)
    )
    cell_distance_heap = cell_distance_logical if cell_distance_storage == "memory" else 0
    cell_distance_mapped = (
        cell_distance_logical if cell_distance_storage == "temporary-memory-map" else 0
    )
    estimates = {
        "validated_expression_float64": expression_bytes,
        "distance_representation_float64": representation_bytes,
        "centroids_float64": int(centroids.memory_usage(index=False, deep=False).sum()),
        "cell_centroid_distances_logical_float64": cell_distance_logical,
        "cell_centroid_distances_heap_float64": cell_distance_heap,
        "cell_centroid_distances_mapped_float64": cell_distance_mapped,
        "centroid_distances_float64": int(
            group_distances.memory_usage(index=False, deep=False).sum()
        ),
        # All-gene inference reuses the validated transpose. An explicit target
        # subset retains only those response columns in an additional contiguous
        # float64 allocation.
        "tree_targets_additional_float64": additional_tree_target_bytes,
        "tree_targets_logical_float64": tree_target_bytes,
        "tf_predictors_float32": predictor_bytes,
    }
    estimates["estimated_heap_core_array_total"] = sum(
        estimates[key]
        for key in (
            "validated_expression_float64",
            "distance_representation_float64",
            "centroids_float64",
            "cell_centroid_distances_heap_float64",
            "centroid_distances_float64",
            "tree_targets_additional_float64",
            "tf_predictors_float32",
        )
    )
    return estimates


def _plan_inference_batches(
    *,
    n_cells: int,
    n_groups: int,
    n_targets: int,
    n_transcription_factors: int,
    predictor_bytes: int,
    numeric_thread_limit: int,
    estimated_model_bytes: int,
    report_retained_bytes: int,
    report_auxiliary_bytes: int,
    report_render_bytes: int,
    checkpoint_enabled: bool,
) -> _BatchMemoryPlan:
    """Maximise safe model exposure while reserving all batch allocations."""

    desired_target_batch_size = min(
        n_targets,
        max(
            _TARGET_BATCH_MIN_MODELS,
            min(_TARGET_BATCH_MAX_MODELS, numeric_thread_limit * 4),
        ),
    )
    desired_group_batch_size = min(
        n_groups,
        max(1, ceil(numeric_thread_limit / max(1, n_targets))),
    )
    detected_memory = available_memory_bytes()
    weight_working_bytes = 4 * n_cells * np.dtype(np.float64).itemsize
    per_group_weight_bytes = n_cells * np.dtype(np.float64).itemsize
    per_group_mask_bytes = n_cells * np.dtype(np.bool_).itemsize

    candidates: list[_BatchMemoryPlan] = []
    minimum_required_bytes: int | None = None
    usable_bytes: int | None = None
    for target_batch_size in range(1, desired_target_batch_size + 1):
        maximum_group_batch = desired_group_batch_size if target_batch_size == n_targets else 1
        for group_batch_size in range(1, maximum_group_batch + 1):
            active_groups = group_batch_size if target_batch_size == n_targets else 1
            models_per_batch = target_batch_size * active_groups
            # The checkpoint path consumes model results as they complete. Without a
            # checkpoint, the complete bounded batch remains in memory until written.
            retained_model_results = (
                min(models_per_batch, numeric_thread_limit)
                if checkpoint_enabled
                else models_per_batch
            )
            result_record_bytes = retained_model_results * (n_transcription_factors * 256 + 768)
            retained_group_bytes = group_batch_size * (
                per_group_weight_bytes + per_group_mask_bytes + predictor_bytes
            )
            inference_reservation = (
                retained_group_bytes
                + weight_working_bytes
                + result_record_bytes
                + report_retained_bytes
            )
            report_render_reservation = (
                group_batch_size * per_group_weight_bytes
                + weight_working_bytes
                + report_retained_bytes
                + report_render_bytes
            )
            report_preparation_reservation = report_retained_bytes + report_auxiliary_bytes
            reserved_bytes = max(
                inference_reservation,
                report_render_reservation,
                report_preparation_reservation,
            )
            model_plan = plan_model_memory(
                estimated_bytes_per_model=estimated_model_bytes,
                available_bytes=detected_memory,
                reserved_bytes=reserved_bytes,
            )
            usable_bytes = model_plan.usable_bytes
            required_bytes = reserved_bytes + estimated_model_bytes
            minimum_required_bytes = (
                required_bytes
                if minimum_required_bytes is None
                else min(minimum_required_bytes, required_bytes)
            )
            if model_plan.max_concurrent_models == 0:
                continue
            memory_concurrency = model_plan.max_concurrent_models or numeric_thread_limit
            concurrent_fits = min(numeric_thread_limit, models_per_batch, memory_concurrency)
            candidates.append(
                _BatchMemoryPlan(
                    desired_group_batch_size=desired_group_batch_size,
                    desired_target_batch_size=desired_target_batch_size,
                    group_batch_size=group_batch_size,
                    target_batch_size=target_batch_size,
                    active_groups_per_inference_batch=active_groups,
                    models_per_inference_batch=models_per_batch,
                    concurrent_fits=concurrent_fits,
                    reserved_bytes=reserved_bytes,
                    model_plan=model_plan,
                )
            )

    if not candidates:
        raise MemoryError(
            "Insufficient available memory for one SPATHI model after reserving the "
            f"smallest inference batch: usable={usable_bytes}, "
            f"estimated_required={minimum_required_bytes} bytes"
        )
    return max(
        candidates,
        key=lambda candidate: (
            candidate.concurrent_fits,
            candidate.target_batch_size,
            candidate.group_batch_size,
            -candidate.reserved_bytes,
        ),
    )


def run_workflow(
    config: SpathiConfig,
    *,
    inputs: RunInputs,
    input_validation_seconds: float,
    run_started_at: datetime,
    run_started: float,
    output_dir: Path,
    checkpoint: ModelCheckpoint | None = None,
    progress_callback: ProgressCallback | None = None,
    resume_requested: bool = False,
) -> WorkflowSummary:
    """Infer and persist one weighted regulatory network per observed cell group.

    The function has no dependency on the CLI or ANDREA. It consumes validated inputs,
    calculates global distance information once, and streams bounded target-group
    batches into the private staging directory supplied by :func:`spathi.core.infer`.
    """

    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise RuntimeError("internal staging output directory must exist and be empty")
    # Disk-backed NumPy mappings must be closed before TemporaryDirectory tries
    # to remove the staging tree on Windows. ExitStack also covers exceptions
    # raised before the ordinary post-inference cleanup point.
    with ExitStack() as cleanup_stack:
        return _run_workflow_impl(
            config,
            inputs=inputs,
            input_validation_seconds=input_validation_seconds,
            run_started_at=run_started_at,
            run_started=run_started,
            output_dir=output_dir,
            checkpoint=checkpoint,
            progress_callback=progress_callback,
            resume_requested=resume_requested,
            cleanup_stack=cleanup_stack,
        )


def validate_group_configuration(config: SpathiConfig, *, group_count: int) -> None:
    """Reject weighting configurations that lose their meaning for one group.

    A group-anchored or group-distance run collapses to unit weights when only one
    group is observed.  That is a valid unweighted ensemble, but it is not the
    requested SPATHI cell-to-centroid model.  Validate this data-dependent contract
    explicitly instead of silently publishing a differently defined network.
    """

    if group_count < 1:
        raise ValueError("at least one observed group is required")
    if group_count != 1:
        return

    incompatible: list[str] = []
    if config.weight_mode != "cell-distance":
        incompatible.append("weight_mode must be 'cell-distance'")
    if config.group_size_correction != "none":
        incompatible.append("group_size_correction must be 'none'")
    centered_distance_space = (
        config.distance_space == "pca" or config.distance_standardization == "standard"
    )
    if centered_distance_space and config.distance_metric != "euclidean":
        incompatible.append("centered distance spaces require distance_metric='euclidean'")
    if incompatible:
        raise ValueError(
            "A single-group dataset must infer one individually cell-weighted network; "
            + "; ".join(incompatible)
        )


def _run_workflow_impl(
    config: SpathiConfig,
    *,
    inputs: RunInputs,
    input_validation_seconds: float,
    run_started_at: datetime,
    run_started: float,
    output_dir: Path,
    checkpoint: ModelCheckpoint | None,
    progress_callback: ProgressCallback | None,
    resume_requested: bool,
    cleanup_stack: ExitStack,
) -> WorkflowSummary:
    """Execute the workflow while ``cleanup_stack`` owns transient resources."""

    phase_times: dict[str, float] = {}
    warning_messages: list[str] = []
    available_threads = available_cpu_count()
    numeric_thread_limit = (
        available_threads if config.threads == -1 else min(config.threads, available_threads)
    )

    input_fingerprints = dict(inputs.input_fingerprints)
    phase_times["input_validation"] = input_validation_seconds
    expression_frame = inputs.take_expression()
    expression_values = expression_frame.to_numpy(dtype=np.float64, copy=False)
    gene_names = list(map(str, expression_frame.index))
    target_names = list(inputs.targets)
    cell_names = list(map(str, expression_frame.columns))
    tf_names = list(inputs.transcription_factors)
    group_ids = sorted(map(str, pd.unique(inputs.groups)))
    validate_group_configuration(config, group_count=len(group_ids))
    weighting_context = prepare_weighting_context(inputs.groups, cell_ids=cell_names)
    group_sizes = dict(
        zip(weighting_context.group_ids, map(int, weighting_context.group_counts), strict=True)
    )
    requested_model_count = len(group_ids) * len(target_names)
    completed_model_keys = frozenset() if checkpoint is None else checkpoint.completed_keys
    expected_groups = frozenset(group_ids)
    expected_targets = frozenset(target_names)
    unexpected_checkpoint_keys = {
        key
        for key in completed_model_keys
        if key[0] not in expected_groups or key[1] not in expected_targets
    }
    if unexpected_checkpoint_keys:
        raise RuntimeError(
            "checkpoint contains models outside the validated run identity: "
            f"{sorted(unexpected_checkpoint_keys)[:5]}"
        )
    resumed_models = len(completed_model_keys)
    remaining_models = requested_model_count - resumed_models
    completed_targets_by_group: defaultdict[str, set[str]] = defaultdict(set)
    if remaining_models:
        for completed_group, completed_target in completed_model_keys:
            completed_targets_by_group[completed_group].add(completed_target)
    completed_by_group = Counter(group for group, _target in completed_model_keys)
    progress_state = {
        "completed_models": resumed_models,
        "completed_groups": sum(
            completed_by_group[group] == len(target_names) for group in group_ids
        ),
    }

    def report_progress(
        phase: ProgressPhase,
        message: str,
        *,
        current_group: str | None = None,
    ) -> None:
        emit_progress(
            progress_callback,
            SpathiProgressEvent(
                phase=phase,
                message=message,
                completed_models=progress_state["completed_models"],
                total_models=requested_model_count,
                completed_groups=progress_state["completed_groups"],
                total_groups=len(group_ids),
                current_group=current_group,
                resumed_models=resumed_models,
            ),
        )

    report_progress(
        "building_representation",
        f"Building the {config.distance_space} distance representation",
    )
    phase_started = perf_counter()
    # Numerical-library reduction order can change PCA coordinates at ~1e-12
    # across thread counts and tree splits can amplify that perturbation. Keep
    # scientific preprocessing single-threaded and spend the configured budget
    # on the much larger collection of independent model fits instead.
    with threadpool_limits(limits=1):
        representation = compute_distance_representation(
            expression_frame,
            distance_space=config.distance_space,
            n_components=config.n_components,
            distance_standardization=config.distance_standardization,
            pca_svd_solver=config.pca_svd_solver,
            random_state=config.random_seed,
        )
    phase_times["distance_representation"] = perf_counter() - phase_started
    if representation.pca_degenerate:
        message = representation.pca_degeneracy_reason or "PCA representation is degenerate"
        warning_messages.append(message)
        LOGGER.warning("%s", message)

    centroid_distance_memory_plan = _plan_centroid_distance_memory(
        n_groups=len(group_ids),
        n_dimensions=representation.values.shape[1],
    )
    cell_distance_output: np.ndarray | None = None
    distance_storage_finalizer: Any = None

    report_progress(
        "computing_distances",
        f"Computing reusable centroids and {config.distance_metric} distances",
    )
    phase_started = perf_counter()
    with threadpool_limits(limits=1):
        centroids = compute_centroids(
            representation,
            inputs.groups,
            group_order=group_ids,
        )
        group_distances = compute_centroid_distances(
            centroids,
            metric=config.distance_metric,
            working_memory=centroid_distance_memory_plan.working_memory_mib,
        )
        # Plan the much larger cell matrix only after centroid and G x G
        # allocations are resident, so the snapshot reflects their real cost.
        distance_memory_plan = _plan_cell_distance_memory(
            n_cells=len(cell_names),
            n_groups=len(group_ids),
            n_dimensions=representation.values.shape[1],
            compute_distances=config.weight_mode != "group-distance",
            distance_metric=config.distance_metric,
        )
        cell_distance_storage = distance_memory_plan.storage
        if config.weight_mode != "group-distance":
            if cell_distance_storage == "temporary-memory-map":
                distance_temporary_file = TemporaryFile(
                    prefix="spathi-cell-distances-",
                    dir=output_dir,
                )
                # If mapping construction itself fails, the backing file is still owned.
                cleanup_stack.callback(distance_temporary_file.close)
                distance_temporary_file.truncate(distance_memory_plan.expected_output_bytes)
                cell_distance_output = np.memmap(
                    distance_temporary_file,
                    mode="r+",
                    dtype=np.float64,
                    shape=(len(cell_names), len(group_ids)),
                    order="F",
                )
                distance_storage_finalizer = finalize(
                    cell_distance_output,
                    _close_distance_backing,
                    cast(Any, cell_distance_output)._mmap,
                    distance_temporary_file,
                )
                cleanup_stack.callback(distance_storage_finalizer)
                message = (
                    "Using chunked, disk-backed storage for the "
                    f"{distance_memory_plan.expected_output_bytes / 1024**2:.1f} MiB "
                    "cell-to-centroid matrix "
                    f"({distance_memory_plan.storage_reason})"
                )
                warning_messages.append(message)
                LOGGER.warning("%s", message)
            else:
                # Weighting consumes one target-group column at a time. Keep
                # columns contiguous in both RAM and disk-backed storage.
                cell_distance_output = np.empty(
                    (len(cell_names), len(group_ids)),
                    dtype=np.float64,
                    order="F",
                )
            if distance_memory_plan.expected_output_bytes > 2 * 1024**3:
                message = (
                    "The cell-to-centroid distance matrix is expected to require "
                    f"{distance_memory_plan.expected_output_bytes / 1024**3:.2f} GiB "
                    "before table overhead"
                )
                warning_messages.append(message)
                LOGGER.warning("%s", message)
            cell_distances = compute_cell_to_centroid_distances(
                representation,
                centroids,
                metric=config.distance_metric,
                working_memory=distance_memory_plan.working_memory_mib,
                output=cell_distance_output,
            )
        else:
            cell_distances = None
    phase_times["centroids_and_distances"] = perf_counter() - phase_started

    phase_started = perf_counter()
    with threadpool_limits(limits=1):
        bandwidth = resolve_bandwidth_for_mode(
            config.weight_mode,
            cell_to_centroid_distances=(
                None
                if cell_distances is None
                else cell_distances.to_numpy(dtype=np.float64, copy=False)
            ),
            centroid_distances=group_distances.to_numpy(dtype=np.float64, copy=False),
            bandwidth=config.bandwidth,
            scratch_dir=output_dir,
        )
    phase_times["bandwidth_selection"] = perf_counter() - phase_started
    if bandwidth.fallback_reason:
        warning_messages.append(f"{bandwidth.fallback_reason}; used bandwidth {bandwidth.value:g}")
    LOGGER.info(
        "Using global %s bandwidth %.6g (%s)",
        config.kernel,
        bandwidth.value,
        bandwidth.method,
    )

    prepared: PreparedInference | None = None
    if remaining_models:
        report_progress("preparing_inference", "Preparing reusable inference matrices")
        phase_started = perf_counter()
        prepared = prepare_inference(
            expression_values.T,
            gene_names,
            tf_names,
            target_names=target_names,
            tree_method=config.tree_method,
            n_estimators=config.n_estimators,
            max_features=config.max_features,
            min_samples_leaf=config.min_samples_leaf,
            max_depth=config.max_depth,
            bootstrap=config.bootstrap,
            random_seed=config.random_seed,
        )
        phase_times["inference_preparation"] = perf_counter() - phase_started
        tree_target_bytes = prepared.expression_nbytes
        additional_tree_target_bytes = prepared.target_expression_additional_nbytes
        predictor_bytes = prepared.predictor_nbytes
        tree_target_dtype = prepared.expression_dtype
        tree_predictor_dtype = prepared.predictor_dtype
        effective_bootstrap = prepared.bootstrap
    else:
        # A complete checkpoint already contains every fitted model. Rebuilding
        # weights and output diagnostics must not allocate the target/TF matrices
        # or require enough memory for a model that will never run.
        phase_times["inference_preparation"] = 0.0
        tree_target_bytes = 0
        additional_tree_target_bytes = 0
        predictor_bytes = 0
        # These describe the model data contract used by the checkpointed fits;
        # the zero storage estimates below record that the arrays were not
        # materialized again during this attempt.
        tree_target_dtype = "float64"
        tree_predictor_dtype = "float32"
        effective_bootstrap = (
            config.tree_method == "random-forest" if config.bootstrap is None else config.bootstrap
        )

    memory_estimate = _estimated_memory_bytes(
        expression_bytes=int(expression_values.nbytes),
        representation_bytes=int(representation.values.nbytes),
        centroids=centroids,
        cell_distances=cell_distances,
        group_distances=group_distances,
        tree_target_bytes=tree_target_bytes,
        additional_tree_target_bytes=additional_tree_target_bytes,
        predictor_bytes=predictor_bytes,
        cell_distance_storage=cell_distance_storage,
    )
    memory_estimate["centroid_distance_planned_persistent_upper_bound"] = (
        centroid_distance_memory_plan.estimated_persistent_bytes
    )
    memory_estimate["centroid_distance_chunk_working_memory_upper_bound"] = (
        centroid_distance_memory_plan.working_memory_bytes
    )
    # The selected responses, TF predictors, and distance representation now own
    # every array still required downstream. With a target subset in PCA space,
    # dropping these final references releases the much larger complete input
    # matrix before any tree ensemble is fitted.
    del expression_values, expression_frame
    if config.report:
        from spathi._report import report_sample_size

        n_cells = len(cell_names)
        n_groups = len(group_ids)
        sampled_cells = report_sample_size(n_cells, n_groups)
        sampled_target_values = n_groups * sampled_cells
        memory_estimate["report_sampled_cells"] = sampled_cells
        memory_estimate["report_sampled_target_cell_values"] = sampled_target_values
        # Four sampled target-by-cell vectors plus eight exact target-by-source
        # summary matrices are retained until the single HTML document is built.
        memory_estimate["report_sampled_vectors_float64"] = int(
            4 * sampled_target_values * np.dtype(np.float64).itemsize
        )
        memory_estimate["report_exact_summary_matrices"] = int(
            8 * n_groups * n_groups * np.dtype(np.float64).itemsize
        )
        memory_estimate["report_retained_rough_bytes"] = int(
            n_cells * (2 * 8 + 8 + 8 + 8)
            + n_groups * 2 * 8
            + memory_estimate["report_sampled_vectors_float64"]
            + memory_estimate["report_exact_summary_matrices"]
        )
        memory_estimate["report_auxiliary_pca_working_rough_bytes"] = int(
            0
            if representation.distance_space == "pca"
            else 2 * representation.values.nbytes + n_cells * 2 * 8
        )
        memory_estimate["report_aggregation_working_rough_bytes"] = int(
            max(group_sizes.values()) * np.dtype(np.float64).itemsize
        )
        # Base64 payloads expand by 4/3 and coexist briefly with the final HTML
        # string and Plotly's vendored browser bundle.
        encoded_numeric_bytes = int(
            (
                memory_estimate["report_sampled_vectors_float64"]
                + memory_estimate["report_exact_summary_matrices"]
                + sampled_cells * (2 * 8 + 4)
                + n_groups * 2 * 8
            )
            * 4
            / 3
        )
        memory_estimate["report_render_working_rough_bytes"] = (
            2 * encoded_numeric_bytes + 24 * 1024**2
        )
        report_reservation = max(
            memory_estimate["report_retained_rough_bytes"]
            + memory_estimate["report_auxiliary_pca_working_rough_bytes"],
            memory_estimate["report_retained_rough_bytes"]
            + memory_estimate["report_render_working_rough_bytes"],
            memory_estimate["report_retained_rough_bytes"]
            + memory_estimate["report_aggregation_working_rough_bytes"],
        )
        report_memory_plan = plan_model_memory(
            estimated_bytes_per_model=1,
            available_bytes=available_memory_bytes(),
            reserved_bytes=report_reservation,
        )
        if report_memory_plan.max_concurrent_models == 0:
            raise MemoryError(
                "Insufficient available memory for the bounded interactive report: "
                f"usable={report_memory_plan.usable_bytes}, "
                f"estimated_required={report_reservation} bytes"
            )
    else:
        memory_estimate["report_sampled_cells"] = 0
        memory_estimate["report_sampled_target_cell_values"] = 0
        memory_estimate["report_sampled_vectors_float64"] = 0
        memory_estimate["report_exact_summary_matrices"] = 0
        memory_estimate["report_retained_rough_bytes"] = 0
        memory_estimate["report_auxiliary_pca_working_rough_bytes"] = 0
        memory_estimate["report_aggregation_working_rough_bytes"] = 0
        memory_estimate["report_render_working_rough_bytes"] = 0
    estimated_model_bytes = estimate_model_memory_bytes(
        n_cells=len(cell_names),
        n_transcription_factors=len(tf_names),
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_depth=config.max_depth,
    )
    batch_memory_plan: _BatchMemoryPlan | None = None
    model_memory_plan: MemoryPlan | None = None
    if remaining_models:
        batch_memory_plan = _plan_inference_batches(
            n_cells=len(cell_names),
            n_groups=len(group_ids),
            n_targets=len(target_names),
            n_transcription_factors=len(tf_names),
            predictor_bytes=predictor_bytes,
            numeric_thread_limit=numeric_thread_limit,
            estimated_model_bytes=estimated_model_bytes,
            report_retained_bytes=memory_estimate["report_retained_rough_bytes"],
            report_auxiliary_bytes=memory_estimate["report_auxiliary_pca_working_rough_bytes"],
            report_render_bytes=max(
                memory_estimate["report_render_working_rough_bytes"],
                memory_estimate["report_aggregation_working_rough_bytes"],
            ),
            checkpoint_enabled=checkpoint is not None,
        )
        group_batch_size = batch_memory_plan.group_batch_size
        target_batch_size = batch_memory_plan.target_batch_size
        active_groups_per_inference_batch = batch_memory_plan.active_groups_per_inference_batch
        models_per_inference_batch = batch_memory_plan.models_per_inference_batch
        concurrent_fits = batch_memory_plan.concurrent_fits
        model_memory_plan = batch_memory_plan.model_plan
        if group_batch_size < batch_memory_plan.desired_group_batch_size:
            message = (
                "Memory planning reduced target groups per batch from "
                f"{batch_memory_plan.desired_group_batch_size} to {group_batch_size}"
            )
            warning_messages.append(message)
            LOGGER.warning("%s", message)
        if target_batch_size < batch_memory_plan.desired_target_batch_size:
            message = (
                "Memory planning reduced targets per batch from "
                f"{batch_memory_plan.desired_target_batch_size} to {target_batch_size}"
            )
            warning_messages.append(message)
            LOGGER.warning("%s", message)
        retained_model_results = (
            min(models_per_inference_batch, concurrent_fits)
            if checkpoint is not None
            else models_per_inference_batch
        )
    else:
        # Weight diagnostics are still rebuilt and compared with their stored
        # hashes, one group at a time. No inference batch exists in this path.
        group_batch_size = 1
        target_batch_size = 0
        active_groups_per_inference_batch = 0
        models_per_inference_batch = 0
        concurrent_fits = 0
        retained_model_results = 0
    memory_estimate["weight_batch_retained_float64"] = int(
        len(cell_names) * group_batch_size * np.dtype(np.float64).itemsize
    )
    memory_estimate["distance_chunk_working_memory_upper_bound"] = (
        distance_memory_plan.working_memory_bytes
    )
    memory_estimate["maximum_target_batch_edge_records"] = int(
        models_per_inference_batch * len(tf_names)
    )
    memory_estimate["maximum_target_batch_edge_records_rough_bytes"] = int(
        memory_estimate["maximum_target_batch_edge_records"] * 256
    )
    memory_estimate["maximum_target_batch_model_records_rough_bytes"] = int(
        models_per_inference_batch * 768
    )
    memory_estimate["retained_model_result_records_rough_bytes"] = int(
        retained_model_results * (len(tf_names) * 256 + 768)
    )
    memory_estimate["weight_result_working_float64"] = int(
        4 * len(cell_names) * np.dtype(np.float64).itemsize
    )
    memory_estimate["group_positive_masks_bool"] = int(
        len(cell_names) * group_batch_size * np.dtype(np.bool_).itemsize
    )
    # If a group contains any constant TF among its positive-weight cells,
    # inference caches one filtered predictor matrix for that group. This is a
    # conservative bound: groups without constant TFs share the core matrix and
    # a filtered matrix can never be larger than it.
    memory_estimate["group_constant_filter_predictors_float32_upper_bound"] = int(
        group_batch_size * predictor_bytes
    )
    concurrent_tf_targets = min(
        concurrent_fits,
        active_groups_per_inference_batch * min(target_batch_size, len(tf_names)),
    )
    memory_estimate["concurrent_self_exclusion_predictors_float32"] = int(
        concurrent_tf_targets
        * len(cell_names)
        * max(0, len(tf_names) - 1)
        * np.dtype(np.float32).itemsize
    )
    memory_estimate["estimated_model_fit_bytes"] = estimated_model_bytes
    memory_estimate["rough_concurrent_model_upper_bound"] = concurrent_fits * estimated_model_bytes
    memory_estimate["batch_planning_reserved_bytes"] = (
        0 if batch_memory_plan is None else batch_memory_plan.reserved_bytes
    )
    inference_temporary = sum(
        memory_estimate[key]
        for key in (
            "weight_batch_retained_float64",
            "weight_result_working_float64",
            "group_positive_masks_bool",
            "group_constant_filter_predictors_float32_upper_bound",
            "retained_model_result_records_rough_bytes",
        )
    )
    report_retained = memory_estimate["report_retained_rough_bytes"]
    report_preparation_temporary = (
        report_retained + memory_estimate["report_auxiliary_pca_working_rough_bytes"]
    )
    report_render_temporary = (
        report_retained
        + max(
            memory_estimate["report_render_working_rough_bytes"],
            memory_estimate["report_aggregation_working_rough_bytes"],
        )
        + memory_estimate["weight_batch_retained_float64"]
        + memory_estimate["weight_result_working_float64"]
    )
    memory_estimate["estimated_peak_heap_before_tree_storage"] = int(
        memory_estimate["estimated_heap_core_array_total"]
        + max(
            memory_estimate["centroid_distance_chunk_working_memory_upper_bound"],
            memory_estimate["distance_chunk_working_memory_upper_bound"],
            inference_temporary + report_retained,
            report_preparation_temporary,
            report_render_temporary,
        )
    )
    memory_estimate["estimated_peak_heap_with_rough_trees"] = int(
        memory_estimate["estimated_peak_heap_before_tree_storage"]
        + memory_estimate["rough_concurrent_model_upper_bound"]
    )
    if memory_estimate["estimated_peak_heap_before_tree_storage"] > 2 * 1024**3:
        message = "Estimated non-tree heap peak exceeds 2 GiB; monitor memory during fitting"
        warning_messages.append(message)
        LOGGER.warning("%s", message)
    if memory_estimate["rough_concurrent_model_upper_bound"] > 2 * 1024**3:
        message = (
            "The conservative concurrent-model memory estimate exceeds 2 GiB; actual tree "
            "memory depends on the fitted depth and leaf count"
        )
        warning_messages.append(message)
        LOGGER.warning("%s", message)

    # Keep Plotly and all report state outside runs that explicitly disable it.
    report_embedding: ReportEmbedding | None = None
    report_builder: InteractiveReportBuilder | None = None
    report_metadata: dict[str, Any] | None = None
    report_seconds = 0.0
    if config.report:
        phase_started = perf_counter()
        from spathi._report import InteractiveReportBuilder, prepare_report_embedding

        # Auxiliary expression-space PCA must obey the same single numerical
        # thread budget as the scientific distance representation. The context
        # is harmless for PCA-space runs, where this call only slices PC1/PC2.
        with threadpool_limits(limits=1):
            report_embedding = prepare_report_embedding(
                representation,
                inputs.groups,
                centroids,
                random_state=config.random_seed,
            )
        report_builder = InteractiveReportBuilder(
            report_embedding,
            group_sizes=group_sizes,
            run_parameters=_report_run_parameters(config),
        )
        if report_builder.sampled_cells != memory_estimate["report_sampled_cells"]:
            raise RuntimeError("report sampling does not match its memory plan")
        report_seconds += perf_counter() - phase_started

    output_started = perf_counter()
    write_tsv_records(
        _iter_centroid_records(centroids, group_ids),
        output_dir / "centroids.tsv",
        _CENTROID_COLUMNS,
    )
    write_tsv_records(
        _iter_group_distance_records(group_distances, group_ids),
        output_dir / "group_distances.tsv",
        _GROUP_DISTANCE_COLUMNS,
    )
    write_tsv_records(
        iter_group_affinity_records(
            group_distances,
            group_sizes,
            bandwidth=bandwidth,
            kernel=config.kernel,
            group_size_correction=config.group_size_correction,
        ),
        output_dir / "group_affinities.tsv",
        _GROUP_AFFINITY_COLUMNS,
    )
    if representation.distance_space == "pca":
        embedding_component_count = min(3, representation.values.shape[1])
        embedding_components = representation.dimension_names[:embedding_component_count]
        embedding_output = pd.DataFrame(
            representation.values[:, :embedding_component_count],
            columns=embedding_components,
        )
        embedding_output.insert(
            0,
            "group",
            inputs.groups.loc[list(representation.cell_ids)].astype(str).to_numpy(copy=False),
        )
        embedding_output.insert(0, "cell", representation.cell_ids)
        write_tsv_gzip(
            embedding_output,
            output_dir / "cell_embedding.tsv.gz",
            ("cell", "group", *embedding_components),
        )

        if representation.explained_variance_ratio is None:
            raise RuntimeError("PCA representation did not report explained variance")
        explained_variance = np.asarray(
            representation.explained_variance_ratio,
            dtype=np.float64,
        )
        if explained_variance.shape != (representation.values.shape[1],):
            raise RuntimeError("PCA explained variance does not match the fitted components")
        variance_output = pd.DataFrame(
            {
                "component": representation.dimension_names,
                "explained_variance_ratio": explained_variance,
                "cumulative_explained_variance_ratio": np.cumsum(
                    explained_variance,
                    dtype=np.float64,
                ),
            }
        )
        write_tsv(
            variance_output,
            output_dir / "pca_explained_variance.tsv",
            (
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance_ratio",
            ),
        )
    elif report_embedding is not None:
        # Expression-space runs use PCA only as an explicitly auxiliary 2D view.
        # Persist its numeric coordinates so every rendered point is auditable.
        auxiliary_component_count = len(report_embedding.explained_variance_ratio)
        auxiliary_component_names = tuple(
            f"AuxiliaryPC{index}" for index in range(1, auxiliary_component_count + 1)
        )
        embedding_output = pd.DataFrame(
            report_embedding.coordinates[:, :auxiliary_component_count],
            columns=auxiliary_component_names,
        )
        embedding_output.insert(0, "group", report_embedding.cell_groups)
        embedding_output.insert(0, "cell", report_embedding.cell_ids)
        write_tsv_gzip(
            embedding_output,
            output_dir / "cell_embedding.tsv.gz",
            ("cell", "group", *auxiliary_component_names),
        )
        auxiliary_explained = np.asarray(
            report_embedding.explained_variance_ratio,
            dtype=np.float64,
        )
        write_tsv(
            pd.DataFrame(
                {
                    "component": auxiliary_component_names,
                    "explained_variance_ratio": auxiliary_explained,
                    "cumulative_explained_variance_ratio": np.cumsum(
                        auxiliary_explained,
                        dtype=np.float64,
                    ),
                }
            ),
            output_dir / "pca_explained_variance.tsv",
            (
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance_ratio",
            ),
        )
    write_json(config.to_dict(), output_dir / "parameters.json")
    phase_times["artifact_writing"] = perf_counter() - output_started
    representation_summary = _RepresentationSummary.from_result(representation)
    # Centroids, distances, optional report coordinates, and persisted PCA
    # artifacts now own everything still needed. In particular, expression-space
    # runs with an explicit target subset no longer retain the complete distance
    # matrix while the tree ensembles are fitted.
    del representation

    weighting_seconds = 0.0
    inference_seconds = 0.0
    dynamic_writing_seconds = 0.0
    n_edges = 0
    completed_models = 0
    trained_models = 0
    skipped_target_records = 0
    model_status_counts: Counter[str] = Counter()
    run_parallel_plan = resolve_thread_budget(
        config.threads,
        remaining_models,
        available_threads=available_threads,
        max_outer_jobs=concurrent_fits or None,
    )
    if (
        remaining_models
        and 1 < concurrent_fits < numeric_thread_limit
        and run_parallel_plan.parallel_level == "estimator"
    ):
        LOGGER.info(
            "The memory cap allows %d simultaneous models; using all %d threads inside "
            "one ensemble at a time instead of leaving CPUs idle",
            concurrent_fits,
            run_parallel_plan.model_n_jobs,
        )
    if (
        model_memory_plan is not None
        and model_memory_plan.max_concurrent_models is not None
        and model_memory_plan.max_concurrent_models < min(numeric_thread_limit, remaining_models)
    ):
        message = (
            "Memory planning limited concurrent model fits to "
            f"{model_memory_plan.max_concurrent_models}"
        )
        warning_messages.append(message)
        LOGGER.warning("%s", message)
    # SQLite's primary key already detects duplicates in the default checkpointed
    # path. Keep an in-memory key set only when checkpointing is explicitly off.
    newly_completed_keys: set[tuple[str, str]] | None = set() if checkpoint is None else None

    def on_model_complete(result: ModelResult) -> None:
        key = (result.stat.target_group, result.stat.target)
        if key in completed_model_keys or (
            newly_completed_keys is not None and key in newly_completed_keys
        ):
            raise RuntimeError(f"model completed more than once: {key!r}")
        if checkpoint is not None:
            checkpoint.record_result(result)
        if newly_completed_keys is not None:
            newly_completed_keys.add(key)
        progress_state["completed_models"] += 1
        completed_by_group[key[0]] += 1
        if completed_by_group[key[0]] == len(target_names):
            progress_state["completed_groups"] += 1
        report_progress(
            "model_inference",
            (
                f"Completed model {progress_state['completed_models']}/"
                f"{requested_model_count}: {key[0]} / {key[1]}"
            ),
            current_group=key[0],
        )

    # Opening the executor also fixes native BLAS/OpenMP pools to one thread.
    # Keep that numerical contract when a resumed checkpoint already contains
    # every model: weights, diagnostics, and the optional report are still rebuilt.
    executor_context = PersistentTaskExecutor(run_parallel_plan)
    with IncrementalRunWriter(output_dir) as writer, executor_context as executor:
        for batch_start in range(0, len(group_ids), group_batch_size):
            batch_groups = group_ids[batch_start : batch_start + group_batch_size]
            batch_weights: dict[object, ArrayLike] = {}
            for batch_offset, target_group in enumerate(batch_groups):
                index = batch_start + batch_offset + 1
                report_progress(
                    "preparing_group",
                    f"Preparing target group {index}/{len(group_ids)}: {target_group}",
                    current_group=target_group,
                )
                phase_started = perf_counter()
                weights = compute_weights(
                    target_group,
                    weighting_context,
                    mode=config.weight_mode,
                    bandwidth=bandwidth,
                    kernel=config.kernel,
                    group_size_correction=config.group_size_correction,
                    cell_distances=(
                        None
                        if cell_distances is None
                        else cell_distances[target_group].to_numpy(dtype=np.float64, copy=False)
                    ),
                    group_distances=group_distances,
                )
                diagnostics = compute_weight_diagnostics(weights, emit_warnings=False)
                for warning in diagnostics.warnings:
                    contextual = f"Target group {target_group!r}: {warning}"
                    warning_messages.append(contextual)
                    LOGGER.warning("%s", contextual)
                if checkpoint is not None:
                    checkpoint.validate_or_record_weights(
                        target_group,
                        weights.final_weight,
                    )
                batch_weights[target_group] = weights.final_weight
                weighting_seconds += perf_counter() - phase_started

                if report_builder is not None:
                    phase_started = perf_counter()
                    report_builder.add_target(weights, diagnostics)
                    report_seconds += perf_counter() - phase_started

                phase_started = perf_counter()
                writer.write_weights(weights)
                writer.write_weight_diagnostics(diagnostics)
                dynamic_writing_seconds += perf_counter() - phase_started

            if not remaining_models:
                continue
            inference_started = perf_counter()
            batch_completed = {
                (group, target)
                for group in batch_groups
                for target in completed_targets_by_group[group]
            }
            if prepared is None:  # pragma: no cover - internal state invariant
                raise RuntimeError("inference matrices were not prepared for pending models")
            if checkpoint is None:
                inference_batches = prepared.iter_group_target_batches(
                    batch_weights,
                    group_order=batch_groups,
                    target_batch_size=target_batch_size,
                    threads=config.threads,
                    completed_models=batch_completed,
                    executor=executor,
                    on_model_complete=on_model_complete,
                )
                for inference_result in inference_batches:
                    inference_seconds += perf_counter() - inference_started
                    completed_models += inference_result.completed_models
                    trained_models += inference_result.trained_models
                    skipped_target_records += len(inference_result.skipped_targets)
                    model_status_counts.update(stat.status for stat in inference_result.model_stats)

                    phase_started = perf_counter()
                    n_edges += writer.write_edges(inference_result.edges)
                    writer.write_skipped_targets(inference_result.skipped_targets)
                    writer.write_model_diagnostics(inference_result.model_stats)
                    dynamic_writing_seconds += perf_counter() - phase_started
                    inference_started = perf_counter()
            else:
                streamed_batches = prepared.stream_group_target_batches(
                    batch_weights,
                    group_order=batch_groups,
                    target_batch_size=target_batch_size,
                    threads=config.threads,
                    completed_models=batch_completed,
                    executor=executor,
                    on_model_complete=on_model_complete,
                )
                for _summary in streamed_batches:
                    inference_seconds += perf_counter() - inference_started
                    inference_started = perf_counter()

        if checkpoint is not None:
            phase_started = perf_counter()
            for model_result in checkpoint.iter_results():
                completed_models += 1
                trained_models += int(model_result.trained)
                model_status_counts[model_result.stat.status] += 1
                n_edges += writer.write_edges(model_result.edges)
                if model_result.skipped is not None:
                    skipped_target_records += 1
                    writer.write_skipped_targets((model_result.skipped,))
                writer.write_model_diagnostics((model_result.stat,))
            dynamic_writing_seconds += perf_counter() - phase_started

    report_progress("writing_outputs", "Finalizing SPATHI output artifacts")
    phase_times["weighting_and_diagnostics"] = weighting_seconds
    phase_times["model_inference"] = inference_seconds
    phase_times["artifact_writing"] += dynamic_writing_seconds
    if distance_storage_finalizer is not None and distance_storage_finalizer.alive:
        distance_storage_finalizer()

    if completed_models != requested_model_count:
        raise RuntimeError(
            f"Inference completed {completed_models} of {requested_model_count} requested models"
        )
    if progress_state["completed_models"] != requested_model_count:
        raise RuntimeError(
            "progress accounting does not match completed inference models: "
            f"{progress_state['completed_models']} != {requested_model_count}"
        )
    effective_threads = run_parallel_plan.effective_threads
    fatal_model_failures = sum(model_status_counts[status] for status in FATAL_MODEL_STATUSES)
    if fatal_model_failures:
        message = (
            f"{fatal_model_failures} model(s) failed during fitting or returned invalid "
            "feature importances"
        )
        warning_messages.append(message)
        LOGGER.error("%s", message)

    if report_builder is not None:
        report_progress("building_report", "Building the interactive HTML report")
        phase_started = perf_counter()
        report_artifact = report_builder.write(
            output_dir / "report.html",
            run_summary={
                "status": "failed" if fatal_model_failures else "complete",
                "input_dimensions": {
                    "genes": len(gene_names),
                    "cells": len(cell_names),
                    "groups": len(group_ids),
                    "transcription_factors": len(tf_names),
                    "targets": len(target_names),
                },
                "models": {
                    "requested": requested_model_count,
                    "trained": trained_models,
                    "skipped": skipped_target_records,
                    "failed": fatal_model_failures,
                    "positive_edges": n_edges,
                },
                "weighting": {
                    "mode": config.weight_mode,
                    "distance_space": config.distance_space,
                    "distance_metric": config.distance_metric,
                    "kernel": config.kernel,
                    "bandwidth": bandwidth.value,
                    "group_size_correction": config.group_size_correction,
                },
            },
        )
        report_metadata = report_artifact.to_metadata()
        report_seconds += perf_counter() - phase_started
    phase_times["report"] = report_seconds

    run_finished_at = datetime.now(UTC)
    total_seconds = perf_counter() - run_started
    phase_times["total"] = total_seconds
    metadata: dict[str, Any] = {
        "status": "failed" if fatal_model_failures else "complete",
        "started_at": run_started_at.isoformat(),
        "finished_at": run_finished_at.isoformat(),
        "spathi_version": __version__,
        "input_dimensions": {
            "genes": len(gene_names),
            "cells": len(cell_names),
            "groups": len(group_ids),
            "transcription_factors": len(tf_names),
            "targets": len(target_names),
        },
        "inputs": input_fingerprints,
        "group_ids": group_ids,
        "group_sizes": {group: int(group_sizes[group]) for group in group_ids},
        "requested_parameters": config.to_dict(),
        "effective_parameters": {
            "centroid_method": "arithmetic_mean",
            "distance_space": representation_summary.distance_space,
            "distance_standardization": representation_summary.standardization,
            "pca_svd_solver_requested": representation_summary.pca_svd_solver,
            "pca_svd_solver_resolution": representation_summary.pca_svd_solver_resolution,
            "effective_n_components": representation_summary.effective_n_components,
            "maximum_informative_n_components": (
                representation_summary.maximum_informative_n_components
            ),
            "pca_explained_variance_ratio": representation_summary.explained_variance_ratio,
            "pca_cumulative_explained_variance_ratio": (
                None
                if representation_summary.explained_variance_ratio is None
                else tuple(
                    np.cumsum(
                        representation_summary.explained_variance_ratio,
                        dtype=np.float64,
                    )
                )
            ),
            "distance_metric": config.distance_metric,
            "centroid_distance_chunk_working_memory_mib": (
                centroid_distance_memory_plan.working_memory_mib
            ),
            "centroid_distance_memory_available_bytes_at_planning": (
                centroid_distance_memory_plan.available_bytes
            ),
            "centroid_distance_memory_usable_bytes_at_planning": (
                centroid_distance_memory_plan.usable_bytes
            ),
            "cell_centroid_distance_storage": cell_distance_storage,
            "cell_centroid_distances_computed": cell_distances is not None,
            "distance_chunk_working_memory_mib": distance_memory_plan.working_memory_mib,
            "distance_storage_reason": distance_memory_plan.storage_reason,
            "distance_memory_available_bytes_at_planning": (distance_memory_plan.available_bytes),
            "distance_memory_usable_bytes_at_planning": distance_memory_plan.usable_bytes,
            "bandwidth": asdict(bandwidth),
            "tree_target_dtype": tree_target_dtype,
            "tree_predictor_dtype": tree_predictor_dtype,
            "inference_preparation_performed": prepared is not None,
            "bootstrap_requested": config.bootstrap,
            "bootstrap_effective": effective_bootstrap,
            "weight_dtype": "float64",
            "pca_degenerate": representation_summary.pca_degenerate,
            "pca_degeneracy_reason": representation_summary.pca_degeneracy_reason,
            "group_processing": "progressive-group-and-target-batches",
            "target_groups_per_batch": group_batch_size,
            "target_groups_per_batch_without_memory_limit": (
                None if batch_memory_plan is None else batch_memory_plan.desired_group_batch_size
            ),
            "targets_per_batch": None if batch_memory_plan is None else target_batch_size,
            "targets_per_batch_without_memory_limit": (
                None if batch_memory_plan is None else batch_memory_plan.desired_target_batch_size
            ),
            "target_selection": (
                "all-expression-genes" if config.target_list is None else "explicit-list"
            ),
            "target_ids": None if config.target_list is None else target_names,
        },
        "random_seed": config.random_seed,
        "parallelism": {
            "threads_requested": config.threads,
            "threads_effective": effective_threads,
            "threads_available": available_threads,
            "preprocessing_thread_limit": 1,
            "inference_thread_budget": numeric_thread_limit,
            "backend": run_parallel_plan.backend,
            "parallel_level": run_parallel_plan.parallel_level,
            "nested_parallelism": False,
            "persistent_worker_pool": run_parallel_plan.outer_jobs > 1,
            "maximum_concurrent_model_fits": (
                0
                if remaining_models == 0
                else run_parallel_plan.outer_jobs
                if run_parallel_plan.parallel_level == "tasks"
                else 1
            ),
            "memory_concurrent_model_cap": concurrent_fits,
            "memory_available_bytes_at_planning": (
                None if model_memory_plan is None else model_memory_plan.available_bytes
            ),
            "memory_usable_bytes_at_planning": (
                None if model_memory_plan is None else model_memory_plan.usable_bytes
            ),
            "memory_reserved_for_batch_bytes": (
                None if model_memory_plan is None else model_memory_plan.reserved_bytes
            ),
            "memory_usable_fraction": (
                None if model_memory_plan is None else model_memory_plan.usable_fraction
            ),
        },
        "models": {
            "requested": requested_model_count,
            "completed": completed_models,
            "trained": trained_models,
            "trained_with_positive_edges": model_status_counts["trained"],
            "trained_without_positive_edges": model_status_counts["trained_no_positive_importance"],
            "preflight_skipped": sum(
                count
                for status, count in model_status_counts.items()
                if status not in TRAINED_MODEL_STATUSES | FATAL_MODEL_STATUSES
            ),
            "fit_or_importance_failures": fatal_model_failures,
            "skipped_target_records": skipped_target_records,
            "positive_edges": n_edges,
            "reused_from_checkpoint": resumed_models,
            "processed_this_attempt": requested_model_count - resumed_models,
        },
        "checkpoint": {
            "enabled": checkpoint is not None,
            "resumed": resume_requested,
            "models_reused": resumed_models,
            "model_storage": None if checkpoint is None else "sqlite-zlib-per-model",
            "weight_identity": (None if checkpoint is None else "sha256-float64-per-group"),
            "included_in_output": False,
        },
        "memory_estimate_bytes": memory_estimate,
        "artifact_semantics": {
            "group_affinities.tsv": {
                "scope": "group-level diagnostic",
                "base_affinity": (
                    "kernel affinity computed from source-to-target centroid distance; "
                    "it is a per-cell base model weight only in group-distance mode"
                ),
                "group_size_factor": (
                    "per-cell multiplicity factor for the source group relative to the target group"
                ),
                "authoritative_model_weights": "cell_weights.tsv.gz:final_weight",
            },
            "cell_embedding.tsv.gz": (
                {
                    "scope": "visual projection of the fitted PCA distance representation",
                    "projection_role": "distance-space",
                    "components": list(representation_summary.displayed_components),
                    "distance_fidelity": (
                        "exact only when all fitted PCA components are included; otherwise this "
                        "is a lower-dimensional view"
                    ),
                }
                if representation_summary.distance_space == "pca"
                else (
                    None
                    if report_embedding is None
                    else {
                        "scope": "auxiliary PCA coordinates used only for the report",
                        "projection_role": "report-only",
                        "components": [
                            f"AuxiliaryPC{index}"
                            for index in range(
                                1,
                                len(report_embedding.explained_variance_ratio) + 1,
                            )
                        ],
                        "distance_fidelity": (
                            "SPATHI weights and distances were calculated in expression space, "
                            "not in this auxiliary projection"
                        ),
                    }
                )
            ),
        },
        "report": {
            "requested": config.report,
            "generated": report_metadata is not None,
            "artifact": report_metadata,
        },
        "dependency_versions": dependency_versions(include_report=config.report),
        "phase_times_seconds": phase_times,
        "warnings": warning_messages,
    }
    write_json(metadata, output_dir / "run_metadata.json")
    LOGGER.info(
        "Finished %d models and %d positive edges in %.3f seconds",
        completed_models,
        n_edges,
        total_seconds,
    )
    return WorkflowSummary(
        n_edges=n_edges,
        total_models=requested_model_count,
        trained_models=trained_models,
        skipped_target_records=skipped_target_records,
        failed_models=fatal_model_failures,
        resumed_models=resumed_models,
        warnings=tuple(warning_messages),
    )
