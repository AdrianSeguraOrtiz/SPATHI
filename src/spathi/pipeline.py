"""End-to-end SPATHI orchestration, independent of the command-line interface."""

from __future__ import annotations

import logging
import platform
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from time import perf_counter
from typing import Any, cast
from weakref import finalize

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

from spathi._version import __version__
from spathi.config import SpathiConfig
from spathi.diagnostics import compute_weight_diagnostics
from spathi.distances import (
    DEFAULT_WORKING_MEMORY_MIB,
    compute_cell_to_centroid_distances,
    compute_centroid_distances,
    iter_cell_to_centroid_distance_chunks,
)
from spathi.inference import prepare_inference
from spathi.io import load_inputs
from spathi.kernels import resolve_bandwidth_for_mode
from spathi.outputs import (
    IncrementalRunWriter,
    create_output_directory,
    write_json,
    write_tsv,
    write_tsv_records,
)
from spathi.parallel import available_cpu_count
from spathi.prototypes import compute_centroids
from spathi.representation import compute_distance_representation
from spathi.weighting import compute_weights, iter_group_affinity_records

LOGGER = logging.getLogger(__name__)
_DISTANCE_MEMMAP_THRESHOLD_BYTES = 512 * 1024**2
_TARGET_BATCH_MIN_MODELS = 32
_TARGET_BATCH_MAX_MODELS = 256
_GROUP_DISTANCE_COLUMNS = ("target_group", "source_group", "centroid_distance")
_GROUP_AFFINITY_COLUMNS = (
    "target_group",
    "source_group",
    "centroid_distance",
    "base_affinity",
    "group_size_factor",
)


@dataclass(frozen=True, slots=True)
class SpathiRunResult:
    """Compact result returned after all run artifacts have been written."""

    output_dir: Path
    network_path: Path
    metadata_path: Path
    n_edges: int
    total_models: int
    trained_models: int
    skipped_models: int
    failed_models: int
    warnings: tuple[str, ...]


def _dependency_versions() -> dict[str, str]:
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


def _should_log_progress(index: int, total: int) -> bool:
    if total <= 10 or index in {1, total}:
        return True
    previous_bucket = ((index - 1) * 10) // total
    current_bucket = (index * 10) // total
    return current_bucket != previous_bucket


def _close_distance_backing(mapping: Any, temporary_file: Any) -> None:
    """Flush and close both layers of one disk-backed distance allocation."""

    try:
        mapping.flush()
    finally:
        try:
            mapping.close()
        finally:
            temporary_file.close()


def _estimated_memory_bytes(
    *,
    expression: np.ndarray,
    representation: np.ndarray,
    centroids: pd.DataFrame,
    cell_distances: pd.DataFrame | None,
    group_distances: pd.DataFrame,
    prepared_inference: Any,
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
        "validated_expression_float64": int(expression.nbytes),
        "distance_representation_float64": int(representation.nbytes),
        "centroids_float64": int(centroids.memory_usage(index=False, deep=False).sum()),
        "cell_centroid_distances_logical_float64": cell_distance_logical,
        "cell_centroid_distances_heap_float64": cell_distance_heap,
        "cell_centroid_distances_mapped_float64": cell_distance_mapped,
        "centroid_distances_float64": int(
            group_distances.memory_usage(index=False, deep=False).sum()
        ),
        # The pipeline supplies a contiguous transpose view of the validated
        # float64 input, so target responses do not allocate another full matrix.
        "tree_targets_additional_float64": 0,
        "tree_targets_logical_float64": int(prepared_inference.expression_nbytes),
        "tf_predictors_float32": int(prepared_inference.predictor_nbytes),
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


def _infer_group_specific_grns_impl(
    config: SpathiConfig,
    *,
    working_output_dir: Path | None = None,
) -> SpathiRunResult:
    """Infer and persist one weighted regulatory network per observed cell group.

    The function has no dependency on the CLI or ANDREA. It validates all three
    input contracts, calculates global distance information once, and streams bounded
    target-group batches into a new self-contained output directory.
    """

    if not isinstance(config, SpathiConfig):
        raise TypeError("config must be a SpathiConfig instance")
    if config.output_dir.exists():
        raise FileExistsError(
            f"Output path already exists and will not be overwritten: {config.output_dir}"
        )

    run_started_at = datetime.now(UTC)
    run_started = perf_counter()
    phase_times: dict[str, float] = {}
    warning_messages: list[str] = []
    available_threads = available_cpu_count()
    numeric_thread_limit = (
        available_threads if config.threads == -1 else min(config.threads, available_threads)
    )

    LOGGER.info("Validating expression, TF-list, and groups inputs")
    phase_started = perf_counter()
    inputs = load_inputs(config.expression, config.tf_list, config.groups)
    input_fingerprints = dict(inputs.input_fingerprints)
    phase_times["input_validation"] = perf_counter() - phase_started
    expression_values = inputs.expression.to_numpy(dtype=np.float64, copy=False)
    gene_names = list(map(str, inputs.expression.index))
    cell_names = list(map(str, inputs.expression.columns))
    tf_names = list(inputs.transcription_factors)
    group_ids = sorted(map(str, pd.unique(inputs.groups)))
    group_sizes = inputs.groups.astype(str).value_counts(sort=False).to_dict()
    expected_cell_distance_bytes = len(cell_names) * len(group_ids) * np.dtype(np.float64).itemsize
    cell_distance_storage = (
        "streamed-discarded" if config.weight_mode == "group-distance" else "memory"
    )
    cell_distance_output: np.ndarray | None = None
    distance_temporary_file: Any = None
    distance_storage_finalizer: Any = None
    if (
        config.weight_mode != "group-distance"
        and expected_cell_distance_bytes >= _DISTANCE_MEMMAP_THRESHOLD_BYTES
    ):
        distance_temporary_file = TemporaryFile(prefix="spathi-cell-distances-")
        distance_temporary_file.truncate(expected_cell_distance_bytes)
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
        cell_distance_storage = "temporary-memory-map"
        message = (
            "Using chunked, disk-backed storage for the "
            f"{expected_cell_distance_bytes / 1024**2:.1f} MiB cell-to-centroid matrix"
        )
        warning_messages.append(message)
        LOGGER.warning("%s", message)
    elif config.weight_mode != "group-distance":
        # Weights are consumed one target-group column at a time. Use the same
        # column-contiguous layout in RAM and in the disk-backed path.
        cell_distance_output = np.empty(
            (len(cell_names), len(group_ids)), dtype=np.float64, order="F"
        )
    if config.weight_mode != "group-distance" and expected_cell_distance_bytes > 2 * 1024**3:
        message = (
            "The cell-to-centroid distance matrix is expected to require "
            f"{expected_cell_distance_bytes / 1024**3:.2f} GiB before table overhead"
        )
        warning_messages.append(message)
        LOGGER.warning("%s", message)
    group_batch_size = min(
        len(group_ids), max(1, ceil(numeric_thread_limit / max(1, len(gene_names))))
    )

    LOGGER.info("Building the %s distance representation", config.distance_space)
    phase_started = perf_counter()
    with threadpool_limits(limits=numeric_thread_limit):
        representation = compute_distance_representation(
            inputs.expression,
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

    LOGGER.info("Computing reusable centroids and %s distances", config.distance_metric)
    phase_started = perf_counter()
    with threadpool_limits(limits=numeric_thread_limit):
        centroids = compute_centroids(
            representation,
            inputs.groups,
            group_order=group_ids,
        )
        group_distances = compute_centroid_distances(
            centroids,
            metric=config.distance_metric,
        )
        if config.weight_mode == "group-distance":
            # The scientific contract still requires cell-to-centroid distances
            # to be calculated once.  This mode never consumes the full matrix,
            # so discard bounded chunks instead of retaining O(cells * groups).
            for _ in iter_cell_to_centroid_distance_chunks(
                representation,
                centroids,
                metric=config.distance_metric,
                working_memory=DEFAULT_WORKING_MEMORY_MIB,
            ):
                pass
            cell_distances = None
        else:
            cell_distances = compute_cell_to_centroid_distances(
                representation,
                centroids,
                metric=config.distance_metric,
                working_memory=DEFAULT_WORKING_MEMORY_MIB,
                output=cell_distance_output,
            )
    phase_times["prototypes_and_distances"] = perf_counter() - phase_started

    phase_started = perf_counter()
    with threadpool_limits(limits=numeric_thread_limit):
        bandwidth = resolve_bandwidth_for_mode(
            config.weight_mode,
            cell_to_centroid_distances=(
                None
                if cell_distances is None
                else cell_distances.to_numpy(dtype=np.float64, copy=False)
            ),
            centroid_distances=group_distances.to_numpy(dtype=np.float64, copy=False),
            bandwidth=config.bandwidth,
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

    phase_started = perf_counter()
    prepared = prepare_inference(
        expression_values.T,
        gene_names,
        tf_names,
        tree_method=config.tree_method,
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        min_samples_leaf=config.min_samples_leaf,
        max_depth=config.max_depth,
        bootstrap=config.bootstrap,
        random_seed=config.random_seed,
    )
    phase_times["inference_preparation"] = perf_counter() - phase_started

    memory_estimate = _estimated_memory_bytes(
        expression=expression_values,
        representation=representation.values,
        centroids=centroids,
        cell_distances=cell_distances,
        group_distances=group_distances,
        prepared_inference=prepared,
        cell_distance_storage=cell_distance_storage,
    )
    memory_estimate["weight_batch_retained_float64"] = (
        len(cell_names) * group_batch_size * np.dtype(np.float64).itemsize
    )
    target_batch_size = min(
        len(gene_names),
        max(
            _TARGET_BATCH_MIN_MODELS,
            min(_TARGET_BATCH_MAX_MODELS, numeric_thread_limit * 4),
        ),
    )
    active_groups_per_inference_batch = (
        group_batch_size if target_batch_size == len(gene_names) else 1
    )
    models_per_inference_batch = target_batch_size * active_groups_per_inference_batch
    memory_estimate["distance_chunk_working_memory_upper_bound"] = int(
        DEFAULT_WORKING_MEMORY_MIB * 1024**2
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
    memory_estimate["weight_result_working_float64"] = int(
        4 * len(cell_names) * np.dtype(np.float64).itemsize
    )
    memory_estimate["group_positive_masks_bool"] = int(
        len(cell_names) * group_batch_size * np.dtype(np.bool_).itemsize
    )
    concurrent_fits = min(numeric_thread_limit, models_per_inference_batch)
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
    memory_estimate["rough_concurrent_tree_upper_bound"] = (
        concurrent_fits * config.n_estimators * max(1, 2 * len(cell_names) - 1) * 64
    )
    inference_temporary = sum(
        memory_estimate[key]
        for key in (
            "weight_batch_retained_float64",
            "weight_result_working_float64",
            "group_positive_masks_bool",
            "maximum_target_batch_edge_records_rough_bytes",
            "maximum_target_batch_model_records_rough_bytes",
            "concurrent_self_exclusion_predictors_float32",
        )
    )
    memory_estimate["estimated_peak_heap_before_tree_storage"] = int(
        memory_estimate["estimated_heap_core_array_total"]
        + max(
            memory_estimate["distance_chunk_working_memory_upper_bound"],
            inference_temporary,
        )
    )
    memory_estimate["estimated_peak_heap_with_rough_trees"] = int(
        memory_estimate["estimated_peak_heap_before_tree_storage"]
        + memory_estimate["rough_concurrent_tree_upper_bound"]
    )
    if memory_estimate["estimated_peak_heap_before_tree_storage"] > 2 * 1024**3:
        message = "Estimated non-tree heap peak exceeds 2 GiB; monitor memory during fitting"
        warning_messages.append(message)
        LOGGER.warning("%s", message)
    if memory_estimate["rough_concurrent_tree_upper_bound"] > 2 * 1024**3:
        message = (
            "A conservative concurrent-tree storage upper bound exceeds 2 GiB; actual tree "
            "memory depends on depth and leaf constraints"
        )
        warning_messages.append(message)
        LOGGER.warning("%s", message)

    if working_output_dir is None:
        output_dir = create_output_directory(config.output_dir)
    else:
        output_dir = working_output_dir
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise RuntimeError("internal staging output directory must exist and be empty")
    output_started = perf_counter()
    centroid_output = centroids.reset_index()
    write_tsv(centroid_output, output_dir / "centroids.tsv")
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
    write_json(config.to_dict(), output_dir / "parameters.json")
    phase_times["artifact_writing"] = perf_counter() - output_started

    weighting_seconds = 0.0
    inference_seconds = 0.0
    dynamic_writing_seconds = 0.0
    n_edges = 0
    completed_models = 0
    trained_models = 0
    skipped_models = 0
    model_status_counts: Counter[str] = Counter()
    parallel_plans: list[dict[str, Any]] = []

    with IncrementalRunWriter(output_dir) as writer:
        for batch_start in range(0, len(group_ids), group_batch_size):
            batch_groups = group_ids[batch_start : batch_start + group_batch_size]
            batch_weights: dict[object, ArrayLike] = {}
            for batch_offset, target_group in enumerate(batch_groups):
                index = batch_start + batch_offset + 1
                if _should_log_progress(index, len(group_ids)):
                    LOGGER.info(
                        "Preparing target group %d/%d: %s",
                        index,
                        len(group_ids),
                        target_group,
                    )

                phase_started = perf_counter()
                weights = compute_weights(
                    target_group,
                    inputs.groups,
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
                batch_weights[target_group] = weights.final_weight
                weighting_seconds += perf_counter() - phase_started

                phase_started = perf_counter()
                writer.write_weight_result(weights)
                writer.write_weight_diagnostics(diagnostics.iter_records())
                dynamic_writing_seconds += perf_counter() - phase_started

            inference_started = perf_counter()
            inference_batches = prepared.iter_group_target_batches(
                batch_weights,
                group_order=batch_groups,
                target_batch_size=target_batch_size,
                threads=config.threads,
            )
            for inference_result in inference_batches:
                inference_seconds += perf_counter() - inference_started
                completed_models += inference_result.completed_models
                trained_models += inference_result.trained_models
                skipped_models += len(inference_result.skipped_targets)
                model_status_counts.update(stat.status for stat in inference_result.model_stats)
                parallel_plans.append(inference_result.parallel_plan.to_dict())

                phase_started = perf_counter()
                n_edges += writer.write_edges(inference_result.edges)
                writer.write_skipped(inference_result.skipped_targets)
                writer.write_model_diagnostics(inference_result.model_stats)
                dynamic_writing_seconds += perf_counter() - phase_started
                inference_started = perf_counter()

    phase_times["weighting_and_diagnostics"] = weighting_seconds
    phase_times["model_inference"] = inference_seconds
    phase_times["artifact_writing"] += dynamic_writing_seconds
    if distance_storage_finalizer is not None and distance_storage_finalizer.alive:
        distance_storage_finalizer()

    requested_model_count = len(group_ids) * len(gene_names)
    if completed_models != requested_model_count:
        raise RuntimeError(
            f"Inference completed {completed_models} of {requested_model_count} requested models"
        )
    effective_threads = max(plan["effective_threads"] for plan in parallel_plans)
    backends = sorted({str(plan["backend"]) for plan in parallel_plans})
    parallel_levels = sorted({str(plan["parallel_level"]) for plan in parallel_plans})
    fatal_model_failures = (
        model_status_counts["model_fit_failed"] + model_status_counts["invalid_feature_importances"]
    )
    if fatal_model_failures:
        message = (
            f"{fatal_model_failures} model(s) failed during fitting or returned invalid "
            "feature importances"
        )
        warning_messages.append(message)
        LOGGER.error("%s", message)

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
            "targets": len(gene_names),
        },
        "inputs": input_fingerprints,
        "group_ids": group_ids,
        "group_sizes": {group: int(group_sizes[group]) for group in group_ids},
        "requested_parameters": config.to_dict(),
        "effective_parameters": {
            "prototype_method": "mean",
            "distance_space": representation.distance_space,
            "distance_standardization": representation.standardization,
            "pca_svd_solver_requested": representation.pca_svd_solver,
            "pca_svd_solver_effective": representation.effective_pca_svd_solver,
            "effective_n_components": representation.effective_n_components,
            "distance_metric": config.distance_metric,
            "cell_centroid_distance_storage": cell_distance_storage,
            "distance_chunk_working_memory_mib": DEFAULT_WORKING_MEMORY_MIB,
            "bandwidth": asdict(bandwidth),
            "tree_target_dtype": prepared.expression_dtype,
            "tree_predictor_dtype": prepared.predictor_dtype,
            "bootstrap_requested": config.bootstrap,
            "bootstrap_effective": prepared.bootstrap,
            "weight_dtype": "float64",
            "pca_degenerate": representation.pca_degenerate,
            "pca_degeneracy_reason": representation.pca_degeneracy_reason,
            "group_processing": "progressive-group-and-target-batches",
            "target_groups_per_batch": group_batch_size,
            "targets_per_batch": target_batch_size,
        },
        "random_seed": config.random_seed,
        "parallelism": {
            "threads_requested": config.threads,
            "threads_effective": effective_threads,
            "threads_available": available_threads,
            "preprocessing_thread_limit": numeric_thread_limit,
            "backends": backends,
            "parallel_levels": parallel_levels,
            "nested_parallelism": False,
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
                if status
                not in {
                    "trained",
                    "trained_no_positive_importance",
                    "model_fit_failed",
                    "invalid_feature_importances",
                }
            ),
            "fit_or_importance_failures": fatal_model_failures,
            "skipped_target_records": skipped_models,
            "positive_edges": n_edges,
        },
        "memory_estimate_bytes": memory_estimate,
        "dependency_versions": _dependency_versions(),
        "phase_times_seconds": phase_times,
        "warnings": warning_messages,
    }
    metadata_write_started = perf_counter()
    write_json(metadata, output_dir / "run_metadata.json")
    phase_times["artifact_writing"] += perf_counter() - metadata_write_started
    total_seconds = perf_counter() - run_started
    phase_times["total"] = total_seconds
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    write_json(metadata, output_dir / "run_metadata.json")
    LOGGER.info(
        "Finished %d models and %d positive edges in %.3f seconds",
        completed_models,
        n_edges,
        total_seconds,
    )
    return SpathiRunResult(
        output_dir=output_dir,
        network_path=output_dir / "network.csv",
        metadata_path=output_dir / "run_metadata.json",
        n_edges=n_edges,
        total_models=requested_model_count,
        trained_models=trained_models,
        skipped_models=skipped_models,
        failed_models=fatal_model_failures,
        warnings=tuple(warning_messages),
    )


def infer_group_specific_grns(config: SpathiConfig) -> SpathiRunResult:
    """Infer a run privately, then publish only a complete output directory.

    Scientific or I/O failures remove the private staging directory, so a user
    can retry the requested path immediately. Completed runs with model-fit
    failures are still published with ``status=failed`` and full diagnostics
    before the function raises an actionable error.
    """

    if not isinstance(config, SpathiConfig):
        raise TypeError("config must be a SpathiConfig instance")
    final_output_dir = config.output_dir
    if final_output_dir.exists():
        raise FileExistsError(
            f"Output path already exists and will not be overwritten: {final_output_dir}"
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{final_output_dir.name or 'spathi'}.staging-"
    with TemporaryDirectory(prefix=prefix, dir=final_output_dir.parent) as staging_name:
        staged_output_dir = Path(staging_name)
        staged_result = _infer_group_specific_grns_impl(
            config,
            working_output_dir=staged_output_dir,
        )
        if final_output_dir.exists():
            raise FileExistsError(
                f"Output path appeared during the run and will not be overwritten: "
                f"{final_output_dir}"
            )
        staged_output_dir.rename(final_output_dir)

    result = replace(
        staged_result,
        output_dir=final_output_dir,
        network_path=final_output_dir / "network.csv",
        metadata_path=final_output_dir / "run_metadata.json",
    )
    if result.failed_models:
        raise RuntimeError(
            f"SPATHI run failed because {result.failed_models} model(s) could not be fitted "
            f"safely; inspect {final_output_dir / 'model_diagnostics.tsv.gz'}"
        )
    return result


__all__ = ["SpathiRunResult", "infer_group_specific_grns"]
