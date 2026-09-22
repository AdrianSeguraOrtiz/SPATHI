"""Exact fixed-forest prefix studies built from ordinary SPATHI inputs.

This module exposes the scientific preparation needed by convergence studies
without changing the one-run/one-network contract of :func:`spathi.infer`.
Every logical estimator count is a prefix of one fitted maximum forest.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

from spathi._workflow import _plan_inference_execution, validate_group_configuration
from spathi.centroids import compute_centroids
from spathi.config import SpathiConfig, ThreadBudget
from spathi.distances import (
    compute_cell_to_centroid_distances,
    compute_centroid_distances,
)
from spathi.inference import (
    InferencePrefixResult,
    PreparedInference,
    prepare_inference,
)
from spathi.io import load_inputs
from spathi.kernels import BandwidthSelection, resolve_bandwidth_for_mode
from spathi.parallel import PersistentTaskExecutor, available_cpu_count
from spathi.representation import RepresentationResult, compute_distance_representation
from spathi.resources import estimate_model_memory_bytes
from spathi.weighting import WeightResult, compute_weights, prepare_weighting_context

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedForestPrefixes:
    """Reusable weighting and inference state for one exact prefix study."""

    prepared_inference: PreparedInference
    weight_results: tuple[WeightResult, ...]
    group_order: tuple[str, ...]
    estimator_counts: tuple[int, ...]
    bandwidth: BandwidthSelection
    representation: RepresentationResult
    config: SpathiConfig

    def __post_init__(self) -> None:
        if not self.group_order or len(set(self.group_order)) != len(self.group_order):
            raise ValueError("group_order must contain unique group identifiers")
        if tuple(result.target_group for result in self.weight_results) != self.group_order:
            raise ValueError("weight_results must occur exactly once in group_order")
        if not self.estimator_counts:
            raise ValueError("estimator_counts cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.estimator_counts
        ):
            raise ValueError("estimator_counts must contain only positive integers")
        if any(
            left >= right
            for left, right in zip(self.estimator_counts, self.estimator_counts[1:], strict=False)
        ):
            raise ValueError("estimator_counts must be strictly increasing")
        if self.estimator_counts[-1] != self.prepared_inference.n_estimators:
            raise ValueError("the final prefix must equal the prepared maximum forest")

    @property
    def group_weights(self) -> Mapping[Any, np.ndarray]:
        """Read-only mapping of group identifiers to final sample weights."""

        return MappingProxyType(
            {result.target_group: result.final_weight for result in self.weight_results}
        )

    def iter_batches(
        self,
        *,
        target_batch_size: int,
        threads: ThreadBudget | None = None,
    ) -> Iterator[tuple[InferencePrefixResult, ...]]:
        """Yield canonical prefix batches; the requested size is a memory-bounded maximum."""

        if type(target_batch_size) is not int or target_batch_size < 1:
            raise ValueError("target_batch_size must be a positive integer")
        config = self.config if threads is None else replace(self.config, threads=threads)
        prepared = self.prepared_inference
        model_bytes = estimate_model_memory_bytes(
            n_cells=prepared.n_cells,
            n_transcription_factors=len(prepared.tf_names),
            n_estimators=prepared.n_estimators,
            min_samples_leaf=prepared.min_samples_leaf,
            max_depth=prepared.max_depth,
            min_weight_fraction_leaf=prepared.min_weight_fraction_leaf,
        )
        model_bytes += prepared.n_estimators * len(prepared.tf_names) * 8
        plan = _plan_inference_execution(
            config=config,
            n_cells=prepared.n_cells,
            n_groups=len(self.group_order),
            n_targets=prepared.n_targets,
            n_transcription_factors=len(prepared.tf_names),
            predictor_bytes=prepared.predictor_nbytes,
            response_bytes=prepared.expression_nbytes,
            remaining_models=len(self.group_order) * prepared.n_targets,
            available_threads=available_cpu_count(),
            estimated_model_bytes=model_bytes,
            checkpoint_enabled=False,
            result_multiplier=len(self.estimator_counts),
        )
        batch_size = min(target_batch_size, plan.batch.target_batch_size)
        group_batch_size = plan.batch.group_batch_size if batch_size == prepared.n_targets else 1
        LOGGER.info(
            "Prefix inference backend %s; up to %d targets and %d groups per batch: %s",
            plan.parallel.backend,
            batch_size,
            group_batch_size,
            plan.backend_reason,
        )
        with PersistentTaskExecutor(
            plan.parallel,
            process_window_size=(
                batch_size * group_batch_size if plan.parallel.backend == "loky" else None
            ),
        ) as executor:
            group_weights = self.group_weights
            for start in range(0, len(self.group_order), group_batch_size):
                groups = self.group_order[start : start + group_batch_size]
                weights: dict[object, ArrayLike] = {group: group_weights[group] for group in groups}
                yield from prepared.iter_group_target_prefix_batches(
                    weights,
                    estimator_counts=self.estimator_counts,
                    target_batch_size=batch_size,
                    group_order=groups,
                    threads=config.threads,
                    executor=executor,
                )


def prepare_forest_prefixes(
    config: SpathiConfig,
    *,
    estimator_counts: Sequence[int],
) -> PreparedForestPrefixes:
    """Prepare exact fixed-forest prefixes from a normal SPATHI configuration.

    Input paths and scientific parameters have precisely the same meaning as in
    :func:`spathi.infer`. ``output_dir`` and ``report`` are intentionally not
    consumed because this function prepares in-memory scientific state and does
    not publish an ordinary SPATHI run. The caller chooses bounded output and
    batching policies when consuming :meth:`PreparedForestPrefixes.iter_batches`.
    """

    if not isinstance(config, SpathiConfig):
        raise TypeError("config must be a SpathiConfig instance")
    counts = tuple(estimator_counts)
    if not counts:
        raise ValueError("estimator_counts cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("estimator_counts must contain only positive integers")
    if any(value < 1 for value in counts):
        raise ValueError("estimator_counts must contain only positive integers")
    if any(left >= right for left, right in zip(counts, counts[1:], strict=False)):
        raise ValueError("estimator_counts must be strictly increasing")
    if counts[-1] != config.n_estimators:
        raise ValueError("the final estimator count must equal config.n_estimators")

    data = load_inputs(
        config.expression,
        config.tf_list,
        config.groups,
        config.target_list,
        config.centroid_weights,
    )
    expression = data.expression
    cells = tuple(map(str, expression.columns))
    genes = tuple(map(str, expression.index))
    groups = tuple(sorted(map(str, pd.unique(data.groups))))
    validate_group_configuration(config, group_count=len(groups))

    # Match ordinary inference: deterministic numerical preprocessing is kept
    # single-threaded and the configured budget is spent on independent models.
    with threadpool_limits(limits=1):
        representation = compute_distance_representation(
            expression,
            distance_space=config.distance_space,
            n_components=config.n_components,
            distance_standardization=config.distance_standardization,
            pca_svd_solver=config.pca_svd_solver,
            random_state=config.random_seed,
        )
        centroids = compute_centroids(
            representation,
            data.groups,
            group_order=groups,
            centroid_weights=data.centroid_weights,
        )
        group_distances = compute_centroid_distances(
            centroids,
            metric=config.distance_metric,
        )
        cell_distances = (
            None
            if config.weight_mode == "group-distance"
            else compute_cell_to_centroid_distances(
                representation,
                centroids,
                metric=config.distance_metric,
            )
        )
        bandwidth = resolve_bandwidth_for_mode(
            config.weight_mode,
            cell_to_centroid_distances=(
                None
                if cell_distances is None
                else cell_distances.to_numpy(dtype=np.float64, copy=False)
            ),
            centroid_distances=group_distances.to_numpy(dtype=np.float64, copy=False),
            bandwidth=config.bandwidth,
            bandwidth_scale=config.bandwidth_scale,
        )

    weighting_context = prepare_weighting_context(data.groups, cell_ids=cells)
    weight_results = tuple(
        compute_weights(
            group,
            weighting_context,
            mode=config.weight_mode,
            bandwidth=bandwidth,
            kernel=config.kernel,
            group_size_correction=config.group_size_correction,
            cell_distances=(
                None
                if cell_distances is None
                else cell_distances[group].to_numpy(dtype=np.float64, copy=False)
            ),
            group_distances=group_distances,
        )
        for group in groups
    )
    values = expression.to_numpy(dtype=np.float64, copy=False)
    prepared = prepare_inference(
        values.T,
        genes,
        data.transcription_factors,
        target_names=data.targets,
        tree_method=config.tree_method,
        n_estimators=config.n_estimators,
        max_features=config.max_features,
        min_samples_leaf=config.min_samples_leaf,
        max_depth=config.max_depth,
        min_weight_fraction_leaf=config.min_weight_fraction_leaf,
        bootstrap=config.bootstrap,
        random_seed=config.random_seed,
    )
    return PreparedForestPrefixes(
        prepared_inference=prepared,
        weight_results=weight_results,
        group_order=groups,
        estimator_counts=counts,
        bandwidth=bandwidth,
        representation=representation,
        config=config,
    )


__all__ = ["PreparedForestPrefixes", "prepare_forest_prefixes"]
