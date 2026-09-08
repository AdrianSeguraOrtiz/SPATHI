"""Exact fixed-forest prefix studies built from ordinary SPATHI inputs.

This module exposes the scientific preparation needed by convergence studies
without changing the one-run/one-network contract of :func:`spathi.infer`.
Every logical estimator count is a prefix of one fitted maximum forest.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spathi._workflow import validate_group_configuration
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
from spathi.representation import RepresentationResult, compute_distance_representation
from spathi.weighting import WeightResult, compute_weights, prepare_weighting_context


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedForestPrefixes:
    """Reusable weighting and inference state for one exact prefix study."""

    prepared_inference: PreparedInference
    weight_results: tuple[WeightResult, ...]
    group_order: tuple[str, ...]
    estimator_counts: tuple[int, ...]
    bandwidth: BandwidthSelection
    representation: RepresentationResult
    threads: ThreadBudget

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
        """Yield all estimator prefixes in bounded canonical target batches."""

        yield from self.prepared_inference.iter_group_target_prefix_batches(
            self.group_weights,
            estimator_counts=self.estimator_counts,
            target_batch_size=target_batch_size,
            group_order=self.group_order,
            threads=self.threads if threads is None else threads,
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
    if config.adaptive_trees:
        raise ValueError("forest prefix studies require adaptive_trees=False")
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
        bootstrap=config.bootstrap,
        adaptive_trees=False,
        adaptive_min_estimators=config.adaptive_min_estimators,
        adaptive_tree_step=config.adaptive_tree_step,
        adaptive_tolerance=config.adaptive_tolerance,
        adaptive_patience=config.adaptive_patience,
        target_eligibility=config.target_eligibility,
        min_target_detected_cells=config.min_target_detected_cells,
        min_target_detected_fraction=config.min_target_detected_fraction,
        min_target_weighted_detected_fraction=config.min_target_weighted_detected_fraction,
        min_target_weighted_detected_ess=config.min_target_weighted_detected_ess,
        random_seed=config.random_seed,
    )
    return PreparedForestPrefixes(
        prepared_inference=prepared,
        weight_results=weight_results,
        group_order=groups,
        estimator_counts=counts,
        bandwidth=bandwidth,
        representation=representation,
        threads=config.threads,
    )


__all__ = ["PreparedForestPrefixes", "prepare_forest_prefixes"]
