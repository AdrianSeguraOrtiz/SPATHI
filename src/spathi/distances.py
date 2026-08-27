"""Vectorized cell-to-centroid and centroid-to-centroid distances."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked

from .representation import RepresentationResult

DistanceMetric = Literal["euclidean", "cosine"]

# Do not inherit scikit-learn's process-wide default (currently 1 GiB).  The
# distance matrix may itself be disk-backed, so allowing an equally large
# transient chunk would defeat that memory-saving path.
DEFAULT_WORKING_MEMORY_MIB = 64.0


@dataclass(frozen=True, slots=True)
class DistanceChunk:
    """A contiguous block of cell-to-centroid distances."""

    start: int
    stop: int
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class DistanceMatrices:
    """Reusable distances for all target groups in one run."""

    cell_to_centroid: pd.DataFrame
    centroid_to_centroid: pd.DataFrame
    metric: DistanceMetric

    @property
    def cell_to_prototype(self) -> pd.DataFrame:
        """Prototype-neutral alias for :attr:`cell_to_centroid`."""

        return self.cell_to_centroid

    @property
    def prototype_to_prototype(self) -> pd.DataFrame:
        """Prototype-neutral alias for :attr:`centroid_to_centroid`."""

        return self.centroid_to_centroid


def _validate_metric(metric: str) -> DistanceMetric:
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    return metric  # type: ignore[return-value]


def _coerce_cells(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    cell_ids: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...] | None]:
    if isinstance(representation, RepresentationResult):
        values = representation.values
        cells = representation.cell_ids
        dimensions: tuple[str, ...] | None = representation.dimension_names
    elif isinstance(representation, pd.DataFrame):
        values = representation.to_numpy(dtype=np.float64, copy=False)
        cells = tuple(map(str, representation.index))
        dimensions = tuple(map(str, representation.columns))
    else:
        values = np.asarray(representation, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("representation must be a two-dimensional matrix")
        cells = (
            tuple(cell_ids)
            if cell_ids is not None
            else tuple(f"cell_{i}" for i in range(values.shape[0]))
        )
        dimensions = None
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("representation must contain at least one cell and one dimension")
    if len(cells) != values.shape[0] or len(set(cells)) != len(cells):
        raise ValueError("cell identifiers must be unique and match representation rows")
    if not np.isfinite(values).all():
        raise ValueError("representation must contain finite values")
    return np.asarray(values, dtype=np.float64, order="C"), tuple(map(str, cells)), dimensions


def _coerce_centroids(
    centroids: pd.DataFrame | np.ndarray,
    group_ids: Sequence[str] | None,
    dimensions: tuple[str, ...] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(centroids, pd.DataFrame):
        if dimensions is not None and tuple(map(str, centroids.columns)) != dimensions:
            raise ValueError(
                "centroid dimensions do not match representation dimensions in the same order"
            )
        values = centroids.to_numpy(dtype=np.float64, copy=False)
        groups = tuple(map(str, centroids.index))
    else:
        values = np.asarray(centroids, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("centroids must be a two-dimensional matrix")
        groups = (
            tuple(group_ids)
            if group_ids is not None
            else tuple(f"group_{i}" for i in range(values.shape[0]))
        )
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("centroids must contain at least one row and one dimension")
    if len(groups) != values.shape[0] or len(set(groups)) != len(groups):
        raise ValueError("group identifiers must be unique and match centroid rows")
    if not np.isfinite(values).all():
        raise ValueError("centroids must contain finite values")
    return np.asarray(values, dtype=np.float64, order="C"), tuple(map(str, groups))


def _cosine_roundoff_tolerance(n_dimensions: int) -> float:
    """Return a forward-error bound for a length-``n_dimensions`` dot product."""

    # Higham's gamma_n bounds accumulated floating-point error.  The additional
    # operations cover the two norms, division, and subtraction from one.  A
    # value below this bound cannot reliably be distinguished from exact
    # collinearity, while genuinely small distances above it remain untouched.
    operation_count = n_dimensions + 4
    scaled_epsilon = operation_count * np.finfo(np.float64).eps
    if scaled_epsilon >= 1.0:  # Defensive: impossible for practical matrices.
        return float(np.finfo(np.float64).eps)
    return float(scaled_epsilon / (1.0 - scaled_epsilon))


def _prepare_cosine_rows(
    values: np.ndarray,
    labels: Sequence[str],
    *,
    kind: str,
) -> np.ndarray:
    """Reject zero vectors and rescale exceptional magnitudes before cosine."""

    # max(abs(row)) is a robust zero-norm test even when squaring tiny finite
    # values would underflow.  Work in blocks to avoid a matrix-sized temporary.
    row_maxima = np.empty(values.shape[0], dtype=np.float64)
    block_size = 65_536
    for start in range(0, values.shape[0], block_size):
        stop = min(start + block_size, values.shape[0])
        row_maxima[start:stop] = np.max(np.abs(values[start:stop]), axis=1)

    zero_rows = np.flatnonzero(row_maxima == 0.0)
    if zero_rows.size:
        preview = ", ".join(repr(str(labels[index])) for index in zero_rows[:5])
        remainder = int(zero_rows.size) - 5
        suffix = f", and {remainder} more" if remainder > 0 else ""
        raise ValueError(
            f"cosine distance is undefined for zero-norm {kind} rows: {preview}{suffix}; "
            "use euclidean distance or remove/transform zero vectors"
        )

    # sklearn computes row norms from sums of squares.  Only rescale rows whose
    # magnitudes could underflow or overflow that operation; cosine is invariant
    # to positive row scaling, and ordinary inputs incur no matrix copy here.
    safe_minimum = np.sqrt(np.finfo(np.float64).tiny)
    safe_maximum = np.sqrt(np.finfo(np.float64).max / values.shape[1])
    exceptional = (row_maxima < safe_minimum) | (row_maxima > safe_maximum)
    if not np.any(exceptional):
        return values
    prepared = np.array(values, dtype=np.float64, order="C", copy=True)
    prepared[exceptional] /= row_maxima[exceptional, np.newaxis]
    return prepared


def _clean_distances(
    values: np.ndarray,
    *,
    metric: DistanceMetric,
    n_dimensions: int,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("distance calculation produced non-finite values")
    # Roundoff can create values a few ulps below zero for some metrics.
    np.maximum(result, 0.0, out=result)
    if metric == "cosine":
        tolerance = _cosine_roundoff_tolerance(n_dimensions)
        result[result <= tolerance] = 0.0
    return result


def iter_cell_to_centroid_distance_chunks(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    centroids: pd.DataFrame | np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    working_memory: float | None = None,
    cell_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
) -> Iterator[DistanceChunk]:
    """Yield cell-to-centroid distances in memory-bounded row chunks.

    ``working_memory`` follows scikit-learn's convention and is expressed in
    MiB.  ``None`` selects SPATHI's bounded 64 MiB default rather than
    scikit-learn's process-wide setting. Consumers that write long-form cell
    weights progressively can use this iterator without materializing another
    full distance matrix.
    """

    validated_metric = _validate_metric(metric)
    if working_memory is not None and (not np.isfinite(working_memory) or working_memory <= 0):
        raise ValueError("working_memory must be a positive number of MiB")
    cell_values, cells, dimensions = _coerce_cells(representation, cell_ids)
    centroid_values, groups = _coerce_centroids(centroids, group_ids, dimensions)
    if cell_values.shape[1] != centroid_values.shape[1]:
        raise ValueError("representation and centroids must have the same number of dimensions")
    if validated_metric == "cosine":
        cell_values = _prepare_cosine_rows(cell_values, cells, kind="representation")
        centroid_values = _prepare_cosine_rows(centroid_values, groups, kind="centroid")

    start = 0
    chunks = pairwise_distances_chunked(
        cell_values,
        centroid_values,
        metric=validated_metric,
        working_memory=(DEFAULT_WORKING_MEMORY_MIB if working_memory is None else working_memory),
    )
    for values in chunks:
        clean = _clean_distances(
            values,
            metric=validated_metric,
            n_dimensions=cell_values.shape[1],
        )
        stop = start + clean.shape[0]
        yield DistanceChunk(start=start, stop=stop, values=clean)
        start = stop


def compute_cell_to_centroid_distances(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    centroids: pd.DataFrame | np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    working_memory: float | None = None,
    cell_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
    output: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate all cell-to-centroid distances once, using vectorized chunks.

    ``output`` may be a writable float64 ndarray or ``numpy.memmap`` with the
    exact result shape. Large pipelines can therefore keep the reusable matrix
    disk-backed while retaining this labelled interface.
    """

    validated_metric = _validate_metric(metric)
    cell_values, cells, dimensions = _coerce_cells(representation, cell_ids)
    centroid_values, groups = _coerce_centroids(centroids, group_ids, dimensions)
    if cell_values.shape[1] != centroid_values.shape[1]:
        raise ValueError("representation and centroids must have the same number of dimensions")

    expected_shape = (cell_values.shape[0], centroid_values.shape[0])
    if output is None:
        # Downstream weighting consumes one target-centroid column at a time.
        # Column-contiguous storage avoids a strided scan for every group.
        result = np.empty(expected_shape, dtype=np.float64, order="F")
    else:
        result = np.asarray(output)
        if result.shape != expected_shape:
            raise ValueError(
                f"output must have shape {expected_shape!r}; received {result.shape!r}"
            )
        if result.dtype != np.float64:
            raise ValueError("output must have dtype float64")
        if not result.flags.writeable:
            raise ValueError("output must be writable")
    for chunk in iter_cell_to_centroid_distance_chunks(
        cell_values,
        centroid_values,
        metric=validated_metric,
        working_memory=working_memory,
        cell_ids=cells,
        group_ids=groups,
    ):
        result[chunk.start : chunk.stop] = chunk.values
    return pd.DataFrame(
        result,
        index=pd.Index(cells, name="cell"),
        columns=pd.Index(groups, name="group"),
        copy=False,
    )


def compute_centroid_distances(
    centroids: pd.DataFrame | np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    group_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate the symmetric centroid-to-centroid distance matrix."""

    validated_metric = _validate_metric(metric)
    centroid_values, groups = _coerce_centroids(centroids, group_ids, dimensions=None)
    if validated_metric == "cosine":
        centroid_values = _prepare_cosine_rows(centroid_values, groups, kind="centroid")
    values = pairwise_distances(centroid_values, metric=validated_metric)
    values = _clean_distances(
        values,
        metric=validated_metric,
        n_dimensions=centroid_values.shape[1],
    )
    # Both supported metrics are symmetric; make exact symmetry/zero diagonal
    # explicit for deterministic TSV output despite floating-point roundoff.
    values = (values + values.T) * 0.5
    np.fill_diagonal(values, 0.0)
    index = pd.Index(groups, name="target_group")
    columns = pd.Index(groups, name="source_group")
    return pd.DataFrame(values, index=index, columns=columns)


def compute_distance_matrices(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    centroids: pd.DataFrame | np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    working_memory: float | None = None,
    cell_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
    cell_distance_output: np.ndarray | None = None,
) -> DistanceMatrices:
    """Compute the two reusable distance matrices required by all modes."""

    cell_distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric=metric,
        working_memory=working_memory,
        cell_ids=cell_ids,
        group_ids=group_ids,
        output=cell_distance_output,
    )
    centroid_distances = compute_centroid_distances(centroids, metric=metric, group_ids=group_ids)
    return DistanceMatrices(
        cell_to_centroid=cell_distances,
        centroid_to_centroid=centroid_distances,
        metric=metric,
    )


# Prototype-neutral and concise aliases.
compute_cell_to_prototype_distances = compute_cell_to_centroid_distances
compute_prototype_distances = compute_centroid_distances
compute_distances = compute_distance_matrices


__all__ = [
    "DEFAULT_WORKING_MEMORY_MIB",
    "DistanceChunk",
    "DistanceMatrices",
    "DistanceMetric",
    "compute_cell_to_centroid_distances",
    "compute_cell_to_prototype_distances",
    "compute_centroid_distances",
    "compute_distance_matrices",
    "compute_distances",
    "compute_prototype_distances",
    "iter_cell_to_centroid_distance_chunks",
]
