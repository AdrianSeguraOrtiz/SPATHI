"""Vectorized cell-to-centroid and centroid-to-centroid distances."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked

from .config import DistanceMetric
from .representation import RepresentationResult

# Do not inherit scikit-learn's process-wide default (currently 1 GiB).  The
# distance matrix may itself be disk-backed, so allowing an equally large
# transient chunk would defeat that memory-saving path.
DEFAULT_WORKING_MEMORY_MIB = 64.0


@dataclass(frozen=True, slots=True, kw_only=True)
class _DistanceChunk:
    """A contiguous block of cell-to-centroid distances."""

    start: int
    stop: int
    values: np.ndarray


def _validate_metric(metric: str) -> DistanceMetric:
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    return metric  # type: ignore[return-value]


def _coerce_cells(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    cell_ids: Sequence[str] | None,
    *,
    working_memory: float,
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
    if not _all_finite(values, working_memory=working_memory):
        raise ValueError("representation must contain finite values")
    return np.asarray(values, dtype=np.float64), tuple(map(str, cells)), dimensions


def _coerce_centroids(
    centroids: pd.DataFrame | np.ndarray,
    group_ids: Sequence[str] | None,
    dimensions: tuple[str, ...] | None,
    *,
    working_memory: float,
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
    if not _all_finite(values, working_memory=working_memory):
        raise ValueError("centroids must contain finite values")
    return np.asarray(values, dtype=np.float64), tuple(map(str, groups))


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


def _normalize_cosine_rows(
    values: np.ndarray,
    labels: Sequence[str],
    *,
    kind: str,
    working_memory: float,
) -> np.ndarray:
    """Return robust unit rows without an unbounded whole-matrix temporary."""

    # Scaling every row by its maximum before the Euclidean norm prevents both
    # underflow and overflow. Callers pass either the small centroid matrix or a
    # working-memory-bounded cell chunk, so this one copy is always planned.
    normalized = np.array(values, dtype=np.float64, order="K", copy=True)
    block_size = _temporary_block_rows(
        n_rows=values.shape[0],
        n_dimensions=values.shape[1],
        working_memory=working_memory,
    )
    zero_count = 0
    zero_preview: list[str] = []
    for start in range(0, values.shape[0], block_size):
        stop = min(start + block_size, values.shape[0])
        block = normalized[start:stop]
        row_maxima = np.max(np.abs(block), axis=1)
        zero_rows = np.flatnonzero(row_maxima == 0.0)
        zero_count += int(zero_rows.size)
        for index in zero_rows:
            if len(zero_preview) == 5:
                break
            zero_preview.append(str(labels[start + int(index)]))
        positive = row_maxima > 0.0
        np.divide(
            block,
            row_maxima[:, np.newaxis],
            out=block,
            where=positive[:, np.newaxis],
        )
        norms = np.linalg.norm(block, axis=1)
        np.divide(
            block,
            norms[:, np.newaxis],
            out=block,
            where=positive[:, np.newaxis],
        )

    if zero_count:
        preview = ", ".join(repr(label) for label in zero_preview)
        remainder = zero_count - len(zero_preview)
        suffix = f", and {remainder} more" if remainder > 0 else ""
        raise ValueError(
            f"cosine distance is undefined for zero-norm {kind} rows: {preview}{suffix}; "
            "use euclidean distance or remove/transform zero vectors"
        )
    return normalized


def _clean_distances(
    values: np.ndarray,
    *,
    metric: DistanceMetric,
    n_dimensions: int,
    working_memory: float,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    block_size = _temporary_block_rows(
        n_rows=result.shape[0],
        n_dimensions=result.shape[1],
        working_memory=working_memory,
    )
    tolerance = _cosine_roundoff_tolerance(n_dimensions) if metric == "cosine" else None
    for start in range(0, result.shape[0], block_size):
        block = result[start : start + block_size]
        if not np.isfinite(block).all():
            raise ValueError("distance calculation produced non-finite values")
        # Roundoff can create values a few ulps below zero for either metric.
        np.maximum(block, 0.0, out=block)
        if tolerance is not None:
            # Cosine distance is mathematically bounded by two. Antiparallel
            # vectors can exceed that bound by a few ulps after the dot product.
            np.minimum(block, 2.0, out=block)
            block[block <= tolerance] = 0.0
    return result


def _validate_working_memory(working_memory: float | None) -> float:
    if working_memory is not None and (not np.isfinite(working_memory) or working_memory <= 0):
        raise ValueError("working_memory must be a positive number of MiB")
    return DEFAULT_WORKING_MEMORY_MIB if working_memory is None else float(working_memory)


def _temporary_block_rows(
    *,
    n_rows: int,
    n_dimensions: int,
    working_memory: float,
) -> int:
    """Bound validation temporaries by the same budget as distance chunks."""

    # ``abs`` retains one float64 block while comparisons/reductions can retain
    # boolean temporaries. Sixteen bytes per input value is conservative for
    # both numeric prechecks below. Reject a budget that cannot safely process
    # even one row instead of silently exceeding the declared upper bound.
    bytes_per_row = max(1, n_dimensions * 16)
    memory_bytes = max(1, int(working_memory * 1024**2))
    if memory_bytes < bytes_per_row:
        raise MemoryError("working_memory cannot hold one bounded numeric validation row")
    return min(n_rows, max(1, memory_bytes // bytes_per_row))


def _all_finite(values: np.ndarray, *, working_memory: float) -> bool:
    """Validate a matrix without allocating a matrix-sized boolean result."""

    block_size = _temporary_block_rows(
        n_rows=values.shape[0],
        n_dimensions=values.shape[1],
        working_memory=working_memory,
    )
    return all(
        np.isfinite(values[start : start + block_size]).all()
        for start in range(0, values.shape[0], block_size)
    )


def _iter_prepared_distance_chunks(
    cell_values: np.ndarray,
    centroid_values: np.ndarray,
    *,
    metric: DistanceMetric,
    working_memory: float,
) -> Iterator[_DistanceChunk]:
    total_working_bytes = max(1, int(working_memory * 1024**2))
    distance_row_bytes = centroid_values.shape[0] * np.dtype(np.float64).itemsize
    cleaning_row_bytes = centroid_values.shape[0] * 16
    if total_working_bytes < distance_row_bytes + cleaning_row_bytes:
        raise MemoryError(
            "working_memory cannot hold one distance row and its bounded validation block"
        )
    # Scikit-learn's budget covers the float64 distance chunk. Reserve enough
    # of the caller's total budget to validate at least one result row while
    # that chunk remains live.
    pairwise_working_bytes = total_working_bytes - cleaning_row_bytes
    start = 0
    chunks = pairwise_distances_chunked(
        cell_values,
        centroid_values,
        metric=metric,
        working_memory=pairwise_working_bytes / 1024**2,
    )
    for values in chunks:
        cleaning_working_bytes = total_working_bytes - values.nbytes
        if cleaning_working_bytes < cleaning_row_bytes:  # pragma: no cover - sklearn guard
            raise MemoryError("distance backend exceeded the configured working-memory budget")
        clean = _clean_distances(
            values,
            metric=metric,
            n_dimensions=cell_values.shape[1],
            working_memory=cleaning_working_bytes / 1024**2,
        )
        stop = start + clean.shape[0]
        yield _DistanceChunk(start=start, stop=stop, values=clean)
        start = stop


def _iter_cosine_distance_chunks(
    cell_values: np.ndarray,
    centroid_values: np.ndarray,
    *,
    cell_labels: Sequence[str],
    centroid_labels: Sequence[str],
    working_memory: float,
) -> Iterator[_DistanceChunk]:
    """Yield robust cosine distances with every full matrix bounded or planned."""

    n_dimensions = cell_values.shape[1]
    n_centroids = centroid_values.shape[0]
    total_working_bytes = max(1, int(working_memory * 1024**2))
    normalized_centroid_bytes = centroid_values.size * np.dtype(np.float64).itemsize
    minimum_validation_bytes = max(1, n_dimensions * 16)
    remaining_bytes = total_working_bytes - normalized_centroid_bytes
    if remaining_bytes < minimum_validation_bytes:
        raise MemoryError(
            "working_memory cannot hold normalized centroids and one cosine validation block"
        )
    normalized_centroids = _normalize_cosine_rows(
        centroid_values,
        centroid_labels,
        kind="centroid",
        working_memory=remaining_bytes / 1024**2,
    )

    normalized_cell_bytes_per_row = n_dimensions * np.dtype(np.float64).itemsize
    distance_bytes_per_row = n_centroids * np.dtype(np.float64).itemsize
    cleaning_row_bytes = n_centroids * 16
    rows_for_normalization = (
        remaining_bytes - minimum_validation_bytes
    ) // normalized_cell_bytes_per_row
    rows_for_distances = (remaining_bytes - cleaning_row_bytes) // (
        normalized_cell_bytes_per_row + distance_bytes_per_row
    )
    rows_per_chunk = min(cell_values.shape[0], rows_for_normalization, rows_for_distances)
    if rows_per_chunk < 1:
        raise MemoryError(
            "working_memory cannot hold normalized centroids and one cosine distance row"
        )

    for start in range(0, cell_values.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, cell_values.shape[0])
        chunk = cell_values[start:stop]
        normalized_cell_bytes = chunk.size * np.dtype(np.float64).itemsize
        normalized_cells = _normalize_cosine_rows(
            chunk,
            cell_labels[start:stop],
            kind="representation",
            working_memory=(remaining_bytes - normalized_cell_bytes) / 1024**2,
        )
        distances = normalized_cells @ normalized_centroids.T
        np.subtract(1.0, distances, out=distances)
        cleaning_working_bytes = remaining_bytes - normalized_cell_bytes - distances.nbytes
        yield _DistanceChunk(
            start=start,
            stop=stop,
            values=_clean_distances(
                distances,
                metric="cosine",
                n_dimensions=n_dimensions,
                working_memory=cleaning_working_bytes / 1024**2,
            ),
        )


def _requires_stable_euclidean_path(
    *arrays: np.ndarray,
    working_memory: float,
) -> bool:
    """Return whether squared norms can lose finite Euclidean distances."""

    n_dimensions = arrays[0].shape[1]
    safe_minimum = np.sqrt(np.finfo(np.float64).tiny)
    # Two valid coordinates can have opposite signs, so their difference can
    # be twice either magnitude.  Keep the squared norm of that worst-case
    # difference within float64 whenever the fast path is selected.
    safe_maximum = np.sqrt(np.finfo(np.float64).max / n_dimensions) / 2.0
    for values in arrays:
        block_size = _temporary_block_rows(
            n_rows=values.shape[0],
            n_dimensions=values.shape[1],
            working_memory=working_memory,
        )
        for start in range(0, values.shape[0], block_size):
            stop = min(start + block_size, values.shape[0])
            absolute = np.abs(values[start:stop])
            if np.any((absolute != 0.0) & (absolute < safe_minimum)) or np.any(
                absolute > safe_maximum
            ):
                return True
    return False


def _iter_stable_euclidean_chunks(
    cell_values: np.ndarray,
    centroid_values: np.ndarray,
    *,
    working_memory: float,
) -> Iterator[_DistanceChunk]:
    """Yield Euclidean distances without forming squared norms.

    ``numpy.hypot.reduce`` scales intermediate values, preserving distances
    whose squares underflow or overflow even though the distances themselves
    are representable.  Each centroid is handled separately so the temporary
    storage remains linear rather than cell x centroid x dimension.
    """

    n_dimensions = cell_values.shape[1]
    n_centroids = centroid_values.shape[0]
    total_working_bytes = max(1, int(working_memory * 1024**2))
    retained_bytes_per_row = np.dtype(np.float64).itemsize * (n_dimensions + n_centroids)
    cleaning_row_bytes = n_centroids * 16
    rows_per_chunk = min(
        cell_values.shape[0],
        (total_working_bytes - cleaning_row_bytes) // retained_bytes_per_row,
    )
    if rows_per_chunk < 1:
        raise MemoryError(
            "working_memory cannot hold one stable Euclidean row and its validation block"
        )

    for start in range(0, cell_values.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, cell_values.shape[0])
        cells = cell_values[start:stop]
        difference = np.empty_like(cells, dtype=np.float64, order="K")
        distances = np.empty((stop - start, n_centroids), dtype=np.float64, order="F")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for centroid_index, centroid in enumerate(centroid_values):
                np.subtract(cells, centroid, out=difference)
                np.hypot.reduce(
                    difference,
                    axis=1,
                    out=distances[:, centroid_index],
                )
        yield _DistanceChunk(
            start=start,
            stop=stop,
            values=_clean_distances(
                distances,
                metric="euclidean",
                n_dimensions=n_dimensions,
                working_memory=(total_working_bytes - difference.nbytes - distances.nbytes)
                / 1024**2,
            ),
        )


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
    memory_limit = _validate_working_memory(working_memory)
    cell_values, cells, dimensions = _coerce_cells(
        representation,
        cell_ids,
        working_memory=memory_limit,
    )
    centroid_values, groups = _coerce_centroids(
        centroids,
        group_ids,
        dimensions,
        working_memory=memory_limit,
    )
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
    if validated_metric == "cosine":
        chunks = _iter_cosine_distance_chunks(
            cell_values,
            centroid_values,
            cell_labels=cells,
            centroid_labels=groups,
            working_memory=memory_limit,
        )
    elif _requires_stable_euclidean_path(
        cell_values,
        centroid_values,
        working_memory=memory_limit,
    ):
        chunks = _iter_stable_euclidean_chunks(
            cell_values,
            centroid_values,
            working_memory=memory_limit,
        )
    else:
        chunks = _iter_prepared_distance_chunks(
            cell_values,
            centroid_values,
            metric=validated_metric,
            working_memory=memory_limit,
        )
    for chunk in chunks:
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
    working_memory: float | None = None,
) -> pd.DataFrame:
    """Calculate the symmetric centroid-to-centroid distance matrix."""

    validated_metric = _validate_metric(metric)
    memory_limit = _validate_working_memory(working_memory)
    centroid_values, groups = _coerce_centroids(
        centroids,
        group_ids,
        dimensions=None,
        working_memory=memory_limit,
    )
    if validated_metric == "cosine":
        normalized_centroids = _normalize_cosine_rows(
            centroid_values,
            groups,
            kind="centroid",
            working_memory=memory_limit,
        )
        values = normalized_centroids @ normalized_centroids.T
        np.subtract(1.0, values, out=values)
    elif _requires_stable_euclidean_path(
        centroid_values,
        working_memory=memory_limit,
    ):
        values = np.empty(
            (centroid_values.shape[0], centroid_values.shape[0]),
            dtype=np.float64,
            order="F",
        )
        for chunk in _iter_stable_euclidean_chunks(
            centroid_values,
            centroid_values,
            working_memory=memory_limit,
        ):
            values[chunk.start : chunk.stop] = chunk.values
    else:
        values = pairwise_distances(centroid_values, metric=validated_metric)
    values = _clean_distances(
        values,
        metric=validated_metric,
        n_dimensions=centroid_values.shape[1],
        working_memory=memory_limit,
    )
    # Both supported metrics are symmetric; make exact symmetry/zero diagonal
    # explicit for deterministic TSV output despite floating-point roundoff.
    # Copying one triangle avoids losing the smallest subnormal distance to an
    # otherwise harmless-looking multiplication or average.
    for row_index in range(values.shape[0]):
        values[row_index + 1 :, row_index] = values[row_index, row_index + 1 :]
    np.fill_diagonal(values, 0.0)
    index = pd.Index(groups, name="target_group")
    columns = pd.Index(groups, name="source_group")
    return pd.DataFrame(values, index=index, columns=columns)


__all__ = [
    "DEFAULT_WORKING_MEMORY_MIB",
    "DistanceMetric",
    "compute_cell_to_centroid_distances",
    "compute_centroid_distances",
]
