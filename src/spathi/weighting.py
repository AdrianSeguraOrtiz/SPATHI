"""The three exact SPATHI distance-to-observation weighting schemes."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GroupSizeCorrection, WeightMode
from .kernels import BandwidthSelection, apply_kernel


class DegenerateWeightsError(ValueError):
    """Raised when no positive sample weight remains for model fitting."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightingContext:
    """Validated cell/group structure shared by every target group in a run."""

    cells: tuple[str, ...]
    cell_groups: tuple[str, ...]
    group_ids: tuple[str, ...]
    group_codes: np.ndarray
    group_counts: np.ndarray

    def __post_init__(self) -> None:
        n_cells = len(self.cells)
        n_groups = len(self.group_ids)
        if n_cells == 0 or len(self.cell_groups) != n_cells:
            raise ValueError("weighting context must contain aligned, non-empty cells and groups")
        if any(
            not isinstance(cell, str) or not cell.strip() or cell != cell.strip()
            for cell in self.cells
        ):
            raise ValueError("weighting context cell identifiers must be non-empty trimmed strings")
        if len(set(self.cells)) != n_cells:
            raise ValueError("weighting context cell identifiers must be unique")
        if (
            n_groups == 0
            or any(
                not isinstance(group, str) or not group.strip() or group != group.strip()
                for group in self.group_ids
            )
            or len(set(self.group_ids)) != n_groups
        ):
            raise ValueError("weighting context group identifiers must be unique and non-empty")
        codes = np.asarray(self.group_codes, dtype=np.intp)
        counts = np.asarray(self.group_counts, dtype=np.int64)
        if codes.shape != (n_cells,) or counts.shape != (n_groups,):
            raise ValueError("weighting context codes and counts do not match its identifiers")
        if np.any(codes < 0) or np.any(codes >= n_groups):
            raise ValueError("weighting context contains an invalid group code")
        if np.any(counts <= 0) or not np.array_equal(
            counts, np.bincount(codes, minlength=n_groups)
        ):
            raise ValueError("weighting context group counts are inconsistent with its codes")
        canonical_groups = tuple(self.group_ids[int(code)] for code in codes)
        if canonical_groups != self.cell_groups:
            raise ValueError("weighting context group labels are inconsistent with its codes")
        # Defensive private copies make the frozen context truly immutable even
        # when a caller retains references to arrays supplied to the constructor.
        codes = codes.copy()
        counts = counts.copy()
        codes.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "group_codes", codes)
        object.__setattr__(self, "group_counts", counts)


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightResult:
    """All interpretable stages of one target group's sample weights.

    ``group_size_factor`` contains only the requested multiplicity correction.
    For ``cell-distance``, ``normalization_factor`` records the additional
    scalar applied after that correction to make the maximum final weight one.
    The other modes use a normalization factor of one so target cells remain
    anchored exactly at one.
    """

    context: WeightingContext
    target_group: str
    distance: np.ndarray
    base_weight: np.ndarray
    group_size_factor: np.ndarray
    final_weight: np.ndarray
    mode: WeightMode
    normalization_factor: float = 1.0

    def __post_init__(self) -> None:
        expected_shape = (len(self.context.cells),)
        vectors = {
            "distance": self.distance,
            "base_weight": self.base_weight,
            "group_size_factor": self.group_size_factor,
            "final_weight": self.final_weight,
        }
        invalid_shapes = {
            name: np.asarray(values).shape
            for name, values in vectors.items()
            if np.asarray(values).shape != expected_shape
        }
        if invalid_shapes:
            raise ValueError(
                f"all WeightResult vectors must have shape {expected_shape!r}; "
                f"received {invalid_shapes!r}"
            )
        if self.target_group not in self.context.group_ids:
            raise ValueError(f"target group {self.target_group!r} has no cells")

    @property
    def cells(self) -> tuple[str, ...]:
        """Cell identifiers shared by every target-group result in the run."""

        return self.context.cells

    @property
    def cell_groups(self) -> tuple[str, ...]:
        """Canonical group label for each cell in :attr:`cells`."""

        return self.context.cell_groups


def _validate_mode(mode: str) -> WeightMode:
    valid = {"cell-distance", "cell-distance-group-anchored", "group-distance"}
    if mode not in valid:
        raise ValueError(f"mode must be one of {sorted(valid)!r}")
    return mode  # type: ignore[return-value]


def _validate_correction(correction: str) -> GroupSizeCorrection:
    if correction not in {"none", "cap-to-target"}:
        raise ValueError("group_size_correction must be 'none' or 'cap-to-target'")
    return correction  # type: ignore[return-value]


def _effective_bandwidth(bandwidth: float | BandwidthSelection) -> float:
    value = bandwidth.value if isinstance(bandwidth, BandwidthSelection) else float(bandwidth)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("bandwidth must be a positive finite number")
    return value


def _is_missing_group_label(value: object) -> bool:
    """Return whether one scalar group label represents a missing value."""

    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _stringify_group_labels(values: Sequence[object], *, field_name: str) -> np.ndarray:
    """Validate group labels before converting them to their canonical strings."""

    raw_values = list(values)
    if any(_is_missing_group_label(value) for value in raw_values):
        raise ValueError(f"{field_name} contains a missing or empty group")
    labels = np.asarray([str(value) for value in raw_values], dtype=str)
    if any(not value.strip() for value in labels):
        raise ValueError(f"{field_name} contains a missing or empty group")
    return labels


def _stringify_target_group(target_group: Hashable) -> str:
    """Validate a target-group scalar before converting it to text."""

    if _is_missing_group_label(target_group):
        raise ValueError("target_group must be a non-missing, non-empty group label")
    target = str(target_group)
    if not target.strip():
        raise ValueError("target_group must be a non-missing, non-empty group label")
    return target


def _coerce_groups_and_cells(
    cell_groups: pd.Series | Sequence[Hashable],
    cell_ids: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(cell_groups, pd.Series):
        if cell_groups.index.has_duplicates:
            raise ValueError("cell_groups has duplicate cell identifiers")
        inferred_cells = tuple(map(str, cell_groups.index))
        if cell_ids is not None:
            requested_cells = tuple(map(str, cell_ids))
            if set(requested_cells) != set(inferred_cells) or len(requested_cells) != len(
                inferred_cells
            ):
                raise ValueError(
                    "cell_ids and cell_groups index must contain the same unique cells"
                )
            aligned = cell_groups.copy()
            aligned.index = pd.Index(inferred_cells)
            raw_groups = aligned.loc[list(requested_cells)].to_numpy(dtype=object).tolist()
            cells = requested_cells
        else:
            raw_groups = cell_groups.to_numpy(dtype=object).tolist()
            cells = inferred_cells
    else:
        raw_groups = list(cell_groups)
        cells = (
            tuple(map(str, cell_ids))
            if cell_ids is not None
            else tuple(f"cell_{i}" for i in range(len(raw_groups)))
        )

    groups = _stringify_group_labels(raw_groups, field_name="cell_groups")
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("cell_groups must be a non-empty one-dimensional vector")
    if len(cells) != groups.size or len(set(cells)) != len(cells):
        raise ValueError("cell identifiers must be unique and match cell_groups")
    return groups, cells


def prepare_weighting_context(
    cell_groups: pd.Series | Sequence[Hashable],
    *,
    cell_ids: Sequence[str] | None = None,
) -> WeightingContext:
    """Validate and encode the cell/group structure once for the complete run."""

    labels, cells = _coerce_groups_and_cells(cell_groups, cell_ids)
    codes, unique_groups = pd.factorize(labels, sort=False)
    group_ids = tuple(str(group) for group in unique_groups.tolist())
    counts = np.bincount(codes, minlength=len(group_ids))
    return WeightingContext(
        cells=cells,
        cell_groups=tuple(labels.tolist()),
        group_ids=group_ids,
        group_codes=codes,
        group_counts=counts,
    )


def _compute_group_size_factors(
    context: WeightingContext,
    target_index: int,
    *,
    correction: GroupSizeCorrection,
) -> np.ndarray:
    r"""Return per-cell factors ``min(1, n_target / n_source)``.

    Target-group cells always receive factor one.  ``none`` returns a vector of
    ones and is therefore a true no-correction path in all three modes.
    """

    factors = np.ones(len(context.cells), dtype=np.float64)
    if correction == "none":
        return factors

    target_count = int(context.group_counts[target_index])
    per_group = np.minimum(1.0, target_count / context.group_counts.astype(np.float64))
    per_group[target_index] = 1.0
    factors[:] = per_group[context.group_codes]
    return factors


def _coerce_cell_distances(
    distances: pd.DataFrame | pd.Series | np.ndarray | Sequence[float] | None,
    *,
    target_group: str,
    cells: tuple[str, ...],
    group_ids: Sequence[str] | None,
) -> np.ndarray:
    if distances is None:
        raise ValueError("cell distances are required for cell-based weighting modes")
    if isinstance(distances, pd.DataFrame):
        frame = distances.copy(deep=False)
        frame.index = frame.index.map(str)
        frame.columns = frame.columns.map(str)
        if frame.index.has_duplicates:
            raise ValueError("cell distance matrix has duplicate cell identifiers")
        if target_group not in frame.columns:
            raise ValueError(f"cell distance matrix has no target-group column {target_group!r}")
        if set(frame.index) != set(cells):
            raise ValueError("cell distance matrix must cover exactly the weighted cells")
        if tuple(frame.index) == cells:
            # The core already stores rows in expression-cell order. Preserve the
            # contiguous column view (notably for a Fortran-order memmap) instead of
            # invoking label-based fancy indexing and copying it unnecessarily.
            values = frame[target_group].to_numpy(dtype=np.float64, copy=False)
        else:
            values = frame.loc[list(cells), target_group].to_numpy(dtype=np.float64)
    elif isinstance(distances, pd.Series):
        series = distances.copy(deep=False)
        series.index = series.index.map(str)
        if series.index.has_duplicates or set(series.index) != set(cells):
            raise ValueError("cell distance series must cover exactly the weighted cells once")
        values = series.loc[list(cells)].to_numpy(dtype=np.float64)
    else:
        array = np.asarray(distances, dtype=np.float64)
        if array.ndim == 1:
            values = array
        elif array.ndim == 2:
            if group_ids is None:
                raise ValueError("group_ids is required for an unlabelled cell distance matrix")
            labels = list(map(str, group_ids))
            if target_group not in labels:
                raise ValueError(f"group_ids does not contain target group {target_group!r}")
            values = array[:, labels.index(target_group)]
        else:
            raise ValueError("cell distances must be a vector or cell-by-group matrix")
    if values.shape != (len(cells),):
        raise ValueError("cell distance vector length does not match cell_groups")
    return np.asarray(values, dtype=np.float64)


def _coerce_group_distance_map(
    distances: pd.DataFrame
    | pd.Series
    | Mapping[Hashable, float]
    | np.ndarray
    | Sequence[float]
    | None,
    *,
    target_group: str,
    observed_groups: Sequence[str],
    group_ids: Sequence[str] | None,
) -> dict[str, float]:
    if distances is None:
        raise ValueError("centroid distances are required for group-distance weighting")
    if isinstance(distances, pd.DataFrame):
        frame = distances.copy(deep=False)
        frame.index = frame.index.map(str)
        frame.columns = frame.columns.map(str)
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError("centroid distance matrix group identifiers must be unique")
        if target_group not in frame.index:
            raise ValueError(f"centroid distance matrix has no target-group row {target_group!r}")
        mapping = {str(group): float(value) for group, value in frame.loc[target_group].items()}
    elif isinstance(distances, pd.Series):
        if distances.index.has_duplicates:
            raise ValueError("group distance series contains duplicate groups")
        mapping = {str(group): float(value) for group, value in distances.items()}
    elif isinstance(distances, Mapping):
        mapping = {str(group): float(value) for group, value in distances.items()}
    else:
        array = np.asarray(distances, dtype=np.float64)
        if group_ids is None:
            raise ValueError("group_ids is required for unlabelled centroid distances")
        labels = list(map(str, group_ids))
        if array.ndim == 1:
            if array.size != len(labels):
                raise ValueError("group distance vector length does not match group_ids")
            mapping = dict(zip(labels, map(float, array), strict=True))
        elif array.ndim == 2:
            if array.shape != (len(labels), len(labels)):
                raise ValueError("centroid distance matrix dimensions must match group_ids")
            if target_group not in labels:
                raise ValueError(f"group_ids does not contain target group {target_group!r}")
            row = array[labels.index(target_group)]
            mapping = dict(zip(labels, map(float, row), strict=True))
        else:
            raise ValueError("centroid distances must be a vector or square matrix")

    missing = [group for group in observed_groups if group not in mapping]
    if missing:
        raise ValueError(f"centroid distances are missing observed groups: {missing!r}")
    return mapping


def compute_weights(
    target_group: Hashable,
    context: WeightingContext,
    *,
    mode: WeightMode,
    bandwidth: float | BandwidthSelection,
    kernel: str = "gaussian",
    group_size_correction: GroupSizeCorrection = "cap-to-target",
    cell_distances: pd.DataFrame | pd.Series | np.ndarray | Sequence[float] | None = None,
    group_distances: (
        pd.DataFrame | pd.Series | Mapping[Hashable, float] | np.ndarray | Sequence[float] | None
    ) = None,
) -> WeightResult:
    """Compute one target group's sample weights according to the selected mode.

    Parameters named ``cell_distances`` and ``group_distances`` may be either a
    target-specific vector or a labelled full matrix.  This allows one global
    distance calculation to be reused across every target group.
    """

    validated_mode = _validate_mode(mode)
    correction = _validate_correction(group_size_correction)
    scale = _effective_bandwidth(bandwidth)
    target = _stringify_target_group(target_group)
    cells = context.cells
    try:
        target_index = context.group_ids.index(target)
    except ValueError:
        raise ValueError(f"target group {target!r} has no cells") from None
    target_mask = context.group_codes == target_index

    factors = _compute_group_size_factors(context, target_index, correction=correction)
    if validated_mode in {"cell-distance", "cell-distance-group-anchored"}:
        used_distances = _coerce_cell_distances(
            cell_distances,
            target_group=target,
            cells=cells,
            group_ids=context.group_ids,
        )
        base = apply_kernel(used_distances, scale, kernel=kernel)
        if validated_mode == "cell-distance-group-anchored":
            base[target_mask] = 1.0
    else:
        observed_groups = list(context.group_ids)
        distance_map = _coerce_group_distance_map(
            group_distances,
            target_group=target,
            observed_groups=observed_groups,
            group_ids=context.group_ids,
        )
        distance_values = np.fromiter(
            (distance_map[group] for group in context.group_ids),
            dtype=np.float64,
            count=len(context.group_ids),
        )
        used_distances = np.take(distance_values, context.group_codes)
        base = apply_kernel(used_distances, scale, kernel=kernel)
        base[target_mask] = 1.0

    corrected = np.asarray(base * factors, dtype=np.float64)
    normalization_factor = 1.0
    if validated_mode == "cell-distance":
        maximum = float(np.max(corrected))
        if not np.isfinite(maximum) or maximum <= 0:
            raise DegenerateWeightsError(
                f"All corrected weights are zero or invalid for target group {target!r}; "
                "increase the bandwidth or inspect distances"
            )
        normalization_factor = 1.0 / maximum
        final = corrected * normalization_factor
        # Multiplication by the reciprocal can leave the maximum one ulp below
        # one. The mode's contract is exact, so pin every exact tie rather than
        # arbitrarily distinguishing the first equivalent observation.
        final[corrected == maximum] = 1.0
    else:
        final = corrected

    if not np.isfinite(final).all() or np.any(final < 0) or np.any(final > 1 + 1e-12):
        raise ValueError("final weights violate the finite [0, 1] weight contract")
    np.clip(final, 0.0, 1.0, out=final)
    if validated_mode != "cell-distance" and not np.all(final[target_mask] == 1.0):
        raise AssertionError(
            "anchored and group-distance target-cell weights must remain exactly one"
        )

    return WeightResult(
        context=context,
        target_group=target,
        distance=np.asarray(used_distances, dtype=np.float64),
        # These arrays are freshly allocated above. Reusing them avoids a second
        # full set of per-cell vectors at the return boundary.
        base_weight=np.asarray(base, dtype=np.float64),
        group_size_factor=np.asarray(factors, dtype=np.float64),
        final_weight=np.asarray(final, dtype=np.float64),
        mode=validated_mode,
        normalization_factor=float(normalization_factor),
    )


def iter_group_affinity_records(
    centroid_distances: pd.DataFrame,
    group_sizes: Mapping[Hashable, int] | pd.Series,
    *,
    bandwidth: float | BandwidthSelection,
    kernel: str = "gaussian",
    group_size_correction: GroupSizeCorrection = "cap-to-target",
) -> Iterator[dict[str, str | float]]:
    """Yield the interpretable group-affinity table without retaining G² rows.

    Affinity is always based on centroid distance, irrespective of the selected
    cell-weight mode; it is a population-level interpretability artifact rather
    than a replacement for individual-cell weights.
    """

    correction = _validate_correction(group_size_correction)
    scale = _effective_bandwidth(bandwidth)
    if centroid_distances.shape[0] != centroid_distances.shape[1]:
        raise ValueError("centroid_distances must be square")
    frame = centroid_distances.copy(deep=False)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if list(frame.index) != list(frame.columns):
        raise ValueError(
            "centroid_distances rows and columns must contain groups in the same order"
        )
    sizes = {str(group): int(size) for group, size in dict(group_sizes).items()}
    missing = [group for group in frame.index if group not in sizes]
    if missing or any(sizes.get(group, 0) <= 0 for group in frame.index):
        raise ValueError(
            f"group_sizes must provide a positive size for every group; missing={missing!r}"
        )

    source_sizes = np.fromiter(
        (sizes[str(source)] for source in frame.columns),
        dtype=np.float64,
        count=len(frame.columns),
    )
    for target in frame.index:
        distances = frame.loc[target].to_numpy(dtype=np.float64)
        base = apply_kernel(distances, scale, kernel=kernel)
        target_index = frame.columns.get_loc(target)
        base[target_index] = 1.0
        factors = np.ones(len(frame.columns), dtype=np.float64)
        if correction == "cap-to-target":
            factors = np.minimum(1.0, sizes[target] / source_sizes)
            factors[target_index] = 1.0
        for source_index, source in enumerate(frame.columns):
            yield {
                "target_group": target,
                "source_group": source,
                "centroid_distance": float(distances[source_index]),
                "base_affinity": float(base[source_index]),
                "group_size_factor": float(factors[source_index]),
            }


__all__ = [
    "DegenerateWeightsError",
    "GroupSizeCorrection",
    "WeightMode",
    "WeightResult",
    "WeightingContext",
    "compute_weights",
    "iter_group_affinity_records",
    "prepare_weighting_context",
]
