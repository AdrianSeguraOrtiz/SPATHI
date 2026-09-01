"""The three exact SPATHI distance-to-observation weighting schemes."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .kernels import BandwidthSelection, apply_kernel

WeightMode = Literal["cell-distance", "cell-distance-group-anchored", "group-distance"]
GroupSizeCorrection = Literal["none", "cap-to-target"]


class DegenerateWeightsError(ValueError):
    """Raised when no positive sample weight remains for model fitting."""


@dataclass(frozen=True, slots=True)
class WeightResult:
    """All interpretable stages of one target group's sample weights.

    ``group_size_factor`` contains only the requested multiplicity correction.
    For ``cell-distance``, ``normalization_factor`` records the additional
    scalar applied after that correction to make the maximum final weight one.
    The other modes use a normalization factor of one so target cells remain
    anchored exactly at one.
    """

    target_group: str
    cells: tuple[str, ...]
    cell_groups: tuple[str, ...]
    distance: np.ndarray
    base_weight: np.ndarray
    group_size_factor: np.ndarray
    final_weight: np.ndarray
    mode: WeightMode
    normalization_factor: float = 1.0

    def __post_init__(self) -> None:
        lengths = {
            len(self.cells),
            len(self.cell_groups),
            self.distance.size,
            self.base_weight.size,
            self.group_size_factor.size,
            self.final_weight.size,
        }
        if len(lengths) != 1:
            raise ValueError("all WeightResult vectors must have the same length")

    @property
    def distances(self) -> np.ndarray:
        """Plural alias for :attr:`distance`."""

        return self.distance

    @property
    def base_weights(self) -> np.ndarray:
        """Plural alias for :attr:`base_weight`."""

        return self.base_weight

    @property
    def group_size_factors(self) -> np.ndarray:
        """Plural alias for :attr:`group_size_factor`."""

        return self.group_size_factor

    @property
    def final_weights(self) -> np.ndarray:
        """Plural alias for :attr:`final_weight`."""

        return self.final_weight

    def to_frame(self) -> pd.DataFrame:
        """Return the exact long-form fields required by ``cell_weights.tsv.gz``."""

        return pd.DataFrame(
            {
                "target_group": self.target_group,
                "cell": self.cells,
                "cell_group": self.cell_groups,
                "distance": self.distance,
                "base_weight": self.base_weight,
                "group_size_factor": self.group_size_factor,
                "final_weight": self.final_weight,
            }
        )


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


def compute_group_size_factors(
    cell_groups: pd.Series | Sequence[Hashable],
    target_group: Hashable,
    *,
    correction: GroupSizeCorrection = "cap-to-target",
) -> np.ndarray:
    r"""Return per-cell factors ``min(1, n_target / n_source)``.

    Target-group cells always receive factor one.  ``none`` returns a vector of
    ones and is therefore a true no-correction path in all three modes.
    """

    validated = _validate_correction(correction)
    raw_groups = (
        cell_groups.to_numpy(dtype=object).tolist()
        if isinstance(cell_groups, pd.Series)
        else list(cell_groups)
    )
    labels = _stringify_group_labels(raw_groups, field_name="cell_groups")
    target = _stringify_target_group(target_group)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("cell_groups must be a non-empty one-dimensional vector")
    target_count = int(np.count_nonzero(labels == target))
    if target_count == 0:
        raise ValueError(f"target group {target!r} has no cells")
    factors = np.ones(labels.size, dtype=np.float64)
    if validated == "none":
        return factors

    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    per_group = np.minimum(1.0, target_count / counts.astype(np.float64))
    per_group[unique == target] = 1.0
    factors[:] = per_group[inverse]
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
            # The pipeline already stores rows in expression-cell order. Preserve the
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
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("cell distances must be finite and non-negative")
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
    used = np.asarray([mapping[group] for group in observed_groups], dtype=np.float64)
    if not np.isfinite(used).all() or np.any(used < 0):
        raise ValueError("centroid distances must be finite and non-negative")
    return mapping


def compute_weights(
    target_group: Hashable,
    cell_groups: pd.Series | Sequence[Hashable],
    *,
    mode: WeightMode,
    bandwidth: float | BandwidthSelection,
    kernel: str = "gaussian",
    group_size_correction: GroupSizeCorrection = "cap-to-target",
    cell_distances: pd.DataFrame | pd.Series | np.ndarray | Sequence[float] | None = None,
    group_distances: (
        pd.DataFrame | pd.Series | Mapping[Hashable, float] | np.ndarray | Sequence[float] | None
    ) = None,
    cell_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
) -> WeightResult:
    """Compute one target group's sample weights according to an exact MVP mode.

    Parameters named ``cell_distances`` and ``group_distances`` may be either a
    target-specific vector or a labelled full matrix.  This allows one global
    distance calculation to be reused across every target group.
    """

    validated_mode = _validate_mode(mode)
    correction = _validate_correction(group_size_correction)
    scale = _effective_bandwidth(bandwidth)
    labels, cells = _coerce_groups_and_cells(cell_groups, cell_ids)
    target = _stringify_target_group(target_group)
    target_mask = labels == target
    if not target_mask.any():
        raise ValueError(f"target group {target!r} has no cells")

    factors = compute_group_size_factors(labels, target, correction=correction)
    if validated_mode in {"cell-distance", "cell-distance-group-anchored"}:
        used_distances = _coerce_cell_distances(
            cell_distances,
            target_group=target,
            cells=cells,
            group_ids=group_ids,
        )
        base = apply_kernel(used_distances, scale, kernel=kernel)
        if validated_mode == "cell-distance-group-anchored":
            base[target_mask] = 1.0
    else:
        observed_groups = list(dict.fromkeys(labels.tolist()))
        distance_map = _coerce_group_distance_map(
            group_distances,
            target_group=target,
            observed_groups=observed_groups,
            group_ids=group_ids,
        )
        distance_groups = tuple(distance_map)
        group_positions = pd.Index(distance_groups).get_indexer(labels)
        if np.any(group_positions < 0):  # guarded by _coerce_group_distance_map
            raise AssertionError("validated group distances lost an observed group")
        distance_values = np.fromiter(
            (distance_map[group] for group in distance_groups),
            dtype=np.float64,
            count=len(distance_groups),
        )
        used_distances = np.take(distance_values, group_positions)
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

    for name, values in (
        ("base weights", base),
        ("group-size factors", factors),
        ("final weights", final),
    ):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1 + 1e-12):
            raise ValueError(f"{name} violate the finite [0, 1] weight contract")
    base = np.clip(base, 0.0, 1.0)
    factors = np.clip(factors, 0.0, 1.0)
    final = np.clip(final, 0.0, 1.0)
    if validated_mode != "cell-distance" and not np.all(final[target_mask] == 1.0):
        raise AssertionError(
            "anchored and group-distance target-cell weights must remain exactly one"
        )

    return WeightResult(
        target_group=target,
        cells=cells,
        cell_groups=tuple(labels.tolist()),
        distance=np.asarray(used_distances, dtype=np.float64).copy(),
        # These arrays are freshly allocated above. Reusing them avoids a second
        # full set of per-cell vectors at the return boundary.
        base_weight=np.asarray(base, dtype=np.float64),
        group_size_factor=np.asarray(factors, dtype=np.float64),
        final_weight=np.asarray(final, dtype=np.float64),
        mode=validated_mode,
        normalization_factor=float(normalization_factor),
    )


def compute_all_weights(
    cell_groups: pd.Series | Sequence[Hashable],
    *,
    mode: WeightMode,
    bandwidth: float | BandwidthSelection,
    kernel: str = "gaussian",
    group_size_correction: GroupSizeCorrection = "cap-to-target",
    cell_to_centroid_distances: pd.DataFrame | np.ndarray | None = None,
    centroid_distances: pd.DataFrame | np.ndarray | None = None,
    cell_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
) -> dict[str, WeightResult]:
    """Compute and cache exactly one weight vector per observed target group."""

    labels, cells = _coerce_groups_and_cells(cell_groups, cell_ids)
    targets = list(dict.fromkeys(labels.tolist()))
    return {
        target: compute_weights(
            target,
            labels,
            mode=mode,
            bandwidth=bandwidth,
            kernel=kernel,
            group_size_correction=group_size_correction,
            cell_distances=cell_to_centroid_distances,
            group_distances=centroid_distances,
            cell_ids=cells,
            group_ids=group_ids,
        )
        for target in targets
    }


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


def compute_group_affinities(
    centroid_distances: pd.DataFrame,
    group_sizes: Mapping[Hashable, int] | pd.Series,
    *,
    bandwidth: float | BandwidthSelection,
    kernel: str = "gaussian",
    group_size_correction: GroupSizeCorrection = "cap-to-target",
) -> pd.DataFrame:
    """Build the complete long-form group-affinity DataFrame for API callers."""

    return pd.DataFrame.from_records(
        iter_group_affinity_records(
            centroid_distances,
            group_sizes,
            bandwidth=bandwidth,
            kernel=kernel,
            group_size_correction=group_size_correction,
        ),
        columns=[
            "target_group",
            "source_group",
            "centroid_distance",
            "base_affinity",
            "group_size_factor",
        ],
    )


calculate_weights = compute_weights


__all__ = [
    "DegenerateWeightsError",
    "GroupSizeCorrection",
    "WeightMode",
    "WeightResult",
    "calculate_weights",
    "compute_all_weights",
    "compute_group_affinities",
    "compute_group_size_factors",
    "compute_weights",
    "iter_group_affinity_records",
]
