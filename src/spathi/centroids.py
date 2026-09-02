"""Arithmetic group centroids in the configured SPATHI distance space."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from .representation import RepresentationResult


def _coerce_representation(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    cell_ids: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if isinstance(representation, RepresentationResult):
        return representation.values, representation.cell_ids, representation.dimension_names
    if isinstance(representation, pd.DataFrame):
        values = representation.to_numpy(dtype=np.float64, copy=False)
        cells = tuple(map(str, representation.index))
        dimensions = tuple(map(str, representation.columns))
    else:
        values = np.asarray(representation, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("representation must be a two-dimensional cell-by-dimension matrix")
        cells = (
            tuple(cell_ids)
            if cell_ids is not None
            else tuple(f"cell_{i}" for i in range(values.shape[0]))
        )
        dimensions = tuple(f"dimension_{i + 1}" for i in range(values.shape[1]))

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("representation must contain at least one cell and one dimension")
    if len(cells) != values.shape[0]:
        raise ValueError("cell_ids length does not match representation rows")
    if len(set(cells)) != len(cells):
        raise ValueError("representation cell identifiers must be unique")
    if not np.isfinite(values).all():
        raise ValueError("representation must contain only finite values")
    return values, tuple(map(str, cells)), dimensions


def _ordered_groups(
    groups: pd.Series | Sequence[Hashable],
    cells: tuple[str, ...],
    group_order: Sequence[Hashable] | None,
) -> tuple[np.ndarray, list[Hashable]]:
    if isinstance(groups, pd.Series):
        if groups.index.has_duplicates:
            raise ValueError("groups index contains duplicate cell identifiers")
        if isinstance(groups.dtype, pd.CategoricalDtype):
            observed_categories = set(groups.dropna())
            empty_categories = [
                category
                for category in groups.cat.categories
                if category not in observed_categories
            ]
            if empty_categories:
                raise ValueError(f"groups contains empty categorical groups: {empty_categories!r}")
        string_index = pd.Index(map(str, groups.index))
        indexed_cells = set(string_index)
        represented_cells = set(cells)
        if indexed_cells != represented_cells:
            missing = [cell for cell in cells if cell not in indexed_cells]
            unexpected = [cell for cell in string_index if cell not in represented_cells]
            raise ValueError(
                f"groups must cover representation cells exactly; missing={missing!r}, unexpected={unexpected!r}"
            )
        aligned = groups.copy()
        aligned.index = string_index
        labels = aligned.loc[list(cells)].to_numpy(dtype=object)
    else:
        labels = np.asarray(list(groups), dtype=object)
        if labels.shape != (len(cells),):
            raise ValueError("groups length does not match the number of represented cells")

    if pd.isna(labels).any() or any(not str(label).strip() for label in labels):
        raise ValueError("groups contains missing or empty labels")
    observed = list(pd.unique(labels))
    if not observed:
        raise ValueError("at least one non-empty group is required")

    if group_order is None:
        order = observed
    else:
        order = list(group_order)
        if len(set(order)) != len(order):
            raise ValueError("group_order contains duplicate labels")
        ordered_groups = set(order)
        observed_groups = set(observed)
        missing_order = [label for label in observed if label not in ordered_groups]
        empty_order = [label for label in order if label not in observed_groups]
        if missing_order or empty_order:
            raise ValueError(
                f"group_order must contain each observed group once; missing={missing_order!r}, empty={empty_order!r}"
            )
    return labels, order


def compute_centroids(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    groups: pd.Series | Sequence[Hashable],
    *,
    cell_ids: Sequence[str] | None = None,
    group_order: Sequence[Hashable] | None = None,
) -> pd.DataFrame:
    """Compute one arithmetic-mean centroid per group."""

    values, cells, dimensions = _coerce_representation(representation, cell_ids)
    labels, order = _ordered_groups(groups, cells, group_order)

    frame = pd.DataFrame(values, index=cells, columns=dimensions, copy=False)
    grouped = frame.groupby(labels, sort=False, observed=True)
    centroids = grouped.mean()
    centroids = centroids.loc[order]
    centroids.index.name = "group"
    centroid_values = centroids.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(centroid_values).all():
        # pandas' fast grouped sum can overflow before division even when the
        # arithmetic mean is finite (for example, mean([float_max, float_max])).
        # Recompute only this exceptional case with an online mean.  When the
        # subtraction in the usual update overflows for opposite-sign extrema,
        # use a convex combination whose products stay within the input range.
        centroid_values = np.empty((len(order), values.shape[1]), dtype=np.float64)
        for group_index, group in enumerate(order):
            positions = grouped.indices[group]
            mean = np.array(values[positions[0]], dtype=np.float64, copy=True)
            for count, position in enumerate(positions[1:], start=2):
                row = values[position]
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = row - mean
                regular = np.isfinite(delta)
                mean[regular] += delta[regular] / count
                exceptional = ~regular
                if np.any(exceptional):
                    previous_weight = (count - 1) / count
                    new_weight = 1.0 / count
                    mean[exceptional] = (
                        mean[exceptional] * previous_weight + row[exceptional] * new_weight
                    )
            centroid_values[group_index] = mean
        centroids = pd.DataFrame(
            centroid_values,
            index=pd.Index(order, name="group"),
            columns=dimensions,
            copy=False,
        )
    if not np.isfinite(centroid_values).all():
        raise ValueError("centroid calculation produced non-finite values")
    return centroids


__all__ = ["compute_centroids"]
