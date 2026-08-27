"""Group-prototype calculation for the SPATHI distance space."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .representation import RepresentationResult

PrototypeMethod = Literal["mean"]


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


def compute_prototypes(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    groups: pd.Series | Sequence[Hashable],
    *,
    method: PrototypeMethod = "mean",
    cell_ids: Sequence[str] | None = None,
    group_order: Sequence[Hashable] | None = None,
) -> pd.DataFrame:
    """Compute one reusable prototype per group.

    ``method`` is intentionally explicit even though the MVP implements only
    arithmetic means.  This keeps distance and weighting code independent of
    the prototype definition so medoids can be introduced later.
    """

    if method != "mean":
        raise ValueError("The MVP supports only the 'mean' prototype method")
    values, cells, dimensions = _coerce_representation(representation, cell_ids)
    labels, order = _ordered_groups(groups, cells, group_order)

    frame = pd.DataFrame(values, index=cells, columns=dimensions, copy=False)
    centroids = frame.groupby(labels, sort=False, observed=True).mean()
    centroids = centroids.loc[order]
    centroids.index.name = "group"
    if not np.isfinite(centroids.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("prototype calculation produced non-finite values")
    return centroids


def compute_centroids(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    groups: pd.Series | Sequence[Hashable],
    *,
    cell_ids: Sequence[str] | None = None,
    group_order: Sequence[Hashable] | None = None,
) -> pd.DataFrame:
    """Compute arithmetic-mean group centroids."""

    return compute_prototypes(
        representation,
        groups,
        method="mean",
        cell_ids=cell_ids,
        group_order=group_order,
    )


__all__ = ["PrototypeMethod", "compute_centroids", "compute_prototypes"]
