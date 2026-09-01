"""Deterministic, bounded-memory visual diagnostics for one SPATHI run.

The inference pipeline is expected to use this module incrementally::

    embedding = prepare_visualization_embedding(representation, groups, centroids)
    records = []
    for target_group in target_groups:
        records.append(
            write_target_weight_panel(
                output_dir,
                weights,
                embedding,
                kernel=config.kernel,
                bandwidth=bandwidth.value,
            )
        )
    records.append(write_effective_mass_heatmap(output_dir, diagnostics))
    result = write_visualization_manifest(output_dir, embedding, records)

``weights`` and ``diagnostics`` are the already computed :class:`WeightResult` and
:class:`WeightDiagnostics` objects.  Consequently no cell-by-group weight table is
retained merely to draw figures: one target panel needs only O(n_cells) working memory,
while the global effective-mass heatmap needs O(n_groups**2) summary values.

The module deliberately owns the plotting dependency and renders through an Agg
canvas without changing Matplotlib's process-wide interactive backend. Inference,
representation, and weighting remain independent from figure construction.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from .diagnostics import WeightDiagnostics
from .kernels import apply_kernel
from .representation import DistanceSpace, RepresentationResult
from .weighting import WeightResult

ProjectionKind = Literal["distance-pca", "auxiliary-pca"]
FigureKind = Literal["target-weight-panel", "effective-mass-heatmap"]

_FIGURE_DPI = 120
_MAX_SCATTER_CELLS = 50_000
_MAX_LEGEND_GROUPS = 12
_MAX_LABELLED_GROUPS = 30
_WEIGHT_COLUMNS = (
    "target_group",
    "cell",
    "cell_group",
    "distance",
    "base_weight",
    "group_size_factor",
    "final_weight",
)
# Matplotlib's stubs enumerate hundreds of literal rcParam keys. Keeping this
# compact, validated mapping typed as Any avoids coupling SPATHI to that generated
# union while ``rc_context`` still validates every key at runtime.
_RC_PARAMS: Any = {
    "axes.grid": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "text.usetex": False,
}


@dataclass(frozen=True, slots=True)
class VisualizationEmbedding:
    """Two-dimensional coordinates used only to display exact SPATHI weights."""

    coordinates: np.ndarray
    centroid_coordinates: np.ndarray
    cell_ids: tuple[str, ...]
    cell_groups: tuple[str, ...]
    group_ids: tuple[str, ...]
    x_label: str
    y_label: str
    projection_kind: ProjectionKind
    distance_space: DistanceSpace
    explained_variance_ratio: tuple[float, ...]
    notes: tuple[str, ...]
    cell_hashes: np.ndarray
    group_codes: np.ndarray
    group_positions: tuple[np.ndarray, ...]
    group_colours: np.ndarray

    def to_metadata(self) -> dict[str, Any]:
        """Return the portable projection description stored in the manifest."""

        return {
            "kind": self.projection_kind,
            "distance_space": self.distance_space,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "n_cells": len(self.cell_ids),
            "n_groups": len(self.group_ids),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class FigureRecord:
    """One deterministic figure and the compact provenance needed by run metadata."""

    kind: FigureKind
    relative_path: str
    sha256: str
    size_bytes: int
    target_group: str | None
    n_cells: int | None
    displayed_cells: int | None
    n_groups: int

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible record in stable field order."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """Manifest and figures produced by the incremental visualization layer."""

    directory: Path
    manifest_path: Path
    figures: tuple[FigureRecord, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return paths relative to the run directory plus figure provenance."""

        return {
            "directory": self.directory.name,
            "manifest": self.manifest_path.relative_to(self.directory.parent).as_posix(),
            "figures": [figure.to_metadata() for figure in self.figures],
        }


def safe_group_filename(group_id: Hashable) -> str:
    """Return a portable, deterministic, collision-resistant PNG filename.

    A readable ASCII slug is followed by the full SHA-256 of the original UTF-8 group
    identifier.  Thus path separators, control characters, Unicode normalization, case
    folding, and truncation cannot make two ordinary group identifiers share a path.
    """

    raw = str(group_id)
    if not raw.strip():
        raise ValueError("group identifiers used for visualization may not be empty")
    ascii_value = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_value).strip("._-").lower()
    slug = slug[:48].rstrip("._-") or "group"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{slug}--{digest}.png"


def _display_label(value: Hashable, *, maximum: int = 72) -> str:
    text = " ".join(str(value).split())
    text = text.replace("$", r"\$")
    if len(text) <= maximum:
        return text
    return f"{text[: maximum - 1]}…"


def _coerce_representation(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    *,
    distance_space: DistanceSpace | None,
    cell_ids: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], DistanceSpace, tuple[float, ...]]:
    if isinstance(representation, RepresentationResult):
        values = representation.values
        cells = representation.cell_ids
        dimensions = representation.dimension_names
        effective_space = representation.distance_space
        explained = representation.explained_variance_ratio or ()
        if distance_space is not None and distance_space != effective_space:
            raise ValueError("distance_space contradicts the supplied RepresentationResult")
    elif isinstance(representation, pd.DataFrame):
        values = representation.to_numpy(dtype=np.float64, copy=False)
        cells = tuple(map(str, representation.index))
        dimensions = tuple(map(str, representation.columns))
        if distance_space is None:
            raise ValueError("distance_space is required for a raw representation DataFrame")
        effective_space = distance_space
        explained = ()
    else:
        values = np.asarray(representation, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("representation must be a two-dimensional matrix")
        if cell_ids is None:
            raise ValueError("cell_ids is required for an unlabelled representation")
        cells = tuple(map(str, cell_ids))
        dimensions = tuple(f"dimension_{index + 1}" for index in range(values.shape[1]))
        if distance_space is None:
            raise ValueError("distance_space is required for an unlabelled representation")
        effective_space = distance_space
        explained = ()

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("representation must contain at least one cell and one dimension")
    if len(cells) != values.shape[0] or len(set(cells)) != len(cells):
        raise ValueError("representation cell identifiers must be unique and match its rows")
    if not np.isfinite(values).all():
        raise ValueError("representation must contain only finite values")
    if len(dimensions) != values.shape[1]:
        raise ValueError("representation dimension names do not match its columns")
    return values, cells, dimensions, effective_space, tuple(map(float, explained))


def _coerce_cell_groups(
    cell_groups: pd.Series | Sequence[Hashable],
    cells: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(cell_groups, pd.Series):
        if cell_groups.index.has_duplicates:
            raise ValueError("cell_groups index contains duplicate cell identifiers")
        aligned = cell_groups.copy(deep=False)
        aligned.index = aligned.index.map(str)
        if set(aligned.index) != set(cells):
            raise ValueError("cell_groups must cover visualization cells exactly")
        values = aligned.loc[list(cells)].to_numpy(dtype=object)
    else:
        values = np.asarray(list(cell_groups), dtype=object)
        if values.shape != (len(cells),):
            raise ValueError("cell_groups length must match visualization cells")
    if pd.isna(values).any() or any(not str(value).strip() for value in values):
        raise ValueError("cell_groups contains missing or empty labels")
    return tuple(map(str, values))


def _coerce_centroids(
    centroids: pd.DataFrame | np.ndarray,
    *,
    dimensions: tuple[str, ...],
    group_ids: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(centroids, pd.DataFrame):
        values = centroids.to_numpy(dtype=np.float64, copy=False)
        groups = tuple(map(str, centroids.index))
        if tuple(map(str, centroids.columns)) != dimensions:
            raise ValueError("centroid dimensions do not match the representation")
    else:
        values = np.asarray(centroids, dtype=np.float64)
        if group_ids is None:
            raise ValueError("group_ids is required for unlabelled centroids")
        groups = tuple(map(str, group_ids))
    if values.ndim != 2 or values.shape != (len(groups), len(dimensions)):
        raise ValueError("centroids must have one row per group and all representation dimensions")
    if len(groups) == 0 or len(set(groups)) != len(groups):
        raise ValueError("centroid group identifiers must be non-empty and unique")
    if not np.isfinite(values).all():
        raise ValueError("centroids must contain only finite values")
    return np.asarray(values, dtype=np.float64), groups


def _two_columns(values: np.ndarray) -> np.ndarray:
    coordinates = np.zeros((values.shape[0], 2), dtype=np.float64)
    coordinates[:, : min(2, values.shape[1])] = values[:, :2]
    return coordinates


def _axis_label(name: str, explained: tuple[float, ...], index: int) -> str:
    if index < len(explained) and np.isfinite(explained[index]):
        return f"{name} ({100.0 * explained[index]:.1f}% variance)"
    return name


def _stable_cell_hashes(cells: Sequence[str]) -> np.ndarray:
    """Hash cell identifiers once for deterministic bounded scatter sampling."""

    hashes = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(cell.encode("utf-8")).digest()[:8], "big")
            for cell in cells
        ),
        dtype=np.uint64,
        count=len(cells),
    )
    hashes.setflags(write=False)
    return hashes


def _build_group_index(
    cell_groups: Sequence[str],
    group_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Build cell-to-group codes and group positions in one reusable pass."""

    group_lookup = {group: index for index, group in enumerate(group_ids)}
    codes = np.fromiter(
        (group_lookup[group] for group in cell_groups),
        dtype=np.int64,
        count=len(cell_groups),
    )
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(group_ids))
    stops = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), stops[:-1]))
    order.setflags(write=False)
    codes.setflags(write=False)
    positions = tuple(order[start:stop] for start, stop in zip(starts, stops, strict=True))
    return codes, positions


def prepare_visualization_embedding(
    representation: RepresentationResult | pd.DataFrame | np.ndarray,
    cell_groups: pd.Series | Sequence[Hashable],
    centroids: pd.DataFrame | np.ndarray,
    *,
    distance_space: DistanceSpace | None = None,
    cell_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
    random_state: int = 0,
) -> VisualizationEmbedding:
    """Prepare one reusable 2D display embedding for all target weight panels.

    A PCA distance representation is displayed directly through PC1/PC2.  When the
    configured distance space is expression, a deterministic two-component PCA is
    calculated solely for display; metadata and axis labels explicitly state that the
    weights were calculated in expression space.  A one-dimensional representation is
    drawn on a zero-valued vertical axis without artificial jitter.
    """

    values, cells, dimensions, effective_space, explained = _coerce_representation(
        representation,
        distance_space=distance_space,
        cell_ids=cell_ids,
    )
    groups_by_cell = _coerce_cell_groups(cell_groups, cells)
    centroid_values, groups = _coerce_centroids(
        centroids,
        dimensions=dimensions,
        group_ids=group_ids,
    )
    observed_groups = set(groups_by_cell)
    if observed_groups != set(groups):
        raise ValueError("centroids must contain every observed cell group exactly once")
    cell_hashes = _stable_cell_hashes(cells)
    group_codes, group_positions = _build_group_index(groups_by_cell, groups)
    group_colours = np.asarray([_group_colour(group) for group in groups], dtype=np.float64)
    group_colours.setflags(write=False)

    notes: list[str] = [
        "Point colour is the exact final weight calculated in the full configured distance space."
    ]
    if effective_space == "pca":
        coordinates = _two_columns(values)
        centroid_coordinates = _two_columns(centroid_values)
        projection_kind: ProjectionKind = "distance-pca"
        x_label = _axis_label("PC1", explained, 0)
        if values.shape[1] >= 2:
            y_label = _axis_label("PC2", explained, 1)
        else:
            y_label = "No second component (fixed at 0)"
            notes.append("The configured PCA distance space has only one component.")
        display_explained = explained[:2]
    else:
        component_count = min(2, values.shape[1], max(1, values.shape[0] - 1))
        solver = "full" if min(values.shape) <= 2 else "randomized"
        pca = PCA(n_components=component_count, svd_solver=solver, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            projected = pca.fit_transform(values)
        coordinates = _two_columns(np.nan_to_num(projected, copy=False))
        centroid_coordinates = _two_columns(
            np.nan_to_num(pca.transform(centroid_values), copy=False)
        )
        auxiliary_explained = tuple(
            map(
                float,
                np.nan_to_num(
                    np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
            )
        )
        projection_kind = "auxiliary-pca"
        x_label = _axis_label("Auxiliary PC1", auxiliary_explained, 0)
        if component_count >= 2:
            y_label = _axis_label("Auxiliary PC2", auxiliary_explained, 1)
        else:
            y_label = "No second auxiliary component (fixed at 0)"
            notes.append("The auxiliary display PCA has only one component.")
        notes.append(
            "The 2D PCA is auxiliary: SPATHI weights and distances were calculated in expression space."
        )
        display_explained = auxiliary_explained[:2]

    if not np.isfinite(coordinates).all() or not np.isfinite(centroid_coordinates).all():
        raise ValueError("visualization projection produced non-finite coordinates")
    return VisualizationEmbedding(
        coordinates=coordinates,
        centroid_coordinates=centroid_coordinates,
        cell_ids=cells,
        cell_groups=groups_by_cell,
        group_ids=groups,
        x_label=x_label,
        y_label=y_label,
        projection_kind=projection_kind,
        distance_space=effective_space,
        explained_variance_ratio=display_explained,
        notes=tuple(notes),
        cell_hashes=cell_hashes,
        group_codes=group_codes,
        group_positions=group_positions,
        group_colours=group_colours,
    )


def _coerce_weight_frame(
    weights: WeightResult | pd.DataFrame,
    embedding: VisualizationEmbedding,
) -> tuple[str, pd.DataFrame]:
    frame = weights.to_frame() if isinstance(weights, WeightResult) else weights.copy(deep=False)
    missing = [column for column in _WEIGHT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"weights are missing required columns: {missing!r}")
    frame = frame.loc[:, list(_WEIGHT_COLUMNS)].copy()
    for column in ("target_group", "cell", "cell_group"):
        frame[column] = frame[column].astype(str)
    targets = tuple(pd.unique(frame["target_group"]))
    if len(targets) != 1:
        raise ValueError("one target panel requires weights for exactly one target group")
    target = targets[0]
    if target not in embedding.group_ids:
        raise ValueError(f"weight target group {target!r} has no visualization centroid")
    if frame["cell"].duplicated().any() or set(frame["cell"]) != set(embedding.cell_ids):
        raise ValueError("target weights must cover every visualization cell exactly once")
    frame = frame.set_index("cell").loc[list(embedding.cell_ids)].reset_index()
    expected_groups = np.asarray(embedding.cell_groups, dtype=str)
    if not np.array_equal(frame["cell_group"].to_numpy(dtype=str), expected_groups):
        raise ValueError("weight cell-group assignments disagree with the visualization embedding")

    for column in ("distance", "base_weight", "group_size_factor", "final_weight"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite values")
        if column == "distance":
            if np.any(values < 0):
                raise ValueError("distance must be non-negative")
        elif np.any(values < 0) or np.any(values > 1 + 1e-12):
            raise ValueError(f"{column} must remain in [0, 1]")
    return target, frame


def _group_colour(group_id: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65_536.0
    saturation = 0.52 + (digest[2] / 255.0) * 0.28
    value = 0.62 + (digest[3] / 255.0) * 0.25
    rgb = mcolors.hsv_to_rgb((hue, saturation, value))
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _sample_indices(
    cell_hashes: np.ndarray,
    maximum: int,
    *,
    group_ids: Sequence[str] = (),
    group_positions: Sequence[np.ndarray] = (),
    priority_group: str | None = None,
) -> np.ndarray:
    """Select a stable bounded subset while retaining rare source groups.

    When strata are supplied, the sample reserves up to three cells per group
    before filling the remaining budget globally by cell hash. If there are
    more groups than available points, the priority stratum is always retained
    and the remaining represented groups are chosen deterministically.
    """

    if maximum <= 0:
        raise ValueError("max_scatter_cells must be a positive integer")
    hashes = np.asarray(cell_hashes, dtype=np.uint64)
    if hashes.ndim != 1 or hashes.size == 0:
        raise ValueError("cell_hashes must be a non-empty one-dimensional vector")
    if hashes.size <= maximum:
        return np.arange(hashes.size, dtype=np.int64)
    if not group_ids and not group_positions:
        selected = np.argpartition(hashes, maximum - 1)[:maximum]
        return np.sort(selected)
    groups = tuple(map(str, group_ids))
    positions_by_group = tuple(
        np.asarray(positions, dtype=np.int64) for positions in group_positions
    )
    if len(groups) != len(positions_by_group) or not groups:
        raise ValueError("group_ids and group_positions must be non-empty and aligned")
    if priority_group is not None and priority_group not in groups:
        raise ValueError("priority sampling group is not present")
    ordered_group_indices = (
        [
            groups.index(priority_group),
            *(index for index, group in enumerate(groups) if group != priority_group),
        ]
        if priority_group is not None
        else list(range(len(groups)))
    )
    if len(ordered_group_indices) > maximum:
        # The priority group remains first; hash the others so truncation is not
        # tied to lexical naming conventions.
        prefix = ordered_group_indices[:1] if priority_group is not None else []
        remaining = ordered_group_indices[len(prefix) :]
        remaining.sort(key=lambda index: hashlib.sha256(groups[index].encode("utf-8")).digest())
        ordered_group_indices = [*prefix, *remaining[: maximum - len(prefix)]]

    per_group = max(1, min(3, maximum // len(ordered_group_indices)))
    reserved: list[np.ndarray] = []
    for group_index in ordered_group_indices:
        candidates = positions_by_group[group_index]
        if candidates.size == 0 or np.any(candidates < 0) or np.any(candidates >= hashes.size):
            raise ValueError("group_positions must contain valid non-empty cell positions")
        count = min(per_group, candidates.size)
        if count == candidates.size:
            chosen = candidates
        else:
            local = np.argpartition(hashes[candidates], count - 1)[:count]
            chosen = candidates[local]
        reserved.append(chosen)

    reserved_indices = np.unique(np.concatenate(reserved))
    remaining_count = maximum - reserved_indices.size
    if remaining_count > 0:
        available = np.ones(hashes.size, dtype=bool)
        available[reserved_indices] = False
        candidates = np.flatnonzero(available)
        if remaining_count < candidates.size:
            local = np.argpartition(hashes[candidates], remaining_count - 1)[:remaining_count]
            candidates = candidates[local]
        reserved_indices = np.concatenate((reserved_indices, candidates))
    return np.sort(reserved_indices)


def _tick_labels(groups: Sequence[str]) -> list[str]:
    if len(groups) <= _MAX_LABELLED_GROUPS:
        return [_display_label(group, maximum=30) for group in groups]
    step = int(np.ceil(len(groups) / _MAX_LABELLED_GROUPS))
    return [
        _display_label(group, maximum=24) if index % step == 0 else ""
        for index, group in enumerate(groups)
    ]


def _add_source_group_legend(axis: Any, groups: Sequence[str]) -> None:
    existing_handles, existing_labels = axis.get_legend_handles_labels()
    if len(groups) > _MAX_LEGEND_GROUPS:
        if existing_handles:
            axis.legend(existing_handles, existing_labels, loc="upper right", fontsize=7)
        axis.text(
            0.99,
            0.02,
            f"{len(groups)} source groups; colour legend omitted",
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=8,
            color="#444444",
        )
        return
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=_group_colour(group),
            markeredgecolor="none",
            markersize=5,
            label=_display_label(group, maximum=36),
        )
        for group in groups
    ]
    axis.legend(
        handles=[*existing_handles, *group_handles],
        labels=[*existing_labels, *[_display_label(group, maximum=36) for group in groups]],
        title="Weight stage / source group",
        loc="best",
        fontsize=7,
        title_fontsize=8,
    )


def _draw_distance_weight_axis(
    axis: Any,
    frame: pd.DataFrame,
    target: str,
    sample: np.ndarray,
    embedding: VisualizationEmbedding,
    *,
    kernel: str | None,
    bandwidth: float | None,
) -> None:
    distances = frame["distance"].to_numpy(dtype=np.float64)
    base = frame["base_weight"].to_numpy(dtype=np.float64)
    final = frame["final_weight"].to_numpy(dtype=np.float64)
    groups = embedding.group_ids

    axis.scatter(
        distances[sample],
        base[sample],
        marker="x",
        s=9,
        linewidths=0.6,
        color="#777777",
        alpha=0.35,
        label="Base weight",
        rasterized=True,
    )
    axis.scatter(
        distances[sample],
        final[sample],
        s=11,
        color=embedding.group_colours[embedding.group_codes[sample]],
        alpha=0.68,
        edgecolors="none",
        rasterized=True,
    )
    if kernel is not None or bandwidth is not None:
        if kernel is None or bandwidth is None:
            raise ValueError("kernel and bandwidth must be supplied together")
        bandwidth_value = float(bandwidth)
        if not np.isfinite(bandwidth_value) or bandwidth_value <= 0:
            raise ValueError("bandwidth must be a positive finite number")
        maximum = float(np.max(distances))
        curve_x = np.linspace(0.0, maximum if maximum > 0 else bandwidth_value, 256)
        curve_y = apply_kernel(curve_x, bandwidth_value, kernel=kernel)
        axis.plot(
            curve_x,
            curve_y,
            color="#111111",
            linestyle="--",
            linewidth=1.2,
            label=f"{kernel} kernel (h={bandwidth_value:.4g})",
        )
    axis.set(xlabel="Weighting distance", ylabel="Weight", ylim=(-0.03, 1.05))
    axis.set_title(f"Distance → weight for target {_display_label(target)}")
    _add_source_group_legend(axis, groups)


def _draw_distribution_axis(
    axis: Any,
    frame: pd.DataFrame,
    target: str,
    embedding: VisualizationEmbedding,
) -> None:
    groups = embedding.group_ids
    final_weights = frame["final_weight"].to_numpy(dtype=np.float64)
    distributions = [final_weights[positions] for positions in embedding.group_positions]
    positions = np.arange(1, len(groups) + 1)
    artists = axis.boxplot(
        distributions,
        positions=positions,
        widths=0.65,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, group in zip(artists["boxes"], groups, strict=True):
        patch.set_facecolor(_group_colour(group))
        patch.set_alpha(0.7 if group != target else 0.95)
        patch.set_edgecolor("#111111" if group == target else "#666666")
        patch.set_linewidth(1.5 if group == target else 0.7)
    axis.set_xticks(positions)
    axis.set_xticklabels(_tick_labels(groups), rotation=45, horizontalalignment="right")
    axis.set(xlabel="Source group", ylabel="Final weight", ylim=(-0.03, 1.05))
    squared_mass = float(np.dot(final_weights, final_weights))
    effective_sample_size = (
        0.0
        if squared_mass == 0.0
        else float(np.sum(final_weights, dtype=np.float64) ** 2 / squared_mass)
    )
    axis.set_title(
        "Final-weight distributions "
        f"(target outlined; effective sample size {effective_sample_size:.1f}/{len(frame)})"
    )


def _draw_embedding_axis(
    axis: Any,
    frame: pd.DataFrame,
    target: str,
    embedding: VisualizationEmbedding,
    sample: np.ndarray,
) -> None:
    weights = frame["final_weight"].to_numpy(dtype=np.float64)
    scatter = axis.scatter(
        embedding.coordinates[sample, 0],
        embedding.coordinates[sample, 1],
        c=weights[sample],
        cmap="viridis",
        norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
        s=13,
        alpha=1.0,
        edgecolors="none",
        rasterized=True,
    )
    target_index = embedding.group_ids.index(target)
    target_positions = sample[embedding.group_codes[sample] == target_index]
    if target_positions.size:
        axis.scatter(
            embedding.coordinates[target_positions, 0],
            embedding.coordinates[target_positions, 1],
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.65,
            s=23,
            label="Target-group cell",
            rasterized=True,
        )
    axis.scatter(
        [embedding.centroid_coordinates[target_index, 0]],
        [embedding.centroid_coordinates[target_index, 1]],
        marker="*",
        s=130,
        facecolor="#ffffff",
        edgecolor="#d62728",
        linewidth=1.5,
        label="Target centroid",
        zorder=5,
    )
    axis.set(xlabel=embedding.x_label, ylabel=embedding.y_label)
    qualifier = (
        "distance-space PCA" if embedding.projection_kind == "distance-pca" else "auxiliary PCA"
    )
    axis.set_title(f"Exact final weights on {qualifier}")
    axis.legend(loc="best", fontsize=7)
    colour_bar = axis.figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colour_bar.set_label("Exact final weight")


def _file_record(
    run_output_dir: Path,
    path: Path,
    *,
    kind: FigureKind,
    target_group: str | None,
    n_cells: int | None,
    displayed_cells: int | None,
    n_groups: int,
) -> FigureRecord:
    digest, size_bytes = _hash_file(path)
    return FigureRecord(
        kind=kind,
        relative_path=path.relative_to(run_output_dir).as_posix(),
        sha256=digest,
        size_bytes=size_bytes,
        target_group=target_group,
        n_cells=n_cells,
        displayed_cells=displayed_cells,
        n_groups=n_groups,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    """Return a streaming SHA-256 and size without retaining file bytes."""

    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def write_target_weight_panel(
    run_output_dir: Path,
    weights: WeightResult | pd.DataFrame,
    embedding: VisualizationEmbedding,
    *,
    kernel: str | None = None,
    bandwidth: float | None = None,
    max_scatter_cells: int = _MAX_SCATTER_CELLS,
) -> FigureRecord:
    """Write one combined distance, distribution, and 2D weight panel.

    Only the supplied target's O(n_cells) vectors are retained.  Scatter panels use a
    stable hash-based subset above ``max_scatter_cells``; boxplots and effective-mass
    diagnostics still consume all cells.
    """

    if type(max_scatter_cells) is not int or max_scatter_cells <= 0:
        raise ValueError("max_scatter_cells must be a positive integer")
    output_root = Path(run_output_dir)
    target, frame = _coerce_weight_frame(weights, embedding)
    sample = _sample_indices(
        embedding.cell_hashes,
        max_scatter_cells,
        group_ids=embedding.group_ids,
        group_positions=embedding.group_positions,
        priority_group=target,
    )
    target_dir = output_root / "visualizations" / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / safe_group_filename(target)

    with matplotlib.rc_context(_RC_PARAMS):
        figure = Figure(figsize=(14.0, 9.5))
        FigureCanvasAgg(figure)
        try:
            grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.78))
            distance_axis = figure.add_subplot(grid[0, 0])
            embedding_axis = figure.add_subplot(grid[0, 1])
            distribution_axis = figure.add_subplot(grid[1, :])
            _draw_distance_weight_axis(
                distance_axis,
                frame,
                target,
                sample,
                embedding,
                kernel=kernel,
                bandwidth=bandwidth,
            )
            _draw_embedding_axis(embedding_axis, frame, target, embedding, sample)
            _draw_distribution_axis(distribution_axis, frame, target, embedding)
            note = (
                embedding.notes[-1]
                if embedding.projection_kind == "auxiliary-pca"
                else (
                    "2D positions are a projection; colours are weights calculated with every retained PC."
                )
            )
            if sample.size < len(frame):
                note += f" Scatter panels display a deterministic {sample.size:,}/{len(frame):,} cell subset."
            figure.suptitle(
                f"SPATHI weight assignment — target {_display_label(target)}",
                fontsize=14,
                fontweight="bold",
            )
            figure.text(0.5, 0.012, note, horizontalalignment="center", fontsize=8, color="#444444")
            figure.subplots_adjust(
                left=0.075, right=0.96, top=0.91, bottom=0.12, hspace=0.42, wspace=0.28
            )
            figure.savefig(
                path,
                dpi=_FIGURE_DPI,
                format="png",
                metadata={"Software": "SPATHI"},
            )
        finally:
            figure.clear()

    return _file_record(
        output_root,
        path,
        kind="target-weight-panel",
        target_group=target,
        n_cells=len(frame),
        displayed_cells=int(sample.size),
        n_groups=len(embedding.group_ids),
    )


def _diagnostic_mass_matrix(
    diagnostics: Sequence[WeightDiagnostics] | Mapping[str, WeightDiagnostics] | pd.DataFrame,
    group_order: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(diagnostics, pd.DataFrame):
        required = {"target_group", "source_group", "source_mass_percent"}
        missing = sorted(required.difference(diagnostics.columns))
        if missing:
            raise ValueError(f"weight diagnostics are missing required columns: {missing!r}")
        frame = diagnostics.loc[:, sorted(required)].copy()
        frame["target_group"] = frame["target_group"].astype(str)
        frame["source_group"] = frame["source_group"].astype(str)
        if frame.duplicated(["target_group", "source_group"]).any():
            raise ValueError("weight diagnostics contain duplicate target/source rows")
        mapping = {
            (row.target_group, row.source_group): float(row.source_mass_percent)
            for row in frame.itertuples(index=False)
        }
        observed_targets = tuple(sorted(frame["target_group"].unique()))
        observed_sources = set(frame["source_group"])
    else:
        values = (
            list(diagnostics.values()) if isinstance(diagnostics, Mapping) else list(diagnostics)
        )
        if not values:
            raise ValueError("at least one weight diagnostic is required")
        if any(not isinstance(value, WeightDiagnostics) for value in values):
            raise TypeError("diagnostics must contain WeightDiagnostics objects")
        mapping = {
            (diagnostic.target_group, source): float(percent)
            for diagnostic in values
            for source, percent in diagnostic.group_mass_percent.items()
        }
        observed_targets = tuple(sorted(diagnostic.target_group for diagnostic in values))
        observed_sources = {
            str(source) for diagnostic in values for source in diagnostic.group_mass_percent
        }

    groups = tuple(map(str, group_order)) if group_order is not None else observed_targets
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("group_order must contain unique group identifiers")
    if set(observed_targets) != set(groups) or observed_sources != set(groups):
        raise ValueError("weight diagnostics must cover every target/source group pair")
    matrix = np.empty((len(groups), len(groups)), dtype=np.float64)
    for target_index, target in enumerate(groups):
        for source_index, source in enumerate(groups):
            try:
                matrix[target_index, source_index] = mapping[(target, source)]
            except KeyError as exc:
                raise ValueError(
                    f"weight diagnostics are missing target/source pair {(target, source)!r}"
                ) from exc
    if not np.isfinite(matrix).all() or np.any(matrix < -1e-10) or np.any(matrix > 100 + 1e-8):
        raise ValueError("source mass percentages must be finite and in [0, 100]")
    if not np.allclose(matrix.sum(axis=1), 100.0, rtol=1e-6, atol=1e-6):
        raise ValueError("source mass percentages must sum to 100 for every target")
    return np.clip(matrix, 0.0, 100.0), groups


def write_effective_mass_heatmap(
    run_output_dir: Path,
    diagnostics: Sequence[WeightDiagnostics] | Mapping[str, WeightDiagnostics] | pd.DataFrame,
    *,
    group_order: Sequence[str] | None = None,
) -> FigureRecord:
    """Write the global target-by-source heatmap of exact final-weight mass."""

    output_root = Path(run_output_dir)
    matrix, groups = _diagnostic_mass_matrix(diagnostics, group_order)
    visualizations_dir = output_root / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    path = visualizations_dir / "effective-weight-mass.png"
    figure_side = min(24.0, max(7.0, 3.5 + 0.42 * len(groups)))
    maximum = max(1.0, float(np.max(matrix)))

    with matplotlib.rc_context(_RC_PARAMS):
        figure = Figure(figsize=(figure_side, figure_side))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        try:
            image = axis.imshow(
                matrix,
                cmap="magma",
                vmin=0.0,
                vmax=maximum,
                interpolation="nearest",
                aspect="auto",
            )
            positions = np.arange(len(groups))
            labels = _tick_labels(groups)
            axis.set_xticks(positions)
            axis.set_yticks(positions)
            axis.set_xticklabels(labels, rotation=45, horizontalalignment="right")
            axis.set_yticklabels(labels)
            axis.set(xlabel="Source group", ylabel="Target network")
            axis.set_title("Effective final-weight mass by target and source group")
            colour_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colour_bar.set_label("Percentage of target model's total weight mass")
            if len(groups) <= 12:
                threshold = maximum * 0.52
                for target_index in range(len(groups)):
                    for source_index in range(len(groups)):
                        value = matrix[target_index, source_index]
                        axis.text(
                            source_index,
                            target_index,
                            f"{value:.1f}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=7,
                            color="white" if value >= threshold else "black",
                        )
            figure.subplots_adjust(left=0.2, right=0.9, top=0.92, bottom=0.2)
            figure.savefig(
                path,
                dpi=_FIGURE_DPI,
                format="png",
                metadata={"Software": "SPATHI"},
            )
        finally:
            figure.clear()

    return _file_record(
        output_root,
        path,
        kind="effective-mass-heatmap",
        target_group=None,
        n_cells=None,
        displayed_cells=None,
        n_groups=len(groups),
    )


def write_visualization_manifest(
    run_output_dir: Path,
    embedding: VisualizationEmbedding,
    figures: Sequence[FigureRecord],
    *,
    verify_hashes: bool = True,
) -> VisualizationResult:
    """Write the deterministic visualization manifest after all figures exist."""

    output_root = Path(run_output_dir)
    visualizations_dir = output_root / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    records = tuple(
        sorted(
            figures,
            key=lambda record: (
                record.kind,
                "" if record.target_group is None else record.target_group,
                record.relative_path,
            ),
        )
    )
    if not records:
        raise ValueError("at least one visualization figure record is required")
    paths = [record.relative_path for record in records]
    if len(set(paths)) != len(paths):
        raise ValueError("visualization figure records contain duplicate paths")
    visualization_root = visualizations_dir.resolve()
    for record in records:
        relative_path = Path(record.relative_path)
        if relative_path.is_absolute():
            raise ValueError("visualization figure paths must be relative")
        path = (output_root / relative_path).resolve()
        try:
            path.relative_to(visualization_root)
        except ValueError as exc:
            raise ValueError(
                "visualization figure paths must remain inside the visualizations directory"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(f"visualization figure is missing: {path}")
        if verify_hashes:
            current_sha256, current_size = _hash_file(path)
        else:
            current_sha256, current_size = record.sha256, path.stat().st_size
        if current_sha256 != record.sha256 or current_size != record.size_bytes:
            raise ValueError(f"visualization figure changed after its record was created: {path}")

    manifest_path = visualizations_dir / "manifest.json"
    payload = {
        "schema_version": "1.0",
        "projection": embedding.to_metadata(),
        "figures": [record.to_metadata() for record in records],
        "interpretation": [
            "Figure colours use exact final model weights, not weights recomputed in two dimensions.",
            "A two-dimensional projection can omit distance carried by additional dimensions.",
            "Weight mass measures statistical contribution and is not a causal or lineage relation.",
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return VisualizationResult(
        directory=visualizations_dir,
        manifest_path=manifest_path,
        figures=records,
    )


__all__ = [
    "FigureRecord",
    "VisualizationEmbedding",
    "VisualizationResult",
    "prepare_visualization_embedding",
    "safe_group_filename",
    "write_effective_mass_heatmap",
    "write_target_weight_panel",
    "write_visualization_manifest",
]
