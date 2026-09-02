"""Offline interactive report generation for one SPATHI inference run.

The workflow prepares one shared two-dimensional embedding, streams each target
group's weights into :class:`InteractiveReportBuilder`, and writes one standalone
``report.html`` after every target group has been observed.  Display points are a
deterministic stratified subset shared by all targets; every aggregate statistic is
calculated from the complete weight vectors.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .diagnostics import WeightDiagnostics
from .representation import DistanceSpace, RepresentationResult
from .weighting import WeightResult

ProjectionKind = Literal["distance-pca", "auxiliary-pca"]

DEFAULT_MAX_DISPLAY_CELLS = 30_000
DEFAULT_MAX_TARGET_CELL_VALUES = 300_000
_REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportEmbedding:
    """One auditable two-dimensional projection shared by the complete report."""

    coordinates: np.ndarray
    centroid_coordinates: np.ndarray
    cell_ids: tuple[str, ...]
    cell_groups: tuple[str, ...]
    group_ids: tuple[str, ...]
    group_codes: np.ndarray
    group_positions: tuple[np.ndarray, ...]
    cell_hashes: np.ndarray
    x_label: str
    y_label: str
    projection_kind: ProjectionKind
    distance_space: DistanceSpace
    explained_variance_ratio: tuple[float, ...]
    notes: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return the projection contract embedded in the report."""

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


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportArtifact:
    """Published HTML report provenance returned to the workflow."""

    path: Path
    sha256: str
    size_bytes: int
    total_cells: int
    sampled_cells: int
    n_groups: int

    def to_metadata(self) -> dict[str, Any]:
        """Return compact JSON-compatible artifact metadata."""

        return {
            "path": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "self_contained": True,
            "total_cells": self.total_cells,
            "sampled_cells": self.sampled_cells,
            "n_groups": self.n_groups,
            "sampling": "deterministic-stratified-sha256",
        }


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def _axis_label(name: str, explained: Sequence[float], index: int) -> str:
    if index < len(explained) and np.isfinite(explained[index]):
        return f"{name} ({100.0 * explained[index]:.1f}% variance)"
    return name


def _two_columns(values: np.ndarray) -> np.ndarray:
    coordinates = np.zeros((values.shape[0], 2), dtype=np.float64)
    coordinates[:, : min(2, values.shape[1])] = values[:, :2]
    return coordinates


def _stable_cell_hashes(cells: Sequence[str]) -> np.ndarray:
    hashes = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(cell.encode("utf-8")).digest()[:8], "big")
            for cell in cells
        ),
        dtype=np.uint64,
        count=len(cells),
    )
    return _readonly(hashes)


def _build_group_index(
    cell_groups: Sequence[str],
    group_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    lookup = {group: index for index, group in enumerate(group_ids)}
    try:
        codes = np.fromiter(
            (lookup[group] for group in cell_groups),
            dtype=np.intp,
            count=len(cell_groups),
        )
    except KeyError as exc:
        raise ValueError(f"cell group has no matching centroid: {exc.args[0]!r}") from None
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(group_ids))
    if np.any(counts == 0):
        missing = [group_ids[int(index)] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"centroids contain groups without cells: {missing!r}")
    stops = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), stops[:-1]))
    positions = tuple(
        _readonly(order[start:stop]) for start, stop in zip(starts, stops, strict=True)
    )
    return _readonly(codes), positions


def _aligned_cell_groups(cell_groups: pd.Series, cells: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(cell_groups, pd.Series):
        raise TypeError("cell_groups must be a pandas Series indexed by cell identifier")
    if cell_groups.index.has_duplicates:
        raise ValueError("cell_groups index contains duplicate cell identifiers")
    aligned = cell_groups.copy(deep=False)
    aligned.index = aligned.index.map(str)
    if len(aligned) != len(cells) or set(aligned.index) != set(cells):
        raise ValueError("cell_groups must cover report cells exactly")
    values = aligned.loc[list(cells)].to_numpy(dtype=object)
    if pd.isna(values).any() or any(not str(value).strip() for value in values):
        raise ValueError("cell_groups contains missing or empty labels")
    return tuple(map(str, values))


def prepare_report_embedding(
    representation: RepresentationResult,
    cell_groups: pd.Series,
    centroids: pd.DataFrame,
    *,
    random_state: int = 0,
) -> ReportEmbedding:
    """Prepare the shared PC1/PC2 view used by the offline report.

    PCA-distance runs reuse the first fitted components. Expression-distance runs
    fit a deterministic auxiliary two-component PCA for display only; this never
    changes the distances or weights used by SPATHI.
    """

    if not isinstance(representation, RepresentationResult):
        raise TypeError("representation must be a RepresentationResult")
    if not isinstance(centroids, pd.DataFrame):
        raise TypeError("centroids must be a pandas DataFrame")
    if type(random_state) is not int or random_state < 0:
        raise ValueError("random_state must be a non-negative integer")

    values = np.asarray(representation.values, dtype=np.float64)
    cells = tuple(map(str, representation.cell_ids))
    dimensions = tuple(map(str, representation.dimension_names))
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("representation must contain at least one cell and one dimension")
    if values.shape != (len(cells), len(dimensions)) or len(set(cells)) != len(cells):
        raise ValueError("representation identifiers do not match its matrix")
    if not np.isfinite(values).all():
        raise ValueError("representation must contain only finite values")

    groups_by_cell = _aligned_cell_groups(cell_groups, cells)
    if centroids.index.has_duplicates:
        raise ValueError("centroid group identifiers must be unique")
    groups = tuple(map(str, centroids.index))
    if not groups or any(not group.strip() for group in groups):
        raise ValueError("centroid group identifiers must be non-empty")
    if tuple(map(str, centroids.columns)) != dimensions:
        raise ValueError("centroid dimensions do not match the representation")
    centroid_values = centroids.to_numpy(dtype=np.float64, copy=False)
    if centroid_values.shape != (len(groups), len(dimensions)):
        raise ValueError("centroids must contain one complete row per group")
    if not np.isfinite(centroid_values).all():
        raise ValueError("centroids must contain only finite values")
    if set(groups_by_cell) != set(groups):
        raise ValueError("centroids must contain every observed cell group exactly once")

    group_codes, group_positions = _build_group_index(groups_by_cell, groups)
    notes = [
        "Final-weight colours always use the exact weights passed to inference.",
        "Spatial separation in this PCA view does not by itself equal the configured "
        "distance; exact weighting distances are shown in Distance to final weight.",
        "A two-dimensional projection can omit separation carried by later dimensions.",
    ]
    if representation.distance_space == "pca":
        explained = tuple(map(float, representation.explained_variance_ratio or ()))
        if len(explained) != values.shape[1]:
            raise ValueError("PCA explained variance does not match the fitted representation")
        coordinates = _two_columns(values)
        centroid_coordinates = _two_columns(centroid_values)
        projection_kind: ProjectionKind = "distance-pca"
        x_label = _axis_label("PC1", explained, 0)
        if values.shape[1] >= 2:
            y_label = _axis_label("PC2", explained, 1)
        else:
            y_label = "No second component (fixed at 0)"
            notes.append("The configured PCA distance space contains one component.")
    else:
        component_count = min(2, values.shape[1], max(1, values.shape[0] - 1))
        solver = "full" if min(values.shape) <= 2 else "randomized"
        pca = PCA(n_components=component_count, svd_solver=solver, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            projected = pca.fit_transform(values)
            projected_centroids = pca.transform(centroid_values)
        coordinates = _two_columns(np.nan_to_num(projected, copy=False))
        centroid_coordinates = _two_columns(np.nan_to_num(projected_centroids, copy=False))
        explained = tuple(
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
        x_label = _axis_label("Auxiliary PC1", explained, 0)
        if component_count >= 2:
            y_label = _axis_label("Auxiliary PC2", explained, 1)
        else:
            y_label = "No second auxiliary component (fixed at 0)"
            notes.append("The auxiliary display PCA contains one component.")
        notes.append(
            "This PCA is auxiliary: SPATHI distances and weights were calculated in expression space."
        )

    if not np.isfinite(coordinates).all() or not np.isfinite(centroid_coordinates).all():
        raise ValueError("report projection produced non-finite coordinates")
    return ReportEmbedding(
        coordinates=_readonly(coordinates),
        centroid_coordinates=_readonly(centroid_coordinates),
        cell_ids=cells,
        cell_groups=groups_by_cell,
        group_ids=groups,
        group_codes=group_codes,
        group_positions=group_positions,
        cell_hashes=_stable_cell_hashes(cells),
        x_label=x_label,
        y_label=y_label,
        projection_kind=projection_kind,
        distance_space=representation.distance_space,
        explained_variance_ratio=explained,
        notes=tuple(notes),
    )


def report_sample_size(
    n_cells: int,
    n_groups: int,
    *,
    max_display_cells: int = DEFAULT_MAX_DISPLAY_CELLS,
    max_target_cell_values: int = DEFAULT_MAX_TARGET_CELL_VALUES,
) -> int:
    """Return the exact display-cell count used by report sampling and memory planning."""

    for name, value in (("n_cells", n_cells), ("n_groups", n_groups)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_groups > n_cells:
        raise ValueError("n_groups cannot exceed n_cells")
    if type(max_display_cells) is not int or max_display_cells <= 0:
        raise ValueError("max_display_cells must be a positive integer")
    if type(max_target_cell_values) is not int or max_target_cell_values <= 0:
        raise ValueError("max_target_cell_values must be a positive integer")
    target_limited = max(n_groups, max_target_cell_values // n_groups)
    budget = min(n_cells, max(max_display_cells, n_groups), target_limited)
    return max(n_groups, budget)


def _deterministic_stratified_sample(
    embedding: ReportEmbedding,
    *,
    max_display_cells: int,
    max_target_cell_values: int,
) -> np.ndarray:
    """Return one proportional, rare-group-preserving sample shared by all targets."""

    n_cells = len(embedding.cell_ids)
    n_groups = len(embedding.group_ids)
    budget = report_sample_size(
        n_cells,
        n_groups,
        max_display_cells=max_display_cells,
        max_target_cell_values=max_target_cell_values,
    )
    if budget >= n_cells:
        return np.arange(n_cells, dtype=np.intp)

    counts = np.asarray([positions.size for positions in embedding.group_positions], dtype=np.int64)
    allocation = np.ones(n_groups, dtype=np.int64)
    remaining = budget - n_groups
    capacities = counts - 1
    capacity_total = int(capacities.sum())
    if remaining > 0 and capacity_total > 0:
        exact = remaining * capacities.astype(np.float64) / capacity_total
        additions = np.floor(exact).astype(np.int64)
        additions = np.minimum(additions, capacities)
        allocation += additions
        still_needed = remaining - int(additions.sum())
        if still_needed:
            remainders = exact - additions
            order = np.lexsort((np.arange(n_groups), -remainders))
            for group_index in order:
                if still_needed == 0:
                    break
                available = int(counts[group_index] - allocation[group_index])
                if available <= 0:
                    continue
                allocation[group_index] += 1
                still_needed -= 1
        if still_needed != 0:  # pragma: no cover - arithmetic invariant
            raise RuntimeError("stratified report sample allocation is incomplete")

    selected: list[np.ndarray] = []
    for positions, count in zip(embedding.group_positions, allocation, strict=True):
        count_int = int(count)
        if count_int == positions.size:
            chosen = positions
        else:
            local = np.argpartition(embedding.cell_hashes[positions], count_int - 1)[:count_int]
            chosen = positions[local]
        selected.append(np.asarray(chosen, dtype=np.intp))
    result = np.sort(np.concatenate(selected))
    if result.size != budget or np.unique(result).size != budget:
        raise RuntimeError("stratified report sample contains an invalid number of cells")
    return result


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"report metadata contains an unsupported value: {type(value).__name__}")


def _typed_array(values: np.ndarray, dtype: str) -> dict[str, Any]:
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    array = np.ascontiguousarray(values, dtype=canonical_dtype)
    return {
        "dtype": canonical_dtype.name,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _script_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _plotly_javascript() -> str:
    """Load the vendored Plotly browser bundle only when a report is requested."""

    from plotly.offline import get_plotlyjs

    javascript = get_plotlyjs()
    if not isinstance(javascript, str) or "Plotly" not in javascript:
        raise RuntimeError("Plotly did not provide its offline JavaScript bundle")
    return re.sub(r"</script", r"<\\/script", javascript, flags=re.IGNORECASE)


class InteractiveReportBuilder:
    """Incrementally collect bounded display data and exact group summaries."""

    def __init__(
        self,
        embedding: ReportEmbedding,
        *,
        group_sizes: Mapping[str, int] | None = None,
        run_parameters: Mapping[str, Any] | None = None,
        max_display_cells: int = DEFAULT_MAX_DISPLAY_CELLS,
        max_target_cell_values: int = DEFAULT_MAX_TARGET_CELL_VALUES,
    ) -> None:
        if not isinstance(embedding, ReportEmbedding):
            raise TypeError("embedding must be a ReportEmbedding")
        self.embedding = embedding
        computed_sizes = {
            group: int(positions.size)
            for group, positions in zip(
                embedding.group_ids,
                embedding.group_positions,
                strict=True,
            )
        }
        if group_sizes is None:
            self.group_sizes = computed_sizes
        else:
            canonical_sizes: dict[str, int] = {}
            for key, value in group_sizes.items():
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError("group_sizes values must be positive integers")
                canonical_sizes[str(key)] = value
            if canonical_sizes != computed_sizes:
                raise ValueError("group_sizes do not match the report embedding")
            self.group_sizes = canonical_sizes
        self.run_parameters = _json_compatible(dict(run_parameters or {}))
        self.sample_indices = _deterministic_stratified_sample(
            embedding,
            max_display_cells=max_display_cells,
            max_target_cell_values=max_target_cell_values,
        )
        n_groups = len(embedding.group_ids)
        n_sample = self.sample_indices.size
        shape = (n_groups, n_sample)
        self._distance = np.empty(shape, dtype=np.float64)
        self._base_weight = np.empty(shape, dtype=np.float64)
        self._size_factor = np.empty(shape, dtype=np.float64)
        self._final_weight = np.empty(shape, dtype=np.float64)
        summary_shape = (n_groups, n_groups)
        self._mass_percent = np.empty(summary_shape, dtype=np.float64)
        self._minimum = np.empty(summary_shape, dtype=np.float64)
        self._q1 = np.empty(summary_shape, dtype=np.float64)
        self._median = np.empty(summary_shape, dtype=np.float64)
        self._q3 = np.empty(summary_shape, dtype=np.float64)
        self._maximum = np.empty(summary_shape, dtype=np.float64)
        self._mean = np.empty(summary_shape, dtype=np.float64)
        self._positive_count = np.empty(summary_shape, dtype=np.int64)
        self._target_metrics: list[dict[str, Any] | None] = [None] * n_groups
        self._seen = np.zeros(n_groups, dtype=bool)

    @property
    def total_cells(self) -> int:
        return len(self.embedding.cell_ids)

    @property
    def sampled_cells(self) -> int:
        """Return cells retained in the shared report payload sample."""

        return int(self.sample_indices.size)

    def add_target(self, weights: WeightResult, diagnostics: WeightDiagnostics) -> None:
        """Add one target group's display vectors and full-data exact summaries."""

        if not isinstance(weights, WeightResult):
            raise TypeError("weights must be a WeightResult")
        if not isinstance(diagnostics, WeightDiagnostics):
            raise TypeError("diagnostics must be a WeightDiagnostics")
        target = weights.target_group
        if target not in self.embedding.group_ids:
            raise ValueError(f"target group has no report centroid: {target!r}")
        target_index = self.embedding.group_ids.index(target)
        if self._seen[target_index]:
            raise ValueError(f"target group was added to the report more than once: {target!r}")
        if weights.cells != self.embedding.cell_ids:
            raise ValueError("target weights do not follow the report cell order")
        if weights.cell_groups != self.embedding.cell_groups:
            raise ValueError("target weights disagree with report cell-group assignments")
        if diagnostics.target_group != target or diagnostics.n_cells != self.total_cells:
            raise ValueError("weight diagnostics do not describe the supplied target weights")

        vectors = {
            "distance": np.asarray(weights.distance, dtype=np.float64),
            "base_weight": np.asarray(weights.base_weight, dtype=np.float64),
            "group_size_factor": np.asarray(weights.group_size_factor, dtype=np.float64),
            "final_weight": np.asarray(weights.final_weight, dtype=np.float64),
        }
        for name, values in vectors.items():
            if values.shape != (self.total_cells,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite vector aligned with report cells")
            if np.any(values < 0):
                raise ValueError(f"{name} must not contain negative values")
            if name != "distance" and np.any(values > 1 + 1e-12):
                raise ValueError(f"{name} must remain in [0, 1]")

        sample = self.sample_indices
        self._distance[target_index] = vectors["distance"][sample]
        self._base_weight[target_index] = vectors["base_weight"][sample]
        self._size_factor[target_index] = vectors["group_size_factor"][sample]
        self._final_weight[target_index] = vectors["final_weight"][sample]

        if set(diagnostics.group_mass_percent) != set(self.embedding.group_ids):
            raise ValueError("weight diagnostics must contain every report source group")
        for source_index, (source_group, positions) in enumerate(
            zip(self.embedding.group_ids, self.embedding.group_positions, strict=True)
        ):
            source_weights = vectors["final_weight"][positions]
            quantiles = np.quantile(
                source_weights,
                (0.25, 0.5, 0.75),
                method="linear",
                overwrite_input=True,
            )
            self._mass_percent[target_index, source_index] = diagnostics.group_mass_percent[
                source_group
            ]
            self._minimum[target_index, source_index] = float(np.min(source_weights))
            self._q1[target_index, source_index] = float(quantiles[0])
            self._median[target_index, source_index] = float(quantiles[1])
            self._q3[target_index, source_index] = float(quantiles[2])
            self._maximum[target_index, source_index] = float(np.max(source_weights))
            self._mean[target_index, source_index] = float(np.mean(source_weights))
            self._positive_count[target_index, source_index] = int(
                np.count_nonzero(source_weights > 0)
            )
        if not np.isfinite(self._mass_percent[target_index]).all() or not np.isclose(
            self._mass_percent[target_index].sum(),
            100.0,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("source-group mass percentages must be finite and sum to 100")

        self._target_metrics[target_index] = {
            "target_group": target,
            "n_target_cells": diagnostics.n_target_cells,
            "total_weight": diagnostics.total_weight,
            "target_mass_percent": diagnostics.target_mass_percent,
            "external_mass_percent": diagnostics.external_mass_percent,
            "min_weight": diagnostics.min_weight,
            "max_weight": diagnostics.max_weight,
            "mean_weight": diagnostics.mean_weight,
            "median_weight": diagnostics.median_weight,
            "positive_cell_count": diagnostics.positive_cell_count,
            "effective_sample_size": diagnostics.effective_sample_size,
            "mode": weights.mode,
            "normalization_factor": weights.normalization_factor,
            "warnings": list(diagnostics.warnings),
        }
        self._seen[target_index] = True

    def _payload(self, run_summary: Mapping[str, Any] | None) -> dict[str, Any]:
        missing = [
            group
            for group, seen in zip(self.embedding.group_ids, self._seen, strict=True)
            if not seen
        ]
        if missing:
            raise RuntimeError(f"report is missing target groups: {missing!r}")
        target_metrics = [metric for metric in self._target_metrics if metric is not None]
        sample = self.sample_indices
        return {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "projection": self.embedding.to_metadata(),
            "groups": list(self.embedding.group_ids),
            "safe_groups": [html.escape(group, quote=True) for group in self.embedding.group_ids],
            "group_sizes": [self.group_sizes[group] for group in self.embedding.group_ids],
            "sample": {
                "method": "deterministic-stratified-sha256",
                "total_cells": self.total_cells,
                "sampled_cells": self.sampled_cells,
                "shared_across_targets": True,
            },
            "safe_sample_cells": [
                html.escape(self.embedding.cell_ids[index], quote=True) for index in sample
            ],
            "target_metrics": _json_compatible(target_metrics),
            "run_parameters": self.run_parameters,
            "run_summary": _json_compatible(dict(run_summary or {})),
            "arrays": {
                "coordinates": _typed_array(self.embedding.coordinates[sample], "<f8"),
                "centroids": _typed_array(self.embedding.centroid_coordinates, "<f8"),
                "group_codes": _typed_array(self.embedding.group_codes[sample], "<u4"),
                "distance": _typed_array(self._distance, "<f8"),
                "base_weight": _typed_array(self._base_weight, "<f8"),
                "group_size_factor": _typed_array(self._size_factor, "<f8"),
                "final_weight": _typed_array(self._final_weight, "<f8"),
                "mass_percent": _typed_array(self._mass_percent, "<f8"),
                "minimum": _typed_array(self._minimum, "<f8"),
                "q1": _typed_array(self._q1, "<f8"),
                "median": _typed_array(self._median, "<f8"),
                "q3": _typed_array(self._q3, "<f8"),
                "maximum": _typed_array(self._maximum, "<f8"),
                "mean": _typed_array(self._mean, "<f8"),
                "positive_count": _typed_array(self._positive_count, "<i8"),
                "explained_variance": _typed_array(
                    np.asarray(self.embedding.explained_variance_ratio, dtype=np.float64),
                    "<f8",
                ),
            },
            "interpretation": [
                "Weights shown in the target explorer are the final sample weights used by inference.",
                "The continuous weight scale is fixed to 0–1 for every target group.",
                "Scatter plots use one deterministic shared sample; aggregate values use all cells.",
                "The PCA cloud is a display projection, not a plot of the configured distance. "
                "Use Distance to final weight for exact values; visual geometry agrees with "
                "Euclidean distance only when the displayed components contain the complete "
                "fitted distance space.",
                "Weight mass is a statistical contribution, not a causal or lineage relationship.",
            ],
        }

    def write(
        self,
        path: Path,
        *,
        run_summary: Mapping[str, Any] | None = None,
    ) -> ReportArtifact:
        """Write one exclusive, standalone HTML report and return its provenance."""

        output_path = Path(path)
        if output_path.suffix.lower() != ".html":
            raise ValueError("report path must use the .html suffix")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _script_json(self._payload(run_summary))
        plotly_javascript = _plotly_javascript()
        document = _render_document(plotly_javascript, payload)
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
        sha256, size_bytes = _hash_file(output_path)
        return ReportArtifact(
            path=output_path,
            sha256=sha256,
            size_bytes=size_bytes,
            total_cells=self.total_cells,
            sampled_cells=self.sampled_cells,
            n_groups=len(self.embedding.group_ids),
        )


_REPORT_CSS = r"""
:root{color-scheme:light;--ink:#132238;--muted:#607087;--line:#dce4ee;--paper:#fff;
--wash:#f5f8fc;--blue:#1769aa;--teal:#008c8c;--shadow:0 10px 30px rgba(35,61,91,.09)}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5
system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:32px max(24px,5vw) 26px;
background:linear-gradient(120deg,#102a43,#1769aa 62%,#008c8c);color:#fff}header h1{margin:0;
font-size:clamp(28px,4vw,46px);letter-spacing:.04em}header p{max-width:900px;margin:8px 0 0;
color:#e5f2ff}.shell{max-width:1500px;margin:0 auto;padding:22px}.tabs{display:flex;gap:8px;
flex-wrap:wrap;margin-bottom:18px}.tab{border:1px solid var(--line);border-radius:999px;background:#fff;
padding:9px 16px;color:var(--ink);font-weight:700;cursor:pointer}.tab[aria-selected="true"]{
background:var(--blue);border-color:var(--blue);color:#fff}.panel{display:none}.panel.active{display:block}
.toolbar,.card{background:var(--paper);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}
.toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:15px 18px;margin-bottom:16px}
label{font-weight:750}select{min-width:220px;padding:9px 12px;border:1px solid #aebbc9;border-radius:9px;
background:#fff;color:var(--ink)}.sample-note{margin-left:auto;color:var(--muted)}.metrics{display:grid;
grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:16px}.metric{padding:15px 17px}
.metric span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.metric strong{display:block;margin-top:4px;font-size:22px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.wide{grid-column:1/-1}.plot{min-height:420px;padding:5px}.plot.hero{min-height:650px}.section-title{margin:0 0 12px;
font-size:20px}.copy{padding:24px}.copy h2{margin-top:0}.copy h3{margin-top:28px}.copy p,.copy li{max-width:980px}
.notice{padding:13px 16px;border-left:4px solid var(--teal);background:#edfafa;border-radius:8px;color:#294b55}
.warning-box{margin:0 0 16px;padding:13px 16px;border-left:4px solid #c67c00;background:#fff8e7;border-radius:8px}
.warning-box:empty{display:none}pre{padding:15px;overflow:auto;border:1px solid var(--line);border-radius:10px;background:#f7f9fc;
white-space:pre-wrap;word-break:break-word}.sr-only{position:absolute;width:1px;height:1px;padding:0;
margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.footer{padding:28px;
text-align:center;color:var(--muted);font-size:13px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.plot.hero{min-height:520px}.sample-note{margin-left:0;width:100%}}
"""


_REPORT_APP_JAVASCRIPT = r"""
"use strict";
const DATA=JSON.parse(document.getElementById("spathi-report-data").textContent);
const COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000","#882255","#44AA99","#999933","#AA4499","#117733"];
const SYMBOLS=["circle","square","triangle-up","triangle-down","cross","x","pentagon","hexagon","hourglass","bowtie"];
const PLOT_CONFIG={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:"svg",filename:"spathi-plot"}};
const BASE_LAYOUT={font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif',color:"#132238"},paper_bgcolor:"#fff",plot_bgcolor:"#fff",margin:{l:72,r:32,t:64,b:72},hovermode:"closest"};
function hasWebGL(){try{const canvas=document.createElement("canvas");return Boolean(canvas.getContext("webgl")||canvas.getContext("experimental-webgl"))}catch(_error){return false}}
function decode(spec){const raw=atob(spec.data),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
 const types={float64:Float64Array,uint32:Uint32Array,int64:BigInt64Array};const Type=types[spec.dtype];if(!Type)throw new Error("Unsupported report dtype "+spec.dtype);return new Type(bytes.buffer)}
const A={};for(const [name,spec] of Object.entries(DATA.arrays))A[name]=decode(spec);
const G=DATA.groups.length,S=DATA.sample.sampled_cells,allGroupRows=Array.from({length:G},()=>[]);
for(let i=0;i<S;i++)allGroupRows[Number(A.group_codes[i])].push(i);
function boundedRows(rows,budget){
 if(S<=budget)return rows;
 const capacity=rows.map(indices=>Math.max(0,indices.length-1)),total=capacity.reduce((sum,value)=>sum+value,0),remaining=budget-G,keep=rows.map(()=>1);
 if(remaining>0&&total>0){const exact=capacity.map(value=>remaining*value/total),extra=exact.map((value,index)=>Math.min(capacity[index],Math.floor(value)));let left=remaining-extra.reduce((sum,value)=>sum+value,0);const order=Array.from({length:G},(_,index)=>index).sort((a,b)=>(exact[b]-extra[b])-(exact[a]-extra[a])||a-b);for(const group of order){if(left===0)break;if(extra[group]<capacity[group]){extra[group]++;left--}}for(let group=0;group<G;group++)keep[group]+=extra[group]}
 return rows.map((indices,group)=>{const count=keep[group];if(count>=indices.length)return indices;if(count===1)return [indices[0]];return Array.from({length:count},(_,index)=>indices[Math.round(index*(indices.length-1)/(count-1))])})
}
const WEBGL_AVAILABLE=hasWebGL(),SVG_POINT_BUDGET=Math.max(5000,G),groupRows=WEBGL_AVAILABLE?allGroupRows:boundedRows(allGroupRows,SVG_POINT_BUDGET),PLOTTED_CELLS=groupRows.reduce((sum,rows)=>sum+rows.length,0);
const POINT_TRACE=S>2000&&WEBGL_AVAILABLE?"scattergl":"scatter";
const row=(array,index,width)=>array.subarray(index*width,(index+1)*width);
const take=(array,indices)=>indices.map(index=>Number(array[index]));
const coord=(indices,column)=>indices.map(index=>Number(A.coordinates[index*2+column]));
const safeGroup=index=>DATA.safe_groups[index];
const groupColor=index=>index<COLORS.length?COLORS[index]:`hsl(${(index*137.508)%360},65%,38%)`;
const groupLegend=index=>safeGroup(index)+" ("+groupRows[index].length.toLocaleString()+"/"+DATA.group_sizes[index].toLocaleString()+" plotted)";
function layout(extra){return Object.assign({},BASE_LAYOUT,extra)}
let overviewRendered=false;
function renderOverview(){
 if(overviewRendered)return;overviewRendered=true;
 const cloud=[];for(let g=0;g<G;g++){const idx=groupRows[g];cloud.push({type:POINT_TRACE,mode:"markers",name:groupLegend(g),x:coord(idx,0),y:coord(idx,1),text:idx.map(i=>DATA.safe_sample_cells[i]),hovertemplate:"<b>%{text}</b><br>Group: "+safeGroup(g)+"<extra></extra>",marker:{size:7,color:groupColor(g),symbol:SYMBOLS[g%SYMBOLS.length],opacity:.76}})}
 cloud.push({type:"scatter",mode:"markers+text",name:"Centroids",x:Array.from({length:G},(_,g)=>Number(A.centroids[g*2])),y:Array.from({length:G},(_,g)=>Number(A.centroids[g*2+1])),text:DATA.safe_groups,textposition:"top center",hovertemplate:"Centroid: %{text}<extra></extra>",marker:{size:13,color:DATA.groups.map((_,g)=>groupColor(g)),symbol:"diamond",line:{color:"#fff",width:1.5}}});
 Plotly.newPlot("overview-pca",cloud,layout({title:{text:"Cell groups in the shared PCA view"},xaxis:{title:{text:DATA.projection.x_label}},yaxis:{title:{text:DATA.projection.y_label}},legend:{orientation:"h",y:-.2}}),PLOT_CONFIG);
 Plotly.newPlot("group-sizes",[{type:"bar",x:DATA.safe_groups,y:DATA.group_sizes,marker:{color:DATA.groups.map((_,g)=>groupColor(g))},hovertemplate:"%{x}: %{y:,} cells<extra></extra>"}],layout({title:{text:"Observed cells by group"},xaxis:{title:{text:"Group"}},yaxis:{title:{text:"Cells"},rangemode:"tozero"}}),PLOT_CONFIG);
 const mass=Array.from({length:G},(_,t)=>Array.from(row(A.mass_percent,t,G),Number));
 Plotly.newPlot("mass-heatmap",[{type:"heatmap",z:mass,x:DATA.safe_groups,y:DATA.safe_groups,zmin:0,zmax:100,colorscale:"Viridis",colorbar:{title:{text:"Weight mass (%)"}},hovertemplate:"Target: %{y}<br>Source: %{x}<br>Mass: %{z:.2f}%<extra></extra>"}],layout({title:{text:"Exact effective weight mass"},xaxis:{title:{text:"Source group"}},yaxis:{title:{text:"Target group"},autorange:"reversed"}}),PLOT_CONFIG);
 const ess=DATA.target_metrics.map(m=>100*m.effective_sample_size/DATA.sample.total_cells),targetMass=DATA.target_metrics.map(m=>m.target_mass_percent);
 Plotly.newPlot("ess-overview",[{type:"bar",name:"ESS / all cells",x:DATA.safe_groups,y:ess,marker:{color:"#1769aa"},hovertemplate:"%{x}<br>ESS fraction: %{y:.1f}%<extra></extra>"},{type:"scatter",mode:"lines+markers",name:"Target-group mass",x:DATA.safe_groups,y:targetMass,line:{color:"#D55E00",width:3},hovertemplate:"%{x}<br>Target mass: %{y:.1f}%<extra></extra>"}],layout({title:{text:"Effective sample size and target-group contribution"},xaxis:{title:{text:"Target group"}},yaxis:{title:{text:"Percentage"},range:[0,100]},legend:{orientation:"h",y:-.22}}),PLOT_CONFIG);
 const variance=Array.from(A.explained_variance,Number),auxiliary=DATA.projection.kind==="auxiliary-pca",components=variance.map((_,i)=>(auxiliary?"Auxiliary PC":"PC")+(i+1));let cumulative=0;const cumulativeValues=variance.map(v=>100*(cumulative+=v));
 Plotly.newPlot("pca-variance",[{type:"bar",name:"Individual",x:components,y:variance.map(v=>100*v),marker:{color:"#56B4E9"}},{type:"scatter",mode:"lines+markers",name:"Cumulative",x:components,y:cumulativeValues,line:{color:"#D55E00",width:3}}],layout({title:{text:auxiliary?"Auxiliary PCA explained variance (report only)":"PCA explained variance"},xaxis:{title:{text:"Component"}},yaxis:{title:{text:"Explained variance (%)"},rangemode:"tozero"},legend:{orientation:"h",y:-.22}}),PLOT_CONFIG)
}
function renderTarget(target){
 const weights=row(A.final_weight,target,S),distances=row(A.distance,target,S),base=row(A.base_weight,target,S),factors=row(A.group_size_factor,target,S),traces=[];
 for(let g=0;g<G;g++){const idx=groupRows[g],custom=idx.map(i=>[DATA.safe_sample_cells[i],safeGroup(g),Number(weights[i]),Number(distances[i]),Number(base[i]),Number(factors[i])]);traces.push({type:POINT_TRACE,mode:"markers",name:groupLegend(g),x:coord(idx,0),y:coord(idx,1),customdata:custom,hovertemplate:"<b>%{customdata[0]}</b><br>Source group: %{customdata[1]}<br>Final weight: %{customdata[2]:.6g}<br>Distance: %{customdata[3]:.6g}<br>Base weight: %{customdata[4]:.6g}<br>Size factor: %{customdata[5]:.6g}<extra></extra>",marker:{size:8,color:take(weights,idx),coloraxis:"coloraxis",symbol:SYMBOLS[g%SYMBOLS.length],line:{color:groupColor(g),width:1.2}}})}
 traces.push({type:"scatter",mode:"markers",name:"Target centroid",x:[Number(A.centroids[target*2])],y:[Number(A.centroids[target*2+1])],hovertemplate:"Target centroid: "+safeGroup(target)+"<extra></extra>",marker:{size:19,symbol:"star",color:"#fff",line:{color:"#c43c35",width:3}}});
 Plotly.react("target-pca",traces,layout({title:{text:"Final weights for target "+safeGroup(target)},xaxis:{title:{text:DATA.projection.x_label}},yaxis:{title:{text:DATA.projection.y_label}},coloraxis:{colorscale:"Cividis",cmin:0,cmax:1,colorbar:{title:{text:"Final weight"}}},legend:{orientation:"h",y:-.2}}),PLOT_CONFIG);
 const distanceTraces=[];for(let g=0;g<G;g++){const idx=groupRows[g];distanceTraces.push({type:POINT_TRACE,mode:"markers",name:groupLegend(g),x:take(distances,idx),y:take(weights,idx),customdata:idx.map(i=>[DATA.safe_sample_cells[i],Number(base[i]),Number(factors[i])]),hovertemplate:"<b>%{customdata[0]}</b><br>Distance: %{x:.6g}<br>Final weight: %{y:.6g}<br>Base weight: %{customdata[1]:.6g}<br>Size factor: %{customdata[2]:.6g}<extra></extra>",marker:{size:7,color:groupColor(g),symbol:SYMBOLS[g%SYMBOLS.length],opacity:.72}})}
 Plotly.react("distance-weight",distanceTraces,layout({title:{text:"Distance to final weight"},xaxis:{title:{text:"Weighting distance"},rangemode:"tozero"},yaxis:{title:{text:"Final weight"},range:[-.02,1.02]},legend:{orientation:"h",y:-.22}}),PLOT_CONFIG);
 const masses=Array.from(row(A.mass_percent,target,G),Number);
 Plotly.react("source-mass",[{type:"bar",x:DATA.safe_groups,y:masses,marker:{color:DATA.groups.map((_,g)=>groupColor(g)),line:{color:DATA.groups.map((_,g)=>g===target?"#111":"rgba(0,0,0,0)"),width:DATA.groups.map((_,g)=>g===target?2:0)}},hovertemplate:"Source: %{x}<br>Exact mass: %{y:.3f}%<extra></extra>"}],layout({title:{text:"Exact source-group contribution"},xaxis:{title:{text:"Source group"}},yaxis:{title:{text:"Total final-weight mass (%)"},range:[0,Math.max(100,Math.max(...masses)*1.08)]}}),PLOT_CONFIG);
 const boxes=[];for(let g=0;g<G;g++){const at=target*G+g;boxes.push({type:"box",name:safeGroup(g),x:[safeGroup(g)],q1:[Number(A.q1[at])],median:[Number(A.median[at])],q3:[Number(A.q3[at])],lowerfence:[Number(A.minimum[at])],upperfence:[Number(A.maximum[at])],boxpoints:false,fillcolor:groupColor(g),line:{color:groupColor(g)},hovertemplate:"Source: "+safeGroup(g)+"<br>Min: "+Number(A.minimum[at]).toPrecision(4)+"<br>Q1: "+Number(A.q1[at]).toPrecision(4)+"<br>Median: "+Number(A.median[at]).toPrecision(4)+"<br>Mean: "+Number(A.mean[at]).toPrecision(4)+"<br>Q3: "+Number(A.q3[at]).toPrecision(4)+"<br>Max: "+Number(A.maximum[at]).toPrecision(4)+"<br>Positive cells: "+Number(A.positive_count[at]).toLocaleString()+"<extra></extra>"})}
 Plotly.react("source-box",boxes,layout({title:{text:"Exact full-data weight summaries"},xaxis:{title:{text:"Source group"}},yaxis:{title:{text:"Final weight"},range:[-.02,1.02]},showlegend:false}),PLOT_CONFIG);
 const metric=DATA.target_metrics[target];document.getElementById("metric-cells").textContent=metric.n_target_cells.toLocaleString();document.getElementById("metric-ess").textContent=metric.effective_sample_size.toLocaleString(undefined,{maximumFractionDigits:1});document.getElementById("metric-target-mass").textContent=metric.target_mass_percent.toFixed(1)+"%";document.getElementById("metric-external-mass").textContent=metric.external_mass_percent.toFixed(1)+"%";document.getElementById("metric-positive").textContent=metric.positive_cell_count.toLocaleString();document.getElementById("metric-mean").textContent=metric.mean_weight.toPrecision(4);
 const targetName=DATA.groups[target],projectionName=DATA.projection.kind==="auxiliary-pca"?"auxiliary report-only PCA":"PCA distance-space projection";document.getElementById("target-pca").setAttribute("aria-label","Interactive "+projectionName+" of sampled cells coloured by final weight for target group "+targetName);document.getElementById("target-status").textContent="Showing target group "+targetName+": "+metric.n_target_cells.toLocaleString()+" target cells, effective sample size "+metric.effective_sample_size.toLocaleString(undefined,{maximumFractionDigits:1})+", and "+metric.target_mass_percent.toFixed(1)+"% target-group weight mass.";
 const warnings=document.getElementById("target-warnings");warnings.replaceChildren();for(const message of metric.warnings){const item=document.createElement("div");item.textContent=message;warnings.appendChild(item)}
}
function populateText(){const retained=DATA.sample.sampled_cells,total=DATA.sample.total_cells;document.getElementById("sample-note").textContent=PLOTTED_CELLS<retained?"Plotting "+PLOTTED_CELLS.toLocaleString()+" of "+retained.toLocaleString()+" retained sample cells ("+total.toLocaleString()+" total); SVG fallback was bounded because WebGL is unavailable; aggregates use all cells.":"Displaying "+retained.toLocaleString()+" of "+total.toLocaleString()+" cells; aggregates use all cells.";document.getElementById("target-projection-note").textContent=(DATA.projection.kind==="auxiliary-pca"?"This is an auxiliary report-only PCA projection. ":"This is a two-dimensional PCA projection. ")+"Use Distance to final weight for the exact configured weighting distance.";document.getElementById("projection-note").textContent=DATA.projection.notes.join(" ");document.getElementById("interpretation-list").replaceChildren(...DATA.interpretation.map(text=>{const li=document.createElement("li");li.textContent=text;return li}));document.getElementById("run-summary").textContent=JSON.stringify(DATA.run_summary,null,2);document.getElementById("run-parameters").textContent=JSON.stringify(DATA.run_parameters,null,2)}
const selector=document.getElementById("target-select");DATA.groups.forEach((group,index)=>{const option=document.createElement("option");option.value=String(index);option.textContent=group;selector.appendChild(option)});selector.addEventListener("change",()=>renderTarget(Number(selector.value)));
const tabs=Array.from(document.querySelectorAll("[role=tab]"));function activateTab(button,focus=false){tabs.forEach(item=>{const selected=item===button;item.setAttribute("aria-selected",String(selected));item.tabIndex=selected?0:-1});document.querySelectorAll("[role=tabpanel]").forEach(panel=>panel.classList.toggle("active",panel.id===button.dataset.tab));if(button.dataset.tab==="overview-panel")renderOverview();if(focus)button.focus();setTimeout(()=>document.querySelectorAll(".plot").forEach(plot=>{if(plot.data)Plotly.Plots.resize(plot)}),0)}
tabs.forEach((button,index)=>{button.addEventListener("click",()=>activateTab(button));button.addEventListener("keydown",event=>{let next=null;if(event.key==="ArrowRight")next=(index+1)%tabs.length;else if(event.key==="ArrowLeft")next=(index-1+tabs.length)%tabs.length;else if(event.key==="Home")next=0;else if(event.key==="End")next=tabs.length-1;if(next!==null){event.preventDefault();activateTab(tabs[next],true)}})});
populateText();renderTarget(0);
"""


def _render_document(plotly_javascript: str, payload: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPATHI interactive report</title><style>{_REPORT_CSS}</style></head><body>
<header><h1>SPATHI</h1><p>Interactive population-aware weighting report. Explore the exact
weights assigned to cells for every target group without leaving this file.</p></header>
<main class="shell"><nav class="tabs" role="tablist" aria-label="Report sections">
<button id="target-tab" class="tab" role="tab" data-tab="target-panel" aria-controls="target-panel" aria-selected="true" tabindex="0">Target explorer</button>
<button id="overview-tab" class="tab" role="tab" data-tab="overview-panel" aria-controls="overview-panel" aria-selected="false" tabindex="-1">Overview</button>
<button id="method-tab" class="tab" role="tab" data-tab="method-panel" aria-controls="method-panel" aria-selected="false" tabindex="-1">Method &amp; provenance</button></nav>
<section id="target-panel" class="panel active" role="tabpanel" aria-labelledby="target-tab"><div class="toolbar"><label for="target-select">Target group</label>
<select id="target-select"></select><span id="sample-note" class="sample-note"></span></div>
<p id="target-status" class="sr-only" aria-live="polite"></p>
<p id="target-projection-note" class="notice"></p>
<div id="target-warnings" class="warning-box" aria-live="polite"></div><div class="metrics">
<div class="card metric"><span>Target cells</span><strong id="metric-cells">—</strong></div>
<div class="card metric"><span>Effective sample size</span><strong id="metric-ess">—</strong></div>
<div class="card metric"><span>Target mass</span><strong id="metric-target-mass">—</strong></div>
<div class="card metric"><span>External mass</span><strong id="metric-external-mass">—</strong></div>
<div class="card metric"><span>Positive-weight cells</span><strong id="metric-positive">—</strong></div>
<div class="card metric"><span>Mean weight</span><strong id="metric-mean">—</strong></div></div>
<div class="grid"><div class="card plot hero wide" id="target-pca" role="region" aria-label="Interactive PCA projection of sampled cells coloured by final weight for the selected target group"></div><div class="card plot" id="distance-weight" role="region" aria-label="Interactive scatter of exact weighting distance against final weight"></div>
<div class="card plot" id="source-mass" role="region" aria-label="Interactive bar chart of exact final-weight mass by source group"></div><div class="card plot wide" id="source-box" role="region" aria-label="Interactive exact full-data final-weight summaries by source group"></div></div></section>
<section id="overview-panel" class="panel" role="tabpanel" aria-labelledby="overview-tab"><div class="grid"><div class="card plot hero wide" id="overview-pca" role="region" aria-label="Interactive PCA projection of sampled cells by observed group and group centroids"></div>
<div class="card plot" id="group-sizes" role="region" aria-label="Interactive bar chart of exact observed cell counts by group"></div><div class="card plot" id="pca-variance" role="region" aria-label="Interactive PCA explained-variance chart"></div>
<div class="card plot wide" id="mass-heatmap" role="region" aria-label="Interactive heatmap of exact target-by-source final-weight mass"></div><div class="card plot wide" id="ess-overview" role="region" aria-label="Interactive effective sample size and target-group contribution by target group"></div></div></section>
<section id="method-panel" class="panel" role="tabpanel" aria-labelledby="method-tab"><article class="card copy"><h2>How to read this report</h2>
<p id="projection-note" class="notice"></p><ul id="interpretation-list"></ul>
<h3>Visual encoding</h3><p>Marker fill uses one shared 0–1 Cividis scale for final weight.
Marker shape and outline identify the observed source group; the star marks the selected
target centroid. Group envelopes are deliberately omitted because a two-dimensional PCA
projection does not establish biological boundaries.</p><h3>Data handling</h3><p class="notice">This file contains
the input identifiers of sampled cells, group labels, and derived coordinates, distances,
weights, and summaries. It omits local input/output paths and the expression matrix. Treat the
report according to the data-governance requirements for those identifiers.</p><h3>Run summary</h3><pre id="run-summary"></pre>
<h3>Run parameters</h3><pre id="run-parameters"></pre></article></section></main>
<noscript><p class="shell notice">JavaScript is required to display the interactive plots.</p></noscript>
<footer class="footer">Generated by SPATHI · self-contained offline report</footer>
<script>{plotly_javascript}</script>
<script id="spathi-report-data" type="application/json">{payload}</script>
<script>{_REPORT_APP_JAVASCRIPT}</script></body></html>
"""


__all__ = [
    "InteractiveReportBuilder",
    "ReportArtifact",
    "ReportEmbedding",
    "prepare_report_embedding",
    "report_sample_size",
]
