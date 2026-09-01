"""Cell representations used exclusively for SPATHI distance calculations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DistanceSpace = Literal["pca", "expression"]
DistanceStandardization = Literal["none", "standard"]
PCASVDSolver = Literal["auto", "randomized", "full"]
PCASVDSolverResolution = Literal["explicit", "delegated-to-scikit-learn"]


@dataclass(frozen=True, slots=True)
class RepresentationResult:
    """A cell-by-dimension distance representation and its effective settings."""

    values: np.ndarray
    cell_ids: tuple[str, ...]
    dimension_names: tuple[str, ...]
    distance_space: DistanceSpace
    standardization: DistanceStandardization
    requested_n_components: int
    effective_n_components: int | None
    maximum_informative_n_components: int | None
    pca_svd_solver: PCASVDSolver
    pca_svd_solver_resolution: PCASVDSolverResolution | None
    explained_variance_ratio: tuple[float, ...] | None = None
    pca_degenerate: bool = False
    pca_degeneracy_reason: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return a labelled cell-by-dimension copy-free view where possible."""

        return pd.DataFrame(self.values, index=self.cell_ids, columns=self.dimension_names)

    @property
    def representation(self) -> np.ndarray:
        """Descriptive alias for :attr:`values`."""

        return self.values


def _coerce_expression(
    expression: pd.DataFrame | np.ndarray,
    cell_ids: list[str] | tuple[str, ...] | None,
    gene_ids: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if isinstance(expression, pd.DataFrame):
        cells = tuple(map(str, expression.columns))
        genes = tuple(map(str, expression.index))
        values = expression.to_numpy(dtype=np.float64, copy=False)
    else:
        values = np.asarray(expression, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("expression must be a two-dimensional genes-by-cells matrix")
        cells = (
            tuple(cell_ids)
            if cell_ids is not None
            else tuple(f"cell_{i}" for i in range(values.shape[1]))
        )
        genes = (
            tuple(gene_ids)
            if gene_ids is not None
            else tuple(f"gene_{i}" for i in range(values.shape[0]))
        )

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("expression must contain at least one gene and one cell")
    if len(cells) != values.shape[1]:
        raise ValueError("cell_ids length does not match the expression columns")
    if len(genes) != values.shape[0]:
        raise ValueError("gene_ids length does not match the expression rows")
    if len(set(cells)) != len(cells) or len(set(genes)) != len(genes):
        raise ValueError("expression cell and gene identifiers must be unique")
    if not np.isfinite(values).all():
        raise ValueError("expression must contain only finite values")
    return values, tuple(map(str, cells)), tuple(map(str, genes))


def compute_distance_representation(
    expression: pd.DataFrame | np.ndarray,
    *,
    distance_space: DistanceSpace = "pca",
    n_components: int = 50,
    distance_standardization: DistanceStandardization = "none",
    pca_svd_solver: PCASVDSolver = "auto",
    random_state: int | None = 0,
    cell_ids: list[str] | tuple[str, ...] | None = None,
    gene_ids: list[str] | tuple[str, ...] | None = None,
) -> RepresentationResult:
    """Construct the representation used to derive cell weights.

    The supplied expression matrix is expected in genes-by-cells orientation.
    It is transposed internally because scikit-learn represents observations in
    rows.  Standardization, when requested, is fitted per gene across cells.
    Neither this representation nor the standardized values replace the
    original expression values used by the inference models.

    ``n_components`` is capped at the centered-data rank bound
    ``min(n_genes, n_cells - 1)``. The one-cell exception retains one structural
    component whose explained variance is diagnosed as zero, while metadata reports
    that its informative rank bound is zero.
    """

    if distance_space not in {"pca", "expression"}:
        raise ValueError("distance_space must be 'pca' or 'expression'")
    if distance_standardization not in {"none", "standard"}:
        raise ValueError("distance_standardization must be 'none' or 'standard'")
    if pca_svd_solver not in {"auto", "randomized", "full"}:
        raise ValueError("pca_svd_solver must be 'auto', 'randomized', or 'full'")
    if (
        isinstance(n_components, bool)
        or not isinstance(n_components, (int, np.integer))
        or n_components <= 0
    ):
        raise ValueError("n_components must be a positive integer")

    genes_by_cells, cells, genes = _coerce_expression(expression, cell_ids, gene_ids)
    cells_by_genes = np.asarray(genes_by_cells.T, dtype=np.float64, order="C")

    if distance_standardization == "standard":
        cells_by_genes = StandardScaler(copy=True).fit_transform(cells_by_genes)

    if distance_space == "expression":
        return RepresentationResult(
            values=np.asarray(cells_by_genes, dtype=np.float64, order="C"),
            cell_ids=cells,
            dimension_names=genes,
            distance_space="expression",
            standardization=distance_standardization,
            requested_n_components=int(n_components),
            effective_n_components=None,
            maximum_informative_n_components=None,
            pca_svd_solver=pca_svd_solver,
            pca_svd_solver_resolution=None,
            explained_variance_ratio=None,
            pca_degenerate=False,
            pca_degeneracy_reason=None,
        )

    n_cells, n_genes = cells_by_genes.shape
    maximum_informative = min(n_genes, max(0, n_cells - 1))
    effective = min(int(n_components), max(1, maximum_informative))
    # Preserve scikit-learn's documented ``auto`` policy instead of copying its
    # version-specific negotiation or reading the private ``_fit_svd_solver`` state.
    # Dependency versions in run metadata make that delegated policy reproducible.
    pca = PCA(n_components=effective, svd_solver=pca_svd_solver, random_state=random_state)
    # scikit-learn reports an expected RuntimeWarning when total variance is
    # zero (including a one-cell matrix). Capture it and expose a structured
    # diagnostic instead of leaking a low-level numerical warning to the CLI.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        transformed = pca.fit_transform(cells_by_genes)
    runtime_warnings = tuple(
        warning for warning in caught_warnings if issubclass(warning.category, RuntimeWarning)
    )
    for warning in caught_warnings:
        if not issubclass(warning.category, RuntimeWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    transformed = np.asarray(transformed, dtype=np.float64, order="C")
    if not np.isfinite(transformed).all():
        raise ValueError("PCA produced non-finite values; inspect the supplied expression matrix")
    dimensions = tuple(f"PC{i}" for i in range(1, effective + 1))
    explained_values = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    non_finite_explained = not np.isfinite(explained_values).all()
    pca_degenerate = non_finite_explained or bool(runtime_warnings)
    if pca_degenerate:
        if cells_by_genes.shape[0] < 2:
            degeneracy_reason = (
                "fewer than two cells; PCA explained variance is undefined and was recorded as zero"
            )
        elif np.all(cells_by_genes == cells_by_genes[0], axis=None):
            degeneracy_reason = (
                "all expression dimensions are constant across cells; PCA explained variance "
                "is undefined and was recorded as zero"
            )
        else:
            degeneracy_reason = (
                "PCA emitted a numerical warning or non-finite explained-variance ratio; "
                "non-finite ratios were recorded as zero"
            )
        explained_values = np.nan_to_num(
            explained_values,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        degeneracy_reason = None
    explained = tuple(float(value) for value in explained_values)
    return RepresentationResult(
        values=transformed,
        cell_ids=cells,
        dimension_names=dimensions,
        distance_space="pca",
        standardization=distance_standardization,
        requested_n_components=int(n_components),
        effective_n_components=effective,
        maximum_informative_n_components=maximum_informative,
        pca_svd_solver=pca_svd_solver,
        pca_svd_solver_resolution=(
            "delegated-to-scikit-learn" if pca_svd_solver == "auto" else "explicit"
        ),
        explained_variance_ratio=explained,
        pca_degenerate=pca_degenerate,
        pca_degeneracy_reason=degeneracy_reason,
    )


# Natural aliases for callers that prefer shorter names.
build_distance_representation = compute_distance_representation
create_distance_representation = compute_distance_representation


__all__ = [
    "DistanceSpace",
    "DistanceStandardization",
    "PCASVDSolver",
    "RepresentationResult",
    "build_distance_representation",
    "compute_distance_representation",
    "create_distance_representation",
]
