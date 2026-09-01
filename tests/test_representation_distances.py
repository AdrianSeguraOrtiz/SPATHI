import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spathi.distances import compute_distance_matrices
from spathi.prototypes import compute_centroids, compute_prototypes
from spathi.representation import compute_distance_representation


def toy_representation() -> tuple[pd.DataFrame, pd.Series]:
    representation = pd.DataFrame(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
        index=["c1", "c2", "c3", "c4"],
        columns=["d1", "d2"],
    )
    groups = pd.Series(["A", "A", "B", "B"], index=representation.index)
    return representation, groups


def test_arithmetic_centroids_are_computed_once_by_group() -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    assert centroids.index.tolist() == ["A", "B"]
    np.testing.assert_allclose(centroids.to_numpy(), [[1.0, 0.0], [5.0, 0.0]])
    pd.testing.assert_frame_equal(centroids, compute_prototypes(representation, groups))


def test_cell_and_centroid_distances_have_expected_values() -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    distances = compute_distance_matrices(representation, centroids, metric="euclidean")
    np.testing.assert_allclose(
        distances.cell_to_centroid.to_numpy(),
        [[1.0, 5.0], [1.0, 3.0], [3.0, 1.0], [5.0, 1.0]],
    )
    assert distances.cell_to_centroid.to_numpy(copy=False).flags.f_contiguous
    np.testing.assert_allclose(distances.centroid_to_centroid.to_numpy(), [[0.0, 4.0], [4.0, 0.0]])


def test_cell_distances_can_use_disk_backed_output(tmp_path: Path) -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    storage = np.memmap(tmp_path / "distances.memmap", mode="w+", dtype=np.float64, shape=(4, 2))
    distances = compute_distance_matrices(
        representation,
        centroids,
        metric="euclidean",
        cell_distance_output=storage,
    )
    np.testing.assert_allclose(
        distances.cell_to_centroid.to_numpy(),
        [[1.0, 5.0], [1.0, 3.0], [3.0, 1.0], [5.0, 1.0]],
    )
    np.testing.assert_allclose(storage, distances.cell_to_centroid.to_numpy())
    assert np.shares_memory(storage, distances.cell_to_centroid.to_numpy())


def test_pca_caps_components_and_does_not_modify_expression() -> None:
    expression = pd.DataFrame(
        [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]],
        index=["G1", "G2"],
        columns=["c1", "c2", "c3"],
    )
    original = expression.copy()
    result = compute_distance_representation(
        expression,
        distance_space="pca",
        n_components=50,
        pca_svd_solver="full",
    )
    assert result.values.shape == (3, 2)
    assert result.effective_n_components == 2
    assert result.requested_n_components == 50
    assert result.pca_svd_solver_resolution == "explicit"
    pd.testing.assert_frame_equal(expression, original)


def test_pca_caps_components_to_centered_informative_rank_and_delegates_auto_solver() -> None:
    expression = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
            [0.0, 4.0, 2.0],
            [8.0, 3.0, 1.0],
        ],
        index=["G1", "G2", "G3", "G4"],
        columns=["c1", "c2", "c3"],
    )

    result = compute_distance_representation(
        expression,
        distance_space="pca",
        n_components=50,
        pca_svd_solver="auto",
    )

    assert result.values.shape == (3, 2)
    assert result.effective_n_components == 2
    assert result.maximum_informative_n_components == 2
    assert result.pca_svd_solver == "auto"
    assert result.pca_svd_solver_resolution == "delegated-to-scikit-learn"


def test_single_cell_pca_retains_one_structural_component() -> None:
    expression = pd.DataFrame(
        [[1.0], [2.0], [3.0]],
        index=["G1", "G2", "G3"],
        columns=["c1"],
    )

    result = compute_distance_representation(expression, n_components=50)

    assert result.values.shape == (1, 1)
    assert result.effective_n_components == 1
    assert result.maximum_informative_n_components == 0
    assert result.explained_variance_ratio == (0.0,)
    assert result.pca_degenerate is True
    assert result.pca_degeneracy_reason is not None
    assert "fewer than two cells" in result.pca_degeneracy_reason


@pytest.mark.parametrize(
    ("expression", "diagnostic_fragment"),
    [
        (pd.DataFrame([[1.0]], index=["G1"], columns=["c1"]), "fewer than two cells"),
        (
            pd.DataFrame(
                [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]],
                index=["G1", "G2"],
                columns=["c1", "c2", "c3"],
            ),
            "constant across cells",
        ),
    ],
)
def test_degenerate_pca_is_finite_and_structurally_diagnosed(
    expression: pd.DataFrame,
    diagnostic_fragment: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compute_distance_representation(
            expression,
            distance_space="pca",
            pca_svd_solver="full",
        )

    assert np.isfinite(result.values).all()
    assert result.explained_variance_ratio is not None
    assert np.isfinite(result.explained_variance_ratio).all()
    assert set(result.explained_variance_ratio) == {0.0}
    assert result.pca_degenerate is True
    assert result.pca_degeneracy_reason is not None
    assert diagnostic_fragment in result.pca_degeneracy_reason


def test_expression_distance_space_is_independent_and_optionally_standardized() -> None:
    expression = pd.DataFrame(
        [[1.0, 2.0, 4.0], [100.0, 200.0, 400.0]],
        index=["G1", "G2"],
        columns=["c1", "c2", "c3"],
    )
    raw = compute_distance_representation(expression, distance_space="expression")
    standardized = compute_distance_representation(
        expression,
        distance_space="expression",
        distance_standardization="standard",
    )
    np.testing.assert_allclose(raw.values, expression.to_numpy().T)
    np.testing.assert_allclose(standardized.values.mean(axis=0), 0.0, atol=1e-12)
    assert standardized.effective_n_components is None


def test_invalid_metric_is_rejected() -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    with pytest.raises(ValueError, match="metric"):
        compute_distance_matrices(representation, centroids, metric="manhattan")  # type: ignore[arg-type]
