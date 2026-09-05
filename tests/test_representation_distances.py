import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import spathi.representation as representation_module
from spathi.centroids import compute_centroids
from spathi.distances import compute_cell_to_centroid_distances, compute_centroid_distances
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


def test_arithmetic_centroids_remain_finite_at_float64_maximum() -> None:
    maximum = np.finfo(np.float64).max
    representation = pd.DataFrame(
        [[maximum, maximum], [maximum, -maximum], [maximum, maximum]],
        index=["c1", "c2", "c3"],
        columns=["same_sign", "mixed_sign"],
    )

    centroids = compute_centroids(representation, ["A", "A", "A"])

    assert centroids.loc["A", "same_sign"] == maximum
    assert centroids.loc["A", "mixed_sign"] == pytest.approx(maximum / 3.0)
    assert np.isfinite(centroids.to_numpy()).all()


def test_centroids_support_explicit_cell_weights_aligned_by_identifier() -> None:
    representation, groups = toy_representation()
    weights = pd.Series([3.0, 1.0, 3.0, 1.0], index=["c4", "c3", "c2", "c1"])

    centroids = compute_centroids(
        representation,
        groups,
        centroid_weights=weights,
    )

    np.testing.assert_allclose(centroids.to_numpy(), [[1.5, 0.0], [5.5, 0.0]])


def test_weighted_centroids_are_invariant_to_group_wide_weight_scaling() -> None:
    representation, groups = toy_representation()
    first = compute_centroids(
        representation,
        groups,
        centroid_weights=[1.0, 3.0, 2.0, 1.0],
    )
    second = compute_centroids(
        representation,
        groups,
        centroid_weights=[10.0, 30.0, 0.5, 0.25],
    )

    np.testing.assert_allclose(first, second)


def test_weighted_centroid_of_one_positive_cell_is_that_cell() -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(
        representation,
        groups,
        centroid_weights=[0.0, 4.0, 0.0, 2.0],
    )

    np.testing.assert_array_equal(centroids.to_numpy(), [[2.0, 0.0], [6.0, 0.0]])


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, 2.0], "length does not match"),
        ([1.0, -1.0, 1.0, 1.0], "must be non-negative"),
        ([1.0, np.nan, 1.0, 1.0], "only finite"),
        ([0.0, 0.0, 1.0, 1.0], "sum to zero for group 'A'"),
    ],
)
def test_weighted_centroids_reject_invalid_weights(
    weights: list[float],
    message: str,
) -> None:
    representation, groups = toy_representation()

    with pytest.raises(ValueError, match=message):
        compute_centroids(representation, groups, centroid_weights=weights)


def test_weighted_centroids_remain_finite_at_float64_extremes() -> None:
    maximum = np.finfo(np.float64).max
    representation = pd.DataFrame(
        [[maximum, maximum], [maximum, -maximum], [maximum, maximum]],
        index=["c1", "c2", "c3"],
        columns=["same_sign", "mixed_sign"],
    )

    centroids = compute_centroids(
        representation,
        ["A", "A", "A"],
        centroid_weights=[maximum, maximum, maximum],
    )

    assert centroids.loc["A", "same_sign"] == maximum
    assert centroids.loc["A", "mixed_sign"] == pytest.approx(maximum / 3.0)
    assert np.isfinite(centroids.to_numpy()).all()


def test_weighted_centroids_tolerate_rescaled_weight_underflow() -> None:
    representation = pd.DataFrame(
        [[1.0], [2.0], [9.0]],
        index=["c1", "c2", "c3"],
        columns=["gene"],
    )

    centroids = compute_centroids(
        representation,
        ["A", "A", "A"],
        centroid_weights=[1e-308, 1e-308, 1e308],
    )

    assert centroids.loc["A", "gene"] == 9.0


def test_cell_and_centroid_distances_have_expected_values() -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    cell_distances = compute_cell_to_centroid_distances(
        representation, centroids, metric="euclidean"
    )
    centroid_distances = compute_centroid_distances(centroids, metric="euclidean")
    np.testing.assert_allclose(
        cell_distances.to_numpy(),
        [[1.0, 5.0], [1.0, 3.0], [3.0, 1.0], [5.0, 1.0]],
    )
    assert cell_distances.to_numpy(copy=False).flags.f_contiguous
    np.testing.assert_allclose(centroid_distances.to_numpy(), [[0.0, 4.0], [4.0, 0.0]])


def test_cell_distances_can_use_disk_backed_output(tmp_path: Path) -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    storage = np.memmap(tmp_path / "distances.memmap", mode="w+", dtype=np.float64, shape=(4, 2))
    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="euclidean",
        output=storage,
    )
    np.testing.assert_allclose(
        distances.to_numpy(),
        [[1.0, 5.0], [1.0, 3.0], [3.0, 1.0], [5.0, 1.0]],
    )
    np.testing.assert_allclose(storage, distances.to_numpy())
    assert np.shares_memory(storage, distances.to_numpy())


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


@pytest.mark.parametrize("standardization", ["none", "standard"])
@pytest.mark.parametrize("solver", ["auto", "full", "randomized"])
def test_pca_in_place_work_buffer_matches_copying_reference(
    standardization: str,
    solver: str,
) -> None:
    rng = np.random.default_rng(20260901)
    expression = pd.DataFrame(
        rng.normal(size=(9, 12)),
        index=[f"G{index}" for index in range(9)],
        columns=[f"c{index}" for index in range(12)],
    )
    cells_by_genes = np.asarray(
        expression.to_numpy(dtype=np.float64, copy=False).T,
        dtype=np.float64,
        order="C",
    )
    if standardization == "standard":
        cells_by_genes = StandardScaler(copy=True).fit_transform(cells_by_genes)
    reference_pca = PCA(
        n_components=4,
        copy=True,
        svd_solver=solver,
        random_state=1729,
    )
    expected = reference_pca.fit_transform(cells_by_genes)

    result = compute_distance_representation(
        expression,
        distance_space="pca",
        n_components=4,
        distance_standardization=standardization,  # type: ignore[arg-type]
        pca_svd_solver=solver,  # type: ignore[arg-type]
        random_state=1729,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        reference_pca.explained_variance_ratio_,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("solver", ["auto", "full", "randomized"])
def test_pca_solver_policy_is_deterministic_for_a_fixed_seed(solver: str) -> None:
    """The exposed solver policy is operational, but every policy must be repeatable."""

    rng = np.random.default_rng(20260906)
    expression = pd.DataFrame(rng.lognormal(mean=1.0, sigma=0.7, size=(8, 24)))

    first = compute_distance_representation(
        expression,
        distance_space="pca",
        n_components=6,
        pca_svd_solver=solver,  # type: ignore[arg-type]
        random_state=31415,
    )
    second = compute_distance_representation(
        expression,
        distance_space="pca",
        n_components=6,
        pca_svd_solver=solver,  # type: ignore[arg-type]
        random_state=31415,
    )

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(
        first.explained_variance_ratio,
        second.explained_variance_ratio,
    )


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_pca_solver_policies_preserve_full_rank_scientific_distances(metric: str) -> None:
    """Exact feasible PCA retains the same distances across numerical solver policies."""

    rng = np.random.default_rng(20260906)
    expression = pd.DataFrame(rng.lognormal(mean=1.0, sigma=0.7, size=(8, 24)))
    groups = [f"group_{index % 4}" for index in range(expression.shape[1])]
    distances: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for solver in ("auto", "full", "randomized"):
        representation = compute_distance_representation(
            expression,
            distance_space="pca",
            n_components=8,
            pca_svd_solver=solver,  # type: ignore[arg-type]
            random_state=31415,
        ).values
        centroids = compute_centroids(representation, groups)
        distances[solver] = (
            compute_cell_to_centroid_distances(
                representation,
                centroids,
                metric=metric,  # type: ignore[arg-type]
            ).to_numpy(),
            compute_centroid_distances(
                centroids,
                metric=metric,  # type: ignore[arg-type]
            ).to_numpy(),
        )

    for solver in ("auto", "randomized"):
        np.testing.assert_allclose(distances[solver][0], distances["full"][0], atol=1e-12)
        np.testing.assert_allclose(distances[solver][1], distances["full"][1], atol=1e-12)


def test_standardization_and_pca_reuse_one_owned_work_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression = np.arange(120, dtype=np.float32).reshape(10, 12)
    expression[:, 1::2] *= 1.5
    original = expression.copy()
    observed: dict[str, object] = {}
    real_scaler = representation_module.StandardScaler
    real_pca = representation_module.PCA

    def tracking_scaler(*args: object, **kwargs: object) -> StandardScaler:
        observed["scaler_copy"] = kwargs.get("copy")
        scaler = real_scaler(*args, **kwargs)
        fit_transform = scaler.fit_transform

        def tracked_fit_transform(
            values: np.ndarray, *fit_args: object, **fit_kwargs: object
        ) -> np.ndarray:
            observed["scaler_input"] = values
            transformed = fit_transform(values, *fit_args, **fit_kwargs)
            observed["scaler_output"] = transformed
            return transformed

        scaler.fit_transform = tracked_fit_transform  # type: ignore[method-assign]
        return scaler

    def tracking_pca(*args: object, **kwargs: object) -> PCA:
        observed["pca_copy"] = kwargs.get("copy")
        pca = real_pca(*args, **kwargs)
        fit_transform = pca.fit_transform

        def tracked_fit_transform(
            values: np.ndarray, *fit_args: object, **fit_kwargs: object
        ) -> np.ndarray:
            observed["pca_input"] = values
            return fit_transform(values, *fit_args, **fit_kwargs)

        pca.fit_transform = tracked_fit_transform  # type: ignore[method-assign]
        return pca

    monkeypatch.setattr(representation_module, "StandardScaler", tracking_scaler)
    monkeypatch.setattr(representation_module, "PCA", tracking_pca)

    compute_distance_representation(
        expression,
        distance_space="pca",
        n_components=4,
        distance_standardization="standard",
        pca_svd_solver="full",
    )

    working = observed["scaler_input"]
    scaled = observed["scaler_output"]
    pca_input = observed["pca_input"]
    assert isinstance(working, np.ndarray)
    assert isinstance(scaled, np.ndarray)
    assert isinstance(pca_input, np.ndarray)
    assert observed["scaler_copy"] is False
    assert observed["pca_copy"] is False
    assert working.dtype == np.float64
    assert working.flags.c_contiguous
    assert working.flags.owndata
    assert not np.shares_memory(working, expression)
    assert np.shares_memory(working, scaled)
    assert np.shares_memory(scaled, pca_input)
    np.testing.assert_array_equal(expression, original)


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
    assert np.shares_memory(raw.values, expression.to_numpy(copy=False))
    assert not np.shares_memory(standardized.values, expression.to_numpy(copy=False))
    assert standardized.effective_n_components is None


def test_invalid_metric_is_rejected() -> None:
    representation, groups = toy_representation()
    centroids = compute_centroids(representation, groups)
    with pytest.raises(ValueError, match="metric"):
        compute_cell_to_centroid_distances(
            representation,
            centroids,
            metric="manhattan",  # type: ignore[arg-type]
        )
