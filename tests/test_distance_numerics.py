import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

import spathi.distances as distance_module
from spathi.distances import (
    DEFAULT_WORKING_MEMORY_MIB,
    compute_cell_to_centroid_distances,
    compute_centroid_distances,
)


def test_cosine_rejects_zero_norm_representation_with_cell_identifier() -> None:
    representation = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.0]],
        index=["empty_cell", "expressed_cell"],
    )
    centroids = pd.DataFrame([[1.0, 0.0]], index=["A"])

    with pytest.raises(ValueError, match=r"zero-norm representation rows: 'empty_cell'"):
        compute_cell_to_centroid_distances(representation, centroids, metric="cosine")


def test_cosine_rejects_zero_norm_centroid_with_group_identifier() -> None:
    centroids = pd.DataFrame(
        [[1.0, 0.0], [0.0, 0.0]],
        index=["A", "empty_group"],
    )

    with pytest.raises(ValueError, match=r"zero-norm centroid rows: 'empty_group'"):
        compute_centroid_distances(centroids, metric="cosine")


def test_cosine_cleans_roundoff_for_collinear_vectors_but_preserves_real_distance() -> None:
    angle = 1e-6
    representation = pd.DataFrame(
        [[0.1, 0.2, 0.3], [1.0, 0.0, 0.0]],
        index=["collinear", "nearby"],
    )
    centroids = pd.DataFrame(
        [[0.2, 0.4, 0.6], [np.cos(angle), np.sin(angle), 0.0]],
        index=["same_direction", "small_angle"],
    )

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="cosine",
    )

    assert distances.loc["collinear", "same_direction"] == 0.0
    assert distances.loc["nearby", "small_angle"] > 0.0
    assert distances.loc["nearby", "small_angle"] == pytest.approx(
        1.0 - np.cos(angle),
        rel=1e-3,
    )


def test_cosine_clamps_antiparallel_roundoff_to_mathematical_upper_bound() -> None:
    representation = pd.DataFrame(
        [[0.2973589782784592, -0.9547657503477964]],
        index=["cell"],
    )
    centroids = pd.DataFrame(
        [[-0.2973589782784592, 0.9547657503477966]],
        index=["opposite"],
    )

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="cosine",
    )

    assert distances.loc["cell", "opposite"] == 2.0


def test_bounded_cosine_matches_reference_pairwise_distances() -> None:
    generator = np.random.default_rng(47)
    representation = generator.normal(size=(17, 9))
    centroids = generator.normal(size=(4, 9))

    observed = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="cosine",
        working_memory=0.002,
    )
    expected = pairwise_distances(representation, centroids, metric="cosine")

    np.testing.assert_allclose(observed.to_numpy(), expected, rtol=2e-14, atol=2e-14)


def test_cosine_rescales_tiny_nonzero_vectors_instead_of_treating_them_as_zero() -> None:
    representation = pd.DataFrame([[1e-300, 2e-300]], index=["tiny_cell"])
    centroids = pd.DataFrame([[2e-300, 4e-300]], index=["tiny_group"])

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="cosine",
    )

    assert distances.loc["tiny_cell", "tiny_group"] == 0.0


@pytest.mark.parametrize(
    "magnitude",
    [np.nextafter(0.0, 1.0), 1e-200, 1e155, np.finfo(np.float64).max],
)
def test_euclidean_preserves_representable_distances_at_float64_extremes(
    magnitude: float,
) -> None:
    representation = pd.DataFrame([[0.0]], index=["cell"])
    centroids = pd.DataFrame([[0.0], [magnitude]], index=["origin", "extreme"])

    cell_distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="euclidean",
    )
    centroid_distances = compute_centroid_distances(centroids, metric="euclidean")

    assert cell_distances.loc["cell", "extreme"] == magnitude
    assert centroid_distances.loc["origin", "extreme"] == magnitude
    assert centroid_distances.loc["extreme", "origin"] == magnitude


def test_euclidean_preserves_opposite_sign_distance_when_only_its_square_overflows() -> None:
    magnitude = 1e154
    representation = pd.DataFrame([[-magnitude]], index=["negative"])
    centroids = pd.DataFrame([[magnitude]], index=["positive"])

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="euclidean",
    )

    assert distances.loc["negative", "positive"] == 2.0 * magnitude


def test_default_distance_chunk_budget_reserves_result_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_working_memory: list[float | None] = []

    def fake_chunks(
        cells: np.ndarray,
        centroids: np.ndarray,
        *,
        metric: str,
        working_memory: float | None,
    ) -> list[np.ndarray]:
        del metric
        received_working_memory.append(working_memory)
        return [np.zeros((cells.shape[0], centroids.shape[0]), dtype=np.float64)]

    monkeypatch.setattr(distance_module, "pairwise_distances_chunked", fake_chunks)
    compute_cell_to_centroid_distances(
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        np.array([[1.5, 0.0]]),
    )

    expected = DEFAULT_WORKING_MEMORY_MIB - 16 / 1024**2
    assert received_working_memory == pytest.approx([expected])


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_numeric_prechecks_obey_the_distance_working_memory_budget(
    metric: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_dimensions = 128
    maximum_rows = 2
    working_bytes = 6_144 if metric == "cosine" else maximum_rows * n_dimensions * 16
    working_memory = working_bytes / 1024**2
    representation = np.ones((11, n_dimensions), dtype=np.float64)
    representation[:, 0] = np.arange(1, 12, dtype=np.float64)
    centroids = np.ones((2, n_dimensions), dtype=np.float64)
    centroids[1, 1] = 2.0
    original_abs = np.abs
    observed_rows: list[int] = []

    def tracked_abs(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        if values.ndim == 2 and values.shape[1] == n_dimensions:
            observed_rows.append(values.shape[0])
        return original_abs(values, *args, **kwargs)

    monkeypatch.setattr(distance_module.np, "abs", tracked_abs)

    compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric=metric,  # type: ignore[arg-type]
        working_memory=working_memory,
    )

    assert observed_rows
    assert max(observed_rows) <= maximum_rows


def test_finite_validation_is_blocked_by_working_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_dimensions = 128
    maximum_rows = 2
    working_memory = maximum_rows * n_dimensions * 16 / 1024**2
    representation = np.ones((11, n_dimensions), dtype=np.float64)
    centroids = np.ones((2, n_dimensions), dtype=np.float64)
    original_isfinite = np.isfinite
    observed_rows: list[int] = []

    def tracked_isfinite(values: object, *args: object, **kwargs: object) -> np.ndarray:
        source = np.asarray(values)
        if source.ndim == 2 and source.shape[1] == n_dimensions:
            observed_rows.append(source.shape[0])
        return original_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(distance_module.np, "isfinite", tracked_isfinite)

    compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="euclidean",
        working_memory=working_memory,
    )

    assert observed_rows
    assert max(observed_rows) <= maximum_rows


def test_extreme_cosine_rows_are_normalized_only_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_dimensions = 128
    representation = np.ones((11, n_dimensions), dtype=np.float64)
    representation[0, 0] = 1e300
    centroids = np.ones((2, n_dimensions), dtype=np.float64)
    centroids[1, 1] = 2.0
    working_memory = 6_144 / 1024**2
    original_array = np.array
    copied_rows: list[int] = []

    def tracked_array(values: object, *args: object, **kwargs: object) -> np.ndarray:
        source = np.asarray(values)
        if source.ndim == 2 and source.shape[1] == n_dimensions and kwargs.get("copy") is True:
            copied_rows.append(source.shape[0])
        return original_array(values, *args, **kwargs)

    monkeypatch.setattr(distance_module.np, "array", tracked_array)

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="cosine",
        working_memory=working_memory,
    )

    assert np.isfinite(distances.to_numpy()).all()
    assert copied_rows
    assert max(copied_rows) <= 2


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_distance_chunks_reserve_cleanup_inside_working_memory(
    metric: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_dimensions = 16
    n_centroids = 3
    working_bytes = 8_192
    representation = np.arange(20 * n_dimensions, dtype=np.float64).reshape(20, n_dimensions)
    representation += 1.0
    centroids = np.arange(n_centroids * n_dimensions, dtype=np.float64).reshape(
        n_centroids, n_dimensions
    )
    centroids += 2.0
    original_clean = distance_module._clean_distances
    observed_peaks: list[int] = []

    def tracked_clean(
        values: np.ndarray,
        *,
        metric: str,
        n_dimensions: int,
        working_memory: float,
    ) -> np.ndarray:
        retained = values.nbytes
        if metric == "cosine":
            retained += centroids.nbytes + values.shape[0] * representation.shape[1] * 8
        observed_peaks.append(retained + int(working_memory * 1024**2))
        return original_clean(
            values,
            metric=metric,  # type: ignore[arg-type]
            n_dimensions=n_dimensions,
            working_memory=working_memory,
        )

    monkeypatch.setattr(distance_module, "_clean_distances", tracked_clean)

    compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric=metric,  # type: ignore[arg-type]
        working_memory=working_bytes / 1024**2,
    )

    assert observed_peaks
    assert max(observed_peaks) <= working_bytes


def test_stable_euclidean_chunks_reserve_cleanup_inside_working_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_dimensions = 16
    working_bytes = 8_192
    representation = np.full((20, n_dimensions), 1e154)
    centroids = np.vstack(
        [
            np.full(n_dimensions, -1e154),
            np.full(n_dimensions, 1e154),
        ]
    )
    original_clean = distance_module._clean_distances
    observed_peaks: list[int] = []

    def tracked_clean(
        values: np.ndarray,
        *,
        metric: str,
        n_dimensions: int,
        working_memory: float,
    ) -> np.ndarray:
        difference_bytes = values.shape[0] * representation.shape[1] * 8
        observed_peaks.append(difference_bytes + values.nbytes + int(working_memory * 1024**2))
        return original_clean(
            values,
            metric=metric,  # type: ignore[arg-type]
            n_dimensions=n_dimensions,
            working_memory=working_memory,
        )

    monkeypatch.setattr(distance_module, "_clean_distances", tracked_clean)

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="euclidean",
        working_memory=working_bytes / 1024**2,
    )

    assert np.isfinite(distances.to_numpy()).all()
    assert observed_peaks
    assert max(observed_peaks) <= working_bytes
