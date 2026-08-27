import numpy as np
import pandas as pd
import pytest

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


def test_cosine_rescales_tiny_nonzero_vectors_instead_of_treating_them_as_zero() -> None:
    representation = pd.DataFrame([[1e-300, 2e-300]], index=["tiny_cell"])
    centroids = pd.DataFrame([[2e-300, 4e-300]], index=["tiny_group"])

    distances = compute_cell_to_centroid_distances(
        representation,
        centroids,
        metric="cosine",
    )

    assert distances.loc["tiny_cell", "tiny_group"] == 0.0


def test_default_distance_chunk_budget_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert received_working_memory == [DEFAULT_WORKING_MEMORY_MIB]
