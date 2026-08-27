import warnings

import numpy as np
import pandas as pd
import pytest

import spathi.kernels as kernels_module
from spathi.kernels import (
    exponential_kernel,
    gaussian_kernel,
    resolve_bandwidth,
    resolve_bandwidth_for_mode,
)
from spathi.weighting import compute_group_size_factors, compute_weights


def test_gaussian_and_exponential_kernels_match_equations() -> None:
    distances = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(gaussian_kernel(distances, 2.0), np.exp(-(distances**2) / 8.0))
    np.testing.assert_allclose(exponential_kernel(distances, 2.0), np.exp(-distances / 2.0))


def test_exponential_kernel_maps_overflowing_ratios_to_zero_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = exponential_kernel(
            np.array([np.finfo(np.float64).max]),
            np.finfo(np.float64).tiny,
        )
    np.testing.assert_array_equal(result, [0.0])


def test_auto_bandwidth_uses_positive_global_median_and_safe_fallback() -> None:
    selection = resolve_bandwidth(np.array([0.0, 1.0, 3.0, 10.0]), "auto")
    assert selection.value == 3.0
    assert selection.method == "auto-median"
    assert selection.positive_distance_count == 3

    with pytest.warns(RuntimeWarning, match="fallback"):
        fallback = resolve_bandwidth(np.zeros(4), "auto")
    assert fallback.value == 1.0
    assert fallback.method == "fallback"


def test_large_auto_bandwidth_path_uses_exact_disk_backed_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels_module, "_BANDWIDTH_IN_MEMORY_ELEMENTS", 1)
    distances = np.array([0.0, 8.0, 2.0, 4.0, 10.0])
    selection = resolve_bandwidth(distances, "auto")
    assert selection.value == 6.0
    assert selection.positive_distance_count == 4


def test_bandwidth_scan_keeps_fortran_distance_storage_copy_free() -> None:
    distances = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    flattened = kernels_module._flat_distance_view(distances)
    assert np.shares_memory(flattened, distances)
    assert resolve_bandwidth(distances).value == 6.0


def test_bandwidth_distance_family_depends_on_weight_mode() -> None:
    cells = np.array([[0.0, 2.0], [4.0, 6.0]])
    groups = np.array([[0.0, 20.0], [20.0, 0.0]])
    assert (
        resolve_bandwidth_for_mode(
            "cell-distance",
            cell_to_centroid_distances=cells,
            centroid_distances=groups,
        ).value
        == 4.0
    )
    assert (
        resolve_bandwidth_for_mode(
            "group-distance",
            cell_to_centroid_distances=cells,
            centroid_distances=groups,
        ).value
        == 20.0
    )


def weighting_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    groups = pd.Series(["A", "A", "B", "B"], index=["a_near", "a_far", "b_near", "b_far"])
    cell_distances = pd.DataFrame(
        {
            "A": [0.2, 2.0, 0.1, 1.0],
            "B": [1.0, 2.0, 0.8, 0.2],
        },
        index=groups.index,
    )
    group_distances = pd.DataFrame([[0.0, 1.5], [1.5, 0.0]], index=["A", "B"], columns=["A", "B"])
    return groups, cell_distances, group_distances


def test_cell_distance_depends_only_on_individual_distance_without_correction() -> None:
    groups, cell_distances, group_distances = weighting_inputs()
    result = compute_weights(
        "A",
        groups,
        mode="cell-distance",
        bandwidth=1.0,
        group_size_correction="none",
        cell_distances=cell_distances,
        group_distances=group_distances,
    )
    expected_base = np.exp(-0.5 * np.square(cell_distances["A"].to_numpy()))
    np.testing.assert_allclose(result.base_weight, expected_base)
    np.testing.assert_allclose(result.group_size_factor, 1.0)
    np.testing.assert_allclose(result.final_weight, expected_base / expected_base.max())
    assert result.final_weight[2] > result.final_weight[1]
    assert result.cell_groups[2] != result.cell_groups[1]
    assert result.final_weight.max() == 1.0


def test_cell_distance_pins_every_maximum_tie_to_exactly_one() -> None:
    groups = pd.Series(["A", "B"], index=["a", "b"])
    result = compute_weights(
        "A",
        groups,
        mode="cell-distance",
        bandwidth=1.0,
        group_size_correction="none",
        cell_distances=np.array([0.25, 0.25]),
    )
    np.testing.assert_array_equal(result.final_weight, [1.0, 1.0])


def test_group_anchored_mode_anchors_target_and_keeps_external_cells_individual() -> None:
    groups, cell_distances, _ = weighting_inputs()
    result = compute_weights(
        "A",
        groups,
        mode="cell-distance-group-anchored",
        bandwidth=1.0,
        group_size_correction="none",
        cell_distances=cell_distances,
    )
    np.testing.assert_array_equal(result.base_weight[:2], [1.0, 1.0])
    np.testing.assert_array_equal(result.final_weight[:2], [1.0, 1.0])
    assert result.final_weight[2] != result.final_weight[3]
    assert result.normalization_factor == 1.0


def test_group_distance_assigns_one_common_external_weight() -> None:
    groups, _, group_distances = weighting_inputs()
    result = compute_weights(
        "A",
        groups,
        mode="group-distance",
        bandwidth=1.0,
        group_size_correction="none",
        group_distances=group_distances,
    )
    np.testing.assert_array_equal(result.final_weight[:2], [1.0, 1.0])
    assert result.final_weight[2] == result.final_weight[3]
    assert result.distance[2] == result.distance[3] == 1.5


def test_cap_to_target_is_separate_and_only_caps_larger_external_groups() -> None:
    groups = pd.Series(["A", "B", "B", "C"], index=["a", "b1", "b2", "c"])
    factors = compute_group_size_factors(groups, "A", correction="cap-to-target")
    np.testing.assert_allclose(factors, [1.0, 0.5, 0.5, 1.0])
    np.testing.assert_array_equal(
        compute_group_size_factors(groups, "A", correction="none"), np.ones(4)
    )

    distances = pd.DataFrame(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    result = compute_weights(
        "A",
        groups,
        mode="group-distance",
        bandwidth=1.0,
        group_size_correction="cap-to-target",
        group_distances=distances,
    )
    np.testing.assert_allclose(result.group_size_factor, factors)
    np.testing.assert_allclose(result.final_weight, result.base_weight * result.group_size_factor)
