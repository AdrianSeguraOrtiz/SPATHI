import warnings
from pathlib import Path
from tempfile import TemporaryFile as real_temporary_file

import numpy as np
import pandas as pd
import pytest

import spathi.kernels as kernels_module
from spathi.kernels import (
    apply_kernel,
    resolve_bandwidth,
    resolve_bandwidth_for_mode,
)
from spathi.weighting import compute_weights, prepare_weighting_context


def test_gaussian_and_exponential_kernels_match_equations() -> None:
    distances = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        apply_kernel(distances, 2.0, kernel="gaussian"),
        np.exp(-(distances**2) / 8.0),
    )
    np.testing.assert_allclose(
        apply_kernel(distances, 2.0, kernel="exponential"),
        np.exp(-distances / 2.0),
    )


def test_exponential_kernel_maps_overflowing_ratios_to_zero_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = apply_kernel(
            np.array([np.finfo(np.float64).max]),
            np.finfo(np.float64).tiny,
            kernel="exponential",
        )
    np.testing.assert_array_equal(result, [0.0])


def test_auto_bandwidth_uses_positive_global_median_and_safe_fallback() -> None:
    selection = resolve_bandwidth(np.array([0.0, 1.0, 3.0, 10.0]), "auto")
    assert selection.value == 3.0
    assert selection.method == "auto-median"
    assert selection.positive_distance_count == 3
    assert selection.automatic_reference_value == 3.0
    assert selection.automatic_scale == 1.0

    fallback = resolve_bandwidth(np.zeros(4), "auto")
    assert fallback.value == 1.0
    assert fallback.method == "fallback"
    assert fallback.automatic_reference_value == 1.0
    assert fallback.automatic_scale == 1.0
    assert fallback.fallback_reason is not None


def test_automatic_bandwidth_scale_multiplies_median_and_fallback() -> None:
    selection = resolve_bandwidth(
        np.array([0.0, 1.0, 3.0, 10.0]),
        "auto",
        bandwidth_scale=0.5,
    )
    assert selection.value == 1.5
    assert selection.automatic_reference_value == 3.0
    assert selection.automatic_scale == 0.5

    fallback = resolve_bandwidth(
        np.zeros(4),
        "auto",
        bandwidth_scale=2.0,
        fallback=1.5,
    )
    assert fallback.value == 3.0
    assert fallback.automatic_reference_value == 1.5
    assert fallback.automatic_scale == 2.0


def test_explicit_bandwidth_is_final_and_cannot_be_rescaled() -> None:
    selection = resolve_bandwidth(np.array([1.0, 2.0]), 2.5)
    assert selection.value == 2.5
    assert selection.method == "explicit"
    assert selection.automatic_reference_value is None
    assert selection.automatic_scale is None

    with pytest.raises(ValueError, match="applies only when bandwidth='auto'"):
        resolve_bandwidth(np.array([1.0, 2.0]), 2.5, bandwidth_scale=2.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_bandwidth_scale_must_be_positive_and_finite(scale: float) -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        resolve_bandwidth(np.array([1.0, 2.0]), "auto", bandwidth_scale=scale)


def test_auto_bandwidth_median_does_not_overflow_for_finite_extremes() -> None:
    maximum = np.finfo(np.float64).max
    selection = resolve_bandwidth(np.array([maximum, maximum]), "auto")

    assert selection.value == maximum
    assert selection.method == "auto-median"
    assert selection.fallback_reason is None


def test_auto_bandwidth_median_does_not_underflow_for_subnormals() -> None:
    smallest_positive = np.nextafter(0.0, 1.0)
    selection = resolve_bandwidth(
        np.array([smallest_positive, smallest_positive]),
        "auto",
    )

    assert selection.value == smallest_positive
    assert selection.method == "auto-median"
    assert selection.fallback_reason is None


def test_large_auto_bandwidth_path_uses_exact_disk_backed_median(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_directories: list[Path | None] = []

    def temporary_file(*args: object, **kwargs: object):
        directory = kwargs.get("dir")
        observed_directories.append(None if directory is None else Path(str(directory)))
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setattr(kernels_module, "_BANDWIDTH_IN_MEMORY_ELEMENTS", 1)
    monkeypatch.setattr(kernels_module, "TemporaryFile", temporary_file)
    distances = np.array([0.0, 8.0, 2.0, 4.0, 10.0])
    selection = resolve_bandwidth(distances, "auto", scratch_dir=tmp_path)
    assert selection.value == 6.0
    assert selection.positive_distance_count == 4
    assert observed_directories == [tmp_path]
    assert list(tmp_path.iterdir()) == []


def test_bandwidth_scan_keeps_fortran_distance_storage_copy_free() -> None:
    distances = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    flattened = kernels_module._flat_distance_view(distances)
    assert np.shares_memory(flattened, distances)
    assert resolve_bandwidth(distances).value == 6.0


def test_disk_backed_bandwidth_closes_scratch_file_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[object] = []

    def temporary_file(*args: object, **kwargs: object):
        handle = real_temporary_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(kernels_module, "_BANDWIDTH_IN_MEMORY_ELEMENTS", 1)
    monkeypatch.setattr(kernels_module, "TemporaryFile", temporary_file)
    monkeypatch.setattr(
        kernels_module,
        "_in_place_median",
        lambda values: (_ for _ in ()).throw(RuntimeError("median failed")),
    )

    with pytest.raises(RuntimeError, match="median failed"):
        resolve_bandwidth(np.array([1.0, 2.0, 3.0]), scratch_dir=tmp_path)

    assert len(opened) == 1
    assert opened[0].closed  # type: ignore[union-attr]
    assert list(tmp_path.iterdir()) == []


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
        prepare_weighting_context(groups),
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
        prepare_weighting_context(groups),
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
        prepare_weighting_context(groups),
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
        prepare_weighting_context(groups),
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
    distances = pd.DataFrame(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    context = prepare_weighting_context(groups)
    result = compute_weights(
        "A",
        context,
        mode="group-distance",
        bandwidth=1.0,
        group_size_correction="cap-to-target",
        group_distances=distances,
    )
    np.testing.assert_allclose(result.group_size_factor, [1.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(result.final_weight, result.base_weight * result.group_size_factor)
    uncorrected = compute_weights(
        "A",
        context,
        mode="group-distance",
        bandwidth=1.0,
        group_size_correction="none",
        group_distances=distances,
    )
    np.testing.assert_array_equal(uncorrected.group_size_factor, np.ones(4))


@pytest.mark.parametrize("missing_group", [None, np.nan, pd.NA], ids=["none", "nan", "pd-na"])
@pytest.mark.parametrize("use_series", [False, True], ids=["sequence", "series"])
def test_weighting_apis_reject_missing_cell_groups_before_string_conversion(
    missing_group: object,
    use_series: bool,
) -> None:
    raw_groups = ["A", missing_group]
    groups = pd.Series(raw_groups, index=["a", "b"]) if use_series else raw_groups

    with pytest.raises(ValueError, match="cell_groups contains a missing"):
        prepare_weighting_context(groups)


@pytest.mark.parametrize("missing_target", [None, np.nan, pd.NA], ids=["none", "nan", "pd-na"])
def test_weighting_apis_reject_missing_target_group_before_string_conversion(
    missing_target: object,
) -> None:
    groups = ["A", "B"]
    context = prepare_weighting_context(groups)

    with pytest.raises(ValueError, match="target_group must be a non-missing"):
        compute_weights(
            missing_target,
            context,
            mode="cell-distance",
            bandwidth=1.0,
            group_size_correction="none",
            cell_distances=np.array([0.0, 1.0]),
        )
