import numpy as np
import pytest

from spathi.diagnostics import compute_weight_diagnostics, effective_sample_size
from spathi.weighting import (
    WeightResult,
    canonicalize_sample_weights,
    prepare_weighting_context,
)


def diagnostic_weights(groups: list[str], weights: np.ndarray, target_group: str) -> WeightResult:
    context = prepare_weighting_context(groups)
    effective, canonicalization = canonicalize_sample_weights(weights)
    return WeightResult(
        context=context,
        target_group=target_group,
        distance=np.zeros_like(weights),
        base_weight=weights.copy(),
        group_size_factor=np.ones_like(weights),
        final_weight=effective,
        mode="cell-distance",
        canonicalization=canonicalization,
    )


def test_effective_sample_size_matches_exact_formula() -> None:
    weights = np.array([1.0, 0.5, 0.0, 0.25])
    expected = weights.sum() ** 2 / np.square(weights).sum()
    assert effective_sample_size(weights) == pytest.approx(expected)


def test_weight_diagnostics_record_mass_for_every_group() -> None:
    groups = ["A", "A", "B", "B", "C"]
    weights = np.array([1.0, 1.0, 0.25, 0.5, 0.25])
    result = compute_weight_diagnostics(
        diagnostic_weights(groups, weights, "A"),
        emit_warnings=False,
        low_ess_fraction=0.0,
    )
    assert result.n_target_cells == 2
    assert result.total_weight == 3.0
    assert result.target_weight == 2.0
    assert result.external_weight == 1.0
    assert result.target_mass_percent == pytest.approx(200.0 / 3.0)
    assert result.external_mass_percent == pytest.approx(100.0 / 3.0)
    assert result.group_weight_mass == {"A": 2.0, "B": 0.75, "C": 0.25}
    assert result.positive_cell_count == 5

    assert list(result.group_weight_mass) == ["A", "B", "C"]


def test_diagnostics_warn_for_external_dominance_and_low_ess() -> None:
    groups = ["A", "B", "B", "B", "B"]
    weights = np.array([0.1, 1.0, 0.0, 0.0, 0.0])
    with pytest.warns(RuntimeWarning) as emitted:
        result = compute_weight_diagnostics(
            diagnostic_weights(groups, weights, "A"),
            low_ess_fraction=0.8,
            dominant_external_fraction=0.5,
        )
    messages = " ".join(str(item.message) for item in emitted)
    assert "Effective sample size" in messages
    assert "more total weight" in messages
    assert "excessive" in messages
    assert len(result.warnings) == 3


def test_group_mass_aggregation_preserves_first_seen_group_order() -> None:
    groups = ["C", "A", "C", "B", "A"]
    weights = np.array([0.25, 1.0, 0.75, 0.5, 0.25])

    result = compute_weight_diagnostics(
        diagnostic_weights(groups, weights, "A"),
        emit_warnings=False,
        low_ess_fraction=0.0,
    )

    assert list(result.group_weight_mass) == ["C", "A", "B"]
    assert result.group_weight_mass == {"C": 1.0, "A": 1.25, "B": 0.5}
    assert result.n_target_cells == 2


def test_external_mass_is_summed_directly_without_cancellation() -> None:
    tiny_external_weight = 1e-12
    result = compute_weight_diagnostics(
        diagnostic_weights(
            ["A", "A", "B"],
            np.array([1.0, 1.0, tiny_external_weight]),
            "A",
        ),
        emit_warnings=False,
        low_ess_fraction=0.0,
    )

    assert result.external_weight == tiny_external_weight
    assert result.external_mass_percent > 0.0


def test_diagnostics_distinguish_raw_and_effective_numerical_support() -> None:
    epsilon = np.finfo(np.float64).eps
    result = compute_weight_diagnostics(
        diagnostic_weights(
            ["A", "A", "B"],
            np.array([1.0, epsilon / 2.0, 4.0 * epsilon]),
            "A",
        ),
        emit_warnings=False,
        low_ess_fraction=0.0,
    )

    assert result.raw_positive_cell_count == 3
    assert result.positive_cell_count == 2
    assert result.canonicalized_cell_count == 1
    assert result.canonicalized_weight_mass == epsilon / 2.0
    assert result.canonicalization_threshold == pytest.approx(epsilon)
    assert result.raw_weight_sum == 1.0 + 4.5 * epsilon
