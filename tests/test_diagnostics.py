import numpy as np
import pandas as pd
import pytest

from spathi.diagnostics import compute_weight_diagnostics, diagnostics_frame, effective_sample_size


def test_effective_sample_size_matches_exact_formula() -> None:
    weights = np.array([1.0, 0.5, 0.0, 0.25])
    expected = weights.sum() ** 2 / np.square(weights).sum()
    assert effective_sample_size(weights) == pytest.approx(expected)


def test_weight_diagnostics_record_mass_for_every_group() -> None:
    groups = pd.Series(["A", "A", "B", "B", "C"])
    weights = np.array([1.0, 1.0, 0.25, 0.5, 0.25])
    result = compute_weight_diagnostics(
        weights, groups, "A", emit_warnings=False, low_ess_fraction=0.0
    )
    assert result.n_target_cells == 2
    assert result.total_weight == 3.0
    assert result.target_weight == 2.0
    assert result.external_weight == 1.0
    assert result.target_mass_percent == pytest.approx(200.0 / 3.0)
    assert result.external_mass_percent == pytest.approx(100.0 / 3.0)
    assert result.group_weight_mass == {"A": 2.0, "B": 0.75, "C": 0.25}
    assert result.positive_cell_count == 5

    frame = diagnostics_frame([result])
    assert frame["source_group"].tolist() == ["A", "B", "C"]
    assert frame.loc[frame["source_group"] == "B", "source_weight"].item() == 0.75


def test_diagnostics_warn_for_external_dominance_and_low_ess() -> None:
    groups = ["A", "B", "B", "B", "B"]
    weights = np.array([0.1, 1.0, 0.0, 0.0, 0.0])
    with pytest.warns(RuntimeWarning) as emitted:
        result = compute_weight_diagnostics(
            weights,
            groups,
            "A",
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
        weights,
        groups,
        "A",
        emit_warnings=False,
        low_ess_fraction=0.0,
    )

    assert list(result.group_weight_mass) == ["C", "A", "B"]
    assert result.group_weight_mass == {"C": 1.0, "A": 1.25, "B": 0.5}
    assert result.n_target_cells == 2


@pytest.mark.parametrize("missing_group", [None, np.nan, pd.NA], ids=["none", "nan", "pd-na"])
@pytest.mark.parametrize("use_series", [False, True], ids=["sequence", "series"])
def test_diagnostics_reject_missing_cell_groups_before_string_conversion(
    missing_group: object,
    use_series: bool,
) -> None:
    raw_groups = ["A", missing_group]
    groups = pd.Series(raw_groups) if use_series else raw_groups

    with pytest.raises(ValueError, match="cell_groups contains a missing"):
        compute_weight_diagnostics(
            np.array([1.0, 0.5]),
            groups,
            "A",
            emit_warnings=False,
        )


@pytest.mark.parametrize("missing_target", [None, np.nan, pd.NA], ids=["none", "nan", "pd-na"])
def test_diagnostics_reject_missing_target_group_before_string_conversion(
    missing_target: object,
) -> None:
    with pytest.raises(ValueError, match="target_group must be a non-missing"):
        compute_weight_diagnostics(
            np.array([1.0, 0.5]),
            ["A", "B"],
            missing_target,
            emit_warnings=False,
        )
