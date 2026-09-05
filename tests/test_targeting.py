from __future__ import annotations

import numpy as np
import pytest

import spathi.targeting as targeting_module
from spathi.targeting import (
    assess_target_eligibility,
    weighted_detected_context_statistics,
)


def test_automatic_eligibility_combines_absolute_fraction_and_variance_requirements() -> None:
    expression = np.zeros((100, 4), dtype=np.float64)
    expression[:19, 0] = 1.0
    expression[:20, 1] = 1.0
    expression[:, 2] = 1.0
    expression[:21, 3] = np.linspace(1.0, 2.0, 21)

    records = assess_target_eligibility(
        expression,
        ["rare", "boundary", "constant", "eligible"],
        mode="automatic",
        min_detected_cells=10,
        min_detected_fraction=0.2,
    )

    assert [record.required_detected_cells for record in records] == [20] * 4
    assert [(record.target, record.eligible, record.reason) for record in records] == [
        ("rare", False, "insufficient_detected_cells"),
        ("boundary", True, "eligible"),
        ("constant", False, "zero_variance"),
        ("eligible", True, "eligible"),
    ]
    assert records[1].detected_fraction == pytest.approx(0.2)


def test_fraction_threshold_does_not_overcount_an_exact_decimal_boundary() -> None:
    expression = np.zeros((100, 1), dtype=np.float64)
    expression[:7, 0] = np.arange(1.0, 8.0)

    (record,) = assess_target_eligibility(
        expression,
        ["boundary"],
        mode="automatic",
        min_detected_cells=1,
        min_detected_fraction=0.07,
    )

    assert record.required_detected_cells == 7
    assert record.detected_fraction == 0.07
    assert record.eligible


def test_all_mode_is_an_unmeasured_unfiltered_reference() -> None:
    records = assess_target_eligibility(
        np.ones((3, 2)),
        ["A", "B"],
        mode="all",
        min_detected_cells=2,
        min_detected_fraction=0.5,
    )

    assert all(record.eligible for record in records)
    assert all(record.reason == "unfiltered_reference" for record in records)
    assert all(record.detected_cells is None for record in records)
    assert all(record.expression_min is None for record in records)
    assert all(record.expression_max is None for record in records)


def test_variability_decision_is_stable_below_and_above_float64_variance_range() -> None:
    maximum = np.finfo(np.float64).max
    expression = np.array(
        [
            [0.0, maximum / 2.0],
            [1.0e-200, maximum],
        ]
    )

    records = assess_target_eligibility(
        expression,
        ["tiny", "huge"],
        mode="automatic",
        min_detected_cells=1,
        min_detected_fraction=0.5,
    )

    assert all(record.eligible for record in records)
    assert records[0].expression_min == 0.0
    assert records[0].expression_max == 1.0e-200
    assert records[1].expression_min == maximum / 2.0
    assert records[1].expression_max == maximum


def test_automatic_eligibility_rejects_centered_expression() -> None:
    with pytest.raises(ValueError, match="non-negative, zero-preserving"):
        assess_target_eligibility(
            np.array([[-1.0], [0.0], [1.0]]),
            ["centered"],
            mode="automatic",
            min_detected_cells=1,
            min_detected_fraction=0.1,
        )


def test_detected_context_statistics_report_weight_fraction_and_effective_size() -> None:
    target = np.array([1.0, 2.0, 0.0, 3.0, 0.0])
    weights = np.array([1.0, 2.0, 100.0, 3.0, 100.0])

    observed = weighted_detected_context_statistics(
        target[:, np.newaxis],
        [0],
        weights,
        weights * weights,
        weight_sum=float(np.sum(weights)),
    )

    assert observed[0][0] == pytest.approx(6.0 / 206.0)
    assert observed[0][1] == pytest.approx(36.0 / 14.0)


def test_detected_context_statistics_are_bitwise_independent_of_batch_width() -> None:
    expression = np.array(
        [
            [1.0, 0.0, 4.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 3.0, 5.0],
        ]
    )
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    squared = weights * weights

    observed = weighted_detected_context_statistics(
        expression,
        [2, 0, 1],
        weights,
        squared,
        weight_sum=float(np.sum(weights)),
    )
    expected = tuple(
        weighted_detected_context_statistics(
            expression,
            [index],
            weights,
            squared,
            weight_sum=float(np.sum(weights)),
        )[0]
        for index in (2, 0, 1)
    )

    assert observed == expected


def test_detected_context_statistics_preserve_reduction_order_at_realistic_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260904)
    expression = (rng.random((101, 17)) < 0.23).astype(np.float64)
    weights = np.exp(rng.normal(size=101))
    squared = weights * weights
    weight_sum = float(np.sum(weights, dtype=np.float64))
    monkeypatch.setattr(
        targeting_module,
        "TARGET_ELIGIBILITY_WORKING_MEMORY_BYTES",
        expression.shape[0] * 2,
    )

    together = weighted_detected_context_statistics(
        expression,
        range(expression.shape[1]),
        weights,
        squared,
        weight_sum=weight_sum,
    )
    separately = tuple(
        weighted_detected_context_statistics(
            expression,
            [index],
            weights,
            squared,
            weight_sum=weight_sum,
        )[0]
        for index in range(expression.shape[1])
    )

    assert together == separately


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"mode": "unknown"}, ValueError),
        ({"min_detected_cells": 0}, ValueError),
        ({"min_detected_fraction": 0.0}, ValueError),
        ({"min_detected_fraction": 1.1}, ValueError),
    ],
)
def test_target_eligibility_rejects_invalid_policy_values(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    options: dict[str, object] = {
        "mode": "automatic",
        "min_detected_cells": 1,
        "min_detected_fraction": 0.1,
        **kwargs,
    }
    with pytest.raises(exception):
        assess_target_eligibility(
            np.ones((3, 1)),
            ["A"],
            **options,  # type: ignore[arg-type]
        )
