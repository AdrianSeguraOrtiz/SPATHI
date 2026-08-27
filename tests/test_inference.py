from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import spathi.inference as inference_module
from spathi.inference import infer_networks, prepare_inference


def inference_data() -> tuple[np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(17)
    n_per_program = 60
    tf1 = rng.normal(size=n_per_program * 2)
    tf2 = rng.normal(size=n_per_program * 2)
    noise = rng.normal(scale=0.03, size=n_per_program * 2)
    target = np.concatenate(
        [tf1[:n_per_program] + noise[:n_per_program], tf2[n_per_program:] + noise[n_per_program:]]
    )
    constant = np.full(n_per_program * 2, 4.0)
    expression = np.column_stack([tf1, tf2, target, constant])
    return expression, ["TF1", "TF2", "G", "CONST"], ["TF1", "TF2"]


def edge_tuples(result: Any) -> list[tuple[str, str, float, str]]:
    return [(edge.context, edge.target, edge.score, edge.source) for edge in result.edges]


def test_estimator_receives_exact_sample_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[np.ndarray] = []

    class FakeEstimator:
        feature_importances_: np.ndarray

        def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> FakeEstimator:
            captured.append(sample_weight.copy())
            self.feature_importances_ = np.full(x.shape[1], 1.0 / x.shape[1])
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FakeEstimator(),
    )
    expression = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    weights = np.array([1.0, 0.25, 0.5])
    infer_networks(
        expression,
        ["TF1", "TF2", "G"],
        ["TF1", "TF2"],
        {"A": weights},
        n_estimators=2,
        threads=1,
    )
    assert len(captured) == 3
    for supplied in captured:
        np.testing.assert_array_equal(supplied, weights)


def test_targets_retain_float64_variation_while_predictors_use_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_targets: list[np.ndarray] = []
    captured_predictor_dtypes: list[np.dtype[np.generic]] = []

    class FakeEstimator:
        feature_importances_: np.ndarray

        def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> FakeEstimator:
            captured_predictor_dtypes.append(x.dtype)
            captured_targets.append(y.copy())
            self.feature_importances_ = np.ones(x.shape[1], dtype=np.float64)
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FakeEstimator(),
    )
    predictor = np.linspace(0.0, 1.0, 64)
    # This is valid float64 variation, but the complete target range is smaller
    # than one float32 ULP around 1.0 and previously collapsed to a constant.
    target = 1.0 + 2.0e-8 * predictor
    assert np.unique(target.astype(np.float32)).size == 1

    result = infer_networks(
        np.column_stack([predictor, target]),
        ["TF", "G"],
        ["TF"],
        {"A": np.ones(predictor.size)},
        n_estimators=2,
        threads=1,
    )

    target_stat = next(stat for stat in result.model_stats if stat.target == "G")
    assert target_stat.status == "trained"
    assert captured_predictor_dtypes == [np.dtype(np.float32)]
    assert captured_targets[0].dtype == np.float64
    np.testing.assert_array_equal(captured_targets[0], target)


def test_unexpected_resource_failures_are_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingEstimator:
        def fit(self, *args: Any, **kwargs: Any) -> None:
            raise MemoryError("simulated resource exhaustion")

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FailingEstimator(),
    )
    expression = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    with pytest.raises(MemoryError, match="resource exhaustion"):
        infer_networks(
            expression,
            ["TF1", "TF2", "G"],
            ["TF1", "TF2"],
            {"A": np.ones(3)},
            n_estimators=2,
            threads=1,
        )


def test_weights_reproducibly_change_target_network() -> None:
    expression, genes, tfs = inference_data()
    first_program = np.r_[np.ones(60), np.zeros(60)]
    second_program = 1.0 - first_program
    result = infer_networks(
        expression,
        genes,
        tfs,
        {"first": first_program, "second": second_program},
        n_estimators=80,
        random_seed=5,
        threads=2,
    )
    scores = {(edge.context, edge.target, edge.source): edge.score for edge in result.edges}
    assert scores[("group:first", "G", "TF1")] > 0.9
    assert scores[("group:first", "G", "TF2")] < 0.1
    assert scores[("group:second", "G", "TF2")] > 0.9
    assert scores[("group:second", "G", "TF1")] < 0.1


def test_all_genes_are_targets_constants_are_recorded_and_autoedges_removed() -> None:
    expression, genes, tfs = inference_data()
    result = infer_networks(
        expression,
        genes,
        tfs,
        {"A": np.ones(expression.shape[0])},
        n_estimators=12,
        random_seed=7,
        threads=1,
    )
    assert result.total_models == len(genes)
    assert {stat.target for stat in result.model_stats} == set(genes)
    assert any(
        skipped.target == "CONST" and skipped.reason == "constant_target"
        for skipped in result.skipped_targets
    )
    assert all(edge.source != edge.target for edge in result.edges)
    assert result.expression_dtype == "float64"
    assert all(edge.sign == "?" for edge in result.edges)


def test_fixed_seed_is_equivalent_across_thread_counts() -> None:
    expression, genes, tfs = inference_data()
    weights = {
        "A": np.linspace(0.2, 1.0, expression.shape[0]),
        "B": np.linspace(1.0, 0.2, expression.shape[0]),
    }
    sequential = infer_networks(
        expression,
        genes,
        tfs,
        weights,
        n_estimators=30,
        random_seed=41,
        threads=1,
    )
    parallel = infer_networks(
        expression,
        genes,
        tfs,
        weights,
        n_estimators=30,
        random_seed=41,
        threads=2,
    )
    assert edge_tuples(sequential) == edge_tuples(parallel)
    assert [item.to_dict() for item in sequential.skipped_targets] == [
        item.to_dict() for item in parallel.skipped_targets
    ]


def test_random_forest_uses_exact_evidence_label() -> None:
    expression, genes, tfs = inference_data()
    result = infer_networks(
        expression,
        genes,
        tfs,
        {"A": np.ones(expression.shape[0])},
        tree_method="random-forest",
        n_estimators=8,
        random_seed=2,
        threads=1,
    )
    assert result.edges
    assert {edge.evidence for edge in result.edges} == {"weighted_random_forest_feature_importance"}


@pytest.mark.parametrize(
    ("tree_method", "expected_bootstrap"),
    [("extra-trees", False), ("random-forest", True)],
)
def test_bootstrap_default_is_resolved_per_estimator(
    tree_method: str, expected_bootstrap: bool
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(
        expression,
        genes,
        tfs,
        tree_method=tree_method,  # type: ignore[arg-type]
        n_estimators=2,
    )
    assert prepared.bootstrap is expected_bootstrap


def test_explicit_bootstrap_overrides_method_default() -> None:
    expression, genes, tfs = inference_data()
    assert (
        prepare_inference(
            expression,
            genes,
            tfs,
            tree_method="random-forest",
            bootstrap=False,
            n_estimators=2,
        ).bootstrap
        is False
    )


def test_inference_rejects_seed_outside_sklearn_range() -> None:
    expression, genes, tfs = inference_data()
    with pytest.raises(ValueError, match="at most"):
        prepare_inference(expression, genes, tfs, random_seed=2**32)


def test_prepared_inference_reuses_matrices_across_progressive_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(
        np.asfortranarray(expression),
        genes,
        tfs,
        n_estimators=20,
        random_seed=29,
    )
    assert prepared.expression_dtype == "float64"
    assert prepared.predictor_dtype == "float32"
    assert prepared.expression_nbytes == expression.size * np.dtype(np.float64).itemsize
    assert prepared.predictor_nbytes == (
        expression.shape[0] * len(tfs) * np.dtype(np.float32).itemsize
    )

    def unexpected_reprepare(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("prepared group inference must not recreate expression matrices")

    monkeypatch.setattr(inference_module, "_prepare_expression", unexpected_reprepare)
    weights = {
        "A": np.linspace(0.1, 1.0, expression.shape[0]),
        "B": np.linspace(1.0, 0.1, expression.shape[0]),
    }
    together = prepared.infer_groups(
        weights,
        group_order=["B", "A"],
        threads=1,
    )
    progressive = [prepared.infer_group(group, weights[group], threads=1) for group in ("B", "A")]
    progressive_edges = sorted(
        (edge.context, edge.target, edge.score, edge.source)
        for result in progressive
        for edge in result.edges
    )
    assert progressive_edges == sorted(edge_tuples(together))
    assert together.group_order == ("B", "A")
    assert [result.group_order for result in progressive] == [("B",), ("A",)]
    assert all(result.total_models == len(genes) for result in progressive)


def test_target_batches_prepare_groups_once_and_match_unbatched_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=12, random_seed=31)
    weights = {"A": np.linspace(0.2, 1.0, expression.shape[0])}
    unbatched = prepared.infer_groups(weights, threads=1)

    prepare_calls = 0
    original_prepare_groups = inference_module._prepare_groups

    def counted_prepare_groups(*args: Any, **kwargs: Any) -> Any:
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare_groups(*args, **kwargs)

    monkeypatch.setattr(inference_module, "_prepare_groups", counted_prepare_groups)
    batches = list(
        prepared.iter_group_target_batches(
            weights,
            target_batch_size=2,
            threads=1,
        )
    )

    assert prepare_calls == 1
    assert all(batch.total_models <= 2 for batch in batches)
    assert sum(batch.completed_models for batch in batches) == len(genes)
    assert [stat.target for batch in batches for stat in batch.model_stats] == sorted(genes)
    assert sorted(edge_tuples(unbatched)) == sorted(
        edge_tuples(batch)[edge_index]
        for batch in batches
        for edge_index in range(len(batch.edges))
    )


def test_complete_target_batch_keeps_multiple_groups_in_one_parallel_plan() -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=4, random_seed=11)
    weights = {
        "A": np.linspace(0.2, 1.0, expression.shape[0]),
        "B": np.linspace(1.0, 0.2, expression.shape[0]),
    }

    batches = list(
        prepared.iter_group_target_batches(
            weights,
            group_order=["A", "B"],
            target_batch_size=len(genes),
            threads=2,
        )
    )

    assert len(batches) == 1
    assert batches[0].group_order == ("A", "B")
    assert batches[0].total_models == len(genes) * len(weights)


@pytest.mark.parametrize("target_batch_size", [0, -1])
def test_target_batch_size_must_be_positive(target_batch_size: int) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=2)
    with pytest.raises(ValueError, match="positive integer"):
        list(
            prepared.iter_group_target_batches(
                {"A": np.ones(expression.shape[0])},
                target_batch_size=target_batch_size,
                threads=1,
            )
        )
