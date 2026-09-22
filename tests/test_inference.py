from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

import spathi.inference as inference_module
from spathi.inference import (
    InferenceResult,
    ModelResult,
    ModelStat,
    SkippedTargetRecord,
    prepare_inference,
)


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


def edge_tuples(result: InferenceResult) -> list[tuple[str, str, float, str]]:
    return [(edge.context, edge.target, edge.score, edge.source) for edge in result.edges]


def attach_fake_forest(
    estimator: object,
    importances: np.ndarray,
    *,
    n_estimators: int,
    node_count: int = 3,
    n_leaves: int = 2,
    max_depth: int = 1,
) -> None:
    """Give estimator doubles the structural surface exposed by sklearn forests."""

    tree = SimpleNamespace(
        tree_=SimpleNamespace(
            node_count=node_count,
            n_leaves=n_leaves,
            max_depth=max_depth,
        ),
        feature_importances_=importances,
    )
    estimator.estimators_ = [tree for _ in range(n_estimators)]  # type: ignore[attr-defined]
    estimator.n_features_in_ = importances.size  # type: ignore[attr-defined]


def run_inference(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    group_weights: dict[str, np.ndarray],
    *,
    target_names: list[str] | None = None,
    group_order: list[str] | None = None,
    threads: str | int = "auto",
    **model_options: Any,
) -> InferenceResult:
    """Exercise the same prepared, bounded-batch engine used by the core."""

    prepared = prepare_inference(
        expression,
        gene_names,
        tf_names,
        target_names=target_names,
        **model_options,
    )
    batches = list(
        prepared.iter_group_target_batches(
            group_weights,
            target_batch_size=prepared.n_targets,
            group_order=group_order,
            threads=threads,
        )
    )
    assert len(batches) == 1
    return batches[0]


def skipped_model_result() -> ModelResult:
    reason = "constant_target"
    return ModelResult(
        edges=(),
        skipped=SkippedTargetRecord(
            target_group="A",
            target="G",
            reason=reason,
            detail="constant",
        ),
        stat=ModelStat(
            target_group="A",
            target="G",
            status=reason,
            random_seed=1,
            n_samples=4,
            n_positive_weight_samples=4,
            weight_sum=4.0,
            n_predictors_input=2,
            n_predictors_used=2,
            discarded_predictors=(),
            constant_predictors=(),
            n_edges=0,
            importance_sum=0.0,
            fit_seconds=0.0,
        ),
        trained=False,
    )


def test_model_result_enforces_status_skip_and_training_invariants() -> None:
    valid = skipped_model_result()
    assert valid.skipped is not None

    with pytest.raises(ValueError, match="trained flag"):
        ModelResult(edges=(), skipped=valid.skipped, stat=valid.stat, trained=True)
    with pytest.raises(ValueError, match="matching its status"):
        ModelResult(
            edges=(),
            skipped=SkippedTargetRecord(
                target_group="A",
                target="G",
                reason="no_variable_predictors",
            ),
            stat=valid.stat,
            trained=False,
        )
    with pytest.raises(ValueError, match="unsupported model status"):
        ModelStat(
            **{**valid.stat.to_dict(), "status": "typo"},  # type: ignore[arg-type]
        )


def test_model_stat_enforces_tree_presence_by_execution_status() -> None:
    base = skipped_model_result().stat.to_dict()
    one_tree = {
        "n_estimators_fitted": 1,
        "tree_nodes_total": 3,
        "tree_nodes_mean": 3.0,
        "tree_nodes_p50": 3.0,
        "tree_nodes_p95": 3.0,
        "tree_nodes_max": 3,
        "tree_leaves_total": 2,
        "tree_leaves_mean": 2.0,
        "tree_leaves_p50": 2.0,
        "tree_leaves_p95": 2.0,
        "tree_leaves_max": 2,
        "tree_depth_total": 1,
        "tree_depth_mean": 1.0,
        "tree_depth_p50": 1.0,
        "tree_depth_p95": 1.0,
        "tree_depth_max": 1,
    }

    for status in (
        "trained",
        "trained_no_positive_importance",
        "invalid_feature_importances",
    ):
        with pytest.raises(ValueError, match="requires at least one fitted tree"):
            ModelStat(**{**base, "status": status})  # type: ignore[arg-type]
        assert (
            ModelStat(  # type: ignore[arg-type]
                **{**base, "status": status, **one_tree},
            ).n_estimators_fitted
            == 1
        )

    for status in (
        "insufficient_positive_weight_samples",
        "constant_target",
        "no_predictors_after_self_exclusion",
        "no_variable_predictors",
    ):
        with pytest.raises(ValueError, match="cannot contain fitted trees"):
            ModelStat(**{**base, "status": status, **one_tree})  # type: ignore[arg-type]
        assert ModelStat(**{**base, "status": status}).n_estimators_fitted == 0  # type: ignore[arg-type]

    zero_tree_failure = ModelStat(**{**base, "status": "model_fit_failed"})
    partial_tree_failure = ModelStat(
        **{**base, "status": "model_fit_failed", **one_tree},  # type: ignore[arg-type]
    )
    assert zero_tree_failure.n_estimators_fitted == 0
    assert partial_tree_failure.n_estimators_fitted == 1


def test_estimator_receives_exact_sample_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[np.ndarray] = []

    class FakeEstimator:
        feature_importances_: np.ndarray

        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators

        def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> FakeEstimator:
            captured.append(sample_weight.copy())
            self.feature_importances_ = np.full(x.shape[1], 1.0 / x.shape[1])
            attach_fake_forest(
                self,
                self.feature_importances_,
                n_estimators=self.n_estimators,
            )
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FakeEstimator(kwargs["n_estimators"]),
    )
    expression = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    weights = np.array([1.0, 0.25, 0.5])
    run_inference(
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


def test_float64_negative_importance_roundoff_is_canonicalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RoundoffEstimator:
        feature_importances_: np.ndarray

        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators

        def fit(
            self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray
        ) -> RoundoffEstimator:
            self.feature_importances_ = np.array(
                [-np.nextafter(0.0, 1.0), 1.0],
                dtype=np.float64,
            )
            attach_fake_forest(
                self,
                self.feature_importances_,
                n_estimators=self.n_estimators,
            )
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: RoundoffEstimator(kwargs["n_estimators"]),
    )
    expression = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 4.0]],
        dtype=np.float64,
    )

    result = run_inference(
        expression,
        ["TF1", "TF2", "G"],
        ["TF1", "TF2"],
        {"A": np.ones(3)},
        target_names=["G"],
        n_estimators=2,
        threads=1,
    )

    assert [(edge.source, edge.score) for edge in result.edges] == [("TF2", 1.0)]
    assert result.model_stats[0].status == "trained"
    assert result.model_stats[0].message == (
        "canonicalized 1 negative feature importance value(s) within float64 roundoff tolerance"
    )


def test_materially_negative_importance_remains_a_fatal_model_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidEstimator:
        feature_importances_: np.ndarray

        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators

        def fit(
            self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray
        ) -> InvalidEstimator:
            self.feature_importances_ = np.array(
                [-2.0 * np.finfo(np.float64).eps, 1.0],
                dtype=np.float64,
            )
            attach_fake_forest(
                self,
                self.feature_importances_,
                n_estimators=self.n_estimators,
            )
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: InvalidEstimator(kwargs["n_estimators"]),
    )
    expression = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 4.0]],
        dtype=np.float64,
    )

    result = run_inference(
        expression,
        ["TF1", "TF2", "G"],
        ["TF1", "TF2"],
        {"A": np.ones(3)},
        target_names=["G"],
        n_estimators=2,
        threads=1,
    )

    assert result.edges == ()
    assert result.model_stats[0].status == "invalid_feature_importances"
    assert result.model_stats[0].message.startswith(
        "estimator returned a materially negative feature importance"
    )


def test_tree_fitting_canonicalizes_machine_precision_weight_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the ExtraTrees impurity failure reproduced by Gaussian tail weights."""

    rng = np.random.default_rng(0)
    regulator = rng.normal(size=10)
    target = rng.normal(size=10)
    weights = np.concatenate((np.ones(5), np.full(5, 1.0e-16)))
    monkeypatch.setattr(inference_module, "stable_task_seed", lambda *_: 42)

    result = run_inference(
        np.column_stack((regulator, target)),
        ["TF", "G"],
        ["TF"],
        {"A": weights},
        target_names=["G"],
        n_estimators=100,
        threads=1,
    )

    assert len(result.model_stats) == 1
    stat = result.model_stats[0]
    assert stat.status == "trained"
    assert stat.n_positive_weight_samples == 5
    assert stat.weight_sum == 5.0
    assert all(np.isfinite(edge.score) for edge in result.edges)


def test_targets_retain_float64_variation_while_predictors_use_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_targets: list[np.ndarray] = []
    captured_predictor_dtypes: list[np.dtype[np.generic]] = []

    class FakeEstimator:
        feature_importances_: np.ndarray

        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators

        def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> FakeEstimator:
            captured_predictor_dtypes.append(x.dtype)
            captured_targets.append(y.copy())
            self.feature_importances_ = np.ones(x.shape[1], dtype=np.float64)
            attach_fake_forest(
                self,
                self.feature_importances_,
                n_estimators=self.n_estimators,
            )
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FakeEstimator(kwargs["n_estimators"]),
    )
    predictor = np.linspace(0.0, 1.0, 64)
    # This is valid float64 variation, but the complete target range is smaller
    # than one float32 ULP around 1.0 and must remain distinct during fitting.
    target = 1.0 + 2.0e-8 * predictor
    assert np.unique(target.astype(np.float32)).size == 1

    result = run_inference(
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


@pytest.mark.parametrize("order", ["C", "F"])
def test_predictors_are_copied_directly_to_the_final_float32_layout(order: str) -> None:
    expression = np.array(
        [[1.25, 2.5, 3.75], [4.0, 5.5, 6.25]],
        dtype=np.float64,
        order=order,
    )

    observed = inference_module._extract_tf_predictors(
        expression,
        np.array([2, 0], dtype=np.intp),
    )

    expected = np.array([[3.75, 1.25], [6.25, 4.0]], dtype=np.float32)
    assert observed.dtype == np.float32
    assert observed.flags.c_contiguous
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_target_subset_is_copied_directly_to_one_fortran_layout(order: str) -> None:
    expression = np.array(
        [[1.25, 2.5, 3.75], [4.0, 5.5, 6.25]],
        dtype=np.float64,
        order=order,
    )

    observed = inference_module._extract_target_responses(
        expression,
        np.array([2, 0], dtype=np.intp),
    )

    expected = np.array([[3.75, 1.25], [6.25, 4.0]], dtype=np.float64)
    assert observed.dtype == np.float64
    assert observed.flags.f_contiguous
    np.testing.assert_array_equal(observed, expected)


def test_bounded_finiteness_scan_checks_every_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_module, "INFERENCE_VALIDATION_WORKING_MEMORY_BYTES", 4)
    values = np.arange(12.0).reshape(3, 4)

    assert inference_module._all_finite_bounded(values)
    values.ravel()[-1] = np.nan
    assert not inference_module._all_finite_bounded(values)


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
        run_inference(
            expression,
            ["TF1", "TF2", "G"],
            ["TF1", "TF2"],
            {"A": np.ones(3)},
            n_estimators=2,
            threads=1,
        )


def test_compact_feature_importance_buffer_is_bitwise_sklearn_equivalent() -> None:
    rng = np.random.default_rng(101)
    predictors = rng.normal(size=(80, 7)).astype(np.float32)
    response = rng.normal(size=80)
    weights = rng.uniform(0.1, 1.0, size=80)
    estimator = inference_module.create_tree_estimator(
        "extra-trees",
        n_estimators=31,
        max_features="sqrt",
        min_samples_leaf=1,
        max_depth=None,
        bootstrap=False,
        random_state=19,
        n_jobs=2,
    )
    estimator.fit(predictors, response, sample_weight=weights)
    expected = np.asarray(estimator.feature_importances_, dtype=np.float64)

    observed = inference_module._extract_feature_importances(estimator)

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("tree_method", ["extra-trees", "random-forest"])
def test_tree_estimator_receives_minimum_leaf_weight_fraction(tree_method: str) -> None:
    estimator = inference_module.create_tree_estimator(
        tree_method,  # type: ignore[arg-type]
        n_estimators=3,
        max_features="sqrt",
        min_samples_leaf=1,
        max_depth=None,
        min_weight_fraction_leaf=0.125,
        bootstrap=tree_method == "random-forest",
        random_state=19,
        n_jobs=1,
    )

    assert estimator.min_weight_fraction_leaf == 0.125


@pytest.mark.parametrize(
    ("tree_method", "bootstrap"),
    [("extra-trees", False), ("random-forest", True)],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("constant_response", [False, True])
def test_fixed_forest_prefix_importances_are_bitwise_independent_fit_equivalent(
    tree_method: str,
    bootstrap: bool,
    n_jobs: int,
    constant_response: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(109)
    predictors = rng.normal(size=(97, 11)).astype(np.float32)
    response = rng.normal(size=97)
    if constant_response:
        response[:] = 1.0
    weights = rng.uniform(0.05, 2.0, size=97)
    weights[::13] = 0.0
    counts = (5, 11, 17)
    context = inference_module._FitContext(
        expression=response[:, np.newaxis],
        tf_names=tuple(f"TF{index}" for index in range(predictors.shape[1])),
        target_to_tf_position=(None,),
        tree_method=tree_method,
        n_estimators=counts[-1],
        max_features="sqrt",
        min_samples_leaf=2,
        max_depth=7,
        bootstrap=bootstrap,
        global_seed=23,
        model_n_jobs=n_jobs,
        min_weight_fraction_leaf=0.0,
    )

    tree_class = ExtraTreeRegressor if tree_method == "extra-trees" else DecisionTreeRegressor
    importance_property = tree_class.feature_importances_
    importance_reads: Counter[int] = Counter()

    def counted_importances(tree: Any) -> np.ndarray:
        importance_reads[id(tree)] += 1
        return importance_property.__get__(tree, type(tree))

    monkeypatch.setattr(tree_class, "feature_importances_", property(counted_importances))
    prefixes = inference_module._fit_fixed_tree_prefixes(
        predictors,
        response,
        weights,
        context=context,
        seed=31,
        estimator_counts=counts,
    )

    assert tuple(outcome.n_estimators_fitted for outcome in prefixes) == counts
    assert tuple(outcome.fit_seconds for outcome in prefixes) == tuple(
        sorted(outcome.fit_seconds for outcome in prefixes)
    )
    assert len(importance_reads) == (0 if constant_response else counts[-1])
    assert all(reads == 1 for reads in importance_reads.values())
    for count, outcome in zip(counts, prefixes, strict=True):
        independent = inference_module.create_tree_estimator(
            tree_method,
            n_estimators=count,
            max_features="sqrt",
            min_samples_leaf=2,
            max_depth=7,
            bootstrap=bootstrap,
            random_state=31,
            n_jobs=n_jobs,
        )
        independent.fit(predictors, response, sample_weight=weights)
        expected = inference_module._extract_feature_importances(independent)
        np.testing.assert_array_equal(outcome.importances, expected)
        structures = tuple(tree.tree_ for tree in independent.estimators_)
        for family, values in (
            ("nodes", np.asarray([tree.node_count for tree in structures], dtype=np.int64)),
            ("leaves", np.asarray([tree.n_leaves for tree in structures], dtype=np.int64)),
            ("depth", np.asarray([tree.max_depth for tree in structures], dtype=np.int64)),
        ):
            summary = inference_module._summarize_tree_measure(values)
            assert getattr(outcome.forest_structure, f"{family}_total") == summary[0]
            assert getattr(outcome.forest_structure, f"{family}_mean") == summary[1]
            assert getattr(outcome.forest_structure, f"{family}_p50") == summary[2]
            assert getattr(outcome.forest_structure, f"{family}_p95") == summary[3]
            assert getattr(outcome.forest_structure, f"{family}_max") == summary[4]
        assert outcome.forest_structure.depth_max <= 7


def test_tree_importance_cache_preserves_prefixes_with_mixed_stumps() -> None:
    class Tree:
        def __init__(self, values: tuple[float, float] | None) -> None:
            self.tree_ = SimpleNamespace(node_count=1 if values is None else 3)
            self.values = values
            self.reads = 0

        @property
        def feature_importances_(self) -> np.ndarray:
            self.reads += 1
            assert self.values is not None
            return np.asarray(self.values, dtype=np.float64)

    trees = [Tree(None), Tree(None), Tree((0.3, 0.7)), Tree(None), Tree((0.1, 0.9))]
    estimator = SimpleNamespace(estimators_=[])
    cache = inference_module._TreeImportanceCache(np.empty((5, 2), dtype=np.float64))
    observed = []
    for count in (2, 4, 5):
        estimator.estimators_ = trees[:count]
        informative = [tree.values for tree in trees[:count] if tree.values is not None]
        if informative:
            expected = np.mean(np.asarray(informative, dtype=np.float64), axis=0, dtype=np.float64)
            expected /= np.sum(expected, dtype=np.float64)
        else:
            expected = np.zeros(2, dtype=np.float64)
        observed.append(cache.extract(estimator))
        np.testing.assert_array_equal(observed[-1], expected)

    assert [tree.reads for tree in trees] == [0, 0, 1, 0, 1]
    np.testing.assert_array_equal(observed[0], np.zeros(2, dtype=np.float64))
    np.testing.assert_array_equal(observed[1], np.array([0.3, 0.7], dtype=np.float64))


def _result_without_runtime(result: InferenceResult) -> tuple[object, ...]:
    stats: list[dict[str, Any]] = []
    for stat in result.model_stats:
        values = stat.to_dict()
        values.pop("fit_seconds")
        stats.append(values)
    return (
        result.edges,
        result.skipped_targets,
        tuple(stats),
        result.group_order,
        result.parallel_plan,
        result.tree_method,
        result.total_models,
        result.completed_models,
        result.trained_models,
        result.expression_dtype,
        result.expression_nbytes,
        result.predictor_nbytes,
    )


@pytest.mark.parametrize("tree_method", ["extra-trees", "random-forest"])
@pytest.mark.parametrize("threads", [1, 2])
def test_prefix_networks_are_exactly_independent_fit_equivalent(
    tree_method: str,
    threads: int,
) -> None:
    expression, genes, tfs = inference_data()
    weights = {
        "A": np.linspace(0.1, 1.0, expression.shape[0]),
        "B": np.linspace(1.0, 0.1, expression.shape[0]),
    }
    counts = (5, 11, 17)
    common = {
        "target_names": ["TF1", "TF2", "G"],
        "tree_method": tree_method,
        "max_features": 1.0,
        "min_samples_leaf": 2,
        "max_depth": 6,
        "random_seed": 43,
    }
    prepared = prepare_inference(
        expression,
        genes,
        tfs,
        n_estimators=counts[-1],
        **common,
    )

    batches = list(
        prepared.iter_group_target_prefix_batches(
            weights,
            estimator_counts=counts,
            target_batch_size=prepared.n_targets,
            group_order=["B", "A"],
            threads=threads,
        )
    )

    assert len(batches) == 1
    assert tuple(prefix.n_estimators for prefix in batches[0]) == counts
    for prefix in batches[0]:
        independent = run_inference(
            expression,
            genes,
            tfs,
            weights,
            group_order=["B", "A"],
            threads=threads,
            n_estimators=prefix.n_estimators,
            **common,
        )
        assert _result_without_runtime(prefix.inference) == _result_without_runtime(independent)
        assert {
            stat.n_estimators_fitted
            for stat in prefix.inference.model_stats
            if stat.status in inference_module.TRAINED_MODEL_STATUSES
        } == {prefix.n_estimators}


@pytest.mark.parametrize(
    ("estimator_counts", "error", "message"),
    [
        ((), ValueError, "cannot be empty"),
        ((0, 17), ValueError, "positive integers"),
        ((5, 5, 17), ValueError, "strictly increasing"),
        ((11, 5, 17), ValueError, "strictly increasing"),
        ((5, 11), ValueError, "final estimator count"),
        ((5, 11.0, 17), TypeError, "positive integers"),
        ((True, 17), TypeError, "positive integers"),
    ],
)
def test_prefix_networks_reject_ambiguous_estimator_schedules(
    estimator_counts: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=17)

    with pytest.raises(error, match=message):
        list(
            prepared.iter_group_target_prefix_batches(
                {"A": np.ones(expression.shape[0])},
                estimator_counts=estimator_counts,  # type: ignore[arg-type]
                target_batch_size=prepared.n_targets,
                threads=1,
            )
        )


def test_fixed_prefix_fit_failure_retains_elapsed_time_and_completed_tree_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTree:
        tree_ = SimpleNamespace(node_count=2, n_leaves=1, max_depth=1)
        feature_importances_ = np.array([1.0])

    class FailingSecondBlockEstimator:
        def __init__(self) -> None:
            self.n_estimators = 10
            self.estimators_: list[FakeTree] = []
            self.calls = 0
            self.n_features_in_ = 1

        def fit(self, *args: Any, **kwargs: Any) -> FailingSecondBlockEstimator:
            self.calls += 1
            if self.calls == 2:
                raise ValueError("simulated second-block failure")
            self.estimators_.extend(FakeTree() for _ in range(self.n_estimators))
            return self

        def set_params(self, *, n_estimators: int) -> FailingSecondBlockEstimator:
            self.n_estimators = n_estimators
            return self

    fake = FailingSecondBlockEstimator()
    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: fake,
    )
    times = iter((10.0, 12.0, 20.0, 23.0))
    monkeypatch.setattr(inference_module, "perf_counter", lambda: next(times))
    context = inference_module._FitContext(
        expression=np.arange(8.0)[:, np.newaxis],
        tf_names=("TF",),
        target_to_tf_position=(None,),
        tree_method="extra-trees",
        n_estimators=30,
        max_features="sqrt",
        min_samples_leaf=1,
        max_depth=None,
        bootstrap=False,
        global_seed=1,
        model_n_jobs=1,
    )

    with pytest.raises(inference_module._TreeFitFailure) as caught:
        inference_module._fit_fixed_tree_prefixes(
            np.arange(8.0, dtype=np.float32)[:, np.newaxis],
            np.arange(8.0),
            np.ones(8),
            context=context,
            seed=1,
            estimator_counts=(10, 20, 30),
        )

    assert str(caught.value.error) == "simulated second-block failure"
    assert caught.value.fit_seconds == 5.0
    assert caught.value.n_estimators_fitted == 10
    assert len(caught.value.completed_outcomes) == 1
    assert caught.value.completed_outcomes[0].n_estimators_fitted == 10
    assert caught.value.forest_structure.nodes_total == 20
    assert caught.value.forest_structure.leaves_total == 10
    assert caught.value.forest_structure.depth_total == 10


def test_prefix_api_preserves_completed_prefixes_after_late_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTree:
        tree_ = SimpleNamespace(node_count=3, n_leaves=2, max_depth=1)
        feature_importances_ = np.array([0.75, 0.25], dtype=np.float64)

    class FailingExtensionEstimator:
        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators
            self.estimators_: list[FakeTree] = []
            self.calls = 0
            self.n_features_in_ = 2

        def fit(self, *args: Any, **kwargs: Any) -> FailingExtensionEstimator:
            self.calls += 1
            if self.calls == 2:
                raise ValueError("simulated late prefix failure")
            self.estimators_.extend(
                FakeTree() for _ in range(self.n_estimators - len(self.estimators_))
            )
            return self

        def set_params(self, *, n_estimators: int) -> FailingExtensionEstimator:
            self.n_estimators = n_estimators
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FailingExtensionEstimator(kwargs["n_estimators"]),
    )
    expression = np.asarray(
        [
            [0.0, 1.0, 0.5],
            [1.0, 0.0, 1.5],
            [2.0, 1.0, 2.5],
            [3.0, 1.5, 3.5],
            [4.0, 2.0, 4.5],
            [5.0, 3.0, 5.5],
        ],
        dtype=np.float64,
    )
    prepared = prepare_inference(
        expression,
        ["TF1", "TF2", "G"],
        ["TF1", "TF2"],
        target_names=["G"],
        n_estimators=6,
        random_seed=17,
    )

    batches = list(
        prepared.iter_group_target_prefix_batches(
            {"A": np.ones(expression.shape[0])},
            estimator_counts=(2, 4, 6),
            target_batch_size=1,
            threads=1,
        )
    )

    assert len(batches) == 1
    prefixes = batches[0]
    assert tuple(prefix.n_estimators for prefix in prefixes) == (2, 4, 6)
    stats = tuple(prefix.inference.model_stats[0] for prefix in prefixes)
    assert tuple(stat.status for stat in stats) == (
        "trained",
        "model_fit_failed",
        "model_fit_failed",
    )
    assert tuple(stat.n_estimators_fitted for stat in stats) == (2, 2, 2)
    assert tuple(prefix.inference.trained_models for prefix in prefixes) == (1, 0, 0)
    assert prefixes[0].inference.skipped_targets == ()
    assert tuple(prefix.inference.skipped_targets[0].reason for prefix in prefixes[1:]) == (
        "model_fit_failed",
        "model_fit_failed",
    )
    assert all("simulated late prefix failure" in stat.message for stat in stats[1:])


def test_weights_reproducibly_change_target_network() -> None:
    expression, genes, tfs = inference_data()
    first_program = np.r_[np.ones(60), np.zeros(60)]
    second_program = 1.0 - first_program
    result = run_inference(
        expression,
        genes,
        tfs,
        {"first": first_program, "second": second_program},
        n_estimators=80,
        max_features=1.0,
        min_weight_fraction_leaf=0.0,
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
    result = run_inference(
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


def test_explicit_target_subset_uses_only_requested_responses_and_all_tf_predictors() -> None:
    expression, genes, tfs = inference_data()
    result = run_inference(
        expression,
        genes,
        tfs,
        {"A": np.ones(expression.shape[0])},
        target_names=["G"],
        n_estimators=12,
        random_seed=7,
        threads=1,
    )

    assert result.total_models == 1
    assert {stat.target for stat in result.model_stats} == {"G"}
    assert {edge.target for edge in result.edges} == {"G"}
    assert {edge.source for edge in result.edges} == {"TF1", "TF2"}
    assert result.expression_nbytes == expression.shape[0] * np.dtype(np.float64).itemsize

    prepared = prepare_inference(
        expression,
        genes,
        tfs,
        target_names=["G"],
        n_estimators=2,
    )
    assert prepared.gene_names == tuple(genes)
    assert prepared.target_names == ("G",)
    assert prepared.n_genes == len(genes)
    assert prepared.n_targets == 1
    assert prepared.target_expression_additional_nbytes == prepared.expression_nbytes


def test_explicit_all_gene_targets_reuse_default_response_storage_and_results() -> None:
    expression, genes, tfs = inference_data()
    weights = {"A": np.ones(expression.shape[0])}
    default = run_inference(
        expression,
        genes,
        tfs,
        weights,
        n_estimators=12,
        random_seed=7,
        threads=1,
    )
    explicit = run_inference(
        expression,
        genes,
        tfs,
        weights,
        target_names=genes,
        n_estimators=12,
        random_seed=7,
        threads=1,
    )
    prepared = prepare_inference(expression, genes, tfs, target_names=genes, n_estimators=2)

    assert edge_tuples(default) == edge_tuples(explicit)
    assert prepared.target_expression_additional_nbytes == 0


def test_low_level_inference_uses_the_stable_model_defaults() -> None:
    expression, genes, tfs = inference_data()

    prepared = prepare_inference(expression, genes, tfs)

    assert prepared.n_estimators == 50
    assert prepared.max_features == 0.5
    assert prepared.min_samples_leaf == 2
    assert prepared.max_depth is None
    assert prepared.min_weight_fraction_leaf == 0.1
    assert prepared.bootstrap is False


@pytest.mark.parametrize(
    ("target_names", "message"),
    [([], "cannot be empty"), (["G", "G"], "unique"), (["MISSING"], "absent")],
)
def test_target_names_must_be_nonempty_unique_expression_genes(
    target_names: list[str],
    message: str,
) -> None:
    expression, genes, tfs = inference_data()
    with pytest.raises(ValueError, match=message):
        prepare_inference(expression, genes, tfs, target_names=target_names, n_estimators=2)


def test_constant_predictors_are_excluded_with_correct_edge_mapping_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted_matrices: dict[tuple[float, ...], np.ndarray] = {}
    fitted_matrix_ids: dict[tuple[float, ...], int] = {}

    class FakeEstimator:
        feature_importances_: np.ndarray

        def __init__(self, n_estimators: int) -> None:
            self.n_estimators = n_estimators

        def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> FakeEstimator:
            fitted_matrices[tuple(y.tolist())] = x.copy()
            fitted_matrix_ids[tuple(y.tolist())] = id(x)
            raw_importances = np.arange(1, x.shape[1] + 1, dtype=np.float64)
            self.feature_importances_ = raw_importances / raw_importances.sum()
            attach_fake_forest(
                self,
                self.feature_importances_,
                n_estimators=self.n_estimators,
            )
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FakeEstimator(kwargs["n_estimators"]),
    )
    tf_left = np.array([0.0, 1.0, 2.0, 3.0])
    tf_constant = np.full(4, 7.0)
    tf_right = np.array([3.0, 1.0, 4.0, 2.0])
    target = np.array([1.0, 4.0, 2.0, 8.0])
    target_2 = np.array([2.0, 5.0, 9.0, 3.0])
    expression = np.column_stack([tf_left, tf_constant, tf_right, target, target_2])

    result = run_inference(
        expression,
        ["TF_LEFT", "TF_CONSTANT", "TF_RIGHT", "TARGET", "TARGET_2"],
        ["TF_LEFT", "TF_CONSTANT", "TF_RIGHT"],
        {"A": np.ones(expression.shape[0])},
        n_estimators=2,
        threads=1,
    )

    target_matrix = fitted_matrices[tuple(target.tolist())]
    assert target_matrix.flags.c_contiguous
    np.testing.assert_array_equal(target_matrix, np.column_stack([tf_left, tf_right]))
    # Non-TF targets share the group-level filtered matrix instead of copying it
    # once per target. TF targets still receive a self-excluded matrix.
    assert fitted_matrix_ids[tuple(target.tolist())] == fitted_matrix_ids[tuple(target_2.tolist())]
    target_edges = {
        edge.source: edge.score
        for edge in result.edges
        if edge.context == "group:A" and edge.target == "TARGET"
    }
    assert target_edges == {
        "TF_LEFT": pytest.approx(1.0 / 3.0),
        "TF_RIGHT": pytest.approx(2.0 / 3.0),
    }

    target_stat = next(stat for stat in result.model_stats if stat.target == "TARGET")
    assert target_stat.n_predictors_input == 3
    assert target_stat.n_predictors_used == 2
    assert target_stat.constant_predictors == ("TF_CONSTANT",)
    assert target_stat.discarded_predictors == ("TF_CONSTANT",)

    right_stat = next(stat for stat in result.model_stats if stat.target == "TF_RIGHT")
    assert right_stat.n_predictors_used == 1
    assert right_stat.constant_predictors == ("TF_CONSTANT",)
    assert right_stat.discarded_predictors == ("TF_CONSTANT", "TF_RIGHT")
    assert all(edge.source != edge.target for edge in result.edges)


def test_all_constant_eligible_predictors_are_discarded_without_fitting() -> None:
    expression = np.column_stack(
        [
            np.full(4, 1.0),
            np.full(4, 2.0),
            np.array([0.0, 1.0, 2.0, 3.0]),
        ]
    )

    result = run_inference(
        expression,
        ["TF1", "TF2", "TARGET"],
        ["TF1", "TF2"],
        {"A": np.ones(expression.shape[0])},
        n_estimators=2,
        threads=1,
    )

    target_stat = next(stat for stat in result.model_stats if stat.target == "TARGET")
    assert target_stat.status == "no_variable_predictors"
    assert target_stat.n_predictors_used == 0
    assert target_stat.constant_predictors == ("TF1", "TF2")
    assert target_stat.discarded_predictors == ("TF1", "TF2")
    assert any(
        skipped.target == "TARGET" and skipped.reason == "no_variable_predictors"
        for skipped in result.skipped_targets
    )


def test_group_weight_statistics_are_calculated_once_per_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=2)
    first = np.zeros(expression.shape[0], dtype=np.float64)
    first[0] = 0.5
    second = np.zeros(expression.shape[0], dtype=np.float64)
    second[:2] = (0.25, 0.75)
    observed: list[tuple[int, float]] = []
    original_statistics = inference_module._group_weight_statistics

    def counted_statistics(
        weights: np.ndarray,
        positive_mask: np.ndarray,
    ) -> tuple[int, float]:
        statistics = original_statistics(weights, positive_mask)
        observed.append(statistics)
        return statistics

    monkeypatch.setattr(inference_module, "_group_weight_statistics", counted_statistics)
    batches = list(
        prepared.iter_group_target_batches(
            {"A": first, "B": second},
            target_batch_size=1,
            threads=1,
        )
    )

    assert observed == [(1, 0.5), (2, 1.0)]
    stats = [stat for batch in batches for stat in batch.model_stats]
    assert len(stats) == 2 * len(genes)
    assert {
        (stat.target_group, stat.n_positive_weight_samples, stat.weight_sum) for stat in stats
    } == {("A", 1, 0.5), ("B", 2, 1.0)}


def test_model_tasks_read_cached_weight_statistics_without_cell_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=2)
    weights = np.zeros(expression.shape[0], dtype=np.float64)
    weights[0] = 0.75
    group = inference_module._prepare_groups(
        {"A": weights},
        None,
        n_cells=prepared.n_cells,
        tf_expression=prepared._tf_expression,
        tf_names=prepared.tf_names,
    )[0]
    execution = prepared._prepare_model_execution(
        (group,),
        target_items=((0, prepared.target_names[0]),),
        completed_models=(),
        threads=1,
        executor=None,
    )

    def unexpected_reduction(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model tasks must reuse cached group weight statistics")

    with monkeypatch.context() as patch:
        patch.setattr(inference_module.np, "count_nonzero", unexpected_reduction)
        patch.setattr(inference_module.np, "sum", unexpected_reduction)
        result = inference_module._fit_model_task(execution.tasks[0], execution.context)

    assert result.stat.status == "insufficient_positive_weight_samples"
    assert result.stat.n_positive_weight_samples == 1
    assert result.stat.weight_sum == 0.75


def test_global_tf_ranges_are_calculated_once_for_all_positive_weight_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf_expression = np.ascontiguousarray(
        np.array(
            [
                [0.0, 5.0, 2.0],
                [1.0, 5.0, 2.0],
                [2.0, 5.0, 8.0],
                [3.0, 5.0, 9.0],
            ],
            dtype=np.float32,
        )
    )
    calls: list[str] = []
    original_positions = inference_module._constant_tf_positions

    def counted_positions(
        values: np.ndarray,
        positive_mask: np.ndarray | None,
    ) -> frozenset[int]:
        calls.append("global" if positive_mask is None else "masked")
        return original_positions(values, positive_mask)

    monkeypatch.setattr(inference_module, "_constant_tf_positions", counted_positions)
    groups = inference_module._prepare_groups(
        {
            "A": np.ones(4),
            "B": np.array([0.25, 0.5, 0.75, 1.0]),
            "C": np.array([1.0, 1.0, 0.0, 0.0]),
        },
        None,
        n_cells=4,
        tf_expression=tf_expression,
        tf_names=("TF1", "TF2", "TF3"),
    )

    assert calls == ["global", "masked"]
    assert groups[0].constant_tf_positions is groups[1].constant_tf_positions
    assert groups[0].constant_tf_positions == frozenset({1})
    assert groups[2].constant_tf_positions == frozenset({1, 2})


def test_fixed_seed_is_equivalent_across_thread_counts() -> None:
    expression, genes, tfs = inference_data()
    weights = {
        "A": np.linspace(0.2, 1.0, expression.shape[0]),
        "B": np.linspace(1.0, 0.2, expression.shape[0]),
    }
    sequential = run_inference(
        expression,
        genes,
        tfs,
        weights,
        n_estimators=30,
        random_seed=41,
        threads=1,
    )
    parallel = run_inference(
        expression,
        genes,
        tfs,
        weights,
        n_estimators=30,
        random_seed=41,
        threads=2,
    )
    assert edge_tuples(sequential) == edge_tuples(parallel)
    assert sequential.skipped_targets == parallel.skipped_targets


def test_random_forest_uses_exact_evidence_label() -> None:
    expression, genes, tfs = inference_data()
    result = run_inference(
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


def test_prepared_inference_preserves_minimum_leaf_weight_fraction() -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(
        expression,
        genes,
        tfs,
        n_estimators=2,
        min_weight_fraction_leaf=0.2,
    )

    assert prepared.min_weight_fraction_leaf == 0.2


@pytest.mark.parametrize("value", [-0.01, 0.500_001, float("nan"), float("inf")])
def test_prepared_inference_rejects_invalid_minimum_leaf_weight_fraction(value: float) -> None:
    expression, genes, tfs = inference_data()
    with pytest.raises(ValueError, match="min_weight_fraction_leaf"):
        prepare_inference(
            expression,
            genes,
            tfs,
            min_weight_fraction_leaf=value,
        )


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("max_features", None, "max_features"),
        ("min_samples_leaf", 0.25, "positive integer"),
        ("min_weight_fraction_leaf", False, "min_weight_fraction_leaf"),
    ],
)
def test_prepared_inference_uses_the_canonical_model_parameter_contract(
    parameter: str,
    value: object,
    message: str,
) -> None:
    expression, genes, tfs = inference_data()
    with pytest.raises(TypeError, match=message):
        prepare_inference(  # type: ignore[arg-type]
            expression,
            genes,
            tfs,
            **{parameter: value},
        )


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
    together = next(
        prepared.iter_group_target_batches(
            weights,
            target_batch_size=prepared.n_targets,
            group_order=["B", "A"],
            threads=1,
        )
    )
    progressive = [
        next(
            prepared.iter_group_target_batches(
                {group: weights[group]},
                target_batch_size=prepared.n_targets,
                group_order=[group],
                threads=1,
            )
        )
        for group in ("B", "A")
    ]
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
    unbatched = next(
        prepared.iter_group_target_batches(
            weights,
            target_batch_size=prepared.n_targets,
            threads=1,
        )
    )

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


def test_streamed_batches_do_not_materialize_inference_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=2, random_seed=13)
    observed: list[ModelResult] = []

    def unexpected_assembly(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("checkpoint streaming must not assemble batch results")

    monkeypatch.setattr(inference_module, "_assemble_result", unexpected_assembly)
    summaries = list(
        prepared.stream_group_target_batches(
            {"A": np.ones(expression.shape[0])},
            target_batch_size=2,
            threads=1,
            on_model_complete=observed.append,
        )
    )

    assert sum(summary.total_models for summary in summaries) == len(genes)
    assert sum(summary.trained_models for summary in summaries) == sum(
        result.trained for result in observed
    )
    assert sum(summary.skipped_target_records for summary in summaries) == sum(
        result.skipped is not None for result in observed
    )
    assert {(result.stat.target_group, result.stat.target) for result in observed} == {
        ("A", gene) for gene in genes
    }


def test_streamed_batches_skip_only_explicit_completed_models() -> None:
    expression, genes, tfs = inference_data()
    prepared = prepare_inference(expression, genes, tfs, n_estimators=2)
    observed: list[ModelResult] = []

    summaries = list(
        prepared.stream_group_target_batches(
            {"A": np.ones(expression.shape[0])},
            target_batch_size=2,
            threads=1,
            completed_models={("A", "G")},
            on_model_complete=observed.append,
        )
    )

    assert sum(summary.total_models for summary in summaries) == len(genes) - 1
    assert {result.stat.target for result in observed} == set(genes).difference({"G"})


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
