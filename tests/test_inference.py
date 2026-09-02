from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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


def run_inference(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    group_weights: dict[str, np.ndarray],
    *,
    target_names: list[str] | None = None,
    group_order: list[str] | None = None,
    threads: int = -1,
    verbose: int = 0,
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
            verbose=verbose,
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


def test_low_level_inference_uses_the_canonical_calibrated_model_defaults() -> None:
    expression, genes, tfs = inference_data()

    prepared = prepare_inference(expression, genes, tfs)

    assert prepared.n_estimators == 250
    assert prepared.max_features == "sqrt"


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

        def fit(self, x: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray) -> FakeEstimator:
            fitted_matrices[tuple(y.tolist())] = x.copy()
            fitted_matrix_ids[tuple(y.tolist())] = id(x)
            raw_importances = np.arange(1, x.shape[1] + 1, dtype=np.float64)
            self.feature_importances_ = raw_importances / raw_importances.sum()
            return self

    monkeypatch.setattr(
        inference_module,
        "create_tree_estimator",
        lambda *args, **kwargs: FakeEstimator(),
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
