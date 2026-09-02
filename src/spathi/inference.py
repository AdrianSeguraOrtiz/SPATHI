"""Weighted, group-specific gene-regulatory network inference.

Each model predicts one target gene from the candidate transcription factors
using every cell and one target-group-specific ``sample_weight`` vector.  The
expression values supplied by the user are not normalized or transformed here.
Target responses are retained in a contiguous ``float64`` matrix without an
unnecessary layout conversion, so small but valid expression differences cannot
disappear during preparation. Candidate
TF predictors are extracted once into the ``float32`` working precision used by
scikit-learn's tree ensembles.  Sample weights and exported importance scores
also remain ``float64``.

Feature importances are relative predictive importances within one target model.
They do not demonstrate causality, and SPATHI does not infer a regulatory sign.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from spathi.config import (
    DEFAULT_MAX_FEATURES,
    DEFAULT_N_ESTIMATORS,
    MAX_RANDOM_SEED,
    MaxFeatures,
    TreeMethod,
)
from spathi.parallel import (
    ParallelPlan,
    PersistentTaskExecutor,
    execute_tasks,
    resolve_thread_budget,
    stable_task_seed,
)

LOGGER = logging.getLogger(__name__)

TreeEstimator: TypeAlias = ExtraTreesRegressor | RandomForestRegressor
UntrainedModelStatus: TypeAlias = Literal[
    "insufficient_positive_weight_samples",
    "constant_target",
    "no_predictors_after_self_exclusion",
    "no_variable_predictors",
    "model_fit_failed",
    "invalid_feature_importances",
]
TrainedModelStatus: TypeAlias = Literal["trained", "trained_no_positive_importance"]
ModelStatus: TypeAlias = UntrainedModelStatus | TrainedModelStatus
SkipReason: TypeAlias = UntrainedModelStatus | Literal["no_positive_feature_importance"]

TRAINED_MODEL_STATUSES = frozenset({"trained", "trained_no_positive_importance"})
UNTRAINED_MODEL_STATUSES = frozenset(
    {
        "insufficient_positive_weight_samples",
        "constant_target",
        "no_predictors_after_self_exclusion",
        "no_variable_predictors",
        "model_fit_failed",
        "invalid_feature_importances",
    }
)
FATAL_MODEL_STATUSES = frozenset({"model_fit_failed", "invalid_feature_importances"})
MODEL_STATUSES = TRAINED_MODEL_STATUSES | UNTRAINED_MODEL_STATUSES
SKIP_REASONS = UNTRAINED_MODEL_STATUSES | {"no_positive_feature_importance"}


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeRecord:
    """One directed, unsigned, predictive network edge."""

    source: str
    target: str
    score: float
    sign: str
    evidence: str
    context: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SkippedTargetRecord:
    """A target for which no usable edge set could be inferred."""

    target_group: str
    target: str
    reason: SkipReason
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in SKIP_REASONS:
            raise ValueError(f"unsupported skipped-target reason: {self.reason!r}")
        if not self.target_group or not self.target:
            raise ValueError("skipped-target identifiers must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "target_group": self.target_group,
            "target": self.target,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelStat:
    """Audit information for one requested ``(group, target)`` model."""

    target_group: str
    target: str
    status: ModelStatus
    random_seed: int
    n_samples: int
    n_positive_weight_samples: int
    weight_sum: float
    n_predictors_input: int
    n_predictors_used: int
    discarded_predictors: tuple[str, ...]
    constant_predictors: tuple[str, ...]
    n_edges: int
    importance_sum: float
    fit_seconds: float
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in MODEL_STATUSES:
            raise ValueError(f"unsupported model status: {self.status!r}")
        if not self.target_group or not self.target:
            raise ValueError("model-stat identifiers must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_group": self.target_group,
            "target": self.target,
            "status": self.status,
            "random_seed": self.random_seed,
            "n_samples": self.n_samples,
            "n_positive_weight_samples": self.n_positive_weight_samples,
            "weight_sum": self.weight_sum,
            "n_predictors_input": self.n_predictors_input,
            "n_predictors_used": self.n_predictors_used,
            "discarded_predictors": self.discarded_predictors,
            "constant_predictors": self.constant_predictors,
            "n_edges": self.n_edges,
            "importance_sum": self.importance_sum,
            "fit_seconds": self.fit_seconds,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceResult:
    """Structured, deterministically ordered output of network inference."""

    edges: tuple[EdgeRecord, ...]
    skipped_targets: tuple[SkippedTargetRecord, ...]
    model_stats: tuple[ModelStat, ...]
    group_order: tuple[str, ...]
    parallel_plan: ParallelPlan
    tree_method: TreeMethod
    total_models: int
    completed_models: int
    trained_models: int
    duration_seconds: float
    expression_dtype: str
    expression_nbytes: int
    predictor_nbytes: int


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceBatchSummary:
    """Lightweight accounting for a streamed model batch.

    Unlike :class:`InferenceResult`, this object never retains edge, skipped-target,
    or model-stat records. It is therefore suitable for checkpoint-backed runs in
    which every :class:`ModelResult` is committed by a completion callback.
    """

    group_order: tuple[str, ...]
    parallel_plan: ParallelPlan
    tree_method: TreeMethod
    total_models: int
    trained_models: int
    skipped_target_records: int
    duration_seconds: float


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedGroup:
    name: str
    weights: NDArray[np.float64]
    positive_mask: NDArray[np.bool_]
    n_positive_weight_samples: int
    weight_sum: float
    constant_tf_positions: frozenset[int]
    variable_tf_positions: tuple[int, ...]
    variable_tf_expression: NDArray[np.float32]


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModelTask:
    group: _PreparedGroup
    target_index: int
    target_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResult:
    """Complete result of one ``(target group, target gene)`` model."""

    edges: tuple[EdgeRecord, ...]
    skipped: SkippedTargetRecord | None
    stat: ModelStat
    trained: bool

    def __post_init__(self) -> None:
        if type(self.trained) is not bool:
            raise TypeError("trained must be a boolean")
        expected_key = (self.stat.target_group, self.stat.target)
        expected_context = f"group:{self.stat.target_group}"
        sources: set[str] = set()
        for edge in self.edges:
            if (edge.context, edge.target) != (expected_context, self.stat.target):
                raise ValueError("model edges must match the model-stat identity")
            if (
                not edge.source
                or edge.source == edge.target
                or edge.source in sources
                or not np.isfinite(edge.score)
                or edge.score <= 0.0
            ):
                raise ValueError("model edges must be unique, non-self, positive, and finite")
            sources.add(edge.source)
        if self.stat.n_edges != len(self.edges):
            raise ValueError("model edge count does not match its diagnostics")
        if (
            self.skipped is not None
            and (
                self.skipped.target_group,
                self.skipped.target,
            )
            != expected_key
        ):
            raise ValueError("skipped-target record must match the model-stat identity")

        status_is_trained = self.stat.status in TRAINED_MODEL_STATUSES
        if self.trained != status_is_trained:
            raise ValueError("trained flag does not match the model status")
        if self.stat.status == "trained":
            if self.skipped is not None or not self.edges:
                raise ValueError("a trained model must have edges and no skipped record")
        elif self.stat.status == "trained_no_positive_importance":
            if (
                self.edges
                or self.skipped is None
                or self.skipped.reason != "no_positive_feature_importance"
            ):
                raise ValueError(
                    "a trained model without positive importances requires its exact skipped record"
                )
        elif self.edges or self.skipped is None or self.skipped.reason != self.stat.status:
            raise ValueError(
                "an untrained model requires no edges and a skipped reason matching its status"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _FitContext:
    expression: NDArray[np.float64]
    tf_names: tuple[str, ...]
    target_to_tf_position: tuple[int | None, ...]
    tree_method: TreeMethod
    n_estimators: int
    max_features: MaxFeatures | None
    min_samples_leaf: int | float
    max_depth: int | None
    bootstrap: bool
    global_seed: int
    model_n_jobs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _InferenceBatchSpec:
    groups: tuple[_PreparedGroup, ...]
    targets: tuple[tuple[int, str], ...]
    completed_models: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModelExecution:
    tasks: tuple[_ModelTask, ...]
    plan: ParallelPlan
    context: _FitContext


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedInference:
    """Reusable, validated tree-inference state.

    Construct instances with :func:`prepare_inference`. Selected target responses
    are retained as ``float64`` storage, while the TF predictor matrix is extracted
    into contiguous ``float32`` exactly once. Both are then shared by every group
    batch. Group-specific weights remain ``float64`` and are never stored
    on this reusable object.
    """

    _expression: NDArray[np.float64]
    _tf_expression: NDArray[np.float32]
    gene_names: tuple[str, ...]
    target_names: tuple[str, ...]
    tf_names: tuple[str, ...]
    _target_to_tf_position: tuple[int | None, ...]
    _target_expression_additional_nbytes: int
    tree_method: TreeMethod
    n_estimators: int
    max_features: MaxFeatures | None
    min_samples_leaf: int | float
    max_depth: int | None
    bootstrap: bool
    random_seed: int

    @property
    def expression_dtype(self) -> str:
        """Storage dtype used by the reusable cells-by-targets response matrix."""

        return str(self._expression.dtype)

    @property
    def predictor_dtype(self) -> str:
        """Storage dtype used by the reusable TF predictor matrix."""

        return str(self._tf_expression.dtype)

    @property
    def expression_nbytes(self) -> int:
        """Bytes occupied by the reusable cells-by-targets response matrix."""

        return int(self._expression.nbytes)

    @property
    def predictor_nbytes(self) -> int:
        """Bytes occupied by the reusable TF predictor matrix."""

        return int(self._tf_expression.nbytes)

    @property
    def target_expression_additional_nbytes(self) -> int:
        """Additional target storage beyond the caller's expression matrix."""

        return self._target_expression_additional_nbytes

    @property
    def n_cells(self) -> int:
        return int(self._expression.shape[0])

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    def iter_group_target_batches(
        self,
        group_weights: Mapping[object, ArrayLike],
        *,
        target_batch_size: int,
        group_order: Sequence[object] | None = None,
        threads: int = -1,
        verbose: int = 0,
        completed_models: Collection[tuple[str, str]] = (),
        executor: PersistentTaskExecutor | None = None,
        on_model_complete: Callable[[ModelResult], None] | None = None,
    ) -> Iterator[InferenceResult]:
        """Yield bounded inference results without re-preparing group weights.

        Groups are visited in ``group_order`` and targets lexicographically.
        Large target sets yield at most ``target_batch_size`` models for a single
        group, preserving globally streamable ordering. If every target fits in
        one batch, all requested groups are fitted together so a small number of
        genes can still use task-level parallelism. Constant-predictor masks and
        other group state are computed only once before the first batch is fitted.
        """

        _validate_verbose(verbose)
        for batch in self._iter_batch_specs(
            group_weights,
            target_batch_size=target_batch_size,
            group_order=group_order,
            completed_models=completed_models,
            threads=threads,
            executor=executor,
        ):
            yield self._infer_prepared_groups(
                batch.groups,
                threads=threads,
                verbose=verbose,
                target_items=batch.targets,
                completed_models=batch.completed_models,
                executor=executor,
                on_model_complete=on_model_complete,
            )

    def stream_group_target_batches(
        self,
        group_weights: Mapping[object, ArrayLike],
        *,
        target_batch_size: int,
        on_model_complete: Callable[[ModelResult], None],
        group_order: Sequence[object] | None = None,
        threads: int = -1,
        verbose: int = 0,
        completed_models: Collection[tuple[str, str]] = (),
        executor: PersistentTaskExecutor | None = None,
    ) -> Iterator[InferenceBatchSummary]:
        """Stream model results into ``on_model_complete`` without retaining them.

        The yielded summaries contain only counters and execution metadata. This
        is the canonical checkpoint path: each model can be durably committed as
        soon as it finishes, while no batch-sized edge or diagnostic collection
        is assembled merely to be discarded afterwards.
        """

        if not callable(on_model_complete):
            raise TypeError("on_model_complete must be callable")
        _validate_verbose(verbose)
        for batch in self._iter_batch_specs(
            group_weights,
            target_batch_size=target_batch_size,
            group_order=group_order,
            completed_models=completed_models,
            threads=threads,
            executor=executor,
        ):
            yield self._stream_prepared_groups(
                batch.groups,
                threads=threads,
                verbose=verbose,
                target_items=batch.targets,
                completed_models=batch.completed_models,
                executor=executor,
                on_model_complete=on_model_complete,
            )

    def _iter_batch_specs(
        self,
        group_weights: Mapping[object, ArrayLike],
        *,
        target_batch_size: int,
        group_order: Sequence[object] | None,
        completed_models: Collection[tuple[str, str]],
        threads: int,
        executor: PersistentTaskExecutor | None,
    ) -> Iterator[_InferenceBatchSpec]:
        if isinstance(target_batch_size, bool) or not isinstance(target_batch_size, Integral):
            raise TypeError("target_batch_size must be a positive integer")
        if target_batch_size < 1:
            raise ValueError("target_batch_size must be a positive integer")
        if executor is not None and executor.plan.requested_threads != threads:
            raise ValueError("executor thread plan does not match the requested threads")
        prepared_groups = _prepare_groups(
            group_weights,
            group_order,
            n_cells=self.n_cells,
            tf_expression=self._tf_expression,
        )
        ordered_targets = tuple(sorted(enumerate(self.target_names), key=lambda item: item[1]))
        completed = frozenset((str(group), str(target)) for group, target in completed_models)
        known_keys = {
            (group.name, target_name)
            for group in prepared_groups
            for _target_index, target_name in ordered_targets
        }
        unexpected_completed = completed.difference(known_keys)
        if unexpected_completed:
            preview = sorted(unexpected_completed)[:5]
            raise ValueError(
                f"completed_models contains identities outside this inference request: {preview}"
            )
        if len(ordered_targets) <= int(target_batch_size):
            if known_keys.difference(completed):
                yield _InferenceBatchSpec(
                    groups=prepared_groups,
                    targets=ordered_targets,
                    completed_models=completed,
                )
            return

        for group in prepared_groups:
            for start in range(0, len(ordered_targets), int(target_batch_size)):
                targets = ordered_targets[start : start + int(target_batch_size)]
                if all((group.name, target_name) in completed for _, target_name in targets):
                    continue
                yield _InferenceBatchSpec(
                    groups=(group,),
                    targets=targets,
                    completed_models=completed,
                )

    def _prepare_model_execution(
        self,
        prepared_groups: tuple[_PreparedGroup, ...],
        *,
        target_items: Sequence[tuple[int, str]],
        completed_models: Collection[tuple[str, str]],
        threads: int,
        executor: PersistentTaskExecutor | None,
    ) -> _ModelExecution:
        tasks = tuple(
            _ModelTask(group=group, target_index=target_index, target_name=target_name)
            for group in prepared_groups
            for target_index, target_name in target_items
            if (group.name, target_name) not in completed_models
        )
        if not tasks:
            raise ValueError("inference batch contains no incomplete models")
        plan = (
            resolve_thread_budget(threads, len(tasks))
            if executor is None
            else replace(executor.plan, total_tasks=len(tasks))
        )
        context = _FitContext(
            expression=self._expression,
            tf_names=self.tf_names,
            target_to_tf_position=self._target_to_tf_position,
            tree_method=self.tree_method,
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            bootstrap=self.bootstrap,
            global_seed=self.random_seed,
            model_n_jobs=plan.model_n_jobs,
        )
        LOGGER.debug(
            "Fitting %d weighted models with %d effective thread(s) at the %s level",
            len(tasks),
            plan.effective_threads,
            plan.parallel_level,
        )
        return _ModelExecution(tasks=tasks, plan=plan, context=context)

    def _infer_prepared_groups(
        self,
        prepared_groups: tuple[_PreparedGroup, ...],
        *,
        threads: int,
        verbose: int,
        target_items: Sequence[tuple[int, str]],
        completed_models: Collection[tuple[str, str]] = (),
        executor: PersistentTaskExecutor | None = None,
        on_model_complete: Callable[[ModelResult], None] | None = None,
    ) -> InferenceResult:
        _validate_verbose(verbose)
        started = perf_counter()
        execution = self._prepare_model_execution(
            prepared_groups,
            target_items=target_items,
            completed_models=completed_models,
            threads=threads,
            executor=executor,
        )

        def fit_task(task: _ModelTask) -> ModelResult:
            return _fit_model_task(task, execution.context)

        if on_model_complete is None:
            if executor is None:
                task_results = execute_tasks(
                    fit_task,
                    execution.tasks,
                    execution.plan,
                    verbose=int(verbose),
                )
            else:
                task_results = executor.execute(fit_task, execution.tasks)
        else:
            task_results = []

            def collect_result(result: ModelResult) -> None:
                on_model_complete(result)
                task_results.append(result)

            if executor is None:
                with PersistentTaskExecutor(execution.plan, verbose=int(verbose)) as owned_executor:
                    owned_executor.consume(fit_task, execution.tasks, on_result=collect_result)
            else:
                executor.consume(fit_task, execution.tasks, on_result=collect_result)
        return _assemble_result(
            task_results,
            prepared_groups=prepared_groups,
            plan=execution.plan,
            tree_method=self.tree_method,
            started=started,
            expression_dtype=self.expression_dtype,
            expression_nbytes=self.expression_nbytes,
            predictor_nbytes=self.predictor_nbytes,
        )

    def _stream_prepared_groups(
        self,
        prepared_groups: tuple[_PreparedGroup, ...],
        *,
        threads: int,
        verbose: int,
        target_items: Sequence[tuple[int, str]],
        completed_models: Collection[tuple[str, str]],
        executor: PersistentTaskExecutor | None,
        on_model_complete: Callable[[ModelResult], None],
    ) -> InferenceBatchSummary:
        _validate_verbose(verbose)
        started = perf_counter()
        execution = self._prepare_model_execution(
            prepared_groups,
            target_items=target_items,
            completed_models=completed_models,
            threads=threads,
            executor=executor,
        )

        def fit_task(task: _ModelTask) -> ModelResult:
            return _fit_model_task(task, execution.context)

        trained_models = 0
        skipped_target_records = 0

        def consume_result(result: ModelResult) -> None:
            nonlocal trained_models, skipped_target_records
            trained_models += int(result.trained)
            skipped_target_records += int(result.skipped is not None)
            on_model_complete(result)

        if executor is None:
            with PersistentTaskExecutor(execution.plan, verbose=int(verbose)) as owned_executor:
                owned_executor.consume(fit_task, execution.tasks, on_result=consume_result)
        else:
            executor.consume(fit_task, execution.tasks, on_result=consume_result)
        duration = perf_counter() - started
        LOGGER.debug(
            "Completed %d models (%d trained, %d skipped) in %.3f s",
            len(execution.tasks),
            trained_models,
            skipped_target_records,
            duration,
        )
        return InferenceBatchSummary(
            group_order=tuple(group.name for group in prepared_groups),
            parallel_plan=execution.plan,
            tree_method=self.tree_method,
            total_models=len(execution.tasks),
            trained_models=trained_models,
            skipped_target_records=skipped_target_records,
            duration_seconds=duration,
        )


def create_tree_estimator(
    tree_method: TreeMethod,
    *,
    n_estimators: int,
    max_features: MaxFeatures | None,
    min_samples_leaf: int | float,
    max_depth: int | None,
    bootstrap: bool,
    random_state: int,
    n_jobs: int,
) -> TreeEstimator:
    """Create a supported scikit-learn estimator with explicit resources."""

    common: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": max_depth,
        "bootstrap": bootstrap,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if tree_method == "extra-trees":
        return ExtraTreesRegressor(**common)
    if tree_method == "random-forest":
        return RandomForestRegressor(**common)
    raise ValueError(
        f"tree_method must be either 'extra-trees' or 'random-forest', got {tree_method!r}"
    )


def _validate_hyperparameters(
    *,
    tree_method: str,
    n_estimators: int,
    max_features: MaxFeatures | None,
    min_samples_leaf: int | float,
    max_depth: int | None,
    bootstrap: bool | None,
    random_seed: int,
) -> None:
    if tree_method not in {"extra-trees", "random-forest"}:
        raise ValueError("tree_method must be either 'extra-trees' or 'random-forest'")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral):
        raise TypeError("n_estimators must be a positive integer")
    if n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer")

    if isinstance(max_features, bool):
        raise TypeError("max_features cannot be a boolean")
    if isinstance(max_features, str) and max_features not in {"sqrt", "log2"}:
        raise ValueError("string max_features must be 'sqrt' or 'log2'")
    if isinstance(max_features, Integral) and max_features < 1:
        raise ValueError("integer max_features must be at least 1")
    if isinstance(max_features, Real) and not isinstance(max_features, Integral):
        if not 0.0 < float(max_features) <= 1.0:
            raise ValueError("float max_features must be in the interval (0, 1]")
    if max_features is not None and not isinstance(max_features, (str, Integral, Real)):
        raise TypeError("max_features must be None, 'sqrt', 'log2', int, or float")

    if isinstance(min_samples_leaf, bool) or not isinstance(min_samples_leaf, Real):
        raise TypeError("min_samples_leaf must be a positive integer or float")
    if isinstance(min_samples_leaf, Integral):
        if min_samples_leaf < 1:
            raise ValueError("integer min_samples_leaf must be at least 1")
    elif not 0.0 < float(min_samples_leaf) <= 0.5:
        raise ValueError("float min_samples_leaf must be in the interval (0, 0.5]")

    if max_depth is not None:
        if isinstance(max_depth, bool) or not isinstance(max_depth, Integral):
            raise TypeError("max_depth must be None or a positive integer")
        if max_depth < 1:
            raise ValueError("max_depth must be None or a positive integer")
    if bootstrap is not None and not isinstance(bootstrap, bool):
        raise TypeError("bootstrap must be a boolean or None")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be a non-negative integer")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if random_seed > MAX_RANDOM_SEED:
        raise ValueError(f"random_seed must be at most {MAX_RANDOM_SEED}")


def _validate_verbose(verbose: int) -> None:
    if isinstance(verbose, bool) or not isinstance(verbose, Integral) or verbose < 0:
        raise ValueError("verbose must be a non-negative integer")


def _normalise_identifiers(
    values: Sequence[object],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names and not allow_empty:
        raise ValueError(f"{label} cannot be empty")
    if any(name == "" for name in names):
        raise ValueError(f"{label} cannot contain empty identifiers")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must contain unique identifiers")
    return names


def _prepare_expression(
    expression: ArrayLike,
    gene_names: Sequence[object],
    tf_names: Sequence[object],
    target_names: Sequence[object] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float32],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int | None, ...],
    int,
]:
    genes = _normalise_identifiers(gene_names, label="gene_names")
    tfs = _normalise_identifiers(tf_names, label="tf_names")
    targets = (
        genes
        if target_names is None
        else _normalise_identifiers(target_names, label="target_names")
    )

    raw = np.asarray(expression)
    if raw.ndim != 2:
        raise ValueError("expression must be a two-dimensional cells-by-genes array")
    if raw.shape[0] < 1:
        raise ValueError("expression must contain at least one cell")
    if raw.shape[1] != len(genes):
        raise ValueError(
            f"expression column count does not match gene_names: {raw.shape[1]} != {len(genes)}"
        )
    try:
        expression64 = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("expression must contain only numeric values") from exc
    if not (expression64.flags.c_contiguous or expression64.flags.f_contiguous):
        # Preserve whichever contiguous layout is cheapest.  In particular, the
        # transpose of the validated genes-by-cells input is naturally
        # Fortran-contiguous, which makes each target column contiguous without
        # duplicating the complete expression matrix.
        expression64 = np.asfortranarray(expression64)
    if not np.isfinite(expression64).all():
        raise ValueError("expression contains non-finite values")

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    absent = [tf for tf in tfs if tf not in gene_to_index]
    if absent:
        raise ValueError(
            "all transcription factors must occur in gene_names; absent: " + ", ".join(absent)
        )
    absent_targets = [target for target in targets if target not in gene_to_index]
    if absent_targets:
        raise ValueError(
            "all targets must occur in gene_names; absent: " + ", ".join(absent_targets)
        )
    tf_indices = np.fromiter((gene_to_index[tf] for tf in tfs), dtype=np.intp, count=len(tfs))
    # Advanced indexing performs the single intentional predictor-matrix copy.
    # scikit-learn's tree implementation consumes float32 predictors, whereas
    # target responses stay float64 to preserve sub-float32 expression changes.
    with np.errstate(over="ignore", invalid="ignore"):
        tf_expression = np.ascontiguousarray(expression64[:, tf_indices], dtype=np.float32)
    if not np.isfinite(tf_expression).all():
        raise ValueError(
            "transcription-factor expression contains values that cannot be represented "
            "as finite float32 predictors"
        )
    if targets == genes:
        target_expression = expression64
        target_expression_additional_nbytes = 0
    else:
        target_indices = np.fromiter(
            (gene_to_index[target] for target in targets),
            dtype=np.intp,
            count=len(targets),
        )
        # Target subsets are intentionally retained in float64 and Fortran order,
        # keeping each response column contiguous without preserving unrelated
        # genes in the tree-inference state.
        target_expression = np.asfortranarray(expression64[:, target_indices], dtype=np.float64)
        target_expression_additional_nbytes = int(target_expression.nbytes)
    tf_position_by_name = {tf: position for position, tf in enumerate(tfs)}
    target_to_tf_position = tuple(tf_position_by_name.get(target) for target in targets)
    return (
        target_expression,
        tf_expression,
        genes,
        targets,
        tfs,
        target_to_tf_position,
        target_expression_additional_nbytes,
    )


def _group_weight_statistics(
    weights: NDArray[np.float64],
    positive_mask: NDArray[np.bool_],
) -> tuple[int, float]:
    """Calculate the group-level weight diagnostics shared by every target model."""

    return (
        int(np.count_nonzero(positive_mask)),
        float(np.sum(weights, dtype=np.float64)),
    )


def _constant_tf_positions(
    tf_expression: NDArray[np.float32],
    positive_mask: NDArray[np.bool_] | None,
) -> frozenset[int]:
    """Return TFs that are constant globally or among positive-weight cells."""

    if positive_mask is None:
        minima = np.min(tf_expression, axis=0)
        maxima = np.max(tf_expression, axis=0)
    else:
        # Boolean indexing would copy the full positive-cells-by-TF matrix.
        # Masked reductions derive the same decision with bounded peak memory.
        positive_rows = positive_mask[:, np.newaxis]
        minima = np.min(
            tf_expression,
            axis=0,
            where=positive_rows,
            initial=np.inf,
        )
        maxima = np.max(
            tf_expression,
            axis=0,
            where=positive_rows,
            initial=-np.inf,
        )
    return frozenset(int(position) for position in np.flatnonzero(minima == maxima))


def _prepare_groups(
    group_weights: Mapping[object, ArrayLike],
    group_order: Sequence[object] | None,
    *,
    n_cells: int,
    tf_expression: NDArray[np.float32],
) -> tuple[_PreparedGroup, ...]:
    if not isinstance(group_weights, Mapping) or not group_weights:
        raise ValueError("group_weights must be a non-empty mapping")

    weights_by_name: dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.bool_], int, float],
    ] = {}
    for raw_group, raw_weights in group_weights.items():
        group = str(raw_group)
        if not group:
            raise ValueError("group_weights cannot contain an empty group identifier")
        if group in weights_by_name:
            raise ValueError(
                f"group identifiers must remain unique after string conversion: {group!r}"
            )
        try:
            weights = np.asarray(raw_weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights for group {group!r} must be numeric") from exc
        if weights.ndim != 1 or weights.shape[0] != n_cells:
            raise ValueError(
                f"weights for group {group!r} must have shape ({n_cells},), got {weights.shape}"
            )
        if not np.isfinite(weights).all():
            raise ValueError(f"weights for group {group!r} contain non-finite values")
        if np.any(weights < 0.0):
            raise ValueError(f"weights for group {group!r} contain negative values")
        weights = np.ascontiguousarray(weights)
        positive_mask = np.ascontiguousarray(weights > 0.0)
        n_positive_weight_samples, weight_sum = _group_weight_statistics(
            weights,
            positive_mask,
        )
        if n_positive_weight_samples == 0:
            raise ValueError(f"weights for group {group!r} are all zero")
        weights_by_name[group] = (
            weights,
            positive_mask,
            n_positive_weight_samples,
            weight_sum,
        )

    if group_order is None:
        ordered_names = tuple(sorted(weights_by_name))
    else:
        ordered_names = _normalise_identifiers(group_order, label="group_order")
        missing = sorted(set(weights_by_name).difference(ordered_names))
        unexpected = sorted(set(ordered_names).difference(weights_by_name))
        if missing or unexpected:
            raise ValueError(
                "group_order must contain every group_weights key exactly once; "
                f"missing={missing}, unexpected={unexpected}"
            )

    prepared: list[_PreparedGroup] = []
    global_constant_positions: frozenset[int] | None = None
    for group in ordered_names:
        weights, positive_mask, n_positive_weight_samples, weight_sum = weights_by_name[group]
        if n_positive_weight_samples == n_cells:
            if global_constant_positions is None:
                global_constant_positions = _constant_tf_positions(tf_expression, None)
            constant_positions = global_constant_positions
        else:
            constant_positions = _constant_tf_positions(tf_expression, positive_mask)
        variable_positions = tuple(
            position
            for position in range(tf_expression.shape[1])
            if position not in constant_positions
        )
        if constant_positions:
            # A constant-TF mask belongs to the target group, not to an
            # individual target gene. Materialize the filtered matrix once and
            # share it across every model for this group. Without this cache a
            # single constant TF caused the same cells-by-TF copy to be rebuilt
            # for every target gene.
            variable_tf_expression = np.take(
                tf_expression,
                np.asarray(variable_positions, dtype=np.intp),
                axis=1,
            )
        else:
            variable_tf_expression = tf_expression
        prepared.append(
            _PreparedGroup(
                name=group,
                weights=weights,
                positive_mask=positive_mask,
                n_positive_weight_samples=n_positive_weight_samples,
                weight_sum=weight_sum,
                constant_tf_positions=constant_positions,
                variable_tf_positions=variable_positions,
                variable_tf_expression=variable_tf_expression,
            )
        )
    return tuple(prepared)


def _make_stat(
    task: _ModelTask,
    context: _FitContext,
    *,
    status: ModelStatus,
    seed: int,
    n_predictors_used: int,
    discarded: tuple[str, ...],
    constant_predictors: tuple[str, ...],
    n_edges: int = 0,
    importance_sum: float = 0.0,
    fit_seconds: float = 0.0,
    message: str = "",
) -> ModelStat:
    return ModelStat(
        target_group=task.group.name,
        target=task.target_name,
        status=status,
        random_seed=seed,
        n_samples=int(context.expression.shape[0]),
        n_positive_weight_samples=task.group.n_positive_weight_samples,
        weight_sum=task.group.weight_sum,
        n_predictors_input=len(context.tf_names),
        n_predictors_used=n_predictors_used,
        discarded_predictors=discarded,
        constant_predictors=constant_predictors,
        n_edges=n_edges,
        importance_sum=importance_sum,
        fit_seconds=fit_seconds,
        message=message,
    )


def _skipped_result(
    task: _ModelTask,
    context: _FitContext,
    *,
    reason: UntrainedModelStatus,
    detail: str,
    seed: int,
    n_predictors_used: int,
    discarded: tuple[str, ...],
    constant_predictors: tuple[str, ...],
    fit_seconds: float = 0.0,
) -> ModelResult:
    skipped = SkippedTargetRecord(
        target_group=task.group.name,
        target=task.target_name,
        reason=reason,
        detail=detail,
    )
    stat = _make_stat(
        task,
        context,
        status=reason,
        seed=seed,
        n_predictors_used=n_predictors_used,
        discarded=discarded,
        constant_predictors=constant_predictors,
        fit_seconds=fit_seconds,
        message=detail,
    )
    return ModelResult(edges=(), skipped=skipped, stat=stat, trained=False)


def _fit_model_task(task: _ModelTask, context: _FitContext) -> ModelResult:
    seed = stable_task_seed(context.global_seed, task.group.name, task.target_name)
    self_position = context.target_to_tf_position[task.target_index]
    eligible_positions = tuple(
        position for position in range(len(context.tf_names)) if position != self_position
    )
    constant_set = task.group.constant_tf_positions
    constant_predictors = tuple(
        context.tf_names[position] for position in eligible_positions if position in constant_set
    )
    selected_positions = tuple(
        position for position in task.group.variable_tf_positions if position != self_position
    )
    selected_names = tuple(context.tf_names[position] for position in selected_positions)
    discarded_positions = constant_set.intersection(eligible_positions)
    if self_position is not None:
        discarded_positions = discarded_positions.union((self_position,))
    discarded = tuple(
        context.tf_names[position]
        for position in range(len(context.tf_names))
        if position in discarded_positions
    )

    n_positive = task.group.n_positive_weight_samples
    if n_positive < 2:
        return _skipped_result(
            task,
            context,
            reason="insufficient_positive_weight_samples",
            detail="fewer than two cells have positive sample weight",
            seed=seed,
            n_predictors_used=len(selected_positions),
            discarded=discarded,
            constant_predictors=constant_predictors,
        )

    y = context.expression[:, task.target_index]
    positive_minimum = np.min(y, where=task.group.positive_mask, initial=np.inf)
    positive_maximum = np.max(y, where=task.group.positive_mask, initial=-np.inf)
    if positive_minimum == positive_maximum:
        return _skipped_result(
            task,
            context,
            reason="constant_target",
            detail="target expression is constant among positive-weight cells",
            seed=seed,
            n_predictors_used=len(selected_positions),
            discarded=discarded,
            constant_predictors=constant_predictors,
        )

    if not eligible_positions:
        return _skipped_result(
            task,
            context,
            reason="no_predictors_after_self_exclusion",
            detail="the target is the only candidate transcription factor",
            seed=seed,
            n_predictors_used=0,
            discarded=discarded,
            constant_predictors=(),
        )

    if not selected_positions:
        return _skipped_result(
            task,
            context,
            reason="no_variable_predictors",
            detail="all eligible predictors are constant among positive-weight cells",
            seed=seed,
            n_predictors_used=0,
            discarded=discarded,
            constant_predictors=constant_predictors,
        )

    if self_position is None or self_position in constant_set:
        x_model = task.group.variable_tf_expression
    else:
        # Only variable TF targets need one per-task copy for self-exclusion.
        # Fill a single C-contiguous allocation directly so no intermediate
        # advanced-indexing array is created.
        source = task.group.variable_tf_expression
        self_variable_position = task.group.variable_tf_positions.index(self_position)
        x_model = np.empty(
            (source.shape[0], len(selected_positions)),
            dtype=np.float32,
            order="C",
        )
        x_model[:, :self_variable_position] = source[:, :self_variable_position]
        x_model[:, self_variable_position:] = source[:, self_variable_position + 1 :]

    estimator = create_tree_estimator(
        context.tree_method,
        n_estimators=context.n_estimators,
        max_features=context.max_features,
        min_samples_leaf=context.min_samples_leaf,
        max_depth=context.max_depth,
        bootstrap=context.bootstrap,
        random_state=seed,
        n_jobs=context.model_n_jobs,
    )
    started = perf_counter()
    try:
        estimator.fit(x_model, y, sample_weight=task.group.weights)
        fit_seconds = perf_counter() - started
        importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
    except (ValueError, FloatingPointError) as exc:
        fit_seconds = perf_counter() - started
        detail = f"{type(exc).__name__}: {exc}"
        return _skipped_result(
            task,
            context,
            reason="model_fit_failed",
            detail=detail,
            seed=seed,
            n_predictors_used=len(selected_positions),
            discarded=discarded,
            constant_predictors=constant_predictors,
            fit_seconds=fit_seconds,
        )

    if importances.shape != (len(selected_names),) or not np.isfinite(importances).all():
        detail = (
            "estimator returned invalid feature_importances_: "
            f"shape={importances.shape}, expected=({len(selected_names)},)"
        )
        return _skipped_result(
            task,
            context,
            reason="invalid_feature_importances",
            detail=detail,
            seed=seed,
            n_predictors_used=len(selected_positions),
            discarded=discarded,
            constant_predictors=constant_predictors,
            fit_seconds=fit_seconds,
        )
    if np.any(importances < 0.0):
        return _skipped_result(
            task,
            context,
            reason="invalid_feature_importances",
            detail="estimator returned a negative feature importance",
            seed=seed,
            n_predictors_used=len(selected_positions),
            discarded=discarded,
            constant_predictors=constant_predictors,
            fit_seconds=fit_seconds,
        )

    evidence = (
        "weighted_extra_trees_feature_importance"
        if context.tree_method == "extra-trees"
        else "weighted_random_forest_feature_importance"
    )
    edge_context = f"group:{task.group.name}"
    edges = tuple(
        EdgeRecord(
            source=source,
            target=task.target_name,
            score=float(score),
            sign="?",
            evidence=evidence,
            context=edge_context,
        )
        for source, score in zip(selected_names, importances, strict=True)
        if score > 0.0
    )
    importance_sum = float(np.sum(importances, dtype=np.float64))
    if not edges:
        skipped = SkippedTargetRecord(
            target_group=task.group.name,
            target=task.target_name,
            reason="no_positive_feature_importance",
            detail="model fitted successfully but produced no positive importances",
        )
        stat = _make_stat(
            task,
            context,
            status="trained_no_positive_importance",
            seed=seed,
            n_predictors_used=len(selected_positions),
            discarded=discarded,
            constant_predictors=constant_predictors,
            n_edges=0,
            importance_sum=importance_sum,
            fit_seconds=fit_seconds,
            message=skipped.detail,
        )
        return ModelResult(edges=(), skipped=skipped, stat=stat, trained=True)

    stat = _make_stat(
        task,
        context,
        status="trained",
        seed=seed,
        n_predictors_used=len(selected_positions),
        discarded=discarded,
        constant_predictors=constant_predictors,
        n_edges=len(edges),
        importance_sum=importance_sum,
        fit_seconds=fit_seconds,
    )
    return ModelResult(edges=edges, skipped=None, stat=stat, trained=True)


def _assemble_result(
    task_results: list[ModelResult],
    *,
    prepared_groups: tuple[_PreparedGroup, ...],
    plan: ParallelPlan,
    tree_method: TreeMethod,
    started: float,
    expression_dtype: str,
    expression_nbytes: int,
    predictor_nbytes: int,
) -> InferenceResult:
    edges = [edge for result in task_results for edge in result.edges]
    skipped = [result.skipped for result in task_results if result.skipped is not None]
    stats = [result.stat for result in task_results]
    # The output contract specifies this lexical order. It is independent of
    # scheduler completion and intentionally distinct from scientific task order.
    edges.sort(key=lambda edge: (edge.context, edge.target, edge.source))
    skipped.sort(key=lambda record: (record.target_group, record.target, record.reason))
    stats.sort(key=lambda record: (record.target_group, record.target))
    trained_models = sum(result.trained for result in task_results)
    duration = perf_counter() - started
    LOGGER.debug(
        "Completed %d/%d models (%d trained, %d skipped) in %.3f s",
        len(task_results),
        plan.total_tasks,
        trained_models,
        len(skipped),
        duration,
    )

    return InferenceResult(
        edges=tuple(edges),
        skipped_targets=tuple(skipped),
        model_stats=tuple(stats),
        group_order=tuple(group.name for group in prepared_groups),
        parallel_plan=plan,
        tree_method=tree_method,
        total_models=plan.total_tasks,
        completed_models=len(task_results),
        trained_models=trained_models,
        duration_seconds=duration,
        expression_dtype=expression_dtype,
        expression_nbytes=expression_nbytes,
        predictor_nbytes=predictor_nbytes,
    )


def prepare_inference(
    expression: ArrayLike,
    gene_names: Sequence[object],
    tf_names: Sequence[object],
    *,
    target_names: Sequence[object] | None = None,
    tree_method: TreeMethod = "extra-trees",
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_features: MaxFeatures | None = DEFAULT_MAX_FEATURES,
    min_samples_leaf: int | float = 1,
    max_depth: int | None = None,
    bootstrap: bool | None = None,
    random_seed: int = 123,
) -> PreparedInference:
    """Validate and materialize reusable state for one or more group networks.

    This is the preferred API for progressive output pipelines. It performs the
    only conversion of the cells-by-genes matrix and the only extraction of the
    selected target responses and TF predictor matrix. Iterating bounded group/target
    batches then reuses both arrays without re-normalizing, re-casting, or
    re-extracting expression data.
    """

    _validate_hyperparameters(
        tree_method=tree_method,
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        bootstrap=bootstrap,
        random_seed=random_seed,
    )
    (
        expression64,
        tf_expression,
        genes,
        targets,
        tfs,
        target_to_tf_position,
        target_expression_additional_nbytes,
    ) = _prepare_expression(expression, gene_names, tf_names, target_names)
    effective_bootstrap = tree_method == "random-forest" if bootstrap is None else bootstrap
    return PreparedInference(
        _expression=expression64,
        _tf_expression=tf_expression,
        gene_names=genes,
        target_names=targets,
        tf_names=tfs,
        _target_to_tf_position=target_to_tf_position,
        _target_expression_additional_nbytes=target_expression_additional_nbytes,
        tree_method=tree_method,
        n_estimators=int(n_estimators),
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=None if max_depth is None else int(max_depth),
        bootstrap=effective_bootstrap,
        random_seed=int(random_seed),
    )


__all__ = [
    "EdgeRecord",
    "FATAL_MODEL_STATUSES",
    "InferenceBatchSummary",
    "InferenceResult",
    "MODEL_STATUSES",
    "ModelResult",
    "ModelStat",
    "ModelStatus",
    "PreparedInference",
    "SKIP_REASONS",
    "SkipReason",
    "SkippedTargetRecord",
    "TRAINED_MODEL_STATUSES",
    "TrainedModelStatus",
    "UNTRAINED_MODEL_STATUSES",
    "UntrainedModelStatus",
    "create_tree_estimator",
    "prepare_inference",
]
