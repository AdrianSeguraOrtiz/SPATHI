"""Weighted, group-specific gene-regulatory network inference.

Each model predicts one target gene from the candidate transcription factors
using every cell and one target-group-specific ``sample_weight`` vector.  The
expression values supplied by the user are not normalized or transformed here.
Target responses are retained in a C-contiguous ``float64`` matrix so small but
valid expression differences cannot disappear during preparation.  Candidate
TF predictors are extracted once into the ``float32`` working precision used by
scikit-learn's tree ensembles.  Sample weights and exported importance scores
also remain ``float64``.

Feature importances are relative predictive importances within one target model.
They do not demonstrate causality, and this MVP deliberately does not infer a
regulatory sign.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from spathi.config import MAX_RANDOM_SEED
from spathi.parallel import (
    ParallelPlan,
    execute_tasks,
    resolve_thread_budget,
    stable_task_seed,
)

LOGGER = logging.getLogger(__name__)

TreeMethod: TypeAlias = Literal["extra-trees", "random-forest"]
TreeEstimator: TypeAlias = ExtraTreesRegressor | RandomForestRegressor


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """One directed, unsigned, predictive network edge."""

    source: str
    target: str
    score: float
    sign: str
    evidence: str
    context: str

    def to_dict(self) -> dict[str, str | float]:
        """Return columns in the exact ANDREA-compatible network order."""

        return {
            "source": self.source,
            "target": self.target,
            "score": self.score,
            "sign": self.sign,
            "evidence": self.evidence,
            "context": self.context,
        }


@dataclass(frozen=True, slots=True)
class SkippedTargetRecord:
    """A target for which no usable edge set could be inferred."""

    target_group: str
    target: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelStat:
    """Audit information for one requested ``(group, target)`` model."""

    target_group: str
    target: str
    status: str
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
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

    @property
    def edge_records(self) -> tuple[EdgeRecord, ...]:
        """Descriptive alias useful to orchestration code."""

        return self.edges

    @property
    def skipped_records(self) -> tuple[SkippedTargetRecord, ...]:
        """Descriptive alias useful to orchestration code."""

        return self.skipped_targets

    def network_rows(self) -> list[dict[str, str | float]]:
        """Return rows ready to construct the exact ``network.csv`` table."""

        return [edge.to_dict() for edge in self.edges]

    def skipped_rows(self) -> list[dict[str, str]]:
        """Return rows ready to construct ``skipped_targets.tsv``."""

        return [record.to_dict() for record in self.skipped_targets]

    def model_stat_rows(self) -> list[dict[str, Any]]:
        """Return per-model audit rows for optional instrumentation output."""

        return [record.to_dict() for record in self.model_stats]


@dataclass(frozen=True, slots=True)
class _PreparedGroup:
    name: str
    weights: NDArray[np.float64]
    positive_mask: NDArray[np.bool_]
    constant_tf_positions: frozenset[int]
    variable_tf_positions: tuple[int, ...]
    variable_tf_expression: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _ModelTask:
    group: _PreparedGroup
    target_index: int
    target_name: str


@dataclass(frozen=True, slots=True)
class _TaskResult:
    edges: tuple[EdgeRecord, ...]
    skipped: SkippedTargetRecord | None
    stat: ModelStat
    trained: bool


@dataclass(frozen=True, slots=True)
class _FitContext:
    expression: NDArray[np.float64]
    tf_expression: NDArray[np.float32]
    tf_names: tuple[str, ...]
    target_to_tf_position: tuple[int | None, ...]
    tree_method: TreeMethod
    n_estimators: int
    max_features: str | int | float | None
    min_samples_leaf: int | float
    max_depth: int | None
    bootstrap: bool
    global_seed: int
    model_n_jobs: int


@dataclass(frozen=True, slots=True)
class PreparedInference:
    """Reusable, validated tree-inference state.

    Construct instances with :func:`prepare_inference`. The cells-by-genes
    expression matrix is retained as contiguous ``float64`` target storage. The
    TF predictor matrix is extracted into contiguous ``float32`` exactly once.
    Both are then shared by every group inference call. Group-specific weights
    remain ``float64`` and are never stored on this reusable object.
    """

    _expression: NDArray[np.float64]
    _tf_expression: NDArray[np.float32]
    gene_names: tuple[str, ...]
    tf_names: tuple[str, ...]
    _target_to_tf_position: tuple[int | None, ...]
    tree_method: TreeMethod
    n_estimators: int
    max_features: str | int | float | None
    min_samples_leaf: int | float
    max_depth: int | None
    bootstrap: bool
    random_seed: int

    @property
    def expression_dtype(self) -> str:
        """Storage dtype used by the reusable cells-by-genes training matrix."""

        return str(self._expression.dtype)

    @property
    def predictor_dtype(self) -> str:
        """Storage dtype used by the reusable TF predictor matrix."""

        return str(self._tf_expression.dtype)

    @property
    def expression_nbytes(self) -> int:
        """Bytes occupied by the reusable cells-by-genes training matrix."""

        return int(self._expression.nbytes)

    @property
    def predictor_nbytes(self) -> int:
        """Bytes occupied by the reusable TF predictor matrix."""

        return int(self._tf_expression.nbytes)

    @property
    def n_cells(self) -> int:
        return int(self._expression.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self._expression.shape[1])

    @property
    def n_transcription_factors(self) -> int:
        return int(self._tf_expression.shape[1])

    def infer_group(
        self,
        group_name: object,
        weights: ArrayLike,
        *,
        threads: int = -1,
        verbose: int = 0,
    ) -> InferenceResult:
        """Infer all target models for one group using the prepared matrices."""

        prepared_groups = _prepare_groups(
            {group_name: weights},
            [group_name],
            n_cells=self.n_cells,
            tf_expression=self._tf_expression,
        )
        return self._infer_prepared_groups(
            prepared_groups,
            threads=threads,
            verbose=verbose,
        )

    def infer_groups(
        self,
        group_weights: Mapping[object, ArrayLike],
        *,
        group_order: Sequence[object] | None = None,
        threads: int = -1,
        verbose: int = 0,
    ) -> InferenceResult:
        """Infer several groups under one global, non-nested parallel plan."""

        prepared_groups = _prepare_groups(
            group_weights,
            group_order,
            n_cells=self.n_cells,
            tf_expression=self._tf_expression,
        )
        return self._infer_prepared_groups(
            prepared_groups,
            threads=threads,
            verbose=verbose,
        )

    def iter_group_target_batches(
        self,
        group_weights: Mapping[object, ArrayLike],
        *,
        target_batch_size: int,
        group_order: Sequence[object] | None = None,
        threads: int = -1,
        verbose: int = 0,
    ) -> Iterator[InferenceResult]:
        """Yield bounded inference results without re-preparing group weights.

        Groups are visited in ``group_order`` and targets lexicographically.
        Large target sets yield at most ``target_batch_size`` models for a single
        group, preserving globally streamable ordering. If every target fits in
        one batch, all requested groups are fitted together so a small number of
        genes can still use task-level parallelism. Constant-predictor masks and
        other group state are computed only once before the first batch is fitted.
        """

        if isinstance(target_batch_size, bool) or not isinstance(target_batch_size, Integral):
            raise TypeError("target_batch_size must be a positive integer")
        if target_batch_size < 1:
            raise ValueError("target_batch_size must be a positive integer")
        _validate_verbose(verbose)
        prepared_groups = _prepare_groups(
            group_weights,
            group_order,
            n_cells=self.n_cells,
            tf_expression=self._tf_expression,
        )
        ordered_targets = tuple(sorted(enumerate(self.gene_names), key=lambda item: item[1]))
        if len(ordered_targets) <= int(target_batch_size):
            # When every target fits, keep the complete group batch together so
            # sparse target sets can still expose enough independent tasks to
            # the outer parallel scheduler. The assembled result remains globally
            # ordered by context, target, and source.
            yield self._infer_prepared_groups(
                prepared_groups,
                threads=threads,
                verbose=verbose,
                target_items=ordered_targets,
            )
            return

        for group in prepared_groups:
            for start in range(0, len(ordered_targets), int(target_batch_size)):
                yield self._infer_prepared_groups(
                    (group,),
                    threads=threads,
                    verbose=verbose,
                    target_items=ordered_targets[start : start + int(target_batch_size)],
                )

    def _infer_prepared_groups(
        self,
        prepared_groups: tuple[_PreparedGroup, ...],
        *,
        threads: int,
        verbose: int,
        target_items: Sequence[tuple[int, str]] | None = None,
    ) -> InferenceResult:
        _validate_verbose(verbose)
        started = perf_counter()
        targets = tuple(enumerate(self.gene_names)) if target_items is None else target_items
        tasks = [
            _ModelTask(group=group, target_index=target_index, target_name=target_name)
            for group in prepared_groups
            for target_index, target_name in targets
        ]
        plan = resolve_thread_budget(threads, len(tasks))
        context = _FitContext(
            expression=self._expression,
            tf_expression=self._tf_expression,
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

        LOGGER.info(
            "Fitting %d weighted models with %d effective thread(s) at the %s level",
            len(tasks),
            plan.effective_threads,
            plan.parallel_level,
        )
        task_results = execute_tasks(
            lambda task: _fit_model_task(task, context),
            tasks,
            plan,
            verbose=int(verbose),
        )
        return _assemble_result(
            task_results,
            prepared_groups=prepared_groups,
            plan=plan,
            tree_method=self.tree_method,
            started=started,
            expression_dtype=self.expression_dtype,
            expression_nbytes=self.expression_nbytes,
            predictor_nbytes=self.predictor_nbytes,
        )


def create_tree_estimator(
    tree_method: TreeMethod,
    *,
    n_estimators: int,
    max_features: str | int | float | None,
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
    max_features: str | int | float | None,
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
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float32],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int | None, ...],
]:
    genes = _normalise_identifiers(gene_names, label="gene_names")
    tfs = _normalise_identifiers(tf_names, label="tf_names")

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
    tf_position_by_target = {gene_to_index[tf]: position for position, tf in enumerate(tfs)}
    target_to_tf_position = tuple(
        tf_position_by_target.get(target_index) for target_index in range(len(genes))
    )
    return expression64, tf_expression, genes, tfs, target_to_tf_position


def _prepare_groups(
    group_weights: Mapping[object, ArrayLike],
    group_order: Sequence[object] | None,
    *,
    n_cells: int,
    tf_expression: NDArray[np.float32],
) -> tuple[_PreparedGroup, ...]:
    if not isinstance(group_weights, Mapping) or not group_weights:
        raise ValueError("group_weights must be a non-empty mapping")

    weights_by_name: dict[str, NDArray[np.float64]] = {}
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
        if not np.any(weights > 0.0):
            raise ValueError(f"weights for group {group!r} are all zero")
        weights_by_name[group] = np.ascontiguousarray(weights)

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
    for group in ordered_names:
        weights = weights_by_name[group]
        positive_mask = weights > 0.0
        if np.all(positive_mask):
            constant_mask = np.all(tf_expression == tf_expression[0], axis=0)
        else:
            # Boolean indexing would copy the full positive-cells-by-TF matrix
            # once per group. Masked reductions keep peak memory bounded while
            # deriving the exact same constant-predictor decision.
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
            constant_mask = minima == maxima
        constant_positions = frozenset(int(position) for position in np.flatnonzero(constant_mask))
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
                positive_mask=np.ascontiguousarray(positive_mask),
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
    status: str,
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
        n_positive_weight_samples=int(np.count_nonzero(task.group.positive_mask)),
        weight_sum=float(np.sum(task.group.weights, dtype=np.float64)),
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
    reason: str,
    detail: str,
    seed: int,
    n_predictors_used: int,
    discarded: tuple[str, ...],
    constant_predictors: tuple[str, ...],
    fit_seconds: float = 0.0,
) -> _TaskResult:
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
    return _TaskResult(edges=(), skipped=skipped, stat=stat, trained=False)


def _fit_model_task(task: _ModelTask, context: _FitContext) -> _TaskResult:
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

    n_positive = int(np.count_nonzero(task.group.positive_mask))
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
    positive_y = y[task.group.positive_mask]
    if np.all(positive_y == positive_y[0]):
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
        return _TaskResult(edges=(), skipped=skipped, stat=stat, trained=True)

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
    return _TaskResult(edges=edges, skipped=None, stat=stat, trained=True)


def _assemble_result(
    task_results: list[_TaskResult],
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
    LOGGER.info(
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
    tree_method: TreeMethod = "extra-trees",
    n_estimators: int = 500,
    max_features: str | int | float | None = 1.0,
    min_samples_leaf: int | float = 1,
    max_depth: int | None = None,
    bootstrap: bool | None = None,
    random_seed: int = 123,
) -> PreparedInference:
    """Validate and materialize reusable state for one or more group networks.

    This is the preferred API for progressive output pipelines. It performs the
    only conversion of the cells-by-genes matrix and the only extraction of the
    TF predictor matrix. Calling :meth:`PreparedInference.infer_group` repeatedly
    then reuses both arrays without re-normalizing, re-casting, or re-extracting
    expression data.
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
        tfs,
        target_to_tf_position,
    ) = _prepare_expression(expression, gene_names, tf_names)
    effective_bootstrap = tree_method == "random-forest" if bootstrap is None else bootstrap
    return PreparedInference(
        _expression=expression64,
        _tf_expression=tf_expression,
        gene_names=genes,
        tf_names=tfs,
        _target_to_tf_position=target_to_tf_position,
        tree_method=tree_method,
        n_estimators=int(n_estimators),
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=None if max_depth is None else int(max_depth),
        bootstrap=effective_bootstrap,
        random_seed=int(random_seed),
    )


def infer_networks(
    expression: ArrayLike,
    gene_names: Sequence[object],
    tf_names: Sequence[object],
    group_weights: Mapping[object, ArrayLike],
    *,
    group_order: Sequence[object] | None = None,
    tree_method: TreeMethod = "extra-trees",
    n_estimators: int = 500,
    max_features: str | int | float | None = 1.0,
    min_samples_leaf: int | float = 1,
    max_depth: int | None = None,
    bootstrap: bool | None = None,
    random_seed: int = 123,
    threads: int = -1,
    verbose: int = 0,
) -> InferenceResult:
    """Infer one weighted network per group across all genes as targets.

    Parameters
    ----------
    expression:
        Numeric matrix with cells in rows and genes in columns. Its supplied
        values are used directly for prediction without a biological
        normalization. Target responses retain ``float64`` precision; only TF
        predictors use scikit-learn's ``float32`` working representation.
    gene_names:
        Unique column identifiers. Every gene is processed as a target.
    tf_names:
        Unique candidate predictors, all present in ``gene_names``. A target TF
        is removed from its own model, preventing autoedges.
    group_weights:
        Mapping from target-group identifier to one non-negative, finite
        ``sample_weight`` vector over all cells. Vectors are computed once by the
        weighting phase and reused for every target in that group.
    group_order:
        Optional explicit deterministic group order. By default group identifiers
        are sorted lexicographically.
    threads:
        The sole parallelism budget: ``-1`` means all available logical CPUs.

    Returns
    -------
    InferenceResult
        Deterministically ordered edges, skipped-target records, model audit
        statistics, the resolved parallel plan, and completion counts.
    """

    started = perf_counter()
    prepared = prepare_inference(
        expression,
        gene_names,
        tf_names,
        tree_method=tree_method,
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        bootstrap=bootstrap,
        random_seed=random_seed,
    )
    result = prepared.infer_groups(
        group_weights,
        group_order=group_order,
        threads=threads,
        verbose=verbose,
    )
    # Preserve the legacy wrapper's timing semantics: preparation is included.
    return replace(result, duration_seconds=perf_counter() - started)


# Explicit aliases keep pipeline code readable while retaining one implementation.
infer_group_specific_networks = infer_networks
infer_group_specific_grns = infer_networks


__all__ = [
    "EdgeRecord",
    "InferenceResult",
    "ModelStat",
    "PreparedInference",
    "SkippedTargetRecord",
    "create_tree_estimator",
    "infer_group_specific_grns",
    "infer_group_specific_networks",
    "infer_networks",
    "prepare_inference",
]
