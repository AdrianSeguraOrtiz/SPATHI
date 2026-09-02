"""Deterministic parallel execution helpers for SPATHI.

The public ``threads`` setting is treated as a single, process-wide budget.  The
automatic strategy spends that budget at exactly one level:

* across independent ``(target group, target gene)`` tasks when there are enough
  tasks to occupy the budget; or
* inside one scikit-learn ensemble at a time for a small task collection.

This prevents every outer worker from creating another full set of workers.  A
threading backend is used for outer parallelism so the read-only expression and
predictor arrays remain shared instead of being copied to child processes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from typing import Any, Literal, TypeVar, cast

from joblib import Parallel, cpu_count, delayed
from threadpoolctl import threadpool_limits

T = TypeVar("T")
R = TypeVar("R")

ParallelLevel = Literal["none", "tasks", "estimator"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParallelPlan:
    """Resolved allocation of SPATHI's one public thread budget.

    ``effective_threads`` is the actual maximum allocation after resolving
    ``-1``, process-visible CPUs, and any outer-concurrency cap. ``outer_jobs``
    and ``model_n_jobs`` show where that allocation is spent. At most one of
    those fields can be greater than one.
    """

    requested_threads: int
    available_threads: int
    effective_threads: int
    total_tasks: int
    max_outer_jobs: int | None
    outer_jobs: int
    model_n_jobs: int
    backend: str
    parallel_level: ParallelLevel

    def __post_init__(self) -> None:
        if self.outer_jobs > 1 and self.model_n_jobs > 1:
            raise ValueError("nested parallelism is not allowed")


def available_cpu_count() -> int:
    """Return the logical CPU capacity visible to joblib, always at least one.

    Joblib accounts for CPU affinity and common container limits, which is safer
    than using ``os.cpu_count`` alone.  The standard-library value is retained as
    a defensive fallback for unusual joblib/runtime combinations.
    """

    try:
        count = int(cpu_count(only_physical_cores=False))
    except (NotImplementedError, TypeError, ValueError):
        count = int(os.cpu_count() or 1)
    return max(1, count)


def resolve_thread_budget(
    threads: int,
    n_tasks: int,
    *,
    available_threads: int | None = None,
    max_outer_jobs: int | None = None,
) -> ParallelPlan:
    """Resolve ``threads`` into a non-nested automatic parallelism plan.

    Parameters
    ----------
    threads:
        ``-1`` requests all available logical CPUs. A positive integer is a
        maximum budget. Zero and values below ``-1`` are invalid.
    n_tasks:
        Number of independent model fits in the run.
    available_threads:
        Optional explicit capacity, primarily useful for deterministic tests.
    max_outer_jobs:
        Optional upper bound for concurrently active model tasks. The caller
        may derive this from its own memory model; this module deliberately
        performs no system-memory detection.

    Notes
    -----
    When the number of tasks is at least the CPU budget, models run concurrently
    and each estimator receives ``n_jobs=1``. For a smaller collection, tasks run
    sequentially and each estimator may use the full budget. Thus the requested
    budget is never multiplied through nested joblib pools.
    """

    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError("threads must be -1 or a positive integer")
    if threads == 0 or threads < -1:
        raise ValueError("threads must be -1 or a positive integer")
    if isinstance(n_tasks, bool) or not isinstance(n_tasks, int):
        raise TypeError("n_tasks must be a non-negative integer")
    if n_tasks < 0:
        raise ValueError("n_tasks must be a non-negative integer")

    capacity = available_cpu_count() if available_threads is None else available_threads
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("available_threads must be a positive integer")
    if capacity < 1:
        raise ValueError("available_threads must be a positive integer")
    if max_outer_jobs is not None and (
        isinstance(max_outer_jobs, bool)
        or not isinstance(max_outer_jobs, int)
        or max_outer_jobs < 1
    ):
        raise ValueError("max_outer_jobs must be a positive integer or None")

    budget = capacity if threads == -1 else min(threads, capacity)
    budget = max(1, budget)

    if n_tasks == 0 or budget == 1:
        return ParallelPlan(
            requested_threads=threads,
            available_threads=capacity,
            effective_threads=1 if n_tasks == 0 else budget,
            total_tasks=n_tasks,
            max_outer_jobs=max_outer_jobs,
            outer_jobs=1,
            model_n_jobs=1,
            backend="sequential",
            parallel_level="none",
        )

    if n_tasks >= budget:
        outer_jobs = min(budget, max_outer_jobs or budget)
        if outer_jobs < budget:
            # A memory cap below the CPU budget would leave processors idle in
            # task-level mode. Fit one ensemble at a time and spend the complete
            # budget across its independent trees instead; this also remains
            # within the memory allowance for a single model.
            return ParallelPlan(
                requested_threads=threads,
                available_threads=capacity,
                effective_threads=budget,
                total_tasks=n_tasks,
                max_outer_jobs=max_outer_jobs,
                outer_jobs=1,
                model_n_jobs=budget,
                backend="threading",
                parallel_level="estimator",
            )
        return ParallelPlan(
            requested_threads=threads,
            available_threads=capacity,
            effective_threads=outer_jobs,
            total_tasks=n_tasks,
            max_outer_jobs=max_outer_jobs,
            outer_jobs=outer_jobs,
            model_n_jobs=1,
            backend="threading",
            parallel_level="tasks",
        )

    return ParallelPlan(
        requested_threads=threads,
        available_threads=capacity,
        effective_threads=budget,
        total_tasks=n_tasks,
        max_outer_jobs=max_outer_jobs,
        outer_jobs=1,
        model_n_jobs=budget,
        backend="threading",
        parallel_level="estimator",
    )


def stable_task_seed(global_seed: int, group_id: object, target: object) -> int:
    """Derive a stable uint32 seed from a task's scientific identity.

    Python's built-in ``hash`` is deliberately randomized between processes, so
    it is unsuitable here. BLAKE2 over a canonical JSON tuple makes seeds
    independent of task enumeration, joblib scheduling, and thread count.
    """

    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if global_seed < 0:
        raise ValueError("global_seed must be non-negative")

    payload = json.dumps(
        [global_seed, str(group_id), str(target)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2s(
        payload,
        digest_size=4,
        person=b"SPATHI-1",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def execute_tasks(
    function: Callable[[T], R],
    tasks: Iterable[T],
    plan: ParallelPlan,
    *,
    verbose: int = 0,
) -> list[R]:
    """Execute tasks according to ``plan`` while limiting native thread pools.

    Joblib preserves input ordering in the returned list, so result ordering does
    not depend on completion timing. Native BLAS/OpenMP pools are constrained to
    one thread; model-level tree parallelism remains controlled by the
    estimator's joblib ``n_jobs`` setting.
    """

    task_list = list(tasks)
    if len(task_list) != plan.total_tasks:
        raise ValueError(
            "parallel plan/task mismatch: "
            f"plan has {plan.total_tasks} tasks, received {len(task_list)}"
        )
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
        raise ValueError("verbose must be a non-negative integer")

    with PersistentTaskExecutor(plan, verbose=verbose) as executor:
        return executor.execute(function, task_list)


class PersistentTaskExecutor:
    """Reuse one worker pool across multiple bounded model batches.

    ``execute`` returns results in input order. ``consume`` instead forwards
    each result on the caller's orchestration thread as soon as it finishes,
    without materializing a result collection.
    """

    def __init__(self, plan: ParallelPlan, *, verbose: int = 0) -> None:
        if not isinstance(plan, ParallelPlan):
            raise TypeError("plan must be a ParallelPlan")
        if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer")
        self.plan = plan
        self.verbose = verbose
        self._stack: ExitStack | None = None
        self._parallel: Parallel | None = None

    def __enter__(self) -> PersistentTaskExecutor:
        if self._stack is not None:
            raise RuntimeError("PersistentTaskExecutor is already open")
        stack = ExitStack()
        try:
            stack.enter_context(threadpool_limits(limits=1))
            if self.plan.outer_jobs > 1:
                parallel = Parallel(
                    n_jobs=self.plan.outer_jobs,
                    backend="threading",
                    prefer="threads",
                    require="sharedmem",
                    verbose=self.verbose,
                    return_as="generator_unordered",
                )
                stack.enter_context(parallel)
                self._parallel = parallel
        except BaseException:
            # ``threadpool_limits`` mutates process-wide native pool settings.
            # Restore them even when constructing/entering Joblib's pool fails.
            self._parallel = None
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        stack = self._stack
        self._stack = None
        self._parallel = None
        if stack is not None:
            stack.__exit__(exc_type, exc, traceback)

    def execute(
        self,
        function: Callable[[T], R],
        tasks: Iterable[T],
    ) -> list[R]:
        """Execute one batch and return its results in the supplied task order."""

        if self._stack is None:
            raise RuntimeError("PersistentTaskExecutor must be opened before use")
        task_list = list(tasks)
        if not task_list:
            return []

        if self.plan.outer_jobs == 1:
            return [function(task) for task in task_list]

        assert self._parallel is not None

        def indexed(task_index: int, task: T) -> tuple[int, R]:
            return task_index, function(task)

        pending = self._parallel(
            delayed(indexed)(task_index, task) for task_index, task in enumerate(task_list)
        )
        missing = object()
        by_index: list[R | object] = [missing] * len(task_list)
        for task_index, result in pending:
            by_index[task_index] = result
        if any(result is missing for result in by_index):
            raise RuntimeError("parallel execution returned an incomplete result set")
        return [cast(R, result) for result in by_index]

    def consume(
        self,
        function: Callable[[T], R],
        tasks: Iterable[T],
        *,
        on_result: Callable[[R], None],
    ) -> None:
        """Process results with a strictly bounded completion backlog.

        Joblib's unordered generator does not apply backpressure while the caller
        handles a completed result. Submit at most one outer-worker wave at a time
        so slow checkpoint or output callbacks can never leave a batch-sized
        collection of completed model results queued in memory.
        """

        if self._stack is None:
            raise RuntimeError("PersistentTaskExecutor must be opened before use")
        if not callable(on_result):
            raise TypeError("on_result must be callable")

        if self.plan.outer_jobs == 1:
            for task in tasks:
                on_result(function(task))
            return

        assert self._parallel is not None
        task_iterator = iter(tasks)
        while task_wave := tuple(islice(task_iterator, self.plan.outer_jobs)):
            pending = self._parallel(delayed(function)(task) for task in task_wave)
            for result in pending:
                on_result(result)


__all__ = [
    "ParallelPlan",
    "PersistentTaskExecutor",
    "available_cpu_count",
    "execute_tasks",
    "resolve_thread_budget",
    "stable_task_seed",
]
