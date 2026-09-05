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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

from joblib import cpu_count
from threadpoolctl import threadpool_limits

from spathi.config import ThreadBudget

T = TypeVar("T")
R = TypeVar("R")

MAX_PENDING_TASKS_PER_WORKER = 2

ParallelLevel = Literal["none", "tasks", "estimator"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParallelPlan:
    """Resolved allocation of SPATHI's one public thread budget.

    ``effective_threads`` is the actual maximum allocation after resolving
    ``auto``, process-visible CPUs, and any outer-concurrency cap. ``outer_jobs``
    and ``model_n_jobs`` show where that allocation is spent. At most one of
    those fields can be greater than one.
    """

    requested_threads: ThreadBudget
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
    threads: ThreadBudget,
    n_tasks: int,
    *,
    available_threads: int | None = None,
    max_outer_jobs: int | None = None,
) -> ParallelPlan:
    """Resolve ``threads`` into a non-nested automatic parallelism plan.

    Parameters
    ----------
    threads:
        ``"auto"`` requests all available logical CPUs. A positive integer is a
        maximum budget.
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

    if threads != "auto" and (isinstance(threads, bool) or not isinstance(threads, int)):
        raise TypeError("threads must be 'auto' or a positive integer")
    if threads != "auto" and threads < 1:
        raise ValueError("threads must be 'auto' or a positive integer")
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

    budget = capacity if threads == "auto" else min(threads, capacity)
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
) -> list[R]:
    """Execute tasks according to ``plan`` while limiting native thread pools.

    :class:`PersistentTaskExecutor` restores input ordering in the returned list,
    so result ordering does not depend on completion timing. Native BLAS/OpenMP
    pools are constrained to one thread; model-level tree parallelism remains
    controlled by the estimator's joblib ``n_jobs`` setting.
    """

    task_list = list(tasks)
    if len(task_list) != plan.total_tasks:
        raise ValueError(
            "parallel plan/task mismatch: "
            f"plan has {plan.total_tasks} tasks, received {len(task_list)}"
        )
    with PersistentTaskExecutor(plan) as executor:
        return executor.execute(function, task_list)


class PersistentTaskExecutor:
    """Reuse one worker pool across multiple bounded model batches.

    ``execute`` returns results in input order. ``consume`` instead forwards
    each result on the caller's orchestration thread as soon as it finishes,
    without materializing a result collection.
    """

    def __init__(self, plan: ParallelPlan) -> None:
        if not isinstance(plan, ParallelPlan):
            raise TypeError("plan must be a ParallelPlan")
        self.plan = plan
        self._stack: ExitStack | None = None
        self._pool: ThreadPoolExecutor | None = None

    def __enter__(self) -> PersistentTaskExecutor:
        if self._stack is not None:
            raise RuntimeError("PersistentTaskExecutor is already open")
        stack = ExitStack()
        try:
            stack.enter_context(threadpool_limits(limits=1))
            if self.plan.outer_jobs > 1:
                pool = stack.enter_context(
                    ThreadPoolExecutor(
                        max_workers=self.plan.outer_jobs,
                        thread_name_prefix="spathi-model",
                    )
                )
                self._pool = pool
        except BaseException:
            # ``threadpool_limits`` mutates process-wide native pool settings.
            # Restore them even when constructing or entering the worker pool fails.
            self._pool = None
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        stack = self._stack
        self._stack = None
        self._pool = None
        if stack is not None:
            stack.__exit__(exc_type, exc, traceback)

    def _consume_rolling(
        self,
        function: Callable[[T], R],
        tasks: Iterable[T],
        *,
        on_result: Callable[[int, R], None],
    ) -> None:
        """Run a bounded rolling task window and consume completions on this thread."""

        pool = self._pool
        if pool is None:
            raise RuntimeError("parallel worker pool is unavailable")

        task_iterator = iter(tasks)
        maximum_pending = self.plan.outer_jobs * MAX_PENDING_TASKS_PER_WORKER
        pending: dict[Future[R], int] = {}
        next_index = 0

        def submit_one() -> bool:
            nonlocal next_index
            try:
                task = next(task_iterator)
            except StopIteration:
                return False
            future = pool.submit(function, task)
            pending[future] = next_index
            next_index += 1
            return True

        try:
            while len(pending) < maximum_pending and submit_one():
                pass
            while pending:
                completed, _not_completed = wait(pending, return_when=FIRST_COMPLETED)
                # Consume exactly one completion before replenishing the window.
                # Keeping a whole completed set alive while refilling every vacancy
                # can temporarily retain two windows of dense ModelResult objects.
                # Submission order is the deterministic tie-breaker when several
                # futures completed before the caller regained control.
                future = min(completed, key=pending.__getitem__)
                task_index = pending.pop(future)
                result = future.result()
                # Drop every executor-owned reference to the completed Future
                # before the callback and before admitting replacement work. This
                # makes the configured rolling-window bound exact for dense results.
                del future
                del completed
                on_result(task_index, result)
                del result
                submit_one()
        except BaseException:
            for future in pending:
                future.cancel()
            raise

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

        missing = object()
        by_index: list[R | object] = [missing] * len(task_list)

        def collect(task_index: int, result: R) -> None:
            by_index[task_index] = result

        self._consume_rolling(function, task_list, on_result=collect)
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
        """Process results through a bounded rolling window.

        At most two tasks per worker are submitted at once. The spare wave keeps
        workers busy when model durations differ or the caller is committing a
        completed result, while bounding both running work and completed results.
        Callbacks always run on the caller's orchestration thread.
        """

        if self._stack is None:
            raise RuntimeError("PersistentTaskExecutor must be opened before use")
        if not callable(on_result):
            raise TypeError("on_result must be callable")

        if self.plan.outer_jobs == 1:
            for task in tasks:
                on_result(function(task))
            return

        self._consume_rolling(
            function,
            tasks,
            on_result=lambda _task_index, result: on_result(result),
        )


__all__ = [
    "ParallelPlan",
    "PersistentTaskExecutor",
    "MAX_PENDING_TASKS_PER_WORKER",
    "available_cpu_count",
    "execute_tasks",
    "resolve_thread_budget",
    "stable_task_seed",
]
