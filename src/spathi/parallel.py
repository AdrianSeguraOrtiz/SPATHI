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
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar

from joblib import Parallel, cpu_count, delayed
from threadpoolctl import threadpool_limits

T = TypeVar("T")
R = TypeVar("R")

ParallelLevel = Literal["none", "tasks", "estimator"]


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """Resolved allocation of SPATHI's one public thread budget.

    ``effective_threads`` is the budget after resolving ``-1`` and respecting
    the logical CPUs available to the Python process. ``outer_jobs`` and
    ``model_n_jobs`` show where that budget is spent.  At most one of those
    fields can be greater than one.
    """

    requested_threads: int
    available_threads: int
    effective_threads: int
    total_tasks: int
    outer_jobs: int
    model_n_jobs: int
    backend: str
    parallel_level: ParallelLevel

    def __post_init__(self) -> None:
        if self.outer_jobs > 1 and self.model_n_jobs > 1:
            raise ValueError("nested parallelism is not allowed")

    @property
    def estimator_n_jobs(self) -> int:
        """Alias matching scikit-learn's estimator parameter terminology."""

        return self.model_n_jobs

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for run metadata."""

        return asdict(self)


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

    budget = capacity if threads == -1 else min(threads, capacity)
    budget = max(1, budget)

    if n_tasks == 0 or budget == 1:
        return ParallelPlan(
            requested_threads=threads,
            available_threads=capacity,
            effective_threads=1 if n_tasks == 0 else budget,
            total_tasks=n_tasks,
            outer_jobs=1,
            model_n_jobs=1,
            backend="sequential",
            parallel_level="none",
        )

    if n_tasks >= budget:
        return ParallelPlan(
            requested_threads=threads,
            available_threads=capacity,
            effective_threads=budget,
            total_tasks=n_tasks,
            outer_jobs=budget,
            model_n_jobs=1,
            backend="threading",
            parallel_level="tasks",
        )

    return ParallelPlan(
        requested_threads=threads,
        available_threads=capacity,
        effective_threads=budget,
        total_tasks=n_tasks,
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

    with threadpool_limits(limits=1):
        if plan.outer_jobs == 1:
            return [function(task) for task in task_list]

        return Parallel(
            n_jobs=plan.outer_jobs,
            backend="threading",
            prefer="threads",
            require="sharedmem",
            verbose=verbose,
        )(delayed(function)(task) for task in task_list)


# Compact compatibility aliases for callers that use shorter terminology.
resolve_threads = resolve_thread_budget
run_tasks = execute_tasks


__all__ = [
    "ParallelPlan",
    "available_cpu_count",
    "execute_tasks",
    "resolve_thread_budget",
    "resolve_threads",
    "run_tasks",
    "stable_task_seed",
]
