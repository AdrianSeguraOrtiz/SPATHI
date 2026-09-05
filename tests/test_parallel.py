from threading import Event, get_ident

import pytest

import spathi.parallel as parallel_module
from spathi.parallel import PersistentTaskExecutor, resolve_thread_budget, stable_task_seed


def test_thread_budget_never_creates_nested_parallelism() -> None:
    task_plan = resolve_thread_budget(4, 20, available_threads=8)
    assert task_plan.outer_jobs == 4
    assert task_plan.model_n_jobs == 1
    assert task_plan.backend == "threading"

    estimator_plan = resolve_thread_budget(4, 2, available_threads=8)
    assert estimator_plan.outer_jobs == 1
    assert estimator_plan.model_n_jobs == 4
    assert estimator_plan.parallel_level == "estimator"


def test_thread_budget_respects_caller_supplied_outer_memory_cap() -> None:
    capped = resolve_thread_budget(
        8,
        20,
        available_threads=8,
        max_outer_jobs=3,
    )
    assert capped.outer_jobs == 1
    assert capped.model_n_jobs == 8
    assert capped.effective_threads == 8
    assert capped.max_outer_jobs == 3
    assert capped.parallel_level == "estimator"

    single_model = resolve_thread_budget(
        8,
        20,
        available_threads=8,
        max_outer_jobs=1,
    )
    assert single_model.outer_jobs == 1
    assert single_model.model_n_jobs == 8
    assert single_model.parallel_level == "estimator"


def test_auto_resolves_to_available_cpu_capacity() -> None:
    plan = resolve_thread_budget("auto", 20, available_threads=6)
    assert plan.requested_threads == "auto"
    assert plan.effective_threads == 6


def test_task_seed_depends_on_identity_not_schedule_order() -> None:
    first = stable_task_seed(123, "B cells", "TP53")
    assert first == stable_task_seed(123, "B cells", "TP53")
    assert first != stable_task_seed(123, "T cells", "TP53")
    assert first != stable_task_seed(123, "B cells", "MYC")


def test_persistent_executor_reuses_pool_and_restores_input_order() -> None:
    plan = resolve_thread_budget(2, 6, available_threads=2)

    with PersistentTaskExecutor(plan) as executor:
        first = executor.execute(lambda value: value * 2, [3, 1, 2])
        second = executor.execute(lambda value: value + 10, [2, 0, 1])

    assert first == [6, 2, 4]
    assert second == [12, 10, 11]


def test_persistent_executor_can_consume_without_collecting_results() -> None:
    plan = resolve_thread_budget(2, 6, available_threads=2)
    callback_threads: list[int] = []
    observed: list[int] = []
    caller_thread = get_ident()

    def record(result: int) -> None:
        callback_threads.append(get_ident())
        observed.append(result)

    with PersistentTaskExecutor(plan) as executor:
        returned = executor.consume(
            lambda value: value * 2,
            [3, 1, 2],
            on_result=record,
        )

    assert returned is None
    assert sorted(observed) == [2, 4, 6]
    assert callback_threads == [caller_thread] * 3


def test_persistent_executor_uses_a_bounded_rolling_window() -> None:
    plan = resolve_thread_budget(3, 8, available_threads=3)
    observed: list[int] = []
    yielded = 0
    consumed = 0
    maximum_ahead = 0

    def tasks():
        nonlocal yielded, maximum_ahead
        for value in range(20):
            yielded += 1
            maximum_ahead = max(maximum_ahead, yielded - consumed)
            yield value

    def record(result: int) -> None:
        nonlocal consumed
        consumed += 1
        observed.append(result)

    with PersistentTaskExecutor(plan) as executor:
        executor.consume(lambda value: value * 2, tasks(), on_result=record)

    assert maximum_ahead == 6
    assert sorted(observed) == [value * 2 for value in range(20)]


def test_persistent_executor_does_not_wait_for_a_complete_worker_wave() -> None:
    plan = resolve_thread_budget(2, 4, available_threads=2)
    third_task_started = Event()

    def work(value: int) -> int:
        if value == 0 and not third_task_started.wait(timeout=2.0):
            raise RuntimeError("rolling scheduler did not start the next task")
        if value == 2:
            third_task_started.set()
        return value

    observed: list[int] = []
    with PersistentTaskExecutor(plan) as executor:
        executor.consume(work, range(4), on_result=observed.append)

    assert sorted(observed) == [0, 1, 2, 3]


def test_persistent_executor_accepts_empty_batches() -> None:
    plan = resolve_thread_budget(2, 2, available_threads=2)
    with PersistentTaskExecutor(plan) as executor:
        assert executor.execute(lambda value: value, []) == []


def test_persistent_executor_restores_thread_limits_when_pool_opening_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = resolve_thread_budget(2, 4, available_threads=2)
    active_limits: list[int] = []

    class FakeLimit:
        def __enter__(self):
            active_limits.append(1)
            return self

        def __exit__(self, *args: object) -> None:
            active_limits.pop()

    class FailingThreadPool:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self):
            raise RuntimeError("simulated pool failure")

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(parallel_module, "threadpool_limits", lambda **kwargs: FakeLimit())
    monkeypatch.setattr(parallel_module, "ThreadPoolExecutor", FailingThreadPool)

    with pytest.raises(RuntimeError, match="simulated pool failure"):
        PersistentTaskExecutor(plan).__enter__()

    assert active_limits == []
