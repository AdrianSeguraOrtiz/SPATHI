from spathi.parallel import resolve_thread_budget, stable_task_seed


def test_thread_budget_never_creates_nested_parallelism() -> None:
    task_plan = resolve_thread_budget(4, 20, available_threads=8)
    assert task_plan.outer_jobs == 4
    assert task_plan.model_n_jobs == 1
    assert task_plan.backend == "threading"

    estimator_plan = resolve_thread_budget(4, 2, available_threads=8)
    assert estimator_plan.outer_jobs == 1
    assert estimator_plan.model_n_jobs == 4
    assert estimator_plan.parallel_level == "estimator"


def test_minus_one_resolves_to_available_cpu_capacity() -> None:
    plan = resolve_thread_budget(-1, 20, available_threads=6)
    assert plan.requested_threads == -1
    assert plan.effective_threads == 6


def test_task_seed_depends_on_identity_not_schedule_order() -> None:
    first = stable_task_seed(123, "B cells", "TP53")
    assert first == stable_task_seed(123, "B cells", "TP53")
    assert first != stable_task_seed(123, "T cells", "TP53")
    assert first != stable_task_seed(123, "B cells", "MYC")
