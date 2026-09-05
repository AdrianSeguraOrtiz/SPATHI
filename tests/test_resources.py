from __future__ import annotations

import pytest

from spathi import resources
from spathi.resources import available_memory_bytes, estimate_model_memory_bytes, plan_model_memory


def _mapping_reader(values: dict[str, str]) -> resources._TextReader:
    return lambda path: values.get(str(path))


def test_cgroup_v2_uses_current_systemd_group_and_restrictive_ancestor() -> None:
    cgroup_text = "0::/user.slice/user-1000.slice/session-4.scope\n"
    mountinfo_text = "36 25 0:32 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup rw\n"
    read_text = _mapping_reader(
        {
            "/sys/fs/cgroup/user.slice/user-1000.slice/session-4.scope/memory.max": "1000\n",
            "/sys/fs/cgroup/user.slice/user-1000.slice/session-4.scope/memory.current": "200\n",
            "/sys/fs/cgroup/user.slice/user-1000.slice/memory.max": "2000\n",
            "/sys/fs/cgroup/user.slice/user-1000.slice/memory.current": "1850\n",
            "/sys/fs/cgroup/user.slice/memory.max": "max\n",
            "/sys/fs/cgroup/user.slice/memory.current": "1900\n",
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/memory.current": "2000\n",
        }
    )

    assert (
        resources._cgroup_available_bytes(
            cgroup_text=cgroup_text,
            mountinfo_text=mountinfo_text,
            read_text=read_text,
        )
        == 150
    )


def test_cgroup_v1_resolves_slurm_subgroup_relative_to_non_root_mount() -> None:
    cgroup_text = "8:cpuset:/slurm/uid_1000/job_42/step_0\n7:memory:/slurm/uid_1000/job_42/step_0\n"
    mountinfo_text = (
        "40 35 0:34 /slurm /sys/fs/cgroup/memory rw,nosuid,nodev,noexec "
        "shared:15 - cgroup cgroup rw,memory\n"
    )
    read_text = _mapping_reader(
        {
            "/sys/fs/cgroup/memory/uid_1000/job_42/step_0/memory.limit_in_bytes": "900",
            "/sys/fs/cgroup/memory/uid_1000/job_42/step_0/memory.usage_in_bytes": "100",
            "/sys/fs/cgroup/memory/uid_1000/job_42/memory.limit_in_bytes": "1000",
            "/sys/fs/cgroup/memory/uid_1000/job_42/memory.usage_in_bytes": "950",
            "/sys/fs/cgroup/memory/uid_1000/memory.limit_in_bytes": str(1 << 62),
            "/sys/fs/cgroup/memory/uid_1000/memory.usage_in_bytes": "960",
        }
    )

    assert (
        resources._cgroup_available_bytes(
            cgroup_text=cgroup_text,
            mountinfo_text=mountinfo_text,
            read_text=read_text,
        )
        == 50
    )


def test_cgroup_exhausted_limit_reports_zero_headroom() -> None:
    assert (
        resources._cgroup_available_bytes(
            cgroup_text="0::/workload\n",
            mountinfo_text="1 2 0:1 / /cg rw - cgroup2 cgroup rw\n",
            read_text=_mapping_reader(
                {
                    "/cg/workload/memory.max": "100",
                    "/cg/workload/memory.current": "125",
                }
            ),
        )
        == 0
    )


def test_cgroup_detection_falls_back_to_conventional_v1_path() -> None:
    read_text = _mapping_reader(
        {
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "500",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": "125",
        }
    )

    assert (
        resources._cgroup_available_bytes(
            cgroup_text="malformed",
            mountinfo_text="unavailable",
            read_text=read_text,
        )
        == 375
    )


def test_available_memory_combines_system_and_cgroup_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "_system_available_bytes", lambda: 400)
    monkeypatch.setattr(resources, "_cgroup_available_bytes", lambda: 250)

    assert available_memory_bytes() == 250


def test_linux_system_memory_prefers_mem_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_sysconf(name: str) -> int:
        raise AssertionError(f"sysconf should not be called for {name}")

    monkeypatch.setattr(resources.os, "sysconf", unexpected_sysconf)

    assert (
        resources._system_available_bytes(
            platform="linux",
            read_text=_mapping_reader(
                {
                    "/proc/meminfo": (
                        "MemTotal:       100000 kB\n"
                        "MemFree:         10000 kB\n"
                        "MemAvailable:    75000 kB\n"
                    )
                }
            ),
        )
        == 75_000 * 1024
    )


def test_linux_system_memory_falls_back_to_available_pages_for_invalid_meminfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sysconf_values = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 12}
    monkeypatch.setattr(resources.os, "sysconf", sysconf_values.__getitem__)

    assert (
        resources._system_available_bytes(
            platform="linux",
            read_text=_mapping_reader({"/proc/meminfo": "MemAvailable: unknown kB\n"}),
        )
        == 49_152
    )


def test_model_memory_estimate_respects_leaf_and_depth_bounds() -> None:
    unconstrained = estimate_model_memory_bytes(
        n_cells=100,
        n_transcription_factors=20,
        n_estimators=50,
        min_samples_leaf=1,
        max_depth=None,
    )
    leaf_limited = estimate_model_memory_bytes(
        n_cells=100,
        n_transcription_factors=20,
        n_estimators=50,
        min_samples_leaf=10,
        max_depth=None,
    )
    depth_limited = estimate_model_memory_bytes(
        n_cells=100,
        n_transcription_factors=20,
        n_estimators=50,
        min_samples_leaf=1,
        max_depth=2,
    )
    assert leaf_limited < unconstrained
    assert depth_limited < unconstrained


def test_memory_plan_caps_concurrency_and_reports_infeasible_budget() -> None:
    plan = plan_model_memory(
        estimated_bytes_per_model=100,
        available_bytes=1_000,
        usable_fraction=0.5,
    )
    assert plan.max_concurrent_models == 5

    constrained = plan_model_memory(
        estimated_bytes_per_model=10_000,
        available_bytes=1_000,
    )
    assert constrained.max_concurrent_models == 0

    exhausted = plan_model_memory(
        estimated_bytes_per_model=100,
        available_bytes=0,
    )
    assert exhausted.usable_bytes == 0
    assert exhausted.max_concurrent_models == 0


def test_memory_plan_reserves_future_batch_allocations_before_models() -> None:
    plan = plan_model_memory(
        estimated_bytes_per_model=100,
        available_bytes=1_000,
        usable_fraction=0.8,
        reserved_bytes=300,
    )

    assert plan.usable_bytes == 800
    assert plan.reserved_bytes == 300
    assert plan.max_concurrent_models == 5


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_memory_plan_rejects_invalid_reservation(value: object) -> None:
    with pytest.raises(ValueError, match="reserved_bytes"):
        plan_model_memory(
            estimated_bytes_per_model=100,
            available_bytes=1_000,
            reserved_bytes=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, -1, True])
def test_model_memory_estimate_rejects_invalid_dimensions(value: int) -> None:
    with pytest.raises(ValueError):
        estimate_model_memory_bytes(
            n_cells=value,
            n_transcription_factors=2,
            n_estimators=2,
            min_samples_leaf=1,
            max_depth=None,
        )
