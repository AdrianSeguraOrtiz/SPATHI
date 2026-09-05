"""Contract tests for the opt-in scaling-suite orchestrator."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


def _package_digest(package_directory: Path) -> str:
    """Independently reproduce the package snapshot digest asserted by the tests."""

    digest = hashlib.sha256()
    for path in sorted(package_directory.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_suite_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_scaling_suite.py"
    spec = importlib.util.spec_from_file_location("spathi_scaling_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def suite() -> ModuleType:
    return _load_suite_module()


def _profile_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "test-profile",
        "description": "A test profile.",
        "limitations": [],
        "source_snapshot": None,
        "defaults": {
            "n_estimators": 5,
            "n_components": 3,
            "targets": [4],
            "threads": [1],
            "warmups": 0,
            "repeats": 1,
            "seed": 17,
            "checkpoint": False,
            "report": False,
            "resource_sample_ms": 20.0,
            "case_timeout_seconds": 60.0,
            "case_process_timeout_seconds": 120.0,
        },
        "cases": [
            {
                "id": "case-one",
                "description": "One case.",
                "tags": ["test"],
                "cells": 12,
                "genes": 8,
                "tfs": 2,
                "groups": 2,
            }
        ],
    }


def _write_profile(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_builtin_profiles_follow_the_current_contract(suite: ModuleType) -> None:
    expected_case_counts = {"smoke": 1, "progressive": 18, "cll-envelope": 4}

    for name, expected_count in expected_case_counts.items():
        profile = suite.load_profile(name)
        assert profile.name == name
        assert len(profile.cases) == expected_count
        assert len(profile.sha256) == 64
        assert profile.source_path.parent.name == "v1"


def test_suite_sigterm_uses_the_interrupt_cleanup_path(
    suite: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_handler = object()
    installed_handlers: list[object] = []
    monkeypatch.setattr(suite.signal, "getsignal", lambda _signal: previous_handler)
    monkeypatch.setattr(
        suite.signal,
        "signal",
        lambda _signal, handler: installed_handlers.append(handler),
    )

    with suite._sigterm_as_keyboard_interrupt():
        with pytest.raises(KeyboardInterrupt):
            installed_handlers[-1](suite.signal.SIGTERM, None)

    assert installed_handlers[-1] is previous_handler


def test_cll_profile_records_dense_and_single_group_limitations(suite: ModuleType) -> None:
    profile = suite.load_profile("cll-envelope")
    limitations = " ".join(profile.limitations).lower()

    assert "dense" in limitations
    assert "one group" in limitations
    assert "eight targets" in limitations
    assert profile.source_snapshot is not None
    assert profile.source_snapshot.relative_path == "sources/cll-observed-v1.json"
    assert len(profile.source_snapshot.sha256) == 64
    assert [(case.cells, case.genes, case.tfs, case.groups) for case in profile.cases] == [
        (319, 14058, 1000, 1),
        (1156, 17188, 1107, 2),
        (4616, 22315, 1249, 3),
        (9806, 21659, 1233, 3),
    ]
    assert [case.id for case in profile.cases] == [
        "et29-cd8-single-group",
        "p34638-cd4-medium",
        "p34638-b-widest-model-universe",
        "e20141-b-largest-cells",
    ]


def test_progressive_profile_measures_all_available_thread_budgets(suite: ModuleType) -> None:
    case = next(
        case for case in suite.load_profile("progressive").cases if case.id == "threads-axis"
    )

    assert case.threads == (1, 2, 4, 8)

    full_target_case = next(
        case for case in suite.load_profile("progressive").cases if case.id == "full-targets-250"
    )
    assert full_target_case.targets is None
    assert "--targets" not in suite.build_case_command(
        full_target_case,
        case_directory=Path("case"),
    )


def test_profile_rejects_a_changed_source_snapshot(
    suite: ModuleType,
    tmp_path: Path,
) -> None:
    source_directory = tmp_path / "sources"
    source_directory.mkdir()
    source_path = source_directory / "observations.json"
    source_path.write_text("{}\n", encoding="utf-8")
    document = _profile_document()
    document["source_snapshot"] = {
        "path": "sources/observations.json",
        "sha256": "0" * 64,
    }
    profile_path = _write_profile(tmp_path / "profile.json", document)

    with pytest.raises(suite.ProfileError, match="source snapshot hash mismatch"):
        suite.load_profile(profile_path)


def test_case_command_is_explicit_and_reuses_one_dataset_matrix(
    suite: ModuleType, tmp_path: Path
) -> None:
    case = next(
        case for case in suite.load_profile("progressive").cases if case.id == "targets-axis"
    )

    command = suite.build_case_command(case, case_directory=tmp_path / case.id)

    target_position = command.index("--targets")
    thread_position = command.index("--threads")
    assert command[target_position + 1 : thread_position] == ["16", "64", "256"]
    assert command[thread_position + 1 : command.index("--warmups")] == ["8"]
    assert "--no-checkpoint" in command
    assert "--no-report" in command
    assert "--keep-work-dir" not in command
    assert command[command.index("--work-dir") + 1] == str(tmp_path / case.id / "work")
    retained_command = suite.build_case_command(
        case,
        case_directory=tmp_path / case.id,
        retain_workspace=True,
    )
    assert "--keep-work-dir" in retained_command


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update({"unknown": True}), "unknown fields"),
        (
            lambda document: document["cases"][0].update({"groups": 0}),  # type: ignore[index,union-attr]
            "groups must be at least 1",
        ),
        (
            lambda document: document["defaults"].update({"targets": [9]}),  # type: ignore[union-attr]
            "targets values must not exceed genes",
        ),
        (
            lambda document: document["defaults"].update({"seed": 2**32}),  # type: ignore[union-attr]
            "seed must be between 0 and 4294967295",
        ),
    ],
)
def test_invalid_profiles_fail_before_execution(
    suite: ModuleType,
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    document = _profile_document()
    mutation(document)  # type: ignore[operator]
    profile_path = _write_profile(tmp_path / "invalid.json", document)

    with pytest.raises(suite.ProfileError, match=message):
        suite.load_profile(profile_path)


def test_profile_outer_timeout_must_cover_all_per_run_budgets(
    suite: ModuleType,
    tmp_path: Path,
) -> None:
    document = _profile_document()
    document["defaults"].update(  # type: ignore[union-attr]
        {
            "targets": [2, 4],
            "threads": [1, 2],
            "case_process_timeout_seconds": 240.0,
        }
    )
    profile_path = _write_profile(tmp_path / "invalid-timeout.json", document)

    with pytest.raises(suite.ProfileError, match="4 per-run timeout budgets"):
        suite.load_profile(profile_path)


def test_dry_run_validates_and_does_not_touch_output(
    suite: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "must-not-exist"

    exit_code = suite.main(["--profile", "smoke", "--output-dir", str(output_dir), "--dry-run"])

    assert exit_code == 0
    assert not output_dir.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert plan["profile"]["name"] == "smoke"
    assert plan["aggregate_case_process_budget_seconds"] == 300.0
    assert plan["cases"][0]["id"] == "smoke"
    assert plan["cases"][0]["generated_input_reuse"]["targets"] == [12]
    assert plan["cases"][0]["estimated_peak_disk_bytes"] > 0


def test_outer_timeout_covers_generation_and_reaps_nested_sessions(
    suite: ModuleType,
    tmp_path: Path,
) -> None:
    psutil = pytest.importorskip("psutil")
    marker = tmp_path / "case-pids.txt"
    program = """
import os
from pathlib import Path
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True,
)
Path(sys.argv[1]).write_text(f"{os.getpid()}\\n{child.pid}\\n", encoding="utf-8")
time.sleep(60)
"""
    stdout_path = tmp_path / "outer.stdout"
    stderr_path = tmp_path / "outer.stderr"
    started = time.monotonic()

    with pytest.raises(suite.CaseProcessTimeoutError, match="outer timeout"):
        suite._execute_case(
            [sys.executable, "-c", program, str(marker)],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label="timeout-case",
            timeout_seconds=0.1,
        )

    assert time.monotonic() - started < 5
    pids = [int(value) for value in marker.read_text(encoding="utf-8").splitlines()]
    assert all(
        not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        for pid in pids
    )


def test_case_execution_keeps_the_package_snapshot_source_only(
    suite: ModuleType,
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "environment.stdout"
    stderr_path = tmp_path / "environment.stderr"

    exit_code = suite._execute_case(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHONDONTWRITEBYTECODE'])",
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        label="environment",
        timeout_seconds=5.0,
    )

    assert exit_code == 0
    assert stdout_path.read_text(encoding="utf-8") == "1\n"


def test_suite_aggregates_rows_and_persists_manifest(
    suite: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_execute(
        command: object,
        *,
        stdout_path: Path,
        stderr_path: Path,
        label: str,
        timeout_seconds: float,
        python_path: Path | None = None,
    ) -> int:
        del label
        assert isinstance(command, list)
        assert timeout_seconds == 300.0
        assert Path(command[1]).name == "benchmark_scaling.py"
        assert Path(command[1]).parent == stdout_path.parents[2]
        assert python_path == stdout_path.parents[2]

        def argument(flag: str) -> str:
            return command[command.index(flag) + 1]

        with stdout_path.open("x", encoding="utf-8", newline="") as stream:
            fields = sorted(suite._REQUIRED_BENCHMARK_FIELDS)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "run_index": "1",
                    "run_type": "measurement",
                    "round": "1",
                    "position": "1",
                    "threads": argument("--threads"),
                    "wall_seconds": "1.25",
                    "peak_rss_bytes": "1000",
                    "sampled_cpu_user_seconds": "1.0",
                    "sampled_cpu_system_seconds": "0.1",
                    "status": "success",
                    "cells": argument("--cells"),
                    "genes": argument("--genes"),
                    "targets": argument("--targets"),
                    "target_list": "--targets" in command,
                    "tfs": argument("--tfs"),
                    "groups": argument("--groups"),
                    "n_estimators": argument("--n-estimators"),
                    "n_components": argument("--n-components"),
                    "checkpoint": str("--checkpoint" in command),
                    "report": str("--report" in command),
                    "show_spathi_output": "False",
                    "seed": argument("--seed"),
                    "resource_sample_ms": argument("--resource-sample-ms"),
                    "case_timeout_seconds": argument("--case-timeout-seconds"),
                    "expression_sha256": "1" * 64,
                    "tf_list_sha256": "2" * 64,
                    "groups_sha256": "3" * 64,
                    "target_list_sha256": "4" * 64,
                    "peak_run_logical_bytes": "2000",
                    "published_output_logical_bytes": "1800",
                    "retained_run_logical_bytes": "1800",
                    "implementation_sha256": _package_digest(stdout_path.parents[2] / "spathi"),
                    "benchmark_sha256": suite._file_sha256(suite.BENCHMARK_SCRIPT),
                }
            )
        stderr_path.write_text("benchmark diagnostic\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(suite, "_execute_case", fake_execute)
    output_dir = tmp_path / "suite-output"

    exit_code = suite.run_suite(suite.load_profile("smoke"), output_dir=output_dir)

    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["cases"][0]["status"] == "success"
    assert manifest["cases"][0]["result_rows"] == 1
    assert manifest["retain_workspaces"] is False
    assert manifest["benchmark"]["snapshot_path"] == "benchmark_scaling.py"
    assert (
        suite._file_sha256(output_dir / "benchmark_scaling.py") == manifest["benchmark"]["sha256"]
    )
    assert _package_digest(output_dir / "spathi") == manifest["implementation"]["sha256"]
    with (output_dir / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["suite_profile"] == "smoke"
    assert rows[0]["suite_case_id"] == "smoke"
    assert rows[0]["suite_case_tags"] == "validation"
    assert rows[0]["status"] == "success"
    assert rows[0]["wall_seconds"] == "1.25"
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_suite_preserves_case_exit_code_when_csv_is_truncated(
    suite: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def truncated_execute(
        command: object,
        *,
        stdout_path: Path,
        stderr_path: Path,
        label: str,
        timeout_seconds: float,
        python_path: Path | None = None,
    ) -> int:
        del command, label, timeout_seconds, python_path
        with stdout_path.open("x", encoding="utf-8", newline="") as stream:
            csv.DictWriter(
                stream,
                fieldnames=sorted(suite._REQUIRED_BENCHMARK_FIELDS),
            ).writeheader()
        stderr_path.write_text("terminated before any row\n", encoding="utf-8")
        return 137

    monkeypatch.setattr(suite, "_execute_case", truncated_execute)
    output_dir = tmp_path / "truncated-suite"

    assert suite.run_suite(suite.load_profile("smoke"), output_dir=output_dir) == 1

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_failures"
    assert manifest["cases"][0]["status"] == "failed"
    assert manifest["cases"][0]["exit_code"] == 137
    assert "no result rows" in manifest["cases"][0]["error"]


def test_suite_marks_active_case_interrupted(
    suite: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted_execute(
        command: object,
        *,
        stdout_path: Path,
        stderr_path: Path,
        label: str,
        timeout_seconds: float,
        python_path: Path | None = None,
    ) -> int:
        del command, label, timeout_seconds, python_path
        stdout_path.touch()
        stderr_path.touch()
        raise KeyboardInterrupt

    monkeypatch.setattr(suite, "_execute_case", interrupted_execute)
    output_dir = tmp_path / "interrupted-suite"

    with pytest.raises(KeyboardInterrupt):
        suite.run_suite(suite.load_profile("smoke"), output_dir=output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["cases"][0]["status"] == "interrupted"
    assert manifest["cases"][0]["exit_code"] == 130
    assert manifest["cases"][0]["completed_at_utc"] is not None
