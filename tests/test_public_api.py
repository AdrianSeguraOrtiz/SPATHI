from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from spathi.core import SpathiRunResult


def test_run_result_is_an_explicit_keyword_only_record() -> None:
    result = SpathiRunResult(
        output_dir=Path("run"),
        network_path=Path("run/network.csv"),
        metadata_path=Path("run/run_metadata.json"),
        report_path=Path("run/report.html"),
        n_edges=3,
        total_models=4,
        trained_models=4,
        skipped_target_records=0,
        resumed_models=0,
        warnings=(),
    )

    assert result.warnings == ()
    assert result.resumed_models == 0
    assert result.report_path == Path("run/report.html")


def test_run_result_rejects_impossible_accounting() -> None:
    with pytest.raises(ValueError, match="trained_models cannot exceed total_models"):
        SpathiRunResult(
            output_dir=Path("run"),
            network_path=Path("run/network.csv"),
            metadata_path=Path("run/run_metadata.json"),
            report_path=None,
            n_edges=0,
            total_models=1,
            trained_models=2,
            skipped_target_records=0,
            resumed_models=0,
            warnings=(),
        )


def test_package_import_is_lazy_but_public_exports_remain_available() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, spathi; "
                "assert 'sklearn' not in sys.modules; "
                "assert 'h5py' not in sys.modules; "
                "from spathi import PrepareConfig, SpathiConfig, SpathiProgressEvent, infer; "
                "assert PrepareConfig.__name__ == 'PrepareConfig'; "
                "assert SpathiConfig.__name__ == 'SpathiConfig'; "
                "assert SpathiProgressEvent.__name__ == 'SpathiProgressEvent'; "
                "assert callable(infer)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_preparation_api_is_lazy_and_public() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, spathi; "
                "assert 'h5py' not in sys.modules; "
                "from spathi import PreparationInputError, PrepareResult, prepare; "
                "assert PreparationInputError.__name__ == 'PreparationInputError'; "
                "assert PrepareResult.__name__ == 'PrepareResult'; "
                "assert callable(prepare); "
                "assert 'h5py' in sys.modules; "
                "assert 'spathi.core' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
