from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.distribution
@pytest.mark.integration
def test_wheel_build_contains_package_and_console_entry_point(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dist"
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(output_dir.glob("spathi-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.namelist()
        assert "spathi/__init__.py" in members
        assert "spathi/_publication.py" in members
        assert "spathi/_report.py" in members
        assert "spathi/core.py" in members
        assert "spathi/preparation.py" in members
        assert "spathi/py.typed" in members
        entry_points = next(name for name in members if name.endswith("entry_points.txt"))
        assert "spathi = spathi.cli:main" in archive.read(entry_points).decode("utf-8")
        metadata_path = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_path).decode("utf-8")
        assert "Import-Name: spathi\n" in metadata
        assert "License-Expression: MIT\n" in metadata
        assert "Requires-Python: >=3.11\n" in metadata
        assert "Requires-Dist: plotly>=6.0\n" in metadata
        assert "Requires-Dist: h5py>=3.10\n" in metadata
        assert "Requires-Dist: scipy>=1.11\n" in metadata
        for classifier in (
            "MacOS",
            "Microsoft :: Windows",
            "POSIX :: Linux",
        ):
            assert f"Classifier: Operating System :: {classifier}\n" in metadata
