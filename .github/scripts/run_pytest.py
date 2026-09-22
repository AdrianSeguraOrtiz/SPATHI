"""Run pytest while exposing failures as GitHub Actions annotations."""

from __future__ import annotations

import os
import sys
from html import escape
from pathlib import Path

import pytest


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


class FailureAnnotations:
    """Translate pytest failures into annotations visible from the Checks API."""

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if not report.failed:
            return
        path, line_index, _ = report.location
        title = _escape_property(f"pytest: {report.nodeid}")
        message = _escape_data(str(report.longrepr)[-8_000:])
        location = _escape_property(Path(path).as_posix())
        annotation = f"::error file={location},line={line_index + 1},title={title}::{message}\n"
        os.write(sys.stdout.fileno(), annotation.encode("utf-8", errors="replace"))
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a", encoding="utf-8") as summary:
                summary.write(
                    f"<details><summary>{escape(report.nodeid)}</summary>\n\n"
                    f"<pre>{escape(str(report.longrepr))}</pre>\n\n</details>\n"
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[FailureAnnotations()]))
