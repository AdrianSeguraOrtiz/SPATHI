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

    def __init__(self) -> None:
        self._reported: set[str] = set()

    def _publish(self, report: object) -> None:
        if not report.failed:
            return
        path, line_index, _ = report.location
        line_number = int(line_index or 0) + 1
        nodeid = str(report.nodeid)
        details = str(report.longrepr)
        identity = f"{nodeid}\0{details}"
        if identity in self._reported:
            return
        self._reported.add(identity)
        title = _escape_property(f"pytest: {nodeid}")
        message = _escape_data(details[-8_000:])
        location = _escape_property(Path(path).as_posix())
        annotation = f"::error file={location},line={line_number},title={title}::{message}\n"
        os.write(sys.stdout.fileno(), annotation.encode("utf-8", errors="replace"))
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a", encoding="utf-8") as summary:
                summary.write(
                    f"<details><summary>{escape(nodeid)}</summary>\n\n"
                    f"<pre>{escape(details)}</pre>\n\n</details>\n"
                )

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._publish(report)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        self._publish(report)


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[FailureAnnotations()]))
