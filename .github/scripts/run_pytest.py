"""Run pytest and expose any failure through GitHub Actions annotations."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from html import escape
from pathlib import Path


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _publish_failure(output_tail: str) -> None:
    message = _escape_data(output_tail[-8_000:])
    annotation = f"::error file=.github/workflows/ci.yml,line=1,title=pytest failed::{message}\n"
    os.write(sys.stdout.fileno(), annotation.encode("utf-8", errors="replace"))
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(
                "<details><summary>pytest failure</summary>\n\n"
                f"<pre>{escape(output_tail)}</pre>\n\n</details>\n"
            )


def main() -> int:
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", *sys.argv[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_tail: deque[str] = deque(maxlen=240)
    for line in process.stdout:
        output_tail.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    return_code = process.wait()
    if return_code != 0:
        _publish_failure("".join(output_tail))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
