from __future__ import annotations

import subprocess
import sys


def test_package_import_is_lazy_but_public_exports_remain_available() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, spathi; "
                "assert 'sklearn' not in sys.modules; "
                "from spathi import SpathiConfig, infer_group_specific_grns; "
                "assert SpathiConfig.__name__ == 'SpathiConfig'; "
                "assert callable(infer_group_specific_grns)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
