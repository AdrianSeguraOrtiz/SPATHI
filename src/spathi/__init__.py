"""SPATHI public Python API.

The scientific stack is imported lazily so lightweight operations such as
``spathi --version`` do not load NumPy, pandas, and scikit-learn. The exported
objects and their type information remain available to library callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spathi._version import __version__

if TYPE_CHECKING:
    from spathi.config import SpathiConfig as SpathiConfig
    from spathi.core import SpathiRunResult as SpathiRunResult
    from spathi.core import infer as infer
    from spathi.progress import ProgressCallback as ProgressCallback
    from spathi.progress import SpathiProgressEvent as SpathiProgressEvent

__all__ = [
    "SpathiConfig",
    "SpathiProgressEvent",
    "SpathiRunResult",
    "ProgressCallback",
    "__version__",
    "infer",
]


def __getattr__(name: str) -> Any:
    """Resolve public API objects on first use without eager scientific imports."""

    if name == "SpathiConfig":
        from spathi.config import SpathiConfig

        globals()[name] = SpathiConfig
        return SpathiConfig
    if name in {"ProgressCallback", "SpathiProgressEvent"}:
        from spathi.progress import ProgressCallback, SpathiProgressEvent

        globals()["ProgressCallback"] = ProgressCallback
        globals()["SpathiProgressEvent"] = SpathiProgressEvent
        return globals()[name]
    if name in {"SpathiRunResult", "infer"}:
        from spathi.core import SpathiRunResult, infer

        globals()["SpathiRunResult"] = SpathiRunResult
        globals()["infer"] = infer
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""

    return sorted(set(globals()).union(__all__))
