"""SPATHI public Python API.

The scientific stack is imported lazily so lightweight operations such as
``spathi --version`` do not load NumPy, pandas, and scikit-learn. The exported
objects and their type information remain available to library callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spathi._version import __version__

if TYPE_CHECKING:
    from spathi.config import PrepareConfig as PrepareConfig
    from spathi.config import SpathiConfig as SpathiConfig
    from spathi.core import SpathiRunResult as SpathiRunResult
    from spathi.core import infer as infer
    from spathi.preparation import PreparationInputError as PreparationInputError
    from spathi.preparation import PrepareResult as PrepareResult
    from spathi.preparation import prepare as prepare
    from spathi.progress import ProgressCallback as ProgressCallback
    from spathi.progress import SpathiProgressEvent as SpathiProgressEvent

__all__ = [
    "PreparationInputError",
    "PrepareConfig",
    "PrepareResult",
    "SpathiConfig",
    "SpathiProgressEvent",
    "SpathiRunResult",
    "ProgressCallback",
    "__version__",
    "infer",
    "prepare",
]


def __getattr__(name: str) -> Any:
    """Resolve public API objects on first use without eager scientific imports."""

    if name in {"PrepareConfig", "SpathiConfig"}:
        from spathi.config import PrepareConfig, SpathiConfig

        globals()["PrepareConfig"] = PrepareConfig
        globals()["SpathiConfig"] = SpathiConfig
        return globals()[name]
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
    if name in {"PreparationInputError", "PrepareResult", "prepare"}:
        from spathi.preparation import PreparationInputError, PrepareResult, prepare

        globals()["PreparationInputError"] = PreparationInputError
        globals()["PrepareResult"] = PrepareResult
        globals()["prepare"] = prepare
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""

    return sorted(set(globals()).union(__all__))
