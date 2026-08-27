"""SPATHI public Python API.

The scientific stack is imported lazily so lightweight operations such as
``spathi --version`` do not load NumPy, pandas, and scikit-learn. The exported
objects and their type information remain unchanged for library callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spathi._version import __version__

if TYPE_CHECKING:
    from spathi.config import SpathiConfig as SpathiConfig
    from spathi.pipeline import SpathiRunResult as SpathiRunResult
    from spathi.pipeline import infer_group_specific_grns as infer_group_specific_grns

__all__ = [
    "SpathiConfig",
    "SpathiRunResult",
    "__version__",
    "infer_group_specific_grns",
]


def __getattr__(name: str) -> Any:
    """Resolve public API objects on first use without eager scientific imports."""

    if name == "SpathiConfig":
        from spathi.config import SpathiConfig

        globals()[name] = SpathiConfig
        return SpathiConfig
    if name in {"SpathiRunResult", "infer_group_specific_grns"}:
        from spathi.pipeline import SpathiRunResult, infer_group_specific_grns

        globals()["SpathiRunResult"] = SpathiRunResult
        globals()["infer_group_specific_grns"] = infer_group_specific_grns
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""

    return sorted(set(globals()).union(__all__))
