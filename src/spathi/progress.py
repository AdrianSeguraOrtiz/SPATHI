"""Structured, orchestration-friendly progress events for SPATHI runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

ProgressPhase: TypeAlias = Literal[
    "validating_inputs",
    "building_representation",
    "computing_distances",
    "preparing_inference",
    "preparing_group",
    "model_inference",
    "writing_outputs",
    "building_report",
    "publishing",
    "complete",
]
PROGRESS_PHASES: tuple[ProgressPhase, ...] = (
    "validating_inputs",
    "building_representation",
    "computing_distances",
    "preparing_inference",
    "preparing_group",
    "model_inference",
    "writing_outputs",
    "building_report",
    "publishing",
    "complete",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpathiProgressEvent:
    """One immutable progress observation emitted by the inference core.

    Model and group counters are global for the complete run.  ``total_models``
    is unknown only while inputs are still being validated.  Callbacks run
    synchronously on the orchestration thread; SPATHI never calls user code from
    a model-fitting worker. Callback exceptions before atomic publication
    deliberately propagate and abort the current attempt. An exception from the
    final ``complete`` notification is logged instead because the output already
    exists and cannot safely be rolled back. With checkpointing enabled, a
    model-inference event is emitted only after its transaction commits; without
    checkpointing it is emitted immediately after the model finishes.
    """

    phase: ProgressPhase
    message: str
    completed_models: int = 0
    total_models: int | None = None
    completed_groups: int = 0
    total_groups: int | None = None
    current_group: str | None = None
    resumed_models: int = 0

    def __post_init__(self) -> None:
        if self.phase not in PROGRESS_PHASES:
            raise ValueError(f"unsupported progress phase: {self.phase!r}")
        for field_name in ("completed_models", "completed_groups", "resumed_models"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for completed_name, total_name in (
            ("completed_models", "total_models"),
            ("completed_groups", "total_groups"),
        ):
            completed = getattr(self, completed_name)
            total = getattr(self, total_name)
            if total is not None:
                if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                    raise ValueError(f"{total_name} must be None or a non-negative integer")
                if completed > total:
                    raise ValueError(f"{completed_name} cannot exceed {total_name}")
        if self.resumed_models > self.completed_models:
            raise ValueError("resumed_models cannot exceed completed_models")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")
        if self.current_group is not None and not isinstance(self.current_group, str):
            raise TypeError("current_group must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for adapters and tests."""

        return asdict(self)


ProgressCallback: TypeAlias = Callable[[SpathiProgressEvent], None]


def emit_progress(callback: ProgressCallback | None, event: SpathiProgressEvent) -> None:
    """Invoke ``callback`` synchronously; callback exceptions are not suppressed."""

    if callback is not None:
        callback(event)


__all__ = [
    "ProgressCallback",
    "ProgressPhase",
    "PROGRESS_PHASES",
    "SpathiProgressEvent",
    "emit_progress",
]
