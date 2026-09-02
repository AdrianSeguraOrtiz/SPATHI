import pytest

from spathi.progress import SpathiProgressEvent


def test_progress_event_exposes_fraction_and_json_fields() -> None:
    event = SpathiProgressEvent(
        phase="model_inference",
        message="Fitted model 3/8",
        completed_models=3,
        total_models=8,
        completed_groups=0,
        total_groups=2,
        current_group="A",
        resumed_models=1,
    )

    assert event.model_fraction == 3 / 8
    assert event.to_dict()["current_group"] == "A"


@pytest.mark.parametrize(
    "overrides",
    [
        {"completed_models": -1},
        {"completed_models": 2, "total_models": 1},
        {"completed_groups": 2, "total_groups": 1},
        {"completed_models": 1, "resumed_models": 2},
    ],
)
def test_progress_event_rejects_inconsistent_counters(overrides: dict[str, int]) -> None:
    values = {
        "phase": "model_inference",
        "message": "working",
        "completed_models": 0,
        "total_models": 4,
        "completed_groups": 0,
        "total_groups": 2,
        "resumed_models": 0,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        SpathiProgressEvent(**values)  # type: ignore[arg-type]


def test_progress_event_rejects_unknown_phase_and_non_text_group() -> None:
    with pytest.raises(ValueError, match="unsupported progress phase"):
        SpathiProgressEvent(phase="unknown", message="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="current_group"):
        SpathiProgressEvent(
            phase="preparing_group",
            message="bad",
            current_group=3,  # type: ignore[arg-type]
        )
