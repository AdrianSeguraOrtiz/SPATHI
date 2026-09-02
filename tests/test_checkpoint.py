import sqlite3
from pathlib import Path

import pytest

import spathi.checkpoint as checkpoint_module
from spathi.checkpoint import ModelCheckpoint, build_checkpoint_identity
from spathi.inference import EdgeRecord, ModelResult, ModelStat


def identity(*, seed: int = 123) -> dict[str, object]:
    return build_checkpoint_identity(
        input_fingerprints={
            "expression": {"path": "/elsewhere/expression.tsv", "size_bytes": 10, "sha256": "a"},
            "tf_list": {"path": "/elsewhere/tf.txt", "size_bytes": 2, "sha256": "b"},
            "groups": {"path": "/elsewhere/groups.tsv", "size_bytes": 5, "sha256": "c"},
        },
        scientific_parameters={"random_seed": seed, "tree_method": "extra-trees"},
        target_names=("G",),
        group_names=("A",),
        dependency_versions={"spathi": "test"},
    )


def test_scientific_fingerprint_excludes_the_report_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    renderer_was_read = False

    def changed_renderer(path: Path) -> bytes:
        nonlocal renderer_was_read
        content = original_read_bytes(path)
        if path.name == "_report.py":
            renderer_was_read = True
            return content + b"simulated report-only change"
        return content

    expected = checkpoint_module.scientific_implementation_fingerprint()
    monkeypatch.setattr(Path, "read_bytes", changed_renderer)

    assert checkpoint_module.scientific_implementation_fingerprint() == expected
    assert renderer_was_read is False


def test_checkpoint_codec_changes_invalidate_the_scientific_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    expected = identity()
    codec_was_read = False

    def changed_codec(path: Path) -> bytes:
        nonlocal codec_was_read
        content = original_read_bytes(path)
        if path.name == "checkpoint.py":
            codec_was_read = True
            return content + b"simulated checkpoint-codec change"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_codec)
    observed = identity()

    assert codec_was_read is True
    assert (
        observed["scientific_implementation_sha256"] != expected["scientific_implementation_sha256"]
    )


def model_result() -> ModelResult:
    return ModelResult(
        edges=(
            EdgeRecord(
                source="TF",
                target="G",
                score=1.0,
                sign="?",
                evidence="test",
                context="group:A",
            ),
        ),
        skipped=None,
        stat=ModelStat(
            target_group="A",
            target="G",
            status="trained",
            random_seed=1,
            n_samples=3,
            n_positive_weight_samples=3,
            weight_sum=3.0,
            n_predictors_input=1,
            n_predictors_used=1,
            discarded_predictors=(),
            constant_predictors=(),
            n_edges=1,
            importance_sum=1.0,
            fit_seconds=0.1,
        ),
        trained=True,
    )


def test_checkpoint_round_trips_one_committed_model(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    expected = model_result()

    with ModelCheckpoint(directory, identity=identity(), resume=False) as checkpoint:
        checkpoint.validate_or_record_weights("A", [1.0, 1.0, 1.0])
        checkpoint.record_result(expected)
        assert checkpoint.completed_keys == frozenset({("A", "G")})
        with pytest.raises(RuntimeError, match="already contains model"):
            checkpoint.record_result(expected)

    with ModelCheckpoint(directory, identity=identity(), resume=True) as resumed:
        resumed.validate_or_record_weights("A", [1.0, 1.0, 1.0])
        assert tuple(resumed.iter_results()) == (expected,)
        resumed.validate_complete()


def test_checkpoint_rejects_changed_identity(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    with ModelCheckpoint(directory, identity=identity(seed=1), resume=False):
        pass

    with pytest.raises(RuntimeError, match="identity does not match"):
        ModelCheckpoint(directory, identity=identity(seed=2), resume=True)


def test_checkpoint_rejects_concurrent_resume(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    with ModelCheckpoint(directory, identity=identity(), resume=False):
        with pytest.raises(RuntimeError, match="already in use"):
            ModelCheckpoint(directory, identity=identity(), resume=True)


def test_resume_rejects_existing_database_without_identity(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    sqlite3.connect(directory / "checkpoint.sqlite3").close()

    with pytest.raises(RuntimeError, match="has no run identity"):
        ModelCheckpoint(directory, identity=identity(), resume=True)


def test_checkpoint_identity_ignores_input_location_but_not_content() -> None:
    first = identity()
    second = build_checkpoint_identity(
        input_fingerprints={
            "expression": {"path": "/moved/expression.tsv", "size_bytes": 10, "sha256": "a"},
            "tf_list": {"path": "/moved/tf.txt", "size_bytes": 2, "sha256": "b"},
            "groups": {"path": "/moved/groups.tsv", "size_bytes": 5, "sha256": "c"},
        },
        scientific_parameters={"random_seed": 123, "tree_method": "extra-trees"},
        target_names=("G",),
        group_names=("A",),
        dependency_versions={"spathi": "test"},
    )
    assert first == second


def test_new_checkpoint_rejects_nonempty_directory_and_symlink(tmp_path: Path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "unrelated.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be empty"):
        ModelCheckpoint(nonempty, identity=identity(), resume=False)
    assert (nonempty / "unrelated.txt").read_text(encoding="utf-8") == "keep me"

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link"):
        ModelCheckpoint(linked, identity=identity(), resume=False)


def test_resume_requires_an_existing_checkpoint_database(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()

    with pytest.raises(RuntimeError, match="resume requires an existing checkpoint.sqlite3"):
        ModelCheckpoint(directory, identity=identity(), resume=True)

    assert list(directory.iterdir()) == []


def test_resume_rejects_and_preserves_unowned_prefixed_files(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    with ModelCheckpoint(directory, identity=identity(), resume=False):
        pass
    unowned = directory / "run-lock.sqlite3.user-data"
    unowned.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned files"):
        ModelCheckpoint(directory, identity=identity(), resume=True)

    assert unowned.read_text(encoding="utf-8") == "keep me"


def test_resume_rejects_owned_named_non_files(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    with ModelCheckpoint(directory, identity=identity(), resume=False):
        pass
    disguised_directory = directory / "checkpoint.sqlite3-wal"
    disguised_directory.mkdir()
    marker = disguised_directory / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned files"):
        ModelCheckpoint(directory, identity=identity(), resume=True)

    assert marker.read_text(encoding="utf-8") == "keep me"


def test_checkpoint_rejects_changed_recalculated_weights(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    with ModelCheckpoint(directory, identity=identity(), resume=False) as checkpoint:
        checkpoint.validate_or_record_weights("A", [1.0, 2.0, 3.0])

    with ModelCheckpoint(directory, identity=identity(), resume=True) as resumed:
        with pytest.raises(RuntimeError, match="weights do not match"):
            resumed.validate_or_record_weights("A", [1.0, 2.0, 4.0])


def test_checkpoint_refuses_incomplete_or_corrupted_results(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    with ModelCheckpoint(directory, identity=identity(), resume=False) as checkpoint:
        checkpoint.validate_or_record_weights("A", [1.0, 1.0, 1.0])
        with pytest.raises(RuntimeError, match="every expected model"):
            checkpoint.validate_complete()
        checkpoint.record_result(model_result())

    connection = sqlite3.connect(directory / "checkpoint.sqlite3")
    with connection:
        connection.execute("UPDATE model_results SET payload_sha256 = 'corrupt'")
    connection.close()

    with ModelCheckpoint(directory, identity=identity(), resume=True) as resumed:
        with pytest.raises(RuntimeError, match="checksum failed"):
            tuple(resumed.iter_results())
