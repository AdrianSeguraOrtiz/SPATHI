from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spathi.diagnostics import compute_weight_diagnostics
from spathi.inference import EdgeRecord, ModelStat, SkippedTargetRecord
from spathi.outputs import (
    IncrementalRunWriter,
    write_json,
    write_tsv_gzip,
    write_tsv_gzip_records,
)
from spathi.weighting import WeightingContext, WeightResult


def _make_output_dir(tmp_path: Path, name: str) -> Path:
    output_dir = tmp_path / name
    output_dir.mkdir()
    return output_dir


def _edge(*, source: str, target: str, score: float = 1.0) -> EdgeRecord:
    return EdgeRecord(
        source=source,
        target=target,
        score=score,
        sign="?",
        evidence="weighted_extra_trees_feature_importance",
        context="group:A",
    )


def _model_stat(
    *,
    target: str,
    n_edges: int,
    discarded_predictors: tuple[str, ...] = (),
    constant_predictors: tuple[str, ...] = (),
) -> ModelStat:
    return ModelStat(
        target_group="A",
        target=target,
        status="trained",
        random_seed=7,
        n_samples=2,
        n_positive_weight_samples=2,
        weight_sum=1.5,
        n_predictors_input=2,
        n_predictors_used=2,
        discarded_predictors=discarded_predictors,
        constant_predictors=constant_predictors,
        n_edges=n_edges,
        importance_sum=1.0,
        fit_seconds=0.1,
    )


def _weights() -> WeightResult:
    return WeightResult(
        context=WeightingContext(
            cells=("cell_1", "cell_2"),
            cell_groups=("A", "B"),
            group_ids=("A", "B"),
            group_codes=np.array([0, 1]),
            group_counts=np.array([1, 1]),
        ),
        target_group="A",
        distance=np.array([0.0, 1.0]),
        base_weight=np.array([1.0, 0.5]),
        group_size_factor=np.ones(2),
        final_weight=np.array([1.0, 0.5]),
        mode="cell-distance",
    )


def test_incremental_writer_produces_exact_schemas_and_canonical_order(tmp_path: Path) -> None:
    output_dir = _make_output_dir(tmp_path, "run")
    weights = _weights()
    diagnostics = compute_weight_diagnostics(weights, emit_warnings=False)

    with IncrementalRunWriter(output_dir) as writer:
        assert (
            writer.write_edges((_edge(source="TF1", target="G1"), _edge(source="TF2", target="G2")))
            == 2
        )
        assert writer.write_weights(weights) == 2
        assert writer.write_weight_diagnostics(diagnostics) == 2
        assert writer.write_model_diagnostics((_model_stat(target="G1", n_edges=1),)) == 1
        assert (
            writer.write_skipped_targets(
                (
                    SkippedTargetRecord(
                        target_group="A",
                        target="G2",
                        reason="constant_target",
                    ),
                )
            )
            == 1
        )

    network = pd.read_csv(output_dir / "network.csv")
    assert network.columns.tolist() == [
        "source",
        "target",
        "score",
        "sign",
        "evidence",
        "context",
    ]
    assert network[["target", "source"]].values.tolist() == [
        ["G1", "TF1"],
        ["G2", "TF2"],
    ]
    with gzip.open(output_dir / "cell_weights.tsv.gz", "rt", encoding="utf-8") as handle:
        assert handle.readline().rstrip().split("\t") == [
            "target_group",
            "cell",
            "cell_group",
            "distance",
            "base_weight",
            "group_size_factor",
            "final_weight",
        ]
    weight_diagnostics = pd.read_csv(output_dir / "weight_diagnostics.tsv", sep="\t")
    assert weight_diagnostics["source_group"].tolist() == ["A", "B"]


def test_json_writer_handles_paths_and_has_terminal_newline(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_json({"path": Path("input.tsv"), "value": 2}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "path": "input.tsv",
        "value": 2,
    }
    assert path.read_bytes().endswith(b"\n")


def test_gzip_tsv_writer_preserves_requested_schema_and_is_reproducible(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "cell": ["c1", "c2"],
            "group": ["A", "B"],
            "PC1": [1.5, -1.5],
            "unused": [1, 2],
        }
    )
    first = tmp_path / "first" / "embedding.tsv.gz"
    second = tmp_path / "second" / "embedding.tsv.gz"
    first.parent.mkdir()
    second.parent.mkdir()

    write_tsv_gzip(frame, first, ("cell", "group", "PC1"))
    write_tsv_gzip(frame, second, ("cell", "group", "PC1"))

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        observed = pd.read_csv(handle, sep="\t")
    assert observed.columns.tolist() == ["cell", "group", "PC1"]
    pd.testing.assert_frame_equal(observed, frame[["cell", "group", "PC1"]])


def test_streaming_gzip_tsv_writer_is_reproducible(tmp_path: Path) -> None:
    records = (
        {"cell": "c1", "centroid_weight": 1.0},
        {"cell": "c2", "centroid_weight": 0.5},
    )
    first = tmp_path / "first" / "centroid_weights.tsv.gz"
    second = tmp_path / "second" / "centroid_weights.tsv.gz"
    first.parent.mkdir()
    second.parent.mkdir()

    assert write_tsv_gzip_records(iter(records), first, ("cell", "centroid_weight")) == 2
    assert write_tsv_gzip_records(iter(records), second, ("cell", "centroid_weight")) == 2

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert handle.read() == "cell\tcentroid_weight\nc1\t1.0\nc2\t0.5\n"


def test_gzip_outputs_are_byte_reproducible_and_have_zero_mtime(tmp_path: Path) -> None:
    outputs: list[Path] = []
    for name in ("first", "second"):
        output_dir = _make_output_dir(tmp_path, name)
        with IncrementalRunWriter(output_dir) as writer:
            writer.write_weights(_weights())
            writer.write_model_diagnostics((_model_stat(target="G", n_edges=1),))
        outputs.append(output_dir)

    for filename in ("cell_weights.tsv.gz", "model_diagnostics.tsv.gz"):
        first = (outputs[0] / filename).read_bytes()
        second = (outputs[1] / filename).read_bytes()
        assert first == second
        assert first[4:8] == b"\x00\x00\x00\x00"


def test_model_diagnostic_identifier_lists_are_unambiguous_json(tmp_path: Path) -> None:
    output_dir = _make_output_dir(tmp_path, "identifier-lists")
    stat = _model_stat(
        target="G",
        n_edges=0,
        discarded_predictors=("TF;A", "TF,B"),
        constant_predictors=("TF;A",),
    )

    with IncrementalRunWriter(output_dir) as writer:
        writer.write_model_diagnostics((stat,))

    diagnostics = pd.read_csv(output_dir / "model_diagnostics.tsv.gz", sep="\t")
    assert json.loads(diagnostics.loc[0, "discarded_predictors_json"]) == ["TF;A", "TF,B"]
    assert json.loads(diagnostics.loc[0, "constant_predictors_json"]) == ["TF;A"]


def test_edge_writer_consumes_canonical_records_once(tmp_path: Path) -> None:
    output_dir = _make_output_dir(tmp_path, "streamed")
    visits = 0

    def records():
        nonlocal visits
        for source, score in (("TF1", 0.7), ("TF2", 0.3)):
            visits += 1
            yield _edge(source=source, target="G", score=score)

    with IncrementalRunWriter(output_dir) as writer:
        assert writer.write_edges(records()) == 2

    assert visits == 2
    network = pd.read_csv(output_dir / "network.csv")
    assert network[["target", "source"]].values.tolist() == [["G", "TF1"], ["G", "TF2"]]


def test_edge_writer_rejects_noncanonical_or_invalid_records(tmp_path: Path) -> None:
    output_dir = _make_output_dir(tmp_path, "invalid")
    with IncrementalRunWriter(output_dir) as writer:
        with pytest.raises(ValueError, match="canonical global order"):
            writer.write_edges((_edge(source="TF2", target="G"), _edge(source="TF1", target="G")))

    output_dir = _make_output_dir(tmp_path, "nonpositive")
    with IncrementalRunWriter(output_dir) as writer:
        with pytest.raises(ValueError, match="positive and finite"):
            writer.write_edges((_edge(source="TF", target="G", score=0.0),))


def test_incremental_writer_never_reopens_or_truncates_existing_output(tmp_path: Path) -> None:
    output_dir = _make_output_dir(tmp_path, "reopened")
    writer = IncrementalRunWriter(output_dir)
    edge = _edge(source="TF1", target="G1")

    with writer:
        writer.write_edges((edge,))

    original = (output_dir / "network.csv").read_bytes()
    with pytest.raises(FileExistsError):
        with writer:
            pass

    assert (output_dir / "network.csv").read_bytes() == original
