import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from spathi.outputs import (
    IncrementalRunWriter,
    create_output_directory,
    write_json,
    write_tsv_gzip,
)


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    path = create_output_directory(tmp_path / "run")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        create_output_directory(path)


def test_incremental_writer_produces_exact_network_schema_and_order(tmp_path: Path) -> None:
    output_dir = create_output_directory(tmp_path / "run")
    edges = [
        {
            "source": "TF2",
            "target": "G2",
            "score": 0.2,
            "sign": "?",
            "evidence": "weighted_extra_trees_feature_importance",
            "context": "group:A",
        },
        {
            "source": "TF1",
            "target": "G1",
            "score": 0.8,
            "sign": "?",
            "evidence": "weighted_extra_trees_feature_importance",
            "context": "group:A",
        },
        {
            "source": "TF2",
            "target": "G1",
            "score": 0.0,
            "sign": "?",
            "evidence": "weighted_extra_trees_feature_importance",
            "context": "group:A",
        },
    ]
    weights = [
        {
            "target_group": "A",
            "cell": "cell_1",
            "cell_group": "A",
            "distance": 0.0,
            "base_weight": 1.0,
            "group_size_factor": 1.0,
            "final_weight": 1.0,
        }
    ]
    with IncrementalRunWriter(output_dir) as writer:
        assert writer.write_edges(edges) == 2
        writer.write_cell_weights(weights)
        writer.write_skipped(
            [{"target_group": "A", "target": "CONST", "reason": "constant_target"}]
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
    assert network["sign"].tolist() == ["?", "?"]
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


def test_gzip_outputs_are_byte_reproducible_and_have_zero_mtime(tmp_path: Path) -> None:
    payload = {
        "target_group": "A",
        "cell": "cell_1",
        "cell_group": "A",
        "distance": 0.0,
        "base_weight": 1.0,
        "group_size_factor": 1.0,
        "final_weight": 1.0,
    }
    outputs: list[Path] = []
    for name in ("first", "second"):
        output_dir = create_output_directory(tmp_path / name)
        with IncrementalRunWriter(output_dir) as writer:
            writer.write_cell_weights([payload])
        outputs.append(output_dir)

    for filename in ("cell_weights.tsv.gz", "model_diagnostics.tsv.gz"):
        first = (outputs[0] / filename).read_bytes()
        second = (outputs[1] / filename).read_bytes()
        assert first == second
        assert first[4:8] == b"\x00\x00\x00\x00"


def test_already_ordered_edge_sequence_is_streamed_without_sorting(tmp_path: Path) -> None:
    output_dir = create_output_directory(tmp_path / "ordered")
    edges = (
        {
            "source": "TF1",
            "target": "G1",
            "score": 0.7,
            "evidence": "test",
            "context": "group:A",
        },
        {
            "source": "TF2",
            "target": "G1",
            "score": 0.3,
            "evidence": "test",
            "context": "group:A",
        },
    )

    with IncrementalRunWriter(output_dir) as writer:
        assert writer.write_edges(edges) == 2

    network = pd.read_csv(output_dir / "network.csv")
    assert network["source"].tolist() == ["TF1", "TF2"]
    assert network["sign"].tolist() == ["?", "?"]
