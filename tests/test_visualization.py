from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import spathi.visualization as visualization_module
from spathi.diagnostics import compute_weight_diagnostics
from spathi.representation import RepresentationResult
from spathi.visualization import (
    prepare_visualization_embedding,
    safe_group_filename,
    write_effective_mass_heatmap,
    write_target_weight_panel,
    write_visualization_manifest,
)
from spathi.weighting import WeightResult

CELLS = ("c1", "c2", "c3", "c4")
GROUPS = ("A", "A", "B", "B")


def pca_representation(*, one_component: bool = False) -> RepresentationResult:
    if one_component:
        values = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        names = ("PC1",)
        explained = (1.0,)
    else:
        values = np.array([[-2.0, 0.5], [-1.0, -0.5], [1.0, -0.4], [2.0, 0.4]])
        names = ("PC1", "PC2")
        explained = (0.8, 0.2)
    return RepresentationResult(
        values=values,
        cell_ids=CELLS,
        dimension_names=names,
        distance_space="pca",
        standardization="none",
        requested_n_components=len(names),
        effective_n_components=len(names),
        maximum_informative_n_components=len(names),
        pca_svd_solver="full",
        pca_svd_solver_resolution="explicit",
        explained_variance_ratio=explained,
    )


def centroids_for(representation: RepresentationResult) -> pd.DataFrame:
    return pd.DataFrame(
        np.vstack(
            [
                representation.values[:2].mean(axis=0),
                representation.values[2:].mean(axis=0),
            ]
        ),
        index=pd.Index(["A", "B"], name="group"),
        columns=representation.dimension_names,
    )


def weights_for(target: str) -> WeightResult:
    if target == "A":
        distance = np.array([0.5, 0.5, 2.5, 3.5])
        base = np.array([1.0, 1.0, 0.4, 0.2])
        final = base.copy()
    else:
        distance = np.array([3.5, 2.5, 0.5, 0.5])
        base = np.array([0.3, 0.2, 1.0, 1.0])
        final = base.copy()
    return WeightResult(
        target_group=target,
        cells=CELLS,
        cell_groups=GROUPS,
        distance=distance,
        base_weight=base,
        group_size_factor=np.ones(4),
        final_weight=final,
        mode="cell-distance-group-anchored",
    )


def prepared_embedding(*, one_component: bool = False):
    representation = pca_representation(one_component=one_component)
    groups = pd.Series(GROUPS, index=CELLS)
    return prepare_visualization_embedding(
        representation,
        groups,
        centroids_for(representation),
    )


def test_group_filenames_are_safe_and_do_not_collide_after_sanitization() -> None:
    filenames = {
        safe_group_filename("A/B"),
        safe_group_filename("A B"),
        safe_group_filename("á b"),
    }
    assert len(filenames) == 3
    assert all("/" not in filename and "\\" not in filename for filename in filenames)
    assert all(filename.endswith(".png") for filename in filenames)


def test_one_component_distance_pca_uses_an_explicit_zero_vertical_axis() -> None:
    embedding = prepared_embedding(one_component=True)
    np.testing.assert_array_equal(embedding.coordinates[:, 1], 0.0)
    np.testing.assert_array_equal(embedding.centroid_coordinates[:, 1], 0.0)
    assert embedding.projection_kind == "distance-pca"
    assert embedding.y_label == "No second component (fixed at 0)"
    assert "100.0% variance" in embedding.x_label


def test_expression_space_uses_a_clearly_labelled_auxiliary_projection() -> None:
    values = np.array(
        [
            [0.0, 0.5, 2.0],
            [0.2, 0.4, 1.7],
            [2.0, 1.0, 0.0],
            [1.7, 1.2, 0.2],
        ]
    )
    representation = RepresentationResult(
        values=values,
        cell_ids=CELLS,
        dimension_names=("G1", "G2", "G3"),
        distance_space="expression",
        standardization="none",
        requested_n_components=50,
        effective_n_components=None,
        maximum_informative_n_components=None,
        pca_svd_solver="auto",
        pca_svd_solver_resolution=None,
    )
    centroids = pd.DataFrame(
        [values[:2].mean(axis=0), values[2:].mean(axis=0)],
        index=["A", "B"],
        columns=representation.dimension_names,
    )
    embedding = prepare_visualization_embedding(
        representation,
        pd.Series(GROUPS, index=CELLS),
        centroids,
    )
    assert embedding.projection_kind == "auxiliary-pca"
    assert embedding.distance_space == "expression"
    assert embedding.x_label.startswith("Auxiliary PC1")
    assert any("visualization only" not in note and "auxiliary" in note for note in embedding.notes)


def test_target_panel_is_byte_deterministic_and_closes_figures(tmp_path: Path) -> None:
    embedding = prepared_embedding()
    first = write_target_weight_panel(
        tmp_path / "first",
        weights_for("A"),
        embedding,
        kernel="gaussian",
        bandwidth=1.25,
    )
    second = write_target_weight_panel(
        tmp_path / "second",
        weights_for("A"),
        embedding,
        kernel="gaussian",
        bandwidth=1.25,
    )
    assert first.relative_path == second.relative_path
    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes > 0
    assert (tmp_path / "first" / first.relative_path).read_bytes() == (
        tmp_path / "second" / second.relative_path
    ).read_bytes()
    assert plt.get_fignums() == []


def test_scatter_sampling_retains_a_rare_target_and_each_source_group() -> None:
    cells = tuple(f"cell-{index}" for index in range(100))
    groups = ("rare", *("common-a" for _ in range(49)), *("common-b" for _ in range(50)))

    selected = visualization_module._sample_indices(
        visualization_module._stable_cell_hashes(cells),
        12,
        group_ids=("rare", "common-a", "common-b"),
        group_positions=tuple(
            np.flatnonzero(np.asarray(groups) == group)
            for group in ("rare", "common-a", "common-b")
        ),
        priority_group="rare",
    )

    assert selected.size == 12
    selected_groups = {groups[index] for index in selected}
    assert selected_groups == {"rare", "common-a", "common-b"}
    assert 0 in selected
    np.testing.assert_array_equal(
        selected,
        visualization_module._sample_indices(
            visualization_module._stable_cell_hashes(cells),
            12,
            group_ids=("rare", "common-a", "common-b"),
            group_positions=tuple(
                np.flatnonzero(np.asarray(groups) == group)
                for group in ("rare", "common-a", "common-b")
            ),
            priority_group="rare",
        ),
    )


def test_manifest_rejects_tampered_files_and_paths_outside_visualizations(
    tmp_path: Path,
) -> None:
    embedding = prepared_embedding()
    panel = write_target_weight_panel(tmp_path, weights_for("A"), embedding)
    panel_path = tmp_path / panel.relative_path
    payload = bytearray(panel_path.read_bytes())
    payload[-1] ^= 1
    panel_path.write_bytes(payload)

    with pytest.raises(ValueError, match="changed after"):
        write_visualization_manifest(tmp_path, embedding, [panel])

    outside = replace(panel, relative_path="../outside.png")
    with pytest.raises(ValueError, match="inside the visualizations"):
        write_visualization_manifest(tmp_path, embedding, [outside])

    absolute = replace(panel, relative_path=str(panel_path.resolve()))
    with pytest.raises(ValueError, match="must be relative"):
        write_visualization_manifest(tmp_path, embedding, [absolute])


def test_mass_heatmap_and_manifest_are_deterministic_and_portable(tmp_path: Path) -> None:
    embedding = prepared_embedding()
    diagnostics = [
        compute_weight_diagnostics(weights_for(target), emit_warnings=False)
        for target in ("A", "B")
    ]

    manifests = []
    for name in ("first", "second"):
        run_dir = tmp_path / name
        panel = write_target_weight_panel(run_dir, weights_for("A"), embedding)
        heatmap = write_effective_mass_heatmap(
            run_dir,
            diagnostics,
            group_order=embedding.group_ids,
        )
        result = write_visualization_manifest(run_dir, embedding, [panel, heatmap])
        manifests.append(result)
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0"
        assert payload["projection"]["kind"] == "distance-pca"
        assert {figure["kind"] for figure in payload["figures"]} == {
            "effective-mass-heatmap",
            "target-weight-panel",
        }
        assert all(not Path(figure["relative_path"]).is_absolute() for figure in payload["figures"])

    assert manifests[0].manifest_path.read_bytes() == manifests[1].manifest_path.read_bytes()
    assert manifests[0].figures[0].sha256 == manifests[1].figures[0].sha256
    assert plt.get_fignums() == []


def test_target_panel_rejects_weights_with_inconsistent_cell_groups(tmp_path: Path) -> None:
    frame = weights_for("A").to_frame()
    frame.loc[frame["cell"] == "c1", "cell_group"] = "B"
    with pytest.raises(ValueError, match="assignments disagree"):
        write_target_weight_panel(tmp_path, frame, prepared_embedding())
