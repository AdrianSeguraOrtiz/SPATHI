from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spathi._report as report_module
from spathi._report import InteractiveReportBuilder, prepare_report_embedding
from spathi.diagnostics import compute_weight_diagnostics
from spathi.representation import RepresentationResult
from spathi.weighting import WeightResult, prepare_weighting_context

CELLS = ("c1", "c2", "c3", "c4")
GROUPS = ("A", "A", "B", "B")
CONTEXT = prepare_weighting_context(GROUPS, cell_ids=CELLS)


def pca_representation(
    *, one_component: bool = False, cells: tuple[str, ...] = CELLS
) -> RepresentationResult:
    if one_component:
        values = np.linspace(-2.0, 2.0, len(cells))[:, None]
        names = ("PC1",)
        explained = (1.0,)
    else:
        values = np.column_stack(
            (
                np.linspace(-2.0, 2.0, len(cells)),
                np.resize(np.array([0.5, -0.5]), len(cells)),
            )
        )
        names = ("PC1", "PC2")
        explained = (0.8, 0.2)
    return RepresentationResult(
        values=values,
        cell_ids=cells,
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


def centroids_for(
    representation: RepresentationResult,
    groups: tuple[str, ...] = GROUPS,
) -> pd.DataFrame:
    group_ids = tuple(sorted(set(groups)))
    values = np.asarray(representation.values)
    return pd.DataFrame(
        np.vstack([values[np.asarray(groups) == group].mean(axis=0) for group in group_ids]),
        index=pd.Index(group_ids, name="group"),
        columns=representation.dimension_names,
    )


def embedding_for(
    *, one_component: bool = False, cells: tuple[str, ...] = CELLS, groups: tuple[str, ...] = GROUPS
):
    representation = pca_representation(one_component=one_component, cells=cells)
    return prepare_report_embedding(
        representation,
        pd.Series(groups, index=cells),
        centroids_for(representation, groups),
    )


def weights_for(target: str) -> WeightResult:
    if target == "A":
        distance = np.array([0.5, 0.5, 2.5, 3.5])
        base = np.array([1.0, 1.0, 0.4, 0.2])
        factor = np.array([1.0, 1.0, 0.8, 0.8])
    else:
        distance = np.array([3.5, 2.5, 0.5, 0.5])
        base = np.array([0.3, 0.2, 1.0, 1.0])
        factor = np.array([0.7, 0.7, 1.0, 1.0])
    final = base * factor
    return WeightResult(
        context=CONTEXT,
        target_group=target,
        distance=distance,
        base_weight=base,
        group_size_factor=factor,
        final_weight=final,
        mode="cell-distance-group-anchored",
    )


def complete_builder(**kwargs: object) -> InteractiveReportBuilder:
    builder = InteractiveReportBuilder(embedding_for(), **kwargs)
    for target in ("A", "B"):
        weights = weights_for(target)
        builder.add_target(
            weights,
            compute_weight_diagnostics(weights, emit_warnings=False),
        )
    return builder


def extract_payload(document: str) -> dict[str, object]:
    match = re.search(
        r'<script id="spathi-report-data" type="application/json">(.*?)</script>',
        document,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def decode_array(spec: dict[str, object]) -> np.ndarray:
    dtype = np.dtype(str(spec["dtype"])).newbyteorder("<")
    shape = tuple(int(value) for value in spec["shape"])  # type: ignore[union-attr]
    return np.frombuffer(base64.b64decode(str(spec["data"])), dtype=dtype).reshape(shape)


def fake_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report_module,
        "_plotly_javascript",
        lambda: "window.Plotly={newPlot(){},react(){},Plots:{resize(){}}};",
    )


def test_one_component_distance_pca_uses_an_explicit_zero_vertical_axis() -> None:
    embedding = embedding_for(one_component=True)

    np.testing.assert_array_equal(embedding.coordinates[:, 1], 0.0)
    np.testing.assert_array_equal(embedding.centroid_coordinates[:, 1], 0.0)
    assert embedding.projection_kind == "distance-pca"
    assert embedding.y_label == "No second component (fixed at 0)"
    assert "100.0% variance" in embedding.x_label


def test_expression_space_uses_a_clearly_labelled_auxiliary_projection() -> None:
    values = np.array([[0.0, 0.5, 2.0], [0.2, 0.4, 1.7], [2.0, 1.0, 0.0], [1.7, 1.2, 0.2]])
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

    embedding = prepare_report_embedding(
        representation,
        pd.Series(GROUPS, index=CELLS),
        centroids_for(representation),
    )

    assert embedding.projection_kind == "auxiliary-pca"
    assert embedding.distance_space == "expression"
    assert embedding.x_label.startswith("Auxiliary PC1")
    assert any("auxiliary" in note.lower() for note in embedding.notes)


def test_global_sample_is_deterministic_proportional_and_keeps_rare_groups() -> None:
    cells = tuple(f"cell-{index}" for index in range(100))
    groups = ("rare", *("common-a" for _ in range(49)), *("common-b" for _ in range(50)))
    embedding = embedding_for(cells=cells, groups=groups)

    first = report_module._deterministic_stratified_sample(
        embedding,
        max_display_cells=12,
        max_target_cell_values=100,
    )
    second = report_module._deterministic_stratified_sample(
        embedding,
        max_display_cells=12,
        max_target_cell_values=100,
    )

    np.testing.assert_array_equal(first, second)
    assert first.size == 12
    assert {groups[index] for index in first} == {"rare", "common-a", "common-b"}
    assert 0 in first


def test_stratified_sample_distributes_largest_remainders_across_groups() -> None:
    cells = tuple(f"cell-{index}" for index in range(40))
    groups = tuple(group for group in ("A", "B", "C", "D") for _ in range(10))
    embedding = embedding_for(cells=cells, groups=groups)

    selected = report_module._deterministic_stratified_sample(
        embedding,
        max_display_cells=7,
        max_target_cell_values=100,
    )

    selected_groups = np.asarray(groups, dtype=object)[selected]
    assert [int(np.count_nonzero(selected_groups == group)) for group in ("A", "B", "C", "D")] == [
        2,
        2,
        2,
        1,
    ]


def test_report_sample_size_is_shared_with_memory_planning() -> None:
    assert (
        report_module.report_sample_size(
            100_000,
            20,
            max_display_cells=30_000,
            max_target_cell_values=300_000,
        )
        == 15_000
    )
    assert (
        report_module.report_sample_size(
            4,
            2,
            max_display_cells=30_000,
            max_target_cell_values=300_000,
        )
        == 4
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        report_module.report_sample_size(
            2,
            3,
            max_display_cells=30_000,
            max_target_cell_values=300_000,
        )


def test_builder_rejects_duplicate_or_missing_targets(tmp_path: Path) -> None:
    builder = InteractiveReportBuilder(embedding_for())
    weights = weights_for("A")
    diagnostics = compute_weight_diagnostics(weights, emit_warnings=False)
    builder.add_target(weights, diagnostics)

    with pytest.raises(ValueError, match="more than once"):
        builder.add_target(weights, diagnostics)
    with pytest.raises(RuntimeError, match="missing target groups"):
        builder.write(tmp_path / "report.html")


def test_report_is_one_offline_html_with_all_interactive_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_plotly(monkeypatch)
    report_path = tmp_path / "report.html"

    artifact = complete_builder(run_parameters={"kernel": "gaussian"}).write(
        report_path,
        run_summary={"cells": 4, "groups": 2},
    )

    document = report_path.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "<script src=" not in document.lower()
    assert "<link href=" not in document.lower()
    assert "Target explorer" in document
    assert "Overview" in document
    assert "Method &amp; provenance" in document
    assert 'role="tablist"' in document
    assert document.count('role="tabpanel"') == 3
    assert document.count('role="region"') == 9
    assert 'role="img"' not in document
    assert 'id="target-status"' in document
    assert "the input identifiers of sampled cells" in document
    assert "not a plot of the configured distance" in document
    assert "cell markers have no outline" in document
    for plot_id in (
        "target-pca",
        "distance-weight",
        "source-mass",
        "source-box",
        "overview-pca",
        "group-sizes",
        "mass-heatmap",
        "ess-overview",
        "pca-variance",
    ):
        assert f'id="{plot_id}"' in document
    assert artifact.path == report_path
    assert artifact.size_bytes == report_path.stat().st_size
    assert len(artifact.sha256) == 64
    assert artifact.to_metadata()["self_contained"] is True


def test_plotly_layout_uses_current_title_objects() -> None:
    """Keep plot, axis, and colour-bar labels visible with current Plotly.js."""

    javascript = report_module._REPORT_APP_JAVASCRIPT
    assert 'title:"' not in javascript
    assert 'title:{text:"Cell groups in the shared PCA view"}' in javascript
    assert "xaxis:{title:{text:DATA.projection.x_label}}" in javascript
    assert 'colorbar:{title:{text:"Final weight"}}' in javascript
    assert "colorscale:WEIGHT_COLORSCALE" in javascript
    assert 'const WEIGHT_COLORSCALE=[[0,"#deebf7"]' in javascript
    assert "Cividis" not in javascript
    assert "line:{color:groupColor(g),width:1.2}" not in javascript
    assert "symbol:SYMBOLS[g%SYMBOLS.length],line:{width:0}" in javascript
    assert "Auxiliary PCA explained variance (report only)" in javascript
    assert "groupLegend" in javascript
    assert 'event.key==="ArrowRight"' in javascript
    assert 'SYMBOLS=["circle","square"' in javascript
    assert 'button.dataset.tab==="overview-panel"' in javascript
    assert "populateText();renderTarget(0);renderOverview()" not in javascript
    assert "SVG_POINT_BUDGET" in javascript


def test_report_application_javascript_has_valid_syntax(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    application = tmp_path / "report-application.js"
    application.write_text(report_module._REPORT_APP_JAVASCRIPT, encoding="utf-8")

    completed = subprocess.run(
        [node, "--check", str(application)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_payload_preserves_exact_weights_and_full_data_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_plotly(monkeypatch)
    path = tmp_path / "report.html"
    complete_builder().write(path)
    payload = extract_payload(path.read_text(encoding="utf-8"))
    arrays = payload["arrays"]
    assert isinstance(arrays, dict)

    final_weights = decode_array(arrays["final_weight"])  # type: ignore[arg-type]
    mass_percent = decode_array(arrays["mass_percent"])  # type: ignore[arg-type]
    medians = decode_array(arrays["median"])  # type: ignore[arg-type]

    np.testing.assert_array_equal(final_weights[0], weights_for("A").final_weight)
    np.testing.assert_allclose(mass_percent.sum(axis=1), 100.0)
    np.testing.assert_allclose(
        medians[0],
        [
            np.median(weights_for("A").final_weight[:2]),
            np.median(weights_for("A").final_weight[2:]),
        ],
    )
    sample = payload["sample"]
    assert isinstance(sample, dict)
    assert sample["shared_across_targets"] is True
    assert sample["sampled_cells"] == 4
    assert sample["total_cells"] == 4


def test_untrusted_identifiers_cannot_close_the_embedded_data_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_plotly(monkeypatch)
    malicious = "</script><script>alert('x')</script>"
    cells = (malicious, "safe-a", "safe-b", "safe-c")
    groups = (malicious, malicious, "B", "B")
    representation = pca_representation(cells=cells)
    embedding = prepare_report_embedding(
        representation,
        pd.Series(groups, index=cells),
        centroids_for(representation, groups),
    )
    context = prepare_weighting_context(groups, cell_ids=cells)
    builder = InteractiveReportBuilder(embedding)
    for target in embedding.group_ids:
        values = np.ones(4, dtype=np.float64)
        weights = WeightResult(
            context=context,
            target_group=target,
            distance=np.zeros(4),
            base_weight=values,
            group_size_factor=values,
            final_weight=values,
            mode="group-distance",
        )
        builder.add_target(weights, compute_weight_diagnostics(weights, emit_warnings=False))

    path = tmp_path / "report.html"
    builder.write(path)
    document = path.read_text(encoding="utf-8")

    assert malicious not in document
    payload = extract_payload(document)
    assert malicious in payload["groups"]  # type: ignore[operator]
    assert html.escape(malicious, quote=True) in payload["safe_sample_cells"]  # type: ignore[operator]


def test_report_bytes_are_deterministic_and_existing_output_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_plotly(monkeypatch)
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    complete_builder().write(first, run_summary={"models": 8})
    complete_builder().write(second, run_summary={"models": 8})

    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        complete_builder().write(first)
    assert first.read_bytes() == second.read_bytes()


def test_builder_validates_group_sizes_and_weight_alignment() -> None:
    with pytest.raises(ValueError, match="do not match"):
        InteractiveReportBuilder(embedding_for(), group_sizes={"A": 1, "B": 3})

    builder = InteractiveReportBuilder(embedding_for())
    inconsistent_context = prepare_weighting_context(("B", "A", "B", "A"), cell_ids=CELLS)
    base = weights_for("A")
    inconsistent = WeightResult(
        context=inconsistent_context,
        target_group="A",
        distance=base.distance,
        base_weight=base.base_weight,
        group_size_factor=base.group_size_factor,
        final_weight=base.final_weight,
        mode=base.mode,
    )
    with pytest.raises(ValueError, match="cell-group assignments"):
        builder.add_target(
            inconsistent,
            compute_weight_diagnostics(inconsistent, emit_warnings=False),
        )
