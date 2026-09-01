# SPATHI

**Similarity-weighted Population-Aware Transcriptional Heterogeneity Inference**

SPATHI infers a gene-regulatory network for each cell group in a preprocessed
single-cell RNA-seq expression matrix. Instead of fitting each group in isolation, it
fits every target model with all cells and changes each observation's contribution
according to its transcriptomic proximity to the group of interest.

> [!IMPORTANT]
> SPATHI is an early-stage scientific tool. It produces hypotheses based on predictive
> feature importance. An inferred edge is not evidence of causality, and the MVP does
> not infer whether regulation is activating or repressing.

The project is independent of ANDREA. Its three input contracts and primary network
output are deliberately compatible with ANDREA so that a future container can install
a pinned SPATHI release without coupling the two codebases.

## Why population-aware inference?

Inferring a separate network from each cluster discards observations from related cell
populations and can make estimates unstable for small groups. SPATHI's working
hypothesis is that transcriptomically close groups may share part of their regulatory
programs. Nearby cells can therefore provide context while receiving less influence
than cells central to the target population.

No temporal order, lineage, hierarchy, or pseudotime is assumed. Similarity is used to
weight observations, not to claim a biological direction between groups.

For each target group, SPATHI:

1. builds a cell representation for distance calculations;
2. computes one centroid per group and reusable distances;
3. converts distances to base weights with a kernel;
4. optionally corrects external groups for multiplicity;
5. trains one weighted tree ensemble per target gene on the supplied expression;
6. writes positive transcription-factor importances as directed candidate edges; and
7. by default, writes diagnostic figures showing the assigned weights and effective
   contribution of each source group.

The PCA or expression representation in steps 1–3 is used **only to define weights**.
Models always use the expression values supplied by the user.

## Installation

SPATHI requires Python 3.11 or newer. For development from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The distribution name is provisionally `spathi`. After a release has actually been
published to PyPI, installation will be:

```bash
python -m pip install spathi==<version>
```

No package has been published as part of the MVP work.

## Input contracts

Inputs are intentionally strict: no CSV input, automatic transposition, optional
target list, or H5AD/AnnData input is supported.

### Expression matrix

`--expression` must point to a tab-separated file with a header, unique genes in the
first column, unique cells in the remaining column names, and finite numeric expression
values. Genes are rows and cells are columns:

```text
gene    cell_1    cell_2    cell_3
G1      0.12      0.20      0.08
G2      1.30      0.95      1.12
```

The fields shown aligned above are separated by tab characters in the actual file.
First-column names reserved by ANDREA for cell-oriented matrices are rejected, as are
joint expression/group identifiers that reveal an inverse cell-by-gene orientation.

The matrix must already be normalized and transformed appropriately for inference.
SPATHI does not silently perform a complete scRNA-seq normalization workflow. Every
gene in this matrix is used as a target.

### Transcription factors

`--tf-list` is plain text with exactly one expression-matrix gene identifier per line:

```text
SOX2
MYC
TP53
```

Blank lines, duplicates, missing genes, and an empty list are errors. When a target is
also a transcription factor, that TF is removed from its own predictor set so no
self-edge is emitted. For each target-group model, any other TF that is constant among
the cells with positive weight is also excluded before fitting. The exact exclusions
and number of predictors actually used are recorded in `model_diagnostics.tsv.gz`.

### Groups

`--groups` is a TSV with a header. Its first column contains cell identifiers and it
must contain a column named exactly `cluster`:

```text
column    cluster
cell_1    B_cells
cell_2    T_cells
cell_3    B_cells
```

The fields shown aligned above are separated by tab characters in the actual file.

Every expression cell must occur exactly once, and no extra cell is allowed. Cell IDs,
cluster values, and group membership must be non-empty and valid.

See the
[`examples/minimal`](https://github.com/AdrianSeguraOrtiz/SPATHI/tree/main/examples/minimal)
directory for a complete small dataset.

## Weighting modes

The `--weight-mode` option provides exactly three definitions:

- `cell-distance`: every base weight is the kernel of that cell's distance to the
  target centroid. Group labels do not affect the base weight; after optional size
  correction, final weights are rescaled to a maximum of one.
- `cell-distance-group-anchored`: target-group cells have base and final weight one;
  external cells receive individual distance-derived weights.
- `group-distance`: target-group cells have weight one; all cells in the same external
  group share the kernel affinity between that group's centroid and the target
  centroid.

The conservative default is `cell-distance-group-anchored`, which fixes the target
population as the unit-weight core while keeping external-cell weights individual.

SPATHI stores distance, base weight, group-size factor, and final model weight
separately. With `--group-size-correction cap-to-target`, an external cell from group
`h` receives the factor `min(1, n_target / n_h)`. The target group is never corrected.
The correction changes population contribution, not biological distance. Use
`--group-size-correction none` to disable it.

The default Gaussian kernel is available alongside an exponential kernel. Bandwidth
can be a positive number or `auto`; automatic selection uses the median of the relevant
strictly positive distances and records any fallback. See
[`docs/methodology.md`](https://github.com/AdrianSeguraOrtiz/SPATHI/blob/main/docs/methodology.md)
for definitions and equations.

Cosine distance requires every vector actually used by the selected mode to have a
non-zero norm: group centroids in all modes, and cell representations in the two
cell-distance modes. SPATHI rejects required zero vectors with their identifiers rather
than assigning an arbitrary similarity. Floating-point residues indistinguishable from
zero under the documented dot-product error bound are canonicalized before bandwidth
selection and weighting.

For PCA, the centered-data informative rank is capped at
`min(n_genes, max(0, n_cells - 1))`; the one-cell case retains one structural
component with diagnosed zero explained variance. Requested and effective component
counts, the informative rank bound, the SVD policy, and explained-variance ratios are
recorded in `run_metadata.json`. With `auto`, concrete solver negotiation remains
delegated to the recorded scikit-learn version rather than copied from private state.

## Command-line usage

Run the bundled example with a deliberately small ensemble:

```bash
spathi infer \
  --expression examples/minimal/expression.tsv \
  --tf-list examples/minimal/tf_list.txt \
  --groups examples/minimal/groups.tsv \
  --output-dir results/minimal-example \
  --weight-mode group-distance \
  --distance-space pca \
  --n-components 3 \
  --distance-metric euclidean \
  --kernel gaussian \
  --bandwidth auto \
  --group-size-correction cap-to-target \
  --tree-method extra-trees \
  --n-estimators 25 \
  --random-seed 123 \
  --threads 1
```

Diagnostic visualizations are generated by default. To run the same inference without
PNG generation, add `--no-visualize`:

```bash
spathi infer \
  --expression examples/minimal/expression.tsv \
  --tf-list examples/minimal/tf_list.txt \
  --groups examples/minimal/groups.tsv \
  --output-dir results/minimal-without-figures \
  --no-visualize
```

Use `--no-visualize` for calibration loops or runtime benchmarks so PNG rendering is
not included in the measured inference cost; numeric networks and weights are unchanged.

For a production-sized run, increase `--n-estimators` after assessing convergence and
resource use. Inspect all options with:

```bash
spathi infer --help
```

The output directory must not already exist; this prevents accidental replacement of a
previous run.

## Python API

The same pipeline is available without the CLI:

```python
from pathlib import Path

from spathi import SpathiConfig, infer_group_specific_grns

config = SpathiConfig(
    expression=Path("examples/minimal/expression.tsv"),
    tf_list=Path("examples/minimal/tf_list.txt"),
    groups=Path("examples/minimal/groups.tsv"),
    output_dir=Path("results/minimal-api"),
    weight_mode="group-distance",
    n_components=3,
    n_estimators=25,
    threads=1,
    visualize=True,
)
result = infer_group_specific_grns(config)
```

`SpathiConfig` is immutable and validates scalar configuration before the pipeline
reads data. Content-level validation remains part of input loading. Set
`visualize=False` to omit visualization artifacts when using the Python API.

## Outputs

Each run directory is self-contained. Its primary artifact is `network.csv` with these
columns in this exact order:

| Column | Meaning |
|---|---|
| `source` | transcription-factor predictor |
| `target` | modeled expression gene |
| `score` | non-negative weighted feature importance |
| `sign` | `?` in the MVP |
| `evidence` | `weighted_extra_trees_feature_importance` or `weighted_random_forest_feature_importance` |
| `context` | `group:<group_id>` |

Only scores strictly greater than zero are retained. Scores keep the estimator's
magnitude and are relative within a target model; they are not normalized across
targets or groups. Rows are deterministically ordered by context, target, and source.

Additional artifacts are:

| File | Purpose |
|---|---|
| `cell_weights.tsv.gz` | authoritative long-form distances, base weights, size factors, and final model weights |
| `group_distances.tsv` | pairwise centroid distances |
| `group_affinities.tsv` | group-level centroid affinities and per-cell size-correction factors |
| `centroids.tsv` | one reusable prototype per group in distance space |
| `weight_diagnostics.tsv` | authoritative effective weight mass, ranges, sample size, and source-group contributions |
| `skipped_targets.tsv` | constant or otherwise non-trainable target models and reasons |
| `model_diagnostics.tsv.gz` | per-model seeds, predictor exclusions, fit status, and timing |
| `cell_embedding.tsv.gz` | cells, groups, and display coordinates: retained PCs for PCA runs, or auxiliary PCs for visualized expression-space runs |
| `pca_explained_variance.tsv` | per-PC and cumulative ratios for that fitted or auxiliary PCA; absent only for expression-space runs with `--no-visualize` |
| `visualizations/targets/*.png` | one weight-assignment panel per target group; omitted with `--no-visualize` |
| `visualizations/effective-weight-mass.png` | exact final-weight mass by target and source group; omitted with `--no-visualize` |
| `visualizations/manifest.json` | projection semantics, figure inventory, hashes, sizes, and counts; omitted with `--no-visualize` |
| `parameters.json` | requested configuration |
| `run_metadata.json` | effective settings, versions, timing, resources, and warnings |

`group_affinities.tsv` is a centroid-level diagnostic. Its `base_affinity` is an
actual per-cell base model weight only in `group-distance` mode; it must not be read as
the effective contribution of a group in either cell-distance mode. The exact weights
passed to the models are `cell_weights.tsv.gz:final_weight`, and their exact aggregate
contributions are in `weight_diagnostics.tsv`.

Each target panel combines distance-to-weight curves, a two-dimensional cell view,
source-group weight distributions, and the effective sample size. Figure colours
always encode the exact `final_weight` used by inference; weights are never recomputed
in two dimensions.
PC1/PC2 are only a projection of the retained PCA space, so separation or overlap in
the panel can omit distance carried by later components. For expression-space runs,
the plotting layer computes a deterministic auxiliary two-component PCA for display
only; inference distances and weights remain in expression space. The global heatmap
uses the exact `source_mass_percent` values from `weight_diagnostics.tsv`.

Run metadata also stores the resolved path, byte size, and SHA-256 digest of every
input so the exact source bytes can be verified later. The expression parser validates
shape and identifiers in a first pass, fills one exact `float64` array row by row in a
second pass, and requires both passes to have the same digest; it never retains a
second table-sized matrix of expression strings.

Gzip tables are written with a fixed header timestamp. Tables whose rows contain only
deterministic values, such as `cell_weights.tsv.gz`, are therefore reproducible byte
for byte for the same inputs and effective settings. `model_diagnostics.tsv.gz`
contains measured fit durations, so its content is intentionally not byte-stable even
though its gzip container has no wall-clock timestamp.

## Models, parallelism, and reproducibility

SPATHI uses scikit-learn's `ExtraTreesRegressor` (default) or
`RandomForestRegressor`, passing the target group's final weight vector through
`sample_weight`. The resulting weighted impurity-based `feature_importances_` values
are the edge scores. Before each fit, SPATHI excludes the target itself and any
remaining TF predictor that is constant among positive-weight cells. Predictor names,
importances, and diagnostic counts retain the resulting filtered-column mapping.

Target responses remain `float64`, preserving valid expression differences smaller
than one `float32` unit. Only the reusable TF predictor matrix is converted once to
the `float32` representation used by scikit-learn's tree implementation. Distance,
weight, and `sample_weight` calculations also remain `float64`.

When neither `--bootstrap` nor `--no-bootstrap` is supplied, SPATHI uses the
estimator-appropriate default: bootstrap sampling is disabled for Extra-Trees and
enabled for Random Forest. An explicit CLI flag overrides that choice.
`parameters.json` retains the requested value (`null` means automatic), while
`run_metadata.json` records the effective boolean used for training.

`--threads` is the only public parallelism budget: `-1` uses all logical CPUs, `1` is
sequential, and a positive integer caps available workers. Independent
`(target group, target gene)` tasks are scheduled with Joblib. Ensembles run with one
worker when outer parallelism is active, while threadpoolctl limits numerical-library
thread pools to avoid nested oversubscription. Per-task seeds depend on the global
seed, group ID, and target ID—not scheduler completion order.

Target groups and their genes are processed in bounded group and target sub-batches
sized from the single thread budget. This permits parallel `(group, target)` tasks
without retaining all run edges or one result object per target in memory.
Cell-to-centroid calculations use chunks capped at 64 MiB; the required group-by-group
centroid matrix is materialized because it is itself a requested output. In the two
cell-distance modes, a large reusable cell-to-centroid matrix is written directly to a
temporary disk-backed memory map. In `group-distance` mode, cell-to-centroid distances
are not computed at all: only the required centroid-to-centroid matrix is calculated,
and each per-cell `distance` recorded for weighting is the applicable shared
centroid-to-centroid distance. Automatic bandwidth still uses the exact global
positive-distance median relevant to the selected mode.

Run artifacts are first written to a private sibling staging directory. The requested
path is published only after all scientific tables and metadata are complete, so an
early validation or weighting failure does not block a retry. If individual model fits
fail, the completed diagnostic run is published with `status=failed` before SPATHI
returns an error pointing to `model_diagnostics.tsv.gz`.

The TF predictor matrix is reused across models, as are representations, centroids,
the distances needed by the selected mode, bandwidth, and one weight vector per target
group. Timing and effective resource decisions are recorded in `run_metadata.json`.

Run the separate synthetic scaling benchmark with, for example:

```bash
python benchmarks/benchmark_scaling.py --threads 1 2 -1 --repeats 2
```

Benchmarks are intentionally excluded from ordinary tests and CI.

## Scientific interpretation and limitations

A high SPATHI score means that a TF was predictively important for one target under one
group-specific weighting. It does not establish direct binding, causal regulation, or
regulatory sign. Impurity importance can favor predictors with greater variation and
can distribute importance unpredictably among correlated TFs.

Results depend on input preprocessing, cell-group assignments, representation choices,
kernel bandwidth, and—most importantly—the biological assumption that transcriptomic
proximity is informative about regulatory similarity. Compare weighting modes and
inspect weight diagnostics, external mass, and effective sample size before biological
interpretation.

## Development and distribution checks

```bash
ruff check .
ruff format --check .
mypy src/spathi
pytest -m "not distribution"
python -m build
python -m twine check dist/*
```

The excluded `distribution` test performs its own wheel build. Run plain `pytest`
when that additional local packaging assertion is desired; CI's dedicated
distribution job is the authoritative build, clean-install, and CLI smoke check.

Run the configured mypy check under Python 3.11, as CI does for the minimum supported
version. Newer interpreters are covered by the runtime test matrix.

To validate a wheel in a clean local environment:

```bash
python -m venv /tmp/spathi-wheel-check
/tmp/spathi-wheel-check/bin/python -m pip install dist/*.whl
/tmp/spathi-wheel-check/bin/spathi --help
```

Release maintainers should follow
[`docs/releasing.md`](https://github.com/AdrianSeguraOrtiz/SPATHI/blob/main/docs/releasing.md).
Building locally does not publish anything and does not require PyPI credentials.

## License

SPATHI is distributed under the
[MIT License](https://github.com/AdrianSeguraOrtiz/SPATHI/blob/main/LICENSE).
