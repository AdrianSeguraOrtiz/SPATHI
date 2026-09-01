# SPATHI methodology

This document specifies the scientific semantics of the SPATHI MVP. It is also a
reference for interpreting diagnostics and testing future implementations.

## Scope and notation

Let the supplied expression matrix contain genes as rows and cells as columns. For
cell \(i\), let \(x_i\) be its expression vector and \(g_i\) its assigned group. Every
gene is modeled as a target. Candidate predictors are the genes listed in `tf_list`,
except that a target is removed from its own predictors when it is also a TF.

For each target group \(c\), SPATHI learns a separate network using every cell. It does
not split the expression matrix into isolated group-specific training sets. A final
weight \(w_i^{(c)}\) controls cell \(i\)'s contribution to the model for group \(c\).

No ordering, trajectory, lineage, or hierarchy among groups is assumed.

## Two deliberately separate spaces

SPATHI keeps the inference space separate from the distance space:

- **Inference space:** the original preprocessed expression values supplied by the
  user. TF columns are predictors and a target gene is the response.
- **Distance space:** either the cell-by-gene expression representation or a PCA
  representation. This space is used only to compute prototypes, distances, and
  weights.

`distance-standardization=standard` fits a standard scaling transform in the distance
pipeline; `none` leaves values unstandardized. This does not modify the values used by
the tree ensembles. SPATHI performs no implicit library-size normalization, log
transformation, scaling, or feature selection on the inference matrix.

When PCA is selected, scikit-learn PCA receives cells as observations. Its centered-data
informative-rank bound is `min(n_genes, max(0, n_cells - 1))`. The one-cell case keeps
one structural component so downstream shapes remain defined; its informative bound
and explained variance are both recorded as zero. Otherwise, the effective component
count is the smaller of the requested count and that bound. `run_metadata.json`
records the requested and effective counts, informative bound, SVD policy, and both
per-component and cumulative explained-variance ratios. When the policy is `auto`,
the concrete solver remains delegated to the recorded scikit-learn version; SPATHI
does not reproduce private solver-selection logic.

PCA distance-space runs write `cell_embedding.tsv.gz` with the first up to three
retained PCs and `pca_explained_variance.tsv` with the complete variance summary. A
visualized expression-space run writes the same artifact names for its clearly named
`AuxiliaryPC` display projection; these auxiliary coordinates do not participate in
distance or weight calculation. They are absent for expression-space runs with
`--no-visualize`.

### Numeric precision

Input validation and all distance, PCA, centroid, kernel, weight, response, and
diagnostic calculations use `float64`. Target responses are read from a contiguous
cells-by-genes `float64` matrix so biologically valid variation smaller than one
`float32` unit is not collapsed before constant-target detection or fitting. SPATHI
extracts only the reusable TF predictor columns into a contiguous `float32` matrix,
matching scikit-learn's tree-oriented numeric path without reducing target precision.
This conversion does not normalize or otherwise transform the supplied values.
Weights remain `float64` when passed as `sample_weight`.

## Group prototypes and distances

Let \(z_i\) denote cell \(i\)'s vector in the configured distance space. The MVP
prototype is the arithmetic centroid:

\[
\mu_c = \frac{1}{n_c}\sum_{i:g_i=c} z_i.
\]

Prototype calculation is isolated behind a dedicated component so that medoids or
other prototypes can be added without redefining weighting or inference. Centroids are
computed once per run.

For the selected Euclidean or cosine metric, SPATHI always calculates the pairwise
centroid distances \(d(\mu_h,\mu_c)\). The dense group-by-group result is materialized
once because it is both small relative to the expression matrix and a requested
output. In `cell-distance` and `cell-distance-group-anchored`, SPATHI additionally
calculates every \(d(z_i,\mu_c)\). These cell-to-centroid calculations are vectorized
through scikit-learn in chunks with a 64 MiB working-memory budget rather than
inheriting scikit-learn's much larger process-wide default. The resulting cell-by-group
matrix is reused; above the in-memory threshold, chunks are written directly into a
temporary disk-backed memory map.

`group-distance` never needs cell-to-centroid distances and therefore does not compute
them. It uses only the centroid-to-centroid matrix. The per-cell `distance` in
`cell_weights.tsv.gz` is consequently the centroid distance shared by that cell's
source group. Automatic-bandwidth calculations retain the exact positive-distance
median relevant to the selected mode; large cell-distance calculations use bounded
scans and a temporary disk-backed selection array.

Cosine distance is undefined for a zero vector. SPATHI therefore rejects zero-norm
centroids in every mode and zero-norm cell representations in the two modes that
actually calculate cell-to-centroid distances. `group-distance` does not inspect cell
norms because those vectors never enter its distance calculation. Required zero rows
are reported by identifier rather than silently assigned distance zero or one. Non-zero
rows at exceptional magnitudes are rescaled before the cosine calculation to avoid norm
underflow or overflow. Finally, non-negative cosine values no larger than the `float64`
forward-error bound for a dot product of the configured dimension are set to exactly
zero. This prevents numerical residue from becoming an artificial automatic bandwidth
while preserving genuinely positive distances above that bound.

## Kernels and global bandwidth

Distances are converted into affinities through a kernel registry. The Gaussian kernel
is

\[
K_{\mathrm{Gaussian}}(d;h)
= \exp\left(-\frac{d^2}{2h^2}\right),
\]

and the optional exponential kernel is

\[
K_{\mathrm{exponential}}(d;h)
= \exp\left(-\frac{d}{h}\right).
\]

The bandwidth \(h\) must be positive. A numeric `bandwidth` is used directly. With
`bandwidth=auto`, one reproducible global bandwidth is selected for the entire run:

- for `cell-distance` and `cell-distance-group-anchored`, \(h\) is the median of all
  strictly positive cell-to-centroid distances;
- for `group-distance`, \(h\) is the median of all strictly positive
  centroid-to-centroid distances.

Zeros are excluded because within-group or coincident prototypes may legitimately have
zero distance. If the relevant family contains no positive finite distance, SPATHI
uses a finite positive fallback, emits a warning, and records that decision. Kernel
outputs are validated to be finite and in \([0,1]\).

## Base weights, multiplicity correction, and final weights

SPATHI never conflates these quantities:

- `distance` is the distance selected by the weighting mode;
- `base_weight` is the direct kernel affinity, or the explicit target-group anchor;
- `group_size_factor` adjusts external population multiplicity;
- `final_weight` is passed to the estimator.

This separation matters because population-size correction changes total group
contribution without changing transcriptomic distance or base biological affinity.

### Cell distance

For `cell-distance`, group identity does not enter the base-weight definition:

\[
w_{i,\mathrm{base}}^{(c)} = K\!\left(d(z_i,\mu_c);h\right).
\]

Consequently, a cell outside \(c\) may receive a greater base weight than a peripheral
cell inside \(c\), and equally distant cells receive the same base weight regardless
of group. If size correction is disabled, the mode depends only on individual
distance.

After any size factor is applied, the complete vector is divided by its maximum so its
largest final weight is one. This rescaling is unique to `cell-distance`. A degenerate
all-zero or non-finite vector is diagnosed explicitly rather than passed silently to a
model.

### Group-anchored cell distance

For `cell-distance-group-anchored`, target-group membership fixes the base weight at
one, while external weights remain cell-specific:

\[
w_{i,\mathrm{base}}^{(c)} =
\begin{cases}
1, & g_i=c,\\
K\!\left(d(z_i,\mu_c);h\right), & g_i\ne c.
\end{cases}
\]

All target cells therefore retain final weight one. External cells from the same group
can have different weights, but none can exceed one. There is no post-correction
normalization that could reduce the target anchor.

### Group distance

For `group-distance`, each external population receives a shared base affinity:

\[
w_{i,\mathrm{base}}^{(c)} =
\begin{cases}
1, & g_i=c,\\
K\!\left(d(\mu_{g_i},\mu_c);h\right), & g_i\ne c.
\end{cases}
\]

Thus all cells from an external group have the same base weight. This mode is the most
directly interpretable at population level.

### Group-size correction

With `group-size-correction=none`, every size factor is one. With `cap-to-target`, an
external cell in group \(h\) receives

\[
f_h^{(c)} = \min\left(1,\frac{n_c}{n_h}\right),
\qquad
\widetilde{w}_i^{(c)} = w_{i,\mathrm{base}}^{(c)} f_h^{(c)}.
\]

The target group always has factor one. This caps the per-cell contribution of a larger
external group; it does not upweight smaller groups and does not modify distances. In
`cell-distance`, the final maximum-one normalization follows this multiplication. In
the two anchored modes, \(\widetilde{w}\) is already the final vector.

## Weight diagnostics

Diagnostics are computed once per target-group weight vector, not once per gene. They
include target-group size, total weight, target and per-external-group mass, target and
external mass percentages, minimum, maximum, mean, median, positive-weight count, and
effective sample size:

\[
\operatorname{ESS}(w) =
\frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
\]

The run warns about degenerate or non-finite weights, very low ESS, external mass that
exceeds target mass, and excessive concentration in one external group. These warnings
are interpretive signals. They do not stop a run unless the resulting model cannot be
trained.

`group_affinities.tsv` is deliberately a group-level centroid diagnostic. Its
`base_affinity` is the kernel of the source-to-target centroid distance, irrespective
of the selected weighting mode, and is a per-cell base model weight only in
`group-distance`. Its `group_size_factor` is the per-cell multiplicity correction for
that source/target pair. The authoritative model-level values are
`cell_weights.tsv.gz:final_weight`; their exact effective source-group mass is reported
in `weight_diagnostics.tsv`.

## Visual diagnostics

`spathi infer` generates visual diagnostics by default; `--no-visualize` (or
`visualize=False` in the Python API) disables the complete `visualizations/` tree.
When enabled, it writes:

- `visualizations/targets/*.png`, one combined panel per target group showing distance
  against base/final weight, a two-dimensional cell projection coloured by exact
  final model weight, final-weight distributions by source group, and the resulting
  effective sample size;
- `visualizations/effective-weight-mass.png`, a target-by-source heatmap built from
  the exact `source_mass_percent` values in the weight diagnostics; and
- `visualizations/manifest.json`, recording projection semantics, figure paths,
  SHA-256 digests, byte sizes, and relevant target/cell/group counts.

For PCA-distance runs, the two-dimensional view is PC1/PC2 from the retained fitted
representation; if only one component exists, the second display coordinate is zero.
PC1/PC2 is a projection, not the space in which weights are recomputed: distance in
later retained components can be invisible in the panel. Colours always use the exact
`final_weight` passed to inference. For expression-distance runs, SPATHI fits a
deterministic auxiliary two-component PCA solely for display; neither inference
distances nor weights change. Scatter plots may use a deterministic subset for very
large inputs. That subset reserves representation for rare source groups and always
includes the target group; distributions and aggregate mass continue to use all cells.
Use `--no-visualize` in calibration or timing runs to exclude rendering cost without
changing numeric networks or weights.

The heatmap quantifies statistical contribution under the chosen weighting rule. It
does not represent causal influence, lineage, or a regulatory edge between groups.

## Weighted network inference

For group \(c\) and target gene \(t\), SPATHI starts from all valid TFs, removes \(t\)
if it is among them, and excludes any remaining predictor that is constant among cells
with positive \(w_i^{(c)}\). This positive-weight restriction matches the observations
that can affect the fitted model. It then fits either scikit-learn's
`ExtraTreesRegressor` or `RandomForestRegressor` on all cells:

\[
\widehat{x_t} = F_c(X_{\mathrm{TF}}; w^{(c)}).
\]

The final vector \(w^{(c)}\) is supplied through the estimator's `sample_weight`
argument. SPATHI uses the fitted `feature_importances_`, based on weighted impurity
reduction, as scores for directed \(\mathrm{TF}\rightarrow t\) candidate edges.

Bootstrap sampling defaults are resolved per estimator: it is disabled for
Extra-Trees and enabled for Random Forest. The API value `bootstrap=None`, or omission
of both `--bootstrap` and `--no-bootstrap` in the CLI, selects that automatic policy;
an explicit boolean overrides it. `parameters.json` preserves the requested nullable
value and `run_metadata.json` records the effective boolean used by every estimator.

All genes are attempted as targets. Constant responses, empty or entirely constant
predictor sets, and other non-trainable models are recorded in `skipped_targets.tsv`.
`model_diagnostics.tsv.gz` distinguishes the self-excluded predictor, constant
predictors, the complete discarded set, and the number actually used, preserving the
mapping from fitted importance columns back to TF names. No self-edges are produced,
no automatic threshold is applied, and every finite score strictly greater than zero
is retained. The feature importances are relative within each target model and are not
renormalized across targets or groups.

Each output edge has `sign=?`. Tree impurity importance does not determine activation
or repression, and SPATHI does not invent a sign.

## Parallel execution and determinism

The main cost grows approximately as

\[
N_{\mathrm{groups}}\,N_{\mathrm{targets}}\,
\operatorname{cost}(\mathrm{ensemble}).
\]

`threads` is the only public resource budget. `-1` resolves to all available logical
CPUs, `1` is sequential, and any positive value is a cap. An internal automatic plan
uses Joblib either across independent `(target group, target gene)` tasks or inside a
single ensemble, never both at full width. When outer task parallelism is active, each
ensemble receives `n_jobs=1`; threadpoolctl constrains BLAS/OpenMP libraries to prevent
nested oversubscription. The threading backend shares the read-only expression array
instead of copying it into worker processes.

A task seed is derived stably from the global seed, group identifier, and target
identifier. It does not depend on scheduler order. Output is sorted by `context`,
`target`, and `source`, so changing `threads` should produce equal or numerically
equivalent results.

The pipeline chooses bounded target-group batches large enough to expose independent
`(group, target)` tasks to Joblib while keeping output progressive. When all genes fit
in one target batch, multiple groups are scheduled together; larger target sets retain
group-major sub-batches so output remains globally sorted while no result object for
every gene is retained at once. The same public thread limit also constrains PCA and
distance-library thread pools, while cell-to-centroid chunks use their separate 64 MiB
working-memory cap.

Representations, centroids, only the distance matrices needed by the selected
weighting mode, bandwidth, TF predictors, size factors, and one weight vector per
target group are reused. In particular, `group-distance` does not allocate or scan a
cell-to-centroid matrix. Phase timings, requested/effective threads, backend, model
counts, relevant dependency versions, and warnings are written to metadata.

Deterministic rows are written in stable order. Compressed tables also use a fixed gzip
timestamp, so a deterministic table such as `cell_weights.tsv.gz` has identical bytes
for identical inputs and effective parameters. `model_diagnostics.tsv.gz` includes
measured fit times; its compression header is reproducible, but its time-bearing
content is not promised to be byte-identical across runs.

## Interpretation boundaries

SPATHI produces predictive regulatory hypotheses, not demonstrated causal networks.
In particular:

- impurity importance can be biased toward predictors with greater variation;
- correlated TFs can divide or exchange importance;
- transcriptomic proximity need not imply shared regulatory mechanisms;
- output quality depends on preprocessing, cell labels, TF coverage, representation,
  metric, kernel, and bandwidth;
- a centroid may summarize a heterogeneous or non-convex group poorly;
- group-size correction changes statistical contribution, not biological similarity.

Use `cell_weights.tsv.gz:final_weight` for the exact observation weights and
`weight_diagnostics.tsv` for their effective group-level contribution.
`group_affinities.tsv` should be used only as the centroid-affinity diagnostic
described above. The figures provide a compact view of those quantities, but their
PC1/PC2 positions are projections and do not replace the tabular values. Independent
evidence and downstream validation remain necessary.
