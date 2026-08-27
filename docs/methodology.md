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

When PCA is selected, scikit-learn PCA receives cells as observations. If the requested
component count is too large, the effective count is bounded safely by the available
cells and genes and recorded in `run_metadata.json`. The selected SVD solver and global
random seed are also recorded.

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

For the selected Euclidean or cosine metric, SPATHI calculates:

1. distances \(d(z_i, \mu_c)\) from every cell to every group centroid; and
2. distances \(d(\mu_h, \mu_c)\) between every pair of group centroids.

Cell-to-centroid calculations are vectorized through scikit-learn in chunks with a
64 MiB working-memory budget rather than inheriting scikit-learn's much larger
process-wide default. The required centroid-to-centroid result is a dense
group-by-group matrix and is materialized once because it is itself an output. The two
cell-distance modes reuse the cell-by-group matrix; when it exceeds the in-memory
threshold, chunks are written directly into a temporary disk-backed memory map.
`group-distance` does not need that matrix for weighting, so it performs the required
cell-to-centroid calculation as a bounded streaming pass and discards each chunk. The
long-form weight artifact continues to record the mode's actual weighting distance,
which is centroid-to-centroid in this mode. Large automatic-bandwidth calculations
likewise use bounded scans and a temporary disk-backed selection array while retaining
the exact positive-distance median.

Cosine distance is undefined for a zero vector. SPATHI therefore rejects any zero-norm
cell representation or centroid and reports the corresponding identifiers; it does
not silently assign distance zero or one. Non-zero rows at exceptional magnitudes are
rescaled before the cosine calculation to avoid norm underflow or overflow. Finally,
non-negative cosine values no larger than the `float64` forward-error bound for a dot
product of the configured dimension are set to exactly zero. This prevents numerical
residue from becoming an artificial automatic bandwidth while preserving genuinely
positive distances above that bound.

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

## Weighted network inference

For group \(c\) and target gene \(t\), SPATHI constructs a predictor matrix from all
valid TFs and removes \(t\) if it is among them. It then fits either scikit-learn's
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

All genes are attempted as targets. Constant responses, empty predictor sets, and other
non-trainable models are recorded in `skipped_targets.tsv`. No self-edges are produced,
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

Representations, centroids, the distances needed by the selected weighting mode,
bandwidth, TF predictors, size factors, and one weight vector per target group are
reused. Phase timings, requested/effective threads, backend, model counts, relevant
dependency versions, and warnings are written to metadata.

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

Use `cell_weights.tsv.gz`, `group_affinities.tsv`, and
`weight_diagnostics.tsv` to assess whether the fitted context matches the intended
biological interpretation. Independent evidence and downstream validation remain
necessary.
