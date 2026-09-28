# GRAM: GAM Rashomon sets with Adaptive MCMC (NeurIPS '26)

Markov-Chain-Monte-Carlo based sampler for uncovering diverse Generalized Additive Models (GAMs) in the Rashomon set.

## Install

```bash
python -m pip install .
```

## Usage

```python
import gram

X, y, feature_names, feature_groups = gram.binarize_data(df)
# df: numeric columns + binary target in the last column.
# Columns of X are grouped by raw feature, thresholds ascending within each group.

sampler = gram.MCMCRashomonSampler(
    X, y,
    eps=0.025,
    sigma2=0.05,
    p_feat=1e-4,
    feature_names=feature_names,
    feature_groups=feature_groups,
)

result = sampler.sample(
    n_steps=5000,
    proposal="adaptive_resize",
    mh_variant="repulsive",
    finetune_coordinate=True,
    diversity_fn="prediction_hamming",
)

print(result.w_models.shape, result.best_error, result.error_bound)
```

### Weight vectors and predictions

Each row of `result.w_models` is a logistic-regression coefficient vector
aligned with the **binarized** design matrix `X` from `binarize_data` (not the
raw CSV). Column 0 is an intercept; remaining columns are cumulative indicators
`feat <= t` (1 if the raw value is at or below `t`, else 0).

Predictions use the usual dot product:

```python
logit = result.X @ result.w_opt
proba = 1.0 / (1.0 + np.exp(-logit))

# Equivalent binned form:
logit = result.X_binned @ result.w_opt_binned
```

Most entries in each row are zero: only thresholds in that model's support set
carry nonzero weights.

### Ladder vs. binned weights

Internally, coefficients are stored on the cumulative ladder (`feat<=t`).
`sample()` also converts them to **disjoint bin weights** automatically
(same column indices, different interpretation):

| Representation | Column meaning | Example (`age` thresholds 34, 35, 36) |
|---|---|---|
| Ladder (`w_models`, `w_opt`) | Increment on `feat<=t` | weights 1, 2, 3 |
| Binned (`w_models_binned`, `w_opt_binned`) | Contribution in bin | weights 6, 5, 3 (`<=34`, `34–35`, `35–36`) |

`result.feature_names` holds ladder names (`age<=34`, …).
`result.feature_names_binned` holds disjoint bin labels aligned with the binned
weights (`age <= 34.0`, `34.0 < age <= 35.0`, …).
`result.bin_labels` is the same information grouped by raw feature name.

### The result

`sample` returns a `RashomonResult`. All samples in the Rashomon set obey
`nll <= nll_bound`, where `nll_bound = best_nll * (1 + eps)`.

| Field | Meaning |
|---|---|
| `w_models` | `(n_models, p)` ladder weights of the kept models. Zero entries = features outside that model's support. |
| `w_opt` | `(p,)` ladder weights of the single lowest-NLL model found. |
| `w_models_binned`, `w_opt_binned` | Same shapes; disjoint-bin weights (populated automatically by `sample()`). |
| `feature_names`, `feature_groups` | Column names and per-feature index lists (pass these to the sampler). |
| `feature_names_binned`, `bin_labels` | Disjoint bin names (flat list and per-feature dict). |
| `X` | `(n, p)` cumulative ladder design matrix used for sampling (`feat<=t` columns). |
| `X_binned` | `(n, p)` disjoint-bin design matrix; `X_binned @ w_binned` equals `X @ w_ladder`. |
| `support_sets` | Column indices active in each kept model — the feature/threshold subset it uses. |
| `errors`, `best_error` | Training 0-1 error per kept model, and the error of `w_opt`. |
| `error_bound` | Worst training error in the set — how much accuracy you give up at the edge of the set. |
| `nlls`, `best_nll`, `nll_bound` | Objective per kept model, the best value, and the threshold that defined the set. |
| `all_nlls` | NLLs of *every* candidate scored, kept or not — compare against `nll_bound` to see how selective the filter was. |
| `n_explored`, `n_accepted` | Candidates scored vs. candidates that fell inside the bound. |
| `acceptance_rate` | MH acceptance rate. Very low (<5%) means the chain is stuck; very high (>60%) means proposals are too timid. |
| `w_mcmc`, `w_ellipsoid`, `ellipsoid_by_support` | Only with `ellipsoid_augment=True`: the MAP models and the ellipsoid samples drawn around them, split out from `w_models`. |

Reading it: a large `w_models` with `error_bound` close to `best_error` means
many near-equally-accurate models exist. Whether they are *meaningfully
different* is a separate question — pass the result to
`gram.plot_gam_shapes` to see the shape functions, or to the
`gram.diversity` metrics to quantify the spread.

### Plotting

```python
# Ladder weights: reverse cumulative sum of threshold coefficients
gram.plot_gam_shapes(result, result.feature_groups, data, save_path="shapes_ladder.png")

# Disjoint bin weights: one step height per bin
gram.plot_gam_shapes(
    result, result.feature_groups, data,
    binned=True,
    save_path="shapes_binned.png",
)
```

With `binned=True`, the plot uses `w_models_binned`, `w_opt_binned`, and
`feature_names_binned`. Both modes use `result.feature_groups` for grouping
columns by raw feature.

## Algorithm

| File | Role |
|---|---|
| `mcmc_rashomon.py` | Core MCMC Rashomon sampler (`MCMCRashomonSampler`), binarization |
| `plotting.py` | GAM shape-function plots |
| `proposal_functions.py` | Support-set proposals (swap, adaptive resize, feature-group, mixture) |
| `diversity.py` | Repulsion kernels used during MCMC + pairwise diversity metrics on sampled models |
| `ellipsoid.py` | Hessian ellipsoid baseline sampler |
| `binarize_augmented_datasets.py` | GBDT-based feature binarization (from \cite{McTavish et al, 2022})|

### Example

`example.py` is the shortest path from the repo to a Rashomon set. 2 datasets, `fico` and `compas` are provided in the `datasets` folder as a starter pack. 

```bash
python example.py     # writes fico_gam_shapes.png
```
