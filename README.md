# GRAM: GAM Rashomon sets with Adaptive MCMC

Markov-Chain-Monte-Carlo based sampler for uncovering diverse Generalized Additive Models (GAMs) in the Rashomon set.

## Install

```bash
python -m pip install .
```

## Usage

```python
import gram

X, y, feature_names, feature_groups = gram.binarize_data(df) 
# df assumed to have numerical values (continuous, binary, or ordinal). The last column is the target.
sampler = gram.MCMCRashomonSampler(X, y, eps=0.025, sigma2=0.05, p_feat=1e-4, feature_groups=feature_groups)
result = sampler.sample(n_steps=5000, proposal="adaptive_resize", mh_variant="repulsive", finetune_coordinate = True, diversity_fn="prediction_hamming")
print(result.w_models.shape, result.best_error, result.error_bound)
# result.w_models[i] is the weight vector for the i-th model. result.w_models[i][j] is the weight for the j-th feature, i.e. feature_names[j]. 
```

### The result

`sample` returns a `RashomonResult`. The Rashomon set is defined on the
**negative log posterior** (NLL + L2 prior): a candidate is kept if
`nll <= nll_bound`, where `nll_bound = best_nll * (1 + eps)` (or the explicit
`rashomon_loss_bound`, if given). Every array below is already filtered to that
set, and row `i` of `w_models`, `support_sets`, `errors`, and `nlls` describes
the same model.

| Field | Meaning |
|---|---|
| `w_models` | `(n_models, p)` weights of the kept models. Zero entries = features outside that model's support. This is the Rashomon set. |
| `w_opt` | `(p,)` weights of the single lowest-NLL model found (the reference the bound is measured against). This is not necessarily the most accurate model, just the model that best traversed the sparsity-accuracy tradeoff.|
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
`gram.plotting.plot_gam_shapes` to see the shape functions, or to the
`gram.diversity` metrics to quantify the spread.

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
