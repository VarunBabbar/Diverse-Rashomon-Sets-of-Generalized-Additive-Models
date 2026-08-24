"""
Minimal GRAM example: load FICO, sample a Rashomon set of GAMs, plot the shapes.

    python example.py
"""

import pandas as pd

import gram

sigma2 = 0.05 # Regularization parameter for a given support set
eps = 0.025 # note that the eps is with respect to the regularized loss, so the accuracy ranges may not reflect the eps you want to use. Adjust accordingly. 
p_feat = 1e-4 # Support set sparsity prior. 

n_steps = 1000 # Total number of steps to run for the MCMC sampler.
diversity_fn = "monotonicity_entropy" # maximize Rashomon set with respect to this diversity function. 
                                    # Options: "prediction_hamming", "jaccard", "monotonicity_entropy", "l1"

dataset = "compas"
# 1. Load dataset. The last column is the binary target.
data = pd.read_csv(f"datasets/{dataset}.csv").reset_index(drop=True)

# 2. Binarize: each raw feature becomes a ladder of `feature <= threshold` columns.
X, y, feature_names, feature_groups = gram.binarize_data(data, num_estimators=200, min_thresholds_per_feat=0.5)
print(X.shape)

# 3. Sample the Rashomon set.
sampler = gram.MCMCRashomonSampler(
    X, y,
    eps=eps,
    sigma2=sigma2,
    p_feat=p_feat,
    feature_groups=feature_groups,
)
result = sampler.sample(
    n_steps=n_steps,
    proposal="adaptive_resize",
    mh_variant="repulsive",
    finetune_coordinate = True,
    diversity_fn=diversity_fn,
    )
print(f"{result.w_models.shape[0]} models, "
      f"training error {result.best_error:.4f} -> {result.error_bound:.4f}")
# result.w_models[i] is the weight vector for the i-th model. result.w_models[i][j] is the weight for the j-th feature, i.e. feature_names[j]. 

# 4. Plot every model's shape function, one panel per feature.
gram.plot_gam_shapes(
    result,
    feature_groups,
    data,
    feature_names=feature_names,
    save_path=f"{dataset}_gam_shapes.png",
    alpha=0.05,
    max_models=5000,
    normalize=False
)
print(f"Saved {dataset}_gam_shapes.png")
