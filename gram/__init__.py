"""GRAM: GAM Rashomon Sets with Adaptive MCMC.

Sample diverse generalized additive models from the Rashomon set via
Metropolis-Hastings over feature-support sets, using a Laplace-approximated
marginal-likelihood target with optional repulsive diversity and Hessian
ellipsoid augmentation.

The algorithm modules live in this package (``gram.mcmc_rashomon``,
``gram.ellipsoid``, ``gram.proposal_functions``, ...). This ``__init__``
re-exports the public API under one namespace::

    import gram

    X, y, feature_names, feature_groups = gram.binarize_data(df)
    sampler = gram.MCMCRashomonSampler(X, y, eps=0.05, feature_groups=feature_groups)
    result = sampler.sample(n_steps=5000, proposal="swap")
"""

from gram.mcmc_rashomon import (
    MCMCRashomonSampler,
    RashomonResult,
    binarize_data,
)
from gram.plotting import plot_gam_shapes
from gram.ellipsoid import EllipsoidSampler
from gram.diversity import (
    DEFAULT_DIVERSITY_FN,
    DIVERSITY_FN_BUILDERS,
    resolve_diversity_fn,
    diversity_min_l1,
    diversity_summary,
    diversity_support_jaccard,
    make_diversity_monotonicity_entropy,
    make_diversity_prediction_hamming,
    monotonicity_diversity,
    pairwise_hamming,
    pairwise_prediction_hamming,
    pairwise_shape_distance,
    pairwise_support_jaccard,
    pairwise_weight_l1,
    variable_importance,
)
from gram.proposal_functions import (
    AdaptiveResizeProposer,
    compute_correlation_scores,
    make_mixture_proposal,
    propose_add_or_remove,
    propose_feature_group_swap,
    propose_multi_swap,
    propose_promise_score_mixture,
    propose_swap,
    propose_swap_or_resize,
    propose_weighted_swap,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core sampler
    "MCMCRashomonSampler",
    "RashomonResult",
    "binarize_data",
    "plot_gam_shapes",
    "EllipsoidSampler",
    # Repulsion kernels (diversity targets during MCMC) — gram.diversity
    "DEFAULT_DIVERSITY_FN",
    "DIVERSITY_FN_BUILDERS",
    "resolve_diversity_fn",
    "diversity_min_l1",
    "diversity_support_jaccard",
    "make_diversity_monotonicity_entropy",
    "make_diversity_prediction_hamming",
    # Support-set proposals
    "AdaptiveResizeProposer",
    "compute_correlation_scores",
    "make_mixture_proposal",
    "propose_add_or_remove",
    "propose_feature_group_swap",
    "propose_multi_swap",
    "propose_promise_score_mixture",
    "propose_swap",
    "propose_swap_or_resize",
    "propose_weighted_swap",
    # Pairwise diversity measures on sampled models — gram.diversity
    "diversity_summary",
    "monotonicity_diversity",
    "pairwise_hamming",
    "pairwise_prediction_hamming",
    "pairwise_shape_distance",
    "pairwise_structural_distance",
    "pairwise_support_jaccard",
    "pairwise_weight_l1",
    "variable_importance",
]
