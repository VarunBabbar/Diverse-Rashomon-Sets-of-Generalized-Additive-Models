"""
Plotting utilities for Rashomon-set GAMs.

Visualisation code lives here so that the sampler module stays free of
plotting dependencies. ``matplotlib`` is imported lazily inside each plotting
function.
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from gram.mcmc_rashomon import RashomonResult, _parse_threshold_from_colname

# Weights below this magnitude are treated as exactly zero when deciding
# whether a feature is used by any model in the Rashomon set.
_ZERO_TOL = 1e-12


def _normalize_shape(shape):
    """Center (subtract mean) and normalize (divide by L2 norm) a shape vector."""
    shape = shape - shape.mean()
    norm = np.linalg.norm(shape)
    if norm > 1e-12:
        shape = shape / norm
    return shape


def plot_gam_shapes(
    result: "RashomonResult",
    feature_groups: Dict[str, List[int]],
    data: pd.DataFrame,
    max_models: int = 200,
    alpha: float = 0.3,
    normalize: bool = False,
    figsize_per_feat: Tuple[float, float] = (5.5, 4.0),
    save_path: Optional[str] = None,
    title_suffix: str = "",
    feature_names: Optional[List[str]] = None,
):
    """
    Plot GAM shape functions for the features that are actually used.

    A feature is plotted only if at least one of the models being drawn (the
    optimal model or one of the sampled Rashomon-set models) gives it a
    non-zero shape function; features whose weights are zero in every such
    model are skipped entirely and do not take up a subplot.

    With threshold binarization (column j = ``feature <= v_j``), the shape
    function at input value x equals the sum of weights whose threshold is
    >= x, i.e. a reverse cumulative sum of the weight vector.

    When ``normalize=True`` (default), each shape function is centered
    (mean-subtracted) and divided by its L2 norm so that all features are
    shown on a comparable scale.

    All Rashomon-set models are drawn in red; the optimal model in dark gray.

    Args:
        feature_names: Optional list of column names (e.g. ``feat<=v``). When
            provided and L1 feature selection reduces columns, thresholds are
            extracted from these names for correct alignment with indices.
    """
    import matplotlib.pyplot as plt

    X_raw = data.iloc[:, :-1]

    w_all = result.w_models
    if w_all.shape[0] > max_models:
        idx = np.random.choice(w_all.shape[0], max_models, replace=False)
        w_all = w_all[idx]

    # Resolve thresholds per feature and drop features that are unused (all
    # weights zero) or whose thresholds cannot be aligned with the indices.
    plotted = []
    for feat, indices in feature_groups.items():
        thresholds_raw = np.sort(np.unique(X_raw[feat].values))
        if feature_names is not None and len(indices) != len(thresholds_raw):
            thresholds = []
            for i in indices:
                if i < len(feature_names):
                    v = _parse_threshold_from_colname(feature_names[i], feat)
                    if v is not None:
                        thresholds.append(v)
            thresholds = np.array(thresholds) if thresholds else np.array([])
        else:
            thresholds = thresholds_raw

        if len(thresholds) != len(indices):
            continue

        used = (np.any(np.abs(w_all[:, indices]) > _ZERO_TOL)
                or np.any(np.abs(result.w_opt[indices]) > _ZERO_TOL))
        if not used:
            continue

        perm = np.argsort(thresholds)
        plotted.append((feat, indices, thresholds[perm], perm))

    if not plotted:
        warnings.warn("No feature has a non-zero shape function; nothing to plot.")
        return

    n_feats = len(plotted)
    n_cols = min(n_feats, 4)
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_feat[0] * n_cols, figsize_per_feat[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (feat, indices, thresholds, perm) in zip(axes_flat, plotted):
        def _shape(w_full):
            """Reverse cumulative sum of the weights in ascending-threshold order.

            The columns of a feature are not stored sorted by threshold, so the
            weights must be permuted *before* accumulating: sorting the partial
            sums afterwards is not the same thing.
            """
            return np.cumsum(w_full[indices][perm][::-1])[::-1]

        for w in w_all:
            shape = _shape(w)
            if normalize:
                shape = _normalize_shape(shape)
            ax.step(thresholds, shape, where="post",
                    color="red", alpha=alpha, linewidth=0.8)

        shape_opt = _shape(result.w_opt)
        if normalize:
            shape_opt = _normalize_shape(shape_opt)
        ax.step(thresholds, shape_opt, where="post",
                color="dimgray", linewidth=2.5, label="optimal")

        ax.set_xlabel(feat, fontsize=14)
        ylabel = "normalized shape" if normalize else "predicted logit"
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10)

    for ax in axes_flat[n_feats:]:
        ax.set_visible(False)

    title = "GAM Shape Functions (Rashomon Set)" + title_suffix
    if normalize:
        title += " — centered & normalized"
    fig.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
