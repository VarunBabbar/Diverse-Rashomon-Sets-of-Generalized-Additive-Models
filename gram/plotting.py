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


def _parse_upper_threshold_from_bin_label(label: str) -> Optional[float]:
    """Upper bound from a disjoint bin label like ``34.0 < age <= 35.0``."""
    if "<=" not in label:
        return None
    try:
        return float(label.rsplit("<=", 1)[1].strip())
    except (ValueError, IndexError):
        return None


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
    binned: bool = True,
):
    """
    Plot GAM shape functions for the features that are actually used.

    A feature is plotted only if at least one of the models being drawn (the
    optimal model or one of the sampled Rashomon-set models) gives it a
    non-zero shape function; features whose weights are zero in every such
    model are skipped entirely and do not take up a subplot.

    With ladder weights (``binned=False``), columns are ``feature <= v_j`` and
    the shape at input value x is a reverse cumulative sum of the weight
    vector. With binned weights (``binned=True``), uses ``w_models_binned``,
    ``w_opt_binned``, and ``feature_names_binned``; each weight is the disjoint
    bin contribution and is plotted directly (no cumsum).

    When ``normalize=True``, each shape function is centered (mean-subtracted)
    and divided by its L2 norm so that all features are shown on a comparable
    scale.

    All Rashomon-set models are drawn in red; the optimal model in dark gray.

    Args:
        feature_names: Optional list of column names (e.g. ``feat<=v``). When
            provided and L1 feature selection reduces columns, thresholds are
            extracted from these names for correct alignment with indices.
            Ignored for naming when ``binned=True`` (uses
            ``result.feature_names_binned`` unless this argument is passed).
        binned: If True, plot disjoint-bin weights from ``result.w_models_binned``
            and ``result.w_opt_binned`` instead of ladder weights.
    """
    import matplotlib.pyplot as plt

    if binned:
        if (
            result.w_models_binned is None
            or result.w_opt_binned is None
            or result.feature_names_binned is None
        ):
            raise ValueError(
                "binned=True requires w_models_binned, w_opt_binned, and "
                "feature_names_binned on RashomonResult (run sample() with "
                "feature_names and feature_groups passed to the sampler)."
            )
        w_all = result.w_models_binned
        w_opt = result.w_opt_binned
        column_names = (
            feature_names if feature_names is not None else result.feature_names_binned
        )
    else:
        w_all = result.w_models
        w_opt = result.w_opt
        column_names = (
            feature_names if feature_names is not None else result.feature_names
        )

    X_raw = data.iloc[:, :-1]

    if w_all.shape[0] > max_models:
        idx = np.random.choice(w_all.shape[0], max_models, replace=False)
        w_all = w_all[idx]

    ladder_names = result.feature_names if result.feature_names is not None else column_names

    # Resolve thresholds per feature and drop features that are unused (all
    # weights zero) or whose thresholds cannot be aligned with the indices.
    plotted = []
    for feat, indices in feature_groups.items():
        thresholds_raw = np.sort(np.unique(X_raw[feat].values))
        if column_names is not None and len(indices) != len(thresholds_raw):
            thresholds = []
            for i in indices:
                if i < len(column_names):
                    if binned:
                        v = _parse_upper_threshold_from_bin_label(column_names[i])
                    else:
                        v = _parse_threshold_from_colname(column_names[i], feat)
                    if v is None and not binned and i < len(ladder_names):
                        v = _parse_threshold_from_colname(ladder_names[i], feat)
                    if v is not None:
                        thresholds.append(v)
            thresholds = np.array(thresholds) if thresholds else np.array([])
        else:
            thresholds = thresholds_raw

        if len(thresholds) != len(indices):
            continue

        used = (np.any(np.abs(w_all[:, indices]) > _ZERO_TOL)
                or np.any(np.abs(w_opt[indices]) > _ZERO_TOL))
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
            weights = w_full[indices][perm]
            if binned:
                return weights
            return np.cumsum(weights[::-1])[::-1]

        for w in w_all:
            shape = _shape(w)
            if normalize:
                shape = _normalize_shape(shape)
            ax.step(thresholds, shape, where="post",
                    color="red", alpha=alpha, linewidth=0.8)

        shape_opt = _shape(w_opt)
        if normalize:
            shape_opt = _normalize_shape(shape_opt)
        ax.step(thresholds, shape_opt, where="post",
                color="dimgray", linewidth=2.5, label="optimal")

        ax.set_xlabel(feat, fontsize=14)
        if normalize:
            ylabel = "normalized shape"
        elif binned:
            ylabel = "bin weight"
        else:
            ylabel = "predicted logit"
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10)

    for ax in axes_flat[n_feats:]:
        ax.set_visible(False)

    title = "GAM Shape Functions (Rashomon Set)" + title_suffix
    if binned:
        title += " — binned weights"
    if normalize:
        title += " — centered & normalized"
    fig.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
