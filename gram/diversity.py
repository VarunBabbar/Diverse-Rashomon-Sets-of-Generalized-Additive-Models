"""
Diversity for Rashomon-set GAMs.

* **Repulsion kernels** — signature ``(accepted_samples, candidate) -> float``,
  evaluated *during* repulsive MH to push proposals away from what has already
  been collected. Higher = more diverse.
* **Pairwise measures** — evaluated *after* sampling on a finished
  ``RashomonResult``, returning ``(dist_matrix, mean_dist)``.

Both operate on the same ``(n_models, p)`` weight matrix, so the shared cores
(``_shape_matrix``, ``_pairwise_l1``, ``_monotonicity_from_W``) are defined once
and used by both sides.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # avoids a cycle: mcmc_rashomon imports the kernels below.
    from gram.mcmc_rashomon import RashomonResult


# ================================================================== #
#  Shared cores — operate on a raw (n_models, p) weight matrix
# ================================================================== #


def _mean_offdiag(dist_matrix: np.ndarray) -> float:
    """Mean over all distinct pairs (strict upper triangle); 0.0 if none."""
    n = dist_matrix.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(dist_matrix[mask].mean()) if mask.sum() > 0 else 0.0


def _pairwise_l1(M: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Chunked pairwise mean-|difference| between rows of ``M``.

    Chunked to avoid the ``(n, n, d)`` allocation, which matters when
    ``n_models`` is large (e.g. 20k+ models from ellipsoid augmentation).
    """
    n = M.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(0, n, chunk):
        ie = min(i + chunk, n)
        diff = np.abs(M[i:ie, None, :] - M[None, :, :])  # (chunk, n, d)
        dist_matrix[i:ie, :] = diff.mean(axis=2)
    return dist_matrix


def _shape_matrix(
    W: np.ndarray, feature_groups: Dict[str, List[int]]
) -> np.ndarray:
    """Concatenated GAM shape functions, one row per model.

    With threshold binarization (column j = ``feature <= v_j``) the shape
    function of a feature group is the reverse cumulative sum of its weights.
    """
    shape_vecs = []
    for _, indices in feature_groups.items():
        cols = W[:, indices]  # (n_models, n_thresholds)
        shape_vecs.append(np.cumsum(cols[:, ::-1], axis=1)[:, ::-1])
    return np.hstack(shape_vecs)  # (n_models, total_thresholds)


def _centered_shapes_per_feature(
    W: np.ndarray, feature_groups: Dict[str, List[int]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-feature ``(centered_shapes, norms)`` used by the structural metrics."""
    out = []
    for _, indices in feature_groups.items():
        cols = W[:, indices]
        shapes = np.cumsum(cols[:, ::-1], axis=1)[:, ::-1]
        centered = shapes - shapes.mean(axis=1, keepdims=True)
        out.append((centered, np.linalg.norm(centered, axis=1)))
    return out


def _shape_distance_from_W(
    W: np.ndarray, feature_groups: Dict[str, List[int]]
) -> Tuple[np.ndarray, float]:
    """Pairwise shape-function L1 distance from a raw weight matrix."""
    dist_matrix = _pairwise_l1(_shape_matrix(W, feature_groups))
    return dist_matrix, _mean_offdiag(dist_matrix)


def _monotonicity_from_W(
    W: np.ndarray,
    feature_groups: Dict[str, List[int]],
    flat_tol: float = 1e-10,
) -> Dict[str, object]:
    """Monotonicity classification + entropy from a raw weight matrix.

    See ``monotonicity_diversity`` for the full description of the output.
    """
    n_models = W.shape[0]
    per_feature = {}
    entropies = []

    for feat, indices in feature_groups.items():
        cols = W[:, indices]
        shapes = np.cumsum(cols[:, ::-1], axis=1)[:, ::-1]  # (n_models, n_thresh)
        centered = shapes - shapes.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)

        n_inc = 0
        n_dec = 0
        n_non = 0
        n_flat = 0

        for m in range(n_models):
            if norms[m] < flat_tol:
                n_flat += 1
                continue
            diffs = np.diff(shapes[m])
            if np.all(diffs >= -1e-14):
                n_inc += 1
            elif np.all(diffs <= 1e-14):
                n_dec += 1
            else:
                n_non += 1

        n_active = n_inc + n_dec + n_non
        if n_active > 0:
            p_inc = n_inc / n_active
            p_dec = n_dec / n_active
            p_non = n_non / n_active
            probs = np.array([p_inc, p_dec, p_non])
            probs = probs[probs > 0]
            ent = float(-np.sum(probs * np.log(probs)) / np.log(3))
            entropies.append(ent)
        else:
            p_inc = p_dec = p_non = 0.0
            ent = 0.0

        per_feature[feat] = {
            "frac_increasing": p_inc,
            "frac_decreasing": p_dec,
            "frac_nonmono": p_non,
            "frac_flat": n_flat / n_models,
            "entropy": ent,
        }

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    if not entropies:
        std_entropy = float("nan")
    elif len(entropies) == 1:
        std_entropy = 0.0
    else:
        std_entropy = float(np.std(np.asarray(entropies, dtype=np.float64), ddof=0))

    return {
        "per_feature": per_feature,
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "n_entropy_features": int(len(entropies)),
    }


# ================================================================== #
#  Pairwise measures on a sampled Rashomon set
# ================================================================== #
# Each returns (dist_matrix, mean_dist) over the models in ``result``.


def pairwise_hamming(result: RashomonResult) -> Tuple[np.ndarray, float]:
    """
    Pairwise Hamming distance on support sets (feature inclusion).

    For each pair of models, computes the fraction of features where one model
    includes it and the other does not.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix of Hamming distances.
        mean_dist:   Average over all distinct pairs.
    """
    active = (result.w_models != 0).astype(np.float64)  # (n_models, p)
    p = active.shape[1]

    # |a XOR b| = |a| + |b| - 2|a AND b|  →  hamming = (XOR count) / p
    overlap = active @ active.T  # (n, n) — counts of shared active features
    counts = active.sum(axis=1)  # (n,)
    xor_counts = counts[:, None] + counts[None, :] - 2 * overlap
    dist_matrix = xor_counts / p
    return dist_matrix, _mean_offdiag(dist_matrix)


def pairwise_support_jaccard(
    result: RashomonResult,
    thresh: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Pairwise Jaccard *distance* on support sets (binary feature inclusion).

    For each pair of models, supports are |w_j| > thresh. Jaccard similarity is
    |A ∩ B| / |A ∪ B|; distance is 1 - similarity. Empty–empty pairs have
    distance 0; empty vs non-empty have distance 1.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    n = W.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64), 0.0
    active = (np.abs(W) > thresh).astype(np.float64)
    overlap = active @ active.T
    counts = active.sum(axis=1)
    union = counts[:, None] + counts[None, :] - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc_sim = np.where(union > 0, overlap / union, 1.0)
    dist_matrix = 1.0 - jacc_sim
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix, _mean_offdiag(dist_matrix)


def pairwise_shape_distance(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
) -> Tuple[np.ndarray, float]:
    """
    Pairwise shape function diversity (average L1 distance).

    For each original feature, the shape function at threshold v_j is the
    reverse cumulative sum of weights for that feature group.  The distance
    between two models is the mean (over features and thresholds) of the
    absolute difference in shape function values.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    return _shape_distance_from_W(result.w_models, feature_groups)


def pairwise_prediction_hamming(
    result: RashomonResult,
    X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Pairwise Hamming distance between model predictions.

    For each pair of models, computes the fraction of samples on which
    they predict different classes.

    Args:
        result: RashomonResult with w_models of shape (n_models, p).
        X:      (n_samples, p) design matrix used for prediction.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    W = result.w_models
    preds = (X @ W.T >= 0).astype(np.float64)   # (n_samples, n_models)
    n_samples = X.shape[0]

    # |pred_a XOR pred_b| = |a| + |b| - 2|a AND b|
    agree = preds.T @ preds                       # (n_models, n_models)
    ones = preds.sum(axis=0)                       # (n_models,)
    disagree = ones[:, None] + ones[None, :] - 2 * agree
    dist_matrix = disagree / n_samples
    return dist_matrix, _mean_offdiag(dist_matrix)


def pairwise_weight_l1(result: RashomonResult) -> Tuple[np.ndarray, float]:
    """
    Pairwise average L1 distance between raw weight vectors.

    Returns:
        dist_matrix: (n_models, n_models) symmetric matrix.
        mean_dist:   Average over all distinct pairs.
    """
    dist_matrix = _pairwise_l1(result.w_models)
    return dist_matrix, _mean_offdiag(dist_matrix)



def monotonicity_diversity(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
    flat_tol: float = 1e-10,
) -> Dict[str, object]:
    """
    For each feature, classify every model's shape function as monotonically
    increasing, monotonically decreasing, non-monotonic, or flat (all weights
    effectively zero).  Report the fractions and an entropy-based diversity
    score.

    A shape is classified by looking at the successive differences of the
    step-function values (reverse cumulative sum of weights).  If all diffs
    are >= 0 it is increasing; if all are <= 0 it is decreasing; otherwise
    non-monotonic.  Shapes whose centered L2 norm is below ``flat_tol`` are
    labelled flat (feature unused).

    The per-feature diversity score is the Shannon entropy of the
    (increasing, decreasing, non-monotonic) distribution (excluding flat),
    normalized by log(3) so that 1.0 = perfectly uniform three-way split.

    Returns:
        Dict with keys:
            per_feature: dict mapping feature name -> {
                "frac_increasing", "frac_decreasing", "frac_nonmono",
                "frac_flat", "entropy"
            }
            mean_entropy: average normalized entropy across features
                          (only features with at least one non-flat model).
            std_entropy: population std of per-feature entropies in that
                         same set; NaN if no such features, 0 if exactly one.
    """
    return _monotonicity_from_W(result.w_models, feature_groups, flat_tol)


def variable_importance(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Permutation importance for each original feature across the Rashomon set.

    For a single model, permutation importance of feature f is the increase
    in misclassification error when the columns belonging to f are randomly
    shuffled (averaged over ``n_repeats`` shuffles).

    This function computes min, max, mean, and std of that quantity across
    all models in the Rashomon set.

    Args:
        result:         RashomonResult.
        feature_groups: Dict mapping feature names to column index lists.
        X:              (n, p) design matrix.
        y:              (n,) binary labels in {0, 1}.
        n_repeats:      Number of random permutations per feature per model.

    Returns:
        Dict mapping feature name -> {"min", "max", "mean", "std", "values"}
        where "values" is the (n_models,) array of per-model importances.
    """
    W = result.w_models  # (n_models, p)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples, _ = X.shape
    n_models = W.shape[0]

    logits_base = X @ W.T  # (n_samples, n_models)
    base_errors = np.mean((logits_base >= 0).astype(float) != y[:, None], axis=0)

    out = {}
    for feat, indices in feature_groups.items():
        importance = np.zeros(n_models)
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm = np.random.permutation(n_samples)
            X_perm[:, indices] = X_perm[perm][:, indices]
            logits_perm = X_perm @ W.T
            perm_errors = np.mean(
                (logits_perm >= 0).astype(float) != y[:, None], axis=0
            )
            importance += perm_errors - base_errors
        importance /= n_repeats

        out[feat] = {
            "min": float(importance.min()),
            "max": float(importance.max()),
            "mean": float(importance.mean()),
            "std": float(importance.std()),
            "values": importance,
        }
    return out


def diversity_summary(
    result: RashomonResult,
    feature_groups: Dict[str, List[int]],
    X: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute all diversity measures and print a summary.

    Args:
        result:         RashomonResult.
        feature_groups: Dict mapping feature names to column index lists.
        X:              Design matrix for prediction Hamming (skipped if None).
        label:          Optional label for the printout.

    Returns a dict with keys: hamming, pred_hamming, shape_l1, weight_l1,
    monotonicity.
    """
    _, h = pairwise_hamming(result)
    _, s = pairwise_shape_distance(result, feature_groups)
    _, w = pairwise_weight_l1(result)
    mono = monotonicity_diversity(result, feature_groups)

    out = {
        "hamming": h, "shape_l1": s, "weight_l1": w,
        "monotonicity": mono,
    }

    tag = f" ({label})" if label else ""
    print(f"Diversity summary{tag}  [{result.n_accepted} models]")
    print(f"  Pairwise Hamming (support)   : {h:.4f}")

    if X is not None:
        _, ph = pairwise_prediction_hamming(result, X)
        out["pred_hamming"] = ph
        print(f"  Pairwise Hamming (prediction): {ph:.4f}")

    print(f"  Pairwise shape L1 distance   : {s:.4f}")
    print(f"  Pairwise weight L1 distance  : {w:.4f}")
    print(f"  Monotonicity entropy (avg)   : {mono['mean_entropy']:.4f}")
    for feat, info in mono["per_feature"].items():
        print(f"    {feat:>20s}:  inc {info['frac_increasing']:.0%}"
              f"  dec {info['frac_decreasing']:.0%}"
              f"  non {info['frac_nonmono']:.0%}"
              f"  flat {info['frac_flat']:.0%}"
              f"  H={info['entropy']:.3f}")
    return out


def plot_variable_importance(
    vi_dict: Dict[str, Dict[str, float]],
    label: str = "",
    ax=None,
):
    """Plot permutation importance ranges (min/max bars with mean marker)."""
    import matplotlib.pyplot as plt

    feats = list(vi_dict.keys())
    means = [vi_dict[f]["mean"] for f in feats]
    mins = [vi_dict[f]["min"] for f in feats]
    maxs = [vi_dict[f]["max"] for f in feats]
    stds = [vi_dict[f]["std"] for f in feats]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(feats))))

    y = np.arange(len(feats))
    ax.barh(y, [mx - mn for mn, mx in zip(mins, maxs)],
            left=mins, color="steelblue", alpha=0.35, edgecolor="steelblue",
            label="range [min, max]")
    ax.errorbar(means, y, xerr=stds, fmt="o", color="black",
                capsize=3, markersize=4, label="mean ± std")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Permutation Importance (error increase)")
    title = "Permutation Importance"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.legend(fontsize=9)
    return ax


# ================================================================== #
#  Repulsion kernels for repulsive MH
# ================================================================== #
# Each has signature (accepted_samples: List[ndarray], candidate: ndarray) -> float.
# Higher return value = more diverse (further from accepted). Used in log ratio:
#   log_alpha += repulsion_weight * (log(diversity(..., w_prop) + eps) - log(diversity(..., w_cur) + eps))


def diversity_min_l1(
    accepted_samples: List[np.ndarray], candidate: np.ndarray
) -> float:
    """
    Minimum mean absolute (L1) distance to any accepted sample.
    Favours proposals that are far from the nearest collected model.
    """
    if not accepted_samples:
        return 0.0
    arr = np.array(accepted_samples)
    dists = np.mean(np.abs(arr - candidate), axis=1)
    return float(dists.min())


def diversity_support_jaccard(
    accepted_samples: List[np.ndarray], candidate: np.ndarray, thresh: float = 1e-8
) -> float:
    """
    One minus the mean Jaccard similarity of support sets.
    Favours proposals whose active (non-zero) features differ from accepted models.
    Supports are derived by thresholding weights: |w[j]| > thresh.
    """
    if not accepted_samples:
        return 0.0
    sup_c = set(np.where(np.abs(candidate) > thresh)[0].tolist())
    if not sup_c:
        return 0.0
    jaccards = []
    for w in accepted_samples:
        sup_w = set(np.where(np.abs(w) > thresh)[0].tolist())
        if not sup_w:
            jaccards.append(1.0)
            continue
        inter = len(sup_c & sup_w)
        union = len(sup_c | sup_w)
        jaccards.append(inter / union if union else 1.0)
    return float(1.0 - np.mean(jaccards))


def make_diversity_prediction_hamming(
    X: np.ndarray,
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a prediction-level Hamming diversity function.

    Computes logits z = X @ w for each model, converts to binary predictions
    (z >= 0 → 1, else 0), and returns the mean Hamming distance between the
    candidate's predictions and those of all accepted models.

    Predictions for reference (accepted) models are cached by ``id(w)`` so that
    each reference's ``X @ w`` is computed at most once over the lifetime of
    the returned callable. This assumes reference arrays stored in the caller's
    ``collected_w`` / ``collected_ellipsoid_samples`` are not mutated after
    being added (which is how ``mh_sample_repulsive`` uses them).

    Usage::

        diversity_fn = make_diversity_prediction_hamming(X)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """
    pred_cache: Dict[int, np.ndarray] = {}
    n_rows = X.shape[0]

    def _predict(w: np.ndarray) -> np.ndarray:
        key = id(w)
        pred = pred_cache.get(key)
        if pred is None:
            pred = (X @ w >= 0).astype(np.int8)
            pred_cache[key] = pred
        return pred

    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not accepted_samples:
            return 0.0
        pred_cand = (X @ candidate >= 0).astype(np.int8)
        P_refs = np.stack([_predict(w) for w in accepted_samples])
        diffs = np.count_nonzero(P_refs != pred_cand[None, :], axis=1)
        return float(diffs.mean() / n_rows)

    return _diversity


def make_diversity_monotonicity_entropy(
    feature_groups: Dict[str, List[int]],
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Factory that returns a monotonicity-entropy diversity function.

    For the set {accepted_samples} ∪ {candidate}, computes the mean entropy
    across shape functions (increasing vs decreasing vs non-monotonic).
    Higher entropy = more diverse monotonicity patterns.

    Usage::

        diversity_fn = make_diversity_monotonicity_entropy(feature_groups)
        result = sampler.sample(..., diversity_fn=diversity_fn)
    """

    def _diversity(accepted_samples: List[np.ndarray], candidate: np.ndarray) -> float:
        if not feature_groups:
            return 0.0
        w_all = np.array(accepted_samples + [candidate])
        if w_all.shape[0] < 2:
            return 0.0
        out = _monotonicity_from_W(w_all, feature_groups)
        return float(out["mean_entropy"])

    return _diversity


# Default repulsion kernel for repulsive MH.
DEFAULT_DIVERSITY_FN = diversity_support_jaccard


# ================================================================== #
#  String -> repulsion kernel
# ================================================================== #

# Each entry builds the kernel from (X, feature_groups); the plain kernels
# ignore both. Keys are matched case-insensitively with '-'/' ' -> '_'.
DIVERSITY_FN_BUILDERS: Dict[str, Callable[..., Callable]] = {
    "l1": lambda X, groups: diversity_min_l1,
    "min_l1": lambda X, groups: diversity_min_l1,
    "jaccard": lambda X, groups: diversity_support_jaccard,
    "support_jaccard": lambda X, groups: diversity_support_jaccard,
    "prediction_hamming": lambda X, groups: make_diversity_prediction_hamming(X),
    "hamming": lambda X, groups: make_diversity_prediction_hamming(X),
    "monotonicity_entropy": lambda X, groups: make_diversity_monotonicity_entropy(groups),
    "monotonicity": lambda X, groups: make_diversity_monotonicity_entropy(groups),
}


def resolve_diversity_fn(
    name: str,
    X: Optional[np.ndarray] = None,
    feature_groups: Optional[Dict[str, List[int]]] = None,
) -> Callable[[List[np.ndarray], np.ndarray], float]:
    """
    Build a repulsion kernel from its name, e.g. ``"monotonicity_entropy"``.

    ``X`` / ``feature_groups`` are the data the factory kernels need; the
    sampler passes its own. See ``DIVERSITY_FN_BUILDERS`` for the valid names.
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in DIVERSITY_FN_BUILDERS:
        raise ValueError(
            f"Unknown diversity_fn {name!r}; choose one of "
            f"{sorted(DIVERSITY_FN_BUILDERS)}"
        )
    return DIVERSITY_FN_BUILDERS[key](X, feature_groups)
