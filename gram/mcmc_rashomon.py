"""
Pure MCMC Rashomon Set Sampler

Samples models from the Rashomon set defined by log posterior:

    R(eps) = { w : nll(w) <= reference_nll * (1 + eps) }

where nll = negative log posterior (NLL + L2 prior). Models are kept if their
log posterior is within the specified bound of the best.

Approach:
  1. Find initial sparse model via L1-regularized logistic regression
  2. Run Metropolis-Hastings over support sets using Laplace-approximated
     marginal likelihood as the MH target
  3. For each sampled support set, fit L2-regularized logistic regression
  4. Keep models whose negative log posterior falls within the bound

Usage:
    sampler = MCMCRashomonSampler(X, y, eps=0.05)
    result = sampler.sample(n_steps=5000, proposal="swap")
    print(result.w_models.shape, result.best_error, result.error_bound)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
from functools import partial
from tqdm import tqdm
from scipy.optimize import minimize
import time
from gram.binarizer import binarize_dataset as binarize_dataset_gbdt, sort_feature_clumps
import random
from gram.diversity import (
    DEFAULT_DIVERSITY_FN,
    resolve_diversity_fn,
    diversity_min_l1,
    diversity_support_jaccard,
    make_diversity_monotonicity_entropy,
    make_diversity_prediction_hamming,
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

def binarize_data(
    data: pd.DataFrame,
    num_estimators: int = 500,
    min_thresholds_per_feat: float = 0.0,
):
    """
    Binarize a dataset into ``feature <= v`` indicator columns.

    Thresholds are learned by a gradient-boosted stump ensemble, so they land
    where the label actually changes. A feature the ensemble rarely split on
    is then topped up with randomly chosen observed values until it has at
    least ``ceil(min_thresholds_per_feat * n_unique_values)`` thresholds, so no
    feature is left with a ladder too coarse to express a shape function.
    An intercept column of ones is prepended.

    Args:
        data: DataFrame where the last column is the target (binary 0/1).
        num_estimators: Number of GBDT estimators used to learn thresholds.
        min_thresholds_per_feat: Proportion in [0, 1] of each numeric feature's
            unique observed values that must appear as thresholds.
            Default: 0.0 (use the GBDT thresholds as-is).

    Returns:
        X: (n, 1 + total_thresholds) numpy array.  Column 0 is the intercept.
        y: (n,) numpy array with values in {0, 1}.
        feature_names: list of column name strings (length = X.shape[1]).
        feature_groups: dict mapping original feature name -> list of column
                        indices in X (useful for group-level proposals).
    """
    if not 0.0 <= min_thresholds_per_feat <= 1.0:
        raise ValueError(
            "min_thresholds_per_feat must be a proportion in [0, 1], got "
            f"{min_thresholds_per_feat}"
        )

    # 1. Supervised thresholds from the GBDT ensemble.
    df_binary, _, header, _ = binarize_dataset_gbdt(
        data, num_estimators=num_estimators,
    )
    X_binary = df_binary.iloc[:, :-1].values.astype(np.float64)
    y = df_binary.iloc[:, -1].values.astype(np.float64)
    header = list(header)

    gbdt_thresholds: Dict[str, List[float]] = {}
    for colname in header:
        if "<=" not in colname:
            continue
        feat = colname.split("<=", 1)[0].strip()
        threshold = _parse_threshold_from_colname(colname, feat)
        if threshold is not None:
            gbdt_thresholds.setdefault(feat, []).append(threshold)

    # 2. Top each feature up to its quota with random unused observed values.
    extra_columns: List[np.ndarray] = []
    extra_headers: List[str] = []
    X_raw = data.iloc[:, :-1]
    for feat in X_raw.columns:
        series = X_raw[feat]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        values = series.to_numpy(dtype=np.float64, copy=False)
        # ``feature <= max`` is constant, so it is never a usable threshold.
        candidates = np.sort(pd.unique(pd.Series(values).dropna()))[:-1]
        used = np.asarray(gbdt_thresholds.get(feat, []), dtype=np.float64)
        n_needed = (
            int(np.ceil(min_thresholds_per_feat * (len(candidates) + 1)))
            - len(used)
        )
        if n_needed <= 0 or len(candidates) == 0:
            continue

        # A GBDT threshold sits between two observed values and so produces the
        # same column as the largest observed value below it; drop those to
        # avoid adding duplicates of columns we already have.
        snapped = candidates[
            np.clip(np.searchsorted(candidates, used, side="right") - 1, 0, None)
        ]
        remaining = np.setdiff1d(candidates, snapped)

        for threshold in np.random.permutation(remaining)[:n_needed]:
            extra_columns.append((values <= threshold).astype(np.float64))
            extra_headers.append(f"{feat}<={float(threshold)}")

    if extra_columns:
        X_binary = np.column_stack([X_binary] + extra_columns)
        header.extend(extra_headers)

    X_binary, header = sort_feature_clumps(X_binary, header)

    # 3. Assemble X (intercept first) and the per-feature column groups.
    X = np.hstack([np.ones((X_binary.shape[0], 1)), X_binary])
    feature_names = ["intercept"] + header

    feature_groups: Dict[str, List[int]] = {}
    for col_idx, colname in enumerate(header):
        if "<=" in colname:
            feat = colname.split("<=", 1)[0].strip()
            feature_groups.setdefault(feat, []).append(col_idx + 1)  # +1: intercept

    return X, y, feature_names, feature_groups


def _parse_threshold_from_colname(colname: str, feat: str) -> Optional[float]:
    """Extract threshold value from column name like 'feat<=v'."""
    if f"{feat}<=" not in colname:
        return None
    try:
        return float(colname.split("<=", 1)[1].strip())
    except (ValueError, IndexError):
        return None


@dataclass
class RashomonResult:
    """Results from MCMC Rashomon set sampling."""
    w_models: np.ndarray
    w_opt: np.ndarray
    support_sets: List[List[int]]
    errors: np.ndarray
    best_error: float
    error_bound: float
    eps: float
    n_explored: int
    n_accepted: int
    runtime: float
    acceptance_rate: float
    w_mcmc: Optional[np.ndarray] = None  # MAP models only (when ellipsoid_augment)
    w_ellipsoid: Optional[np.ndarray] = None  # Ellipsoid samples only
    ellipsoid_by_support: Optional[Dict[tuple, List[np.ndarray]]] = None  # support_key -> ellipsoid samples
    log_alpha_history: Optional[List[Dict[str, float]]] = None  # Per-step terms when mh_variant="repulsive"
    best_nll: Optional[float] = None  # Best NLL found so far
    nlls: Optional[np.ndarray] = None  # NLLs for returned Rashomon-set models
    all_nlls: Optional[np.ndarray] = None  # NLLs for all explored candidate models
    nll_bound: Optional[float] = None  # NLL threshold used to filter Rashomon-set models
    feature_names: Optional[List[str]] = None
    feature_groups: Optional[Dict[str, List[int]]] = None
    w_models_binned: Optional[np.ndarray] = None  # disjoint-bin weights (see convert_gams_to_binned_representation)
    w_opt_binned: Optional[np.ndarray] = None
    bin_labels: Optional[Dict[str, List[str]]] = None
    feature_names_binned: Optional[List[str]] = None  # column-aligned disjoint bin names
    X: Optional[np.ndarray] = None  # (n, p) cumulative ladder design matrix
    X_binned: Optional[np.ndarray] = None  # (n, p) disjoint-bin design matrix

    @staticmethod
    def _ladder_weights_to_bin_weights(ladder_w: np.ndarray) -> np.ndarray:
        """Reverse cumulative sum along the last axis: bin_i = sum(ladder_w[i:])."""
        ladder_w = np.asarray(ladder_w, dtype=np.float64)
        if ladder_w.ndim == 1:
            if ladder_w.size == 0:
                return ladder_w
            return np.cumsum(ladder_w[::-1])[::-1]
        return np.cumsum(ladder_w[..., ::-1], axis=-1)[..., ::-1]

    @staticmethod
    def _convert_weights_to_binned(
        weights: np.ndarray,
        feature_names: List[str],
        feature_groups: Dict[str, List[int]],
    ) -> np.ndarray:
        """Convert ladder weights to disjoint bin weights for a (n_models, p) matrix."""
        out = np.asarray(weights, dtype=np.float64).copy()
        if out.ndim != 2:
            raise ValueError(
                f"weights must be 2D (n_models, n_features), got shape {out.shape}"
            )
        for feat, indices in feature_groups.items():
            pairs = []
            for i in indices:
                t = _parse_threshold_from_colname(feature_names[i], feat)
                if t is not None:
                    pairs.append((i, t))
            if not pairs:
                continue
            pairs.sort(key=lambda x: x[1])
            ladder_idx = np.array([i for i, _ in pairs], dtype=int)
            out[:, ladder_idx] = RashomonResult._ladder_weights_to_bin_weights(
                out[:, ladder_idx]
            )
        return out

    @staticmethod
    def _bin_labels_for_feature(
        feat: str,
        indices: List[int],
        feature_names: List[str],
    ) -> List[str]:
        pairs = []
        for i in indices:
            t = _parse_threshold_from_colname(feature_names[i], feat)
            if t is not None:
                pairs.append((i, t))
        pairs.sort(key=lambda x: x[1])
        labels: List[str] = []
        for k, (_, t) in enumerate(pairs):
            if k == 0:
                labels.append(f"{feat} <= {t}")
            else:
                labels.append(f"{pairs[k - 1][1]} < {feat} <= {t}")
        return labels

    @staticmethod
    def _build_feature_names_binned(
        feature_names: List[str],
        feature_groups: Dict[str, List[int]],
    ) -> List[str]:
        """Flat column names aligned with ``w_*_binned`` (disjoint bins per feature)."""
        names_binned = list(feature_names)
        for feat, indices in feature_groups.items():
            labels = RashomonResult._bin_labels_for_feature(
                feat, indices, feature_names,
            )
            pairs = []
            for i in indices:
                t = _parse_threshold_from_colname(feature_names[i], feat)
                if t is not None:
                    pairs.append((i, t))
            pairs.sort(key=lambda x: x[1])
            for idx, label in zip([i for i, _ in pairs], labels):
                names_binned[idx] = label
        return names_binned

    @staticmethod
    def _build_X_binned(
        X: np.ndarray,
        feature_names: List[str],
        feature_groups: Dict[str, List[int]],
    ) -> np.ndarray:
        """Disjoint-bin design matrix aligned with binned weights (same shape as ``X``)."""
        X_b = np.asarray(X, dtype=np.float64).copy()
        for feat, indices in feature_groups.items():
            pairs = []
            for i in indices:
                t = _parse_threshold_from_colname(feature_names[i], feat)
                if t is not None:
                    pairs.append((i, t))
            if not pairs:
                continue
            pairs.sort(key=lambda x: x[1])
            ladder_idx = np.array([i for i, _ in pairs], dtype=int)
            ladder = X_b[:, ladder_idx]
            bins = np.zeros_like(ladder)
            for j in range(ladder.shape[1]):
                if j == 0:
                    bins[:, j] = ladder[:, j]
                else:
                    bins[:, j] = ladder[:, j] * (1.0 - ladder[:, j - 1])
            X_b[:, ladder_idx] = bins
        return X_b

    def convert_gams_to_binned_representation(self) -> "RashomonResult":
        """
        Convert cumulative ``feat<=t`` ladder weights to disjoint bin weights.

        Populates ``w_models_binned``, ``w_opt_binned``, ``bin_labels``,
        ``feature_names_binned``, and ``X_binned`` (when ``X`` is set) on this
        result. Uses ``self.feature_names`` and
        ``self.feature_groups``. Columns stay aligned with ``feature_names``
        (intercept and plain binary features unchanged). For ascending thresholds
        ``t1, t2, ...`` with ladder weights ``w1, w2, ...`` on ``feat<=t1``,
        ``feat<=t2``, ...::

            bin1 = w1 + w2 + ...   # feat <= t1
            bin2 = w2 + ...        # t1 < feat <= t2
            bin3 = w3 + ...        # t2 < feat <= t3
            ...

        Example: thresholds 34, 35, 36 with ladder weights 1, 2, 3 become
        bin weights 6, 5, 3.

        Returns:
            ``self``, with binned-weight attributes set.
        """
        if self.feature_names is None or self.feature_groups is None:
            raise ValueError(
                "feature_names and feature_groups must be set on RashomonResult "
                "(pass them to MCMCRashomonSampler and run sample())."
            )
        feature_names = self.feature_names
        feature_groups = self.feature_groups

        self.w_models_binned = self._convert_weights_to_binned(
            self.w_models, feature_names, feature_groups,
        )
        self.w_opt_binned = self._convert_weights_to_binned(
            self.w_opt.reshape(1, -1), feature_names, feature_groups,
        )[0]
        self.bin_labels = {
            feat: self._bin_labels_for_feature(feat, indices, feature_names)
            for feat, indices in feature_groups.items()
        }
        self.feature_names_binned = self._build_feature_names_binned(
            feature_names, feature_groups,
        )
        if self.X is not None:
            self.X_binned = self._build_X_binned(
                self.X, feature_names, feature_groups,
            )
        return self


class MCMCRashomonSampler:
    """
    MCMC sampler for the misclassification-error Rashomon set.

    Uses Metropolis-Hastings over support sets (feature subsets) with the
    Laplace-approximated marginal likelihood as the MH target distribution.
    For each sampled support set, fits L2-regularized logistic regression
    and keeps models whose 0-1 error is within (1+eps) of the best.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 0.05,
        sigma2: float = 10.0,
        p_feat: float = 0.01,
        feature_names: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[int]]] = None,
        lambda_smooth: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            X: (n, p) design matrix. Include a column of ones for an intercept.
            y: (n,) binary labels in {0, 1}.
            eps: Rashomon set tolerance.
            sigma2: Prior variance on weights (= C in sklearn LogisticRegression).
            p_feat: Prior probability of including each feature (for Laplace score).
            feature_groups: Optional dict mapping group names to column index lists.
                            Enables group-level swap proposals. For the smoothness
                            prior, columns within each group are assumed to be
                            ordered by threshold.
            feature_names: Optional list of column names aligned with ``X``
                           (index 0 is typically ``intercept``). Stored on
                           ``RashomonResult`` after ``sample()``.
            lambda_smooth: Smoothness penalty applied to the posterior score
                          (NOT to the fit). The logistic regression is still
                          fitted with the plain L2 prior via sklearn; after
                          fitting, the term
                          ``0.5 * λ * Σ_{(i,j) adj} (θ_i - θ_j)^2`` is added
                          to the negative log posterior (equivalently,
                          subtracted from ``lp_theta`` in the Laplace score).
                          Boundary thresholds whose adjacent neighbour is
                          absent from the current support contribute
                          ``0.5 * λ * θ_i^2`` (missing neighbour treated as 0).
                          Effect: MH is discouraged from accepting supports
                          whose L2-MAP fits happen to be jumpy, even when
                          those fits achieve comparable likelihood. Note that
                          the Laplace curvature (``log|H|``) is still the L2
                          Hessian, so this is not a strictly Bayesian
                          posterior — it is an L2 posterior with a smoothness
                          regulariser tacked on at scoring time, which is
                          what you want if you only care about pushing the
                          sampler toward smoother GAMs post-fit.
                          Default 0.0 (disabled).
            seed: Optional integer seed for reproducibility. When provided, seeds
                  both the ``random`` and ``numpy.random`` global RNGs so that
                  subsequent calls to ``sample`` produce deterministic results.
                  If ``None`` (default), the existing RNG state is left untouched.
        """
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.n, self.p = self.X.shape
        self.eps = eps
        self.sigma2 = sigma2
        self.p_feat = p_feat
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.feature_groups = feature_groups
        if self.feature_names is not None and len(self.feature_names) != self.p:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) must match "
                f"X.shape[1] ({self.p})"
            )
        self.lambda_smooth = float(lambda_smooth)
        self._smooth_enabled: bool = self.lambda_smooth > 0.0
        # Precompute smoothness-prior structure: ordered column indices per
        # group and the list of (i, j) adjacency pairs in global column space.
        self._smooth_adj_pairs: List[Tuple[int, int]] = []
        if feature_groups is not None:
            for _name, cols in feature_groups.items():
                cols_sorted = [int(c) for c in cols]
                for a, b in zip(cols_sorted[:-1], cols_sorted[1:]):
                    self._smooth_adj_pairs.append((int(a), int(b)))
        # Precompute |corr(feature_i, feature_j)| once for fast lookups in MH.
        self.abs_feature_corr = self.compute_feature_correlation_matrix(self.X)
        # Per-run caches populated during `sample`.
        self._mh_hessian_cache: Dict[tuple, np.ndarray] = {}
        self._rashomon_hessian_cache: Dict[tuple, np.ndarray] = {}

    def _smooth_penalty_structure(
        self, idx: Optional[List[int]]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Precompute the adjacency / boundary structure for ``idx``.

        Returns a dict with keys:
          - ``pair_local``: ``(m, 2)`` local index pairs ``(i, j)`` for adjacent
            thresholds both present in the support; contribute
            ``0.5 * λ * (θ_i - θ_j)^2``.
          - ``boundary_local``: local indices whose adjacency partner is absent
            from the support; contribute ``0.5 * λ * θ_i^2`` (missing neighbour
            is treated as 0).
        Returns ``None`` if the penalty is disabled or no structure applies.
        """
        if not self._smooth_enabled or idx is None:
            return None
        pos: Dict[int, int] = {int(c): i for i, c in enumerate(idx)}
        pair_local: List[Tuple[int, int]] = []
        boundary_local: List[int] = []
        for a, b in self._smooth_adj_pairs:
            a_in = a in pos
            b_in = b in pos
            if a_in and b_in:
                pair_local.append((pos[a], pos[b]))
            elif a_in:
                boundary_local.append(pos[a])
            elif b_in:
                boundary_local.append(pos[b])
        if not pair_local and not boundary_local:
            return None
        return {
            "pair_local": (
                np.asarray(pair_local, dtype=np.int64)
                if pair_local
                else np.zeros((0, 2), dtype=np.int64)
            ),
            "boundary_local": np.asarray(boundary_local, dtype=np.int64),
        }

    def _smooth_penalty_value(
        self, theta: np.ndarray, struct: Optional[Dict[str, np.ndarray]]
    ) -> float:
        """``0.5 * λ * [Σ (θ_i − θ_j)^2 + Σ θ_i^2]`` on the support."""
        if struct is None:
            return 0.0
        total = 0.0
        pl = struct["pair_local"]
        bl = struct["boundary_local"]
        if pl.shape[0] > 0:
            diffs = theta[pl[:, 0]] - theta[pl[:, 1]]
            total += float(np.dot(diffs, diffs))
        if bl.size > 0:
            total += float(np.dot(theta[bl], theta[bl]))
        return 0.5 * self.lambda_smooth * total

    def fit_logistic(
        self,
        X_S: np.ndarray,
        y: np.ndarray,
        sigma2: float = 10.0,
        max_iter: int = 1000,
        idx: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit L2-regularised logistic regression and return ``(theta, proba)``.

        The smoothness prior is intentionally NOT used during optimisation: it
        only appears in the posterior *score* (via ``total_nll`` and the
        ``lp_theta`` term in ``_laplace_score_from_hessian``) to discourage MH
        from accepting supports whose L2-MAP fits happen to be jumpy. ``idx``
        is accepted purely for API symmetry with callers that thread it
        through; it is not consulted here.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2", C=sigma2, fit_intercept=False,
                solver="liblinear", max_iter=max_iter,
            )
            clf.fit(X_S, y)
        return clf.coef_.ravel(), clf.predict_proba(X_S)[:, 1]

    @staticmethod
    def compute_feature_correlation_matrix(X: np.ndarray) -> np.ndarray:
        """Compute NaN-safe absolute pairwise feature correlation matrix."""
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.abs(corr).astype(np.float32, copy=False)

    @staticmethod
    def misclassification_error(proba, y):
        """0-1 misclassification rate from predicted probabilities."""
        return float(np.mean((proba >= 0.5).astype(float) != y))

    @staticmethod
    def predict_error(X, w, y):
        """Misclassification error from a full (p,) weight vector."""
        preds = (X @ w >= 0).astype(float)
        return float(np.mean(preds != y))

    def total_nll(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        sigma2: float,
        idx: Optional[List[int]] = None,
    ) -> float:
        """Total negative log-posterior (NLL + L2 prior [+ smoothness penalty]).

        The smoothness contribution is a *post-fit* penalty and is NOT used
        when optimising ``theta`` (see ``fit_logistic``). Passing ``idx`` adds
        the smoothness term; omitting it leaves the output L2-only.
        """
        _e = 1e-12
        nll = -np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        val = nll + 0.5 * np.dot(theta, theta) / sigma2
        if self._smooth_enabled and idx is not None:
            struct = self._smooth_penalty_structure(idx)
            val += self._smooth_penalty_value(theta, struct)
        return float(val)

    def _hessian_matrix(
        self,
        X_S: np.ndarray,
        proba: np.ndarray,
        sigma2: float,
        jitter: float = 0.0,
        theta: Optional[np.ndarray] = None,
        idx: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Hessian of NLL + L2 prior at ``theta``.

        The smoothness prior is deliberately excluded from the Hessian: since
        the model is fit to the L2-only posterior, ``θ̂`` is the MAP of the
        L2 posterior, and the Laplace curvature used in the score should be
        the Hessian of that same L2 posterior. The smoothness term is added
        only to ``lp_theta`` / ``total_nll`` as an additive penalty on the
        fitted weights. ``theta`` and ``idx`` are accepted for API symmetry
        with callers that thread them through.
        """
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        d = X_S.shape[1]
        H = (X_S.T * W) @ X_S + (1 / sigma2 + jitter) * np.eye(d)
        return H

    def _laplace_score_from_hessian(
        self,
        X_S: np.ndarray,
        theta: np.ndarray,
        proba: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        p_feat: float,
        K: int,
        hessian: np.ndarray,
        idx: Optional[List[int]] = None,
    ) -> float:
        """Laplace score using a precomputed Hessian matrix.

        ``hessian`` is the L2-only Hessian of the fitted logistic posterior;
        the smoothness prior does NOT enter the Hessian (the model is fit to
        the L2 posterior). When ``idx`` is given and ``lambda_smooth > 0``
        the smoothness term is subtracted from ``lp_theta`` so that MH is
        pushed away from non-smooth shape functions at fixed fit quality.
        """
        _e = 1e-12
        ll = np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        lp_theta = -0.5 * np.dot(theta, theta) / sigma2
        if self._smooth_enabled and idx is not None:
            struct = self._smooth_penalty_structure(idx)
            lp_theta -= self._smooth_penalty_value(theta, struct)
        _, logdet = np.linalg.slogdet(hessian)
        k = X_S.shape[1]
        lp_S = k * np.log(p_feat) + (K - k) * np.log(1 - p_feat)
        return float(ll + lp_theta - 0.5 * logdet + lp_S)

    def _hessian_logdet(
        self,
        X_S,
        proba,
        sigma2,
        jitter=1e-6,
        idx_new: Optional[List[int]] = None,
        theta: Optional[np.ndarray] = None,
    ):
        H = self._hessian_matrix(
            X_S, proba, sigma2, jitter=jitter,
            theta=theta, idx=idx_new,
        )
        _, logdet = np.linalg.slogdet(H)
        return logdet, H

    def laplace_score(
        self,
        X_S,
        theta,
        proba,
        y,
        sigma2,
        p_feat,
        K,
        idx_new: Optional[List[int]] = None,
        return_hessian: bool = False,
    ):
        """
        Laplace-approximated log marginal likelihood + structure prior.
        J(S) = log p(D|S,theta) + log p(theta|S) - 0.5 log|H_S| + log p(S).
        """
        _e = 1e-12
        ll = np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        lp_theta = -0.5 * np.dot(theta, theta) / sigma2
        if self._smooth_enabled and idx_new is not None:
            struct = self._smooth_penalty_structure(idx_new)
            lp_theta -= self._smooth_penalty_value(theta, struct)
        logdet, hessian = self._hessian_logdet(
            X_S,
            proba,
            sigma2,
            idx_new=idx_new,
            theta=theta,
        )
        k = X_S.shape[1]
        lp_S = k * np.log(p_feat) + (K - k) * np.log(1 - p_feat)
        sc = ll + lp_theta - 0.5 * logdet + lp_S
        if return_hessian:
            return sc, hessian
        return sc

    def score_support(self, S):
        """Fit logistic regression on support S and return (score, theta, proba)."""
        idx = sorted(S)
        return self.score_support_idx(idx)

    def score_support_idx(self, idx: List[int]):
        """Fit logistic regression on a precomputed support index list."""
        sc, theta, proba, _ = self.score_support_idx_with_hessian(idx)
        return sc, theta, proba

    def score_support_idx_with_hessian(self, idx: List[int]):
        """Like score_support_idx but also returns the Hessian matrix."""
        X_S = self.X[:, idx]
        theta, proba = self.fit_logistic(X_S, self.y, self.sigma2, idx=idx)
        sc, hessian = self.laplace_score(
            X_S, theta, proba, self.y, self.sigma2, self.p_feat, self.p,
            idx_new=idx, return_hessian=True,
        )
        return sc, theta, proba, hessian

    def _fit_logistic_warm_start(
        self,
        X_S: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        x0: np.ndarray,
        max_iter: int = 500,
        idx: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit L2-regularized logistic regression with warm-started feature coefficients.

        Uses ``LogisticRegression`` with ``warm_start=True``, same hyperparameters as
        ``fit_logistic``. Initializes ``coef_`` from ``x0`` and ``intercept_`` to zero
        before ``fit`` so sag starts from the warm start (sklearn appends intercept
        to the coefficient vector internally when optimizing).

        ``idx`` is accepted for API symmetry; the smoothness prior is not used
        during optimisation (see ``fit_logistic``'s docstring).
        """
        _ = idx  # unused by design; kept for signature symmetry
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        n_features = X_S.shape[1]
        if x0.shape[0] != n_features:
            raise ValueError(f"x0 length {x0.shape[0]} != n_features {n_features}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2",
                C=sigma2,
                fit_intercept=False,
                solver="liblinear",
                max_iter=max_iter,
                warm_start=True,
            )
            clf.coef_ = x0.reshape(1, n_features)
            clf.fit(X_S, y)
        theta = clf.coef_.ravel()
        proba = clf.predict_proba(X_S)[:, 1]
        return theta, proba

    def _score_support_warm_start_add_remove(
        self,
        S: set,
        S_p: set,
        theta_cur: np.ndarray,
        idx_cur: List[int],
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Warm-start fit for add or remove (exactly one feature changed).
        - Delete: x0 = current weights (excluding removed feature).
        - Add: x0 = current weights + weight of most correlated existing feature.
        """
        added = S_p - S
        removed = S - S_p
        if len(added) == 0 and len(removed) == 1:
            # Delete one feature
            idx_p = sorted(S_p)
            feat_to_coef_dict = dict(zip(idx_cur, theta_cur))
            x0 = np.array([feat_to_coef_dict.get(j, 0.0) for j in idx_p], dtype=np.float64)
        elif len(added) == 1 and len(removed) == 0:
            # Add one feature: use weight of most correlated existing feature
            f_add = next(iter(added))
            feat_to_coef = dict(zip(idx_cur, theta_cur))
            best_w = 0.0
            max_corr = 0.0
            for f in S:
                corr = float(self.abs_feature_corr[f_add, f])
                if corr > max_corr:
                    max_corr = corr
                    best_w = feat_to_coef.get(f, 0.0)
            idx_p = sorted(S_p)
            x0 = np.zeros(len(idx_p), dtype=np.float64)
            for i, j in enumerate(idx_p):
                if j in S:
                    x0[i] = theta_cur[idx_cur.index(j)]
                else:
                    x0[i] = best_w
        else:
            return None

        X_Sp = self.X[:, idx_p]
        theta_p, proba_p = self._fit_logistic_warm_start(
            X_Sp, self.y, self.sigma2, x0, idx=idx_p,
        )
        sc = self.laplace_score(
            X_Sp, theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p,
            idx_new=idx_p,
        )
        return sc, theta_p, proba_p

    def _score_support_finetune_swap(
        self,
        S: set,
        S_p: set,
        theta_cur: np.ndarray,
        idx_cur: List[int],
        corr_threshold: float,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Fast approximation for k-swap proposals (k=1,2,3) when swapped features
        are correlated. Fit only the incoming features' weights; keep others fixed.

        Finetune is applied only if for each incoming feature f_add, there exists
        at least one outgoing feature f_rem with |corr(f_add, f_rem)| >= tau.

        Returns (sc, theta_p, proba_p) or None if finetune cannot be applied.
        """
        added = S_p - S
        removed = S - S_p
        k = len(added)
        if k == 0 or k != len(removed) or k > 3:
            return None

        added_list = sorted(added)
        removed_list = sorted(removed)

        # Correlation check: each incoming feature must have corr >= tau with
        # at least one outgoing feature
        X_add = self.X[:, added_list].astype(np.float64)
        for f_add in added_list:
            max_corr = float(np.max(self.abs_feature_corr[f_add, removed_list]))
            if max_corr < corr_threshold:
                return None

        feat_to_coef = dict(zip(idx_cur, theta_cur))
        fixed_outcome = self.X[:, idx_cur] @ theta_cur
        for f_rem in removed_list:
            w_rem = feat_to_coef.get(f_rem, 0.0)
            fixed_outcome = fixed_outcome - self.X[:, f_rem] * w_rem
        prior_sum_sq = np.dot(theta_cur, theta_cur) - sum(
            feat_to_coef.get(f, 0.0) ** 2 for f in removed_list
        )
        _e = 1e-12

        def objective(theta_add_arr: np.ndarray):
            theta_add = theta_add_arr.astype(np.float64)
            logits = np.clip(fixed_outcome + X_add @ theta_add, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            nll = -np.sum(
                self.y * np.log(p + _e) + (1 - self.y) * np.log(1 - p + _e)
            )
            prior = 0.5 * (prior_sum_sq + np.dot(theta_add, theta_add)) / self.sigma2
            fval = nll + prior
            grad_nll = X_add.T @ (p - self.y)
            grad_prior = theta_add / self.sigma2
            grad = grad_nll + grad_prior
            return fval, grad

        x0 = np.array([feat_to_coef.get(f, 0.0) for f in removed_list], dtype=np.float64)
        result = minimize(
            objective,
            x0=x0,
            method="BFGS",
            jac=True,
            options={"maxiter": 50})
        theta_add_new = result.x.astype(np.float64)

        idx_p = sorted(S_p)
        add_to_idx = {f: i for i, f in enumerate(added_list)}
        theta_p = np.zeros(len(idx_p))
        for i, j in enumerate(idx_p):
            if j in added_list:
                theta_p[i] = theta_add_new[add_to_idx[j]]
            elif j in S:
                theta_p[i] = feat_to_coef.get(j, 0.0)

        X_Sp = self.X[:, idx_p]
        logits = np.clip(X_Sp @ theta_p, -20.0, 20.0)
        proba_p = 1.0 / (1.0 + np.exp(-logits))

        hessian_p = self._hessian_matrix(
            X_Sp, proba_p, self.sigma2, jitter=1e-6,
            theta=theta_p, idx=idx_p,
        )
        sc_p = self._laplace_score_from_hessian(
            X_Sp, theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p,
            idx=idx_p,
        )
        return sc_p, theta_p, proba_p, hessian_p

    def _score_support_add_remove_support(
        self,
        S: set,
        S_p: set,
        idx_cur: List[int],
        theta_cur: np.ndarray,
        corr_threshold: float = 0.3,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Score support with add or remove (exactly one feature changed).
        """
        added = S_p - S
        removed = S - S_p
        idx_p = sorted(S_p)
        if len(added) == 1 and len(removed) == 0:
            feature_to_add = list(added)[0]
            corr_vector = self.abs_feature_corr[feature_to_add, idx_cur].astype(np.float64, copy=False)
            max_corr = np.max(corr_vector)
            if max_corr > corr_threshold:
                return None
            fixed_outcome = self.X[:, idx_cur] @ theta_cur
            X_add = self.X[:, feature_to_add].astype(np.float64)
            theta_add_init = np.random.randn(1)
            _e = 1e-12
            prior_sum_sq = np.dot(theta_cur, theta_cur)
            def objective(theta_add: np.ndarray):
                logits = np.clip(fixed_outcome + X_add * theta_add, -20.0, 20.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                nll = -np.sum(
                    self.y * np.log(p + _e) + (1 - self.y) * np.log(1 - p + _e)
                )
                prior = 0.5 * (prior_sum_sq + np.dot(theta_add, theta_add)) / self.sigma2
                fval = nll + prior
                grad_nll = X_add @ (p - self.y)
                grad_prior = theta_add / self.sigma2
                grad = grad_nll + grad_prior
                return fval, grad
            result = minimize(
                objective,
                x0=theta_add_init,
                method="BFGS",
                jac=True,
                options={"maxiter": 50},
            )
            theta_add_new = float(np.asarray(result.x, dtype=np.float64).reshape(-1)[0])
            theta_p = np.zeros(len(idx_p))
            feat_to_coef_p = {f:i for i, f in enumerate(idx_cur)}
            for i, f in enumerate(idx_p):
                if f in S:
                    theta_p[i] = theta_cur[feat_to_coef_p[f]]
                else:
                    theta_p[i] = theta_add_new

            logits = np.clip(self.X[:, idx_p] @ theta_p, -20.0, 20.0)
            proba_p = 1.0 / (1.0 + np.exp(-logits))
            hessian_p = self._hessian_matrix(
                self.X[:, idx_p], proba_p, self.sigma2, jitter=1e-6,
                theta=theta_p, idx=idx_p,
            )
            sc_p = self._laplace_score_from_hessian(
                self.X[:, idx_p], theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p,
                idx=idx_p,
            )
            return sc_p, theta_p, proba_p, hessian_p

        elif len(added) == 0 and len(removed) == 1:
            feature_to_remove = list(removed)[0]
            remaining_features = [f for f in idx_cur if f != feature_to_remove]
            corr_vector = self.abs_feature_corr[feature_to_remove, remaining_features].astype(np.float64, copy=False)
            max_corr = np.max(corr_vector)
            if max_corr > corr_threshold:
                return None
            k = 3
            k_eff = min(k, len(remaining_features))
            top_k_linear = np.argsort(corr_vector)[::-1][:k_eff]
            top_k_correlated_features = sorted(
                remaining_features[i] for i in top_k_linear
            )
            remaining_features_fixed = [
                i for i in remaining_features if i not in top_k_correlated_features
            ]
            feat_to_coef_cur = {f:i for i, f in enumerate(idx_cur)}
            theta_rf = theta_cur[
                np.array([feat_to_coef_cur[f] for f in remaining_features_fixed], dtype=np.intp)
            ]
            fixed_outcome = self.X[:, remaining_features_fixed] @ theta_rf
            theta_modified_init = np.random.randn(k_eff)
            _e = 1e-12
            prior_sum_sq = np.dot(theta_rf, theta_rf)
            X_topk = self.X[:, top_k_correlated_features]
            def objective(theta_modified: np.ndarray):
                logits = np.clip(fixed_outcome + X_topk @ theta_modified, -20.0, 20.0)
                p = 1.0 / (1.0 + np.exp(-logits))
                nll = -np.sum(
                    self.y * np.log(p + _e) + (1 - self.y) * np.log(1 - p + _e)
                )
                prior = 0.5 * (prior_sum_sq + np.dot(theta_modified, theta_modified)) / self.sigma2
                fval = nll + prior
                grad_nll = X_topk.T @ (p - self.y)
                grad_prior = theta_modified / self.sigma2
                grad = grad_nll + grad_prior
                return fval, grad
            result = minimize(
                objective,
                x0=theta_modified_init,
                method="BFGS",
                jac=True,
                options={"maxiter": 50},
            )
            pos_in_cur = {f: i for i, f in enumerate(idx_cur)}
            theta_modified_new = result.x.astype(np.float64)
            theta_from_topk = {
                f: theta_modified_new[i] for i, f in enumerate(top_k_correlated_features)
            }
            theta_p = np.empty(len(idx_p), dtype=np.float64)
            for i, f in enumerate(idx_p):
                theta_p[i] = theta_from_topk.get(f, theta_cur[pos_in_cur[f]])
            logits = np.clip(self.X[:, idx_p] @ theta_p, -20.0, 20.0)
            proba_p = 1.0 / (1.0 + np.exp(-logits))
            hessian_p = self._hessian_matrix(
                self.X[:, idx_p], proba_p, self.sigma2, jitter=1e-6,
                theta=theta_p, idx=idx_p,
            )
            sc_p = self._laplace_score_from_hessian(
                self.X[:, idx_p], theta_p, proba_p, self.y, self.sigma2, self.p_feat, self.p, hessian_p,
                idx=idx_p,
            )
            return sc_p, theta_p, proba_p, hessian_p
        else:
            return None

    def score_support_with_error(self, S):
        """Like score_support but also returns misclassification error."""
        idx = sorted(S)
        X_S = self.X[:, idx]
        theta, proba = self.fit_logistic(X_S, self.y, self.sigma2, idx=idx)
        sc = self.laplace_score(
            X_S, theta, proba, self.y, self.sigma2, self.p_feat, self.p,
            idx_new=idx,
        )
        err = self.misclassification_error(proba, self.y)
        return sc, theta, proba, err

    propose_swap = staticmethod(propose_swap)
    propose_multi_swap = staticmethod(propose_multi_swap)
    propose_add_or_remove = staticmethod(propose_add_or_remove)
    propose_weighted_swap = staticmethod(propose_weighted_swap)
    propose_swap_or_resize = staticmethod(propose_swap_or_resize)
    propose_promise_score_mixture = staticmethod(propose_promise_score_mixture)
    make_mixture_proposal = staticmethod(make_mixture_proposal)
    compute_correlation_scores = staticmethod(compute_correlation_scores)

    def propose_feature_group_swap(self, S, K, score):
        return propose_feature_group_swap(
            S,
            K,
            score,
            feature_groups=self.feature_groups,
            fallback_proposal=self.propose_swap,
        )

    def mh_sample(
        self,
        init_support: List[int],
        n_steps: int = 5000,
        burn_in: int = 1000,
        skip: int = 80,
        beta: float = 1.0,
        proposal_fn: Optional[Callable] = None,
        finetune_coordinate: bool = False,
        finetune_corr_threshold: float = 0.3,
    ) -> Tuple[List[List[int]], List[float], float, List[np.ndarray], int, int]:
        """
        Standard MH over support sets.

        Returns (support_sets, scores, acceptance_rate, collected_w, n_finetune, n_full).
        Collects unique support sets at evenly spaced intervals after burn-in.
        collected_w[i] is the MAP weight vector (full p-vector) for support_sets[i].
        """
        if proposal_fn is None:
            proposal_fn = self.propose_swap

        S = set(init_support)
        idx_cur = sorted(S)
        sc, theta_cur, proba_cur, hessian_cur = self.score_support_idx_with_hessian(idx_cur)

        skip = max(1, int(skip))
        n_acc = 0
        n_finetune, n_full = 0, 0
        results, result_scores, result_thetas = [], [], []
        result_hessians: Dict[tuple, np.ndarray] = {}
        seen = set()

        for t in tqdm(range(n_steps), desc="MH sampling"):
            S_p, lq_f, lq_b = proposal_fn(S, self.p, sc)
            if not S_p:
                continue
            idx_p = sorted(S_p)
            # try:
            hessian_p = None
            if finetune_coordinate:
                if len(S_p) == len(S): # swapping
                    # t0 = time.perf_counter()
                    t0 = time.perf_counter()
                    finetune_result_swap = self._score_support_finetune_swap(
                        S,
                        S_p,
                        theta_cur,
                        idx_cur,
                        finetune_corr_threshold,
                    )
                    time_taken = time.perf_counter() - t0
                    
                    if finetune_result_swap is not None:
                        # print(f"Time taken for finetune_swap: {time_taken:.4f} seconds")
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_swap
                        n_finetune += 1
                    else:
                        t0 = time.perf_counter()
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        time_taken = time.perf_counter() - t0
                        # print(f"Time taken for full finetune: {time_taken:.4f} seconds")
                        n_full += 1
                else:
                    t0 = time.perf_counter()
                    finetune_result_add_remove = self._score_support_add_remove_support(
                        S,
                        S_p,
                        idx_cur,
                        theta_cur,
                        finetune_corr_threshold,
                    )
                    time_taken = time.perf_counter() - t0
                    # print(f"Time taken for finetune_add_remove: {time_taken:.4f} seconds")
                    if finetune_result_add_remove is not None:
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_add_remove
                        n_finetune += 1
                    else:
                        t0 = time.perf_counter()
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        time_taken = time.perf_counter() - t0
                        # print(f"Time taken for full finetune: {time_taken:.4f} seconds")
                        n_full += 1
            else:
                t0 = time.perf_counter()
                sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                time_taken = time.perf_counter() - t0
                # print(f"Time taken for full finetune: {time_taken:.4f} seconds")
                n_full += 1

            log_a = beta * (sc_p - sc) + lq_b - lq_f
            accepted = np.log(np.random.random() + 1e-300) < log_a
            if hasattr(proposal_fn, "observe"):
                try:
                    proposal_fn.observe(float(sc_p - sc), bool(accepted))
                except Exception:
                    pass
            if accepted:
                S, sc = S_p, sc_p
                theta_cur, idx_cur = theta_p, idx_p
                proba_cur = proba_p
                if hessian_p is None:
                    hessian_cur = self._hessian_matrix(
                        self.X[:, idx_cur], proba_cur, self.sigma2,
                        theta=theta_cur, idx=idx_cur,
                    )
                else:
                    hessian_cur = hessian_p
                n_acc += 1

            if t >= burn_in and (t - burn_in) % skip == 0:
                key = tuple(idx_cur)
                _e = 1e-12
                _ll = float(np.sum(
                    self.y * np.log(proba_cur + _e)
                    + (1 - self.y) * np.log(1 - proba_cur + _e)
                ))
                _lp_theta_l2 = float(
                    -0.5 * np.dot(theta_cur, theta_cur) / self.sigma2
                )
                _smooth_pen = (
                    float(self._smooth_penalty_value(
                        theta_cur,
                        self._smooth_penalty_structure(idx_cur),
                    ))
                    if self._smooth_enabled
                    else 0.0
                )
                _lp_theta = _lp_theta_l2 - _smooth_pen
                _, _logdet = np.linalg.slogdet(hessian_cur)
                _curv = float(-0.5 * _logdet)
                _k = len(idx_cur)
                _lp_S = float(
                    _k * np.log(self.p_feat)
                    + (self.p - _k) * np.log(1 - self.p_feat)
                )
                _total = _ll + _lp_theta + _curv + _lp_S
                if key not in seen:
                    seen.add(key)
                    results.append(idx_cur.copy())
                    result_scores.append(sc)
                    w = np.zeros(self.p)
                    w[idx_cur] = theta_cur
                    result_thetas.append(w)
                    result_hessians[key] = hessian_cur.copy()

        if hasattr(proposal_fn, "probs"):
            try:
                print(f"Final adaptive proposal probs: {proposal_fn.probs}")
            except Exception:
                pass
        return results, result_scores, n_acc / max(n_steps, 1), result_thetas, n_finetune, n_full

    def mh_sample_repulsive(
        self,
        init_support: List[int],
        n_steps: int = 5000,
        burn_in: int = 1000,
        skip: int = 80,
        beta: float = 1.0, 
        proposal_fn: Optional[Callable] = None,
        repulsion_weight: float = 1.0,
        diversity_fn: Optional[Union[str, Callable[[List[np.ndarray], np.ndarray], float]]] = None,
        ellipsoid_augment: bool = False,
        ellipsoid_n_samples: int = 50,
        repulsion_reference: str = "map_plus_ellipsoid",
        rashomon_nll_bound: Optional[float] = None,
        finetune_coordinate: bool = False,
        finetune_corr_threshold: float = 0.25,
        max_repulsion_refs: int = 200,
    ) -> Tuple[
        List[List[int]],
        List[float],
        float,
        List[np.ndarray],
        List[Dict[str, float]],
        int,
        int,
        Dict[Tuple[int, ...], Dict[str, np.ndarray]],
    ]:
        """
        Diversity-seeking MH over support sets.

        Like ``mh_sample`` but the acceptance probability includes a repulsion
        term that favours proposals far from already collected models::

            log alpha = beta * (score_proposed - score_current)
                      + (log_q_bwd - log_q_fwd)
                      + repulsion_weight * (log d_prop - log d_cur)

        where d = diversity(accepted_samples, sample). Higher diversity value =
        more diverse. Before any samples are collected (during burn-in) the
        repulsion term is zero.

        Returns (support_sets, scores, acceptance_rate, collected_w, log_alpha_history, n_finetune, n_full).
        collected_w[i] is the MAP weight vector for support_sets[i].
        log_alpha_history is a list of dicts per step with keys: term_score,
        term_proposal, term_repulsion, log_alpha, accepted, d_prop, d_cur.

        Args:
            repulsion_weight: Strength of the diversity bonus (0 = standard MH).
            diversity_fn: Callable (accepted_samples, candidate) -> float;
                higher = more diverse. May also be a name: "l1", "jaccard",
                "prediction_hamming" or "monotonicity_entropy" (see
                gram.diversity.DIVERSITY_FN_BUILDERS).
                Default: diversity_support_jaccard.
            ellipsoid_augment: If True, compute repulsion diversity as the mean
                diversity over ellipsoid samples around candidate centers.
            ellipsoid_n_samples: Number of ellipsoid samples used for repulsion
                diversity averaging when ``ellipsoid_augment=True``.
            repulsion_reference: Controls the repulsion reference set / candidate
                representation when ``ellipsoid_augment=True``.

                - ``"map_only"``: Always compare MAPs to MAPs. The reference
                  pool is just ``collected_w`` (MAP weights of previously
                  accepted supports) and the candidate is its own MAP. No
                  ellipsoid sampling is done on either side.
                - ``"map_plus_ellipsoid"`` (default): Compare ellipsoid cloud
                  to ellipsoid cloud. The reference pool is the union of
                  collected MAPs and every ellipsoid sample drawn around
                  previously accepted supports, and the candidate side is
                  represented by an ellipsoid cloud around its MAP (mean
                  diversity over the cloud).

                When ``ellipsoid_augment=False`` this flag is ignored and the
                repulsion reduces to plain MAP-vs-MAPs. The reference pool is
                capped at ``max_repulsion_refs`` samples via uniform
                subsampling.
            rashomon_nll_bound: Absolute NLL bound used to size the ellipsoids
                drawn at accept time. If provided, each ellipsoid uses
                ``ub = max(rashomon_nll_bound - nll_center, 1e-6)``, matching
                the quadratic-approx Rashomon region. If ``None``, falls back
                to the heuristic ``ub = max(1e-6, 0.05 * nll_center)``.
        """
        if proposal_fn is None:
            proposal_fn = self.propose_swap
        if diversity_fn is None:
            diversity_fn = DEFAULT_DIVERSITY_FN
        elif isinstance(diversity_fn, str):
            diversity_fn = resolve_diversity_fn(
                diversity_fn, self.X, self.feature_groups
            )
        if repulsion_reference not in ("map_only", "map_plus_ellipsoid"):
            raise ValueError(
                "repulsion_reference must be 'map_only' or 'map_plus_ellipsoid'; "
                f"got {repulsion_reference!r}"
            )

        S = set(init_support)
        idx_cur = sorted(S)
        sc_cur, theta_cur, proba_cur, hessian_cur = self.score_support_idx_with_hessian(idx_cur)
        w_cur = np.zeros(self.p)
        w_cur[idx_cur] = theta_cur

        skip = max(1, int(skip))
        n_acc = 0
        n_finetune, n_full = 0, 0
        results, result_scores = [], []
        result_hessians: Dict[tuple, np.ndarray] = {}
        seen = set()
        collected_w: List[np.ndarray] = []
        collected_ellipsoid_samples: List[np.ndarray] = []
        collected_ellipsoid_by_support: Dict[Tuple[int, ...], Dict[str, np.ndarray]] = {}
        log_alpha_history: List[Dict[str, float]] = []
        repulsion_ellipsoid_n = max(1, int(ellipsoid_n_samples))
        max_repulsion_refs = max(1, int(max_repulsion_refs))
        use_ellipsoid_repulsion = bool(ellipsoid_augment) and repulsion_reference == "map_plus_ellipsoid"

        def _draw_ellipsoid_cloud(
            center_idx: List[int],
            center_theta: np.ndarray,
            center_proba: np.ndarray,
            center_hessian: Optional[np.ndarray],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Draw an ellipsoid cloud around a center.

            Returns (w_samples_full, errors, nlls) where w_samples_full has
            shape (k, p), and errors/nlls are per-sample evaluated on the full
            training data. The ellipsoid radius uses the Rashomon NLL bound
            when provided (via ``rashomon_nll_bound``), otherwise falls back
            to the heuristic ``0.05 * nll_center``.
            """
            hessian_use = center_hessian
            if hessian_use is None:
                hessian_use = self._hessian_matrix(
                    self.X[:, center_idx], center_proba, self.sigma2, jitter=1e-6,
                    theta=center_theta, idx=center_idx,
                )
            eigvals, eigvecs = self._hessian_eig_from_matrix(hessian_use)
            # Ellipsoid radii use the data + L2 objective only. Smoothness is
            # only a support-scoring prior in the MH/Laplace score.
            nll_center = self.total_nll(
                center_proba, self.y, center_theta, self.sigma2, idx=None,
            )
            if rashomon_nll_bound is not None:
                ub = max(1e-6, float(rashomon_nll_bound) - float(nll_center))
            else:
                ub = max(1e-6, 0.05 * float(nll_center))
            theta_samples, errors, nlls = self._sample_ellipsoid(
                theta_center=center_theta,
                X_S=self.X[:, center_idx],
                y=self.y,
                sigma2=self.sigma2,
                n_samples=repulsion_ellipsoid_n,
                ub=ub,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            if theta_samples.shape[0] == 0:
                return (
                    np.zeros((0, self.p), dtype=np.float64),
                    np.zeros((0,), dtype=np.float64),
                    np.zeros((0,), dtype=np.float64),
                )
            w_samples_full = np.zeros((theta_samples.shape[0], self.p), dtype=np.float64)
            w_samples_full[:, center_idx] = theta_samples
            return w_samples_full, np.asarray(errors, dtype=np.float64), np.asarray(nlls, dtype=np.float64)

        def _subsample_refs(pool: List[np.ndarray]) -> List[np.ndarray]:
            if len(pool) > max_repulsion_refs:
                sampled_idx = np.random.choice(
                    len(pool), size=max_repulsion_refs, replace=False
                )
                return [pool[int(i)] for i in sampled_idx]
            return pool

        def _mean_repulsion_diversity(
            center_w: np.ndarray,
            center_idx: List[int],
            center_theta: np.ndarray,
            center_proba: np.ndarray,
            center_hessian: Optional[np.ndarray],
        ) -> float:
            if not use_ellipsoid_repulsion:
                # MAPs vs MAPs, capped at max_repulsion_refs
                repulsion_refs = _subsample_refs(collected_w)
                return float(diversity_fn(repulsion_refs, center_w))

            # Ellipsoid cloud vs (MAPs + previously collected ellipsoid samples),
            # with the reference pool capped at max_repulsion_refs total samples.
            pool = collected_w + collected_ellipsoid_samples
            repulsion_refs = _subsample_refs(pool)

            w_samples_full, _, _ = _draw_ellipsoid_cloud(
                center_idx=center_idx,
                center_theta=center_theta,
                center_proba=center_proba,
                center_hessian=center_hessian,
            )
            if w_samples_full.shape[0] == 0:
                return float(diversity_fn(repulsion_refs, center_w))
            d_vals = [float(diversity_fn(repulsion_refs, w_s)) for w_s in w_samples_full]
            return float(np.mean(d_vals))

        for t in tqdm(range(n_steps), desc="MH repulsive"):
            S_p, lq_f, lq_b = proposal_fn(S, self.p, sc_cur)
            if not S_p:
                continue
            idx_p = sorted(S_p)
            hessian_p = None
            if finetune_coordinate:
                if len(S_p) == len(S): # swapping
                    finetune_result_swap = self._score_support_finetune_swap(
                            S,
                            S_p,
                            theta_cur,
                            idx_cur,
                            finetune_corr_threshold,
                    )
                    if finetune_result_swap is not None:
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_swap
                        n_finetune += 1
                    else:
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        n_full += 1
                else:
                    finetune_result_add_remove = self._score_support_add_remove_support(
                        S,
                        S_p,
                        idx_cur,
                        theta_cur,
                        finetune_corr_threshold,
                    )
                    if finetune_result_add_remove is not None:
                        sc_p, theta_p, proba_p, hessian_p = finetune_result_add_remove
                        n_finetune += 1
                    else:
                        sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                        n_full += 1
            else:
                sc_p, theta_p, proba_p, hessian_p = self.score_support_idx_with_hessian(idx_p)
                n_full += 1

            w_prop = np.zeros(self.p)
            w_prop[idx_p] = theta_p

            # Standard MH ratio
            term_score = beta * (sc_p - sc_cur)
            term_proposal = lq_b - lq_f
            log_a = term_score + term_proposal

            # Repulsion bonus
            term_repulsion = 0.0
            d_prop, d_cur = float("nan"), float("nan")
            if collected_w:
                d_prop = _mean_repulsion_diversity(
                    center_w=w_prop,
                    center_idx=idx_p,
                    center_theta=theta_p,
                    center_proba=proba_p,
                    center_hessian=hessian_p,
                )
                d_cur = _mean_repulsion_diversity(
                    center_w=w_cur,
                    center_idx=idx_cur,
                    center_theta=theta_cur,
                    center_proba=proba_cur,
                    center_hessian=hessian_cur,
                )
                term_repulsion = repulsion_weight * self.n * (
                    np.log(d_prop + 1e-12) - np.log(d_cur + 1e-12)
                )
                log_a += term_repulsion

            u = np.random.random()
            accepted = np.log(u + 1e-300) < log_a
            if accepted:
                S, sc_cur = S_p, sc_p
                theta_cur, idx_cur = theta_p, idx_p
                proba_cur = proba_p
                if hessian_p is None:
                    hessian_cur = self._hessian_matrix(
                        self.X[:, idx_cur], proba_cur, self.sigma2,
                        theta=theta_cur, idx=idx_cur,
                    )
                else:
                    hessian_cur = hessian_p
                w_cur = w_prop
                n_acc += 1

            if t >= burn_in and (t - burn_in) % skip == 0:
                key = tuple(idx_cur)
                if key not in seen:
                    seen.add(key)
                    results.append(idx_cur.copy())
                    result_scores.append(sc_cur)
                    collected_w.append(w_cur.copy())
                    result_hessians[key] = hessian_cur.copy()
                    if use_ellipsoid_repulsion:
                        cloud_w, cloud_err, cloud_nll = _draw_ellipsoid_cloud(
                            center_idx=idx_cur,
                            center_theta=theta_cur,
                            center_proba=proba_cur,
                            center_hessian=hessian_cur,
                        )
                        if cloud_w.shape[0] > 0:
                            collected_ellipsoid_by_support[key] = {
                                "w_samples": cloud_w,
                                "errors": cloud_err,
                                "nlls": cloud_nll,
                            }
                            for i in range(cloud_w.shape[0]):
                                collected_ellipsoid_samples.append(cloud_w[i])

            log_alpha_history.append({
                "term_score": float(term_score),
                "term_proposal": float(term_proposal),
                "term_repulsion": float(term_repulsion),
                "log_alpha": float(log_a),
                "accepted": accepted,
                "d_prop": float(d_prop),
                "d_cur": float(d_cur),
            })

        # Summary of log_alpha terms (helps debug repulsion vs score scale)
        if log_alpha_history:
            hist = log_alpha_history
            ts = [h["term_score"] for h in hist]
            tp = [h["term_proposal"] for h in hist]
            tr = [h["term_repulsion"] for h in hist]
            la = [h["log_alpha"] for h in hist]
            print(
                "log_alpha terms (mean ± std): "
                f"term_score={np.mean(ts):.3f}±{np.std(ts):.3f}, "
                f"term_proposal={np.mean(tp):.3f}±{np.std(tp):.3f}, "
                f"term_repulsion={np.mean(tr):.3f}±{np.std(tr):.3f}, "
                f"log_alpha={np.mean(la):.3f}±{np.std(la):.3f}"
            )

        self._mh_hessian_cache = result_hessians
        return (
            results,
            result_scores,
            n_acc / max(n_steps, 1),
            collected_w,
            log_alpha_history,
            n_finetune,
            n_full,
            collected_ellipsoid_by_support,
        )

    def random_sample(self, support_size, n_samples=50):
        """Uniformly random support sets (baseline comparison)."""
        results, scores = [], []
        seen = set()
        for _ in range(n_samples * 20):
            if len(results) >= n_samples:
                break
            S = sorted(np.random.choice(self.p, support_size, replace=False).tolist())
            key = tuple(S)
            if key in seen:
                continue
            seen.add(key)
            try:
                sc, _, _ = self.score_support(S)
                results.append(S)
                scores.append(sc)
            except Exception:
                pass
        return results, scores

    def find_initial_support(self, c_value=2):
        """Find a sparse starting support via L1-regularized logistic regression."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l1", C=c_value, solver="liblinear",
                fit_intercept=True, max_iter=5000,
            )
            clf.fit(self.X, self.y)
        support = np.nonzero(clf.coef_.ravel())[0].tolist()
        if not support:
            corr = self.compute_correlation_scores(self.X, self.y)
            support = np.argsort(corr)[-5:].tolist()
        return support

    @staticmethod
    def _hessian_eig(X_S, proba, sigma2):
        """Compute Hessian eigendecomposition for ellipsoid sampling.

        This legacy path uses the isotropic L2 Hessian only (no fused-lasso
        contribution); callers that need the penalised Hessian should build it
        via ``_hessian_matrix`` on a sampler instance and then pass it through
        ``_hessian_eig_from_matrix``.
        """
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        d = X_S.shape[1]
        H = (X_S.T * W) @ X_S + (1.0 / sigma2) * np.eye(d)
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    @staticmethod
    def _hessian_eig_from_matrix(H):
        """Eigendecomposition from a precomputed Hessian matrix."""
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    @staticmethod
    def _sample_ellipsoid(theta_center, X_S, y, sigma2, n_samples, ub,
                         eigvals=None, eigvecs=None):
        """
        Sample n_samples weight vectors from the Hessian-defined ellipsoid
        around theta_center with bound ub, and return them with their
        misclassification errors.

        When eigvals, eigvecs are provided (from _hessian_eig), skips Hessian
        computation to avoid redundant work.

        Returns (theta_samples, errors, nlls) where theta_samples is (n_samples, d),
        errors is (n_samples,), nlls is (n_samples,) for Rashomon filtering.
        """
        if n_samples <= 0 or ub <= 0:
            return np.zeros((0, len(theta_center))), np.array([]), np.array([])

        d = len(theta_center)
        if eigvals is not None and eigvecs is not None:
            pass  # use provided
        else:
            proba_center = 1.0 / (1.0 + np.exp(-np.clip(X_S @ theta_center, -20, 20)))
            eigvals, eigvecs = MCMCRashomonSampler._hessian_eig(
                X_S, proba_center, sigma2
            )
        a = np.sqrt(2 * ub / np.maximum(eigvals, 1e-12))

        u = np.random.randn(n_samples, d)
        u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
        r = np.random.random(n_samples) ** (1.0 / d)
        z = u * r[:, None]  # uniform in unit ball

        # Rotate then scale using eigvecs as columns
        deltas = (eigvecs * a) @ z.T  # shape (d, n_samples)
        theta_samples = theta_center[:, None] + deltas  # (d, n_samples)
        theta_samples = theta_samples.T  # (n_samples, d)

        logits = X_S @ theta_samples.T
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        preds = (logits >= 0).astype(np.float64)
        errors = np.mean(preds != y[:, None], axis=0)
        _eps = 1e-12
        nll_data = -np.sum(
        y[:, None] * np.log(proba + _eps) + (1 - y[:, None]) * np.log(1 - proba + _eps),axis=0)                                                    # (n_samples,)
        nll_reg  = 0.5 * np.sum(theta_samples**2, axis=1) / sigma2  # (n_samples,)
        nlls     = nll_data + nll_reg  
        return theta_samples, errors, nlls

    def sample(
        self,
        n_steps: int = 500,
        beta: float = 0.5,
        proposal: str = "adaptive_resize",
        init_support: Optional[List[int]] = None,
        n_swaps: int = 2,
        eps: Optional[float] = None,
        mh_variant: str = "repulsive",
        ellipsoid_augment: bool = True,
        ellipsoid_n_samples: int = 25,
        repulsion_weight: float = 0.05,
        # "l1" | "jaccard" | "prediction_hamming" | "monotonicity_entropy"
        # (or any callable)
        diversity_fn: Optional[Union[str, Callable[[List[np.ndarray], np.ndarray], float]]] = None,
        repulsion_reference: str = "map_only",
        finetune_coordinate: bool = True,
        finetune_corr_threshold: float = 0.1,
        rashomon_loss_bound: Optional[float] = None,
        max_repulsion_refs: int = 200,
    ) -> RashomonResult:
        """
        Sample models from a Rashomon set.
        Args:
            n_steps:              Total MH steps.
            burn_in:              Steps to discard before collecting.
            target_models:        Target number of unique support sets.
            beta:                 Inverse temperature (lower = more exploration).
            proposal:             One of "swap", "multi_swap", "multi_swap_3",
                                  "add_remove", "swap_or_resize",
                                  "weighted_swap", "promise_score_mixture", "feature_group",
                                  "mixture", or "random".
            init_support:         Starting support set (auto if None).
            n_swaps:              Swaps per step for multi_swap.
            eps:                  Override self.eps if provided.
            mh_variant:           "standard" or "repulsive".
            ellipsoid_augment:    If True, sample additional models from a
                                  Hessian-defined ellipsoid around each support
                                  set's MAP estimate.
            ellipsoid_n_samples:  Candidates to draw per support set (default 20).
            repulsion_weight:     Strength of diversity bonus in repulsive MH
                                  (0 = standard MH). Only used when
                                  mh_variant="repulsive".
            diversity_fn:         For mh_variant="repulsive", the repulsion
                                  kernel: a callable
                                  (accepted_samples, candidate) -> float, or one
                                  of these names (case-insensitive, '-'/' ' == '_'):
                                    "l1"                   min weight-L1 distance to
                                      (alias "min_l1")     the nearest accepted model
                                    "jaccard"              support-set Jaccard distance
                                      (alias "support_jaccard")
                                    "prediction_hamming"   label disagreement
                                      (alias "hamming")
                                    "monotonicity_entropy" shape-monotonicity entropy
                                      (alias "monotonicity")
                                  Names are resolved by
                                  gram.diversity.resolve_diversity_fn against
                                  DIVERSITY_FN_BUILDERS, using this sampler's X
                                  and feature_groups.
                                  Default: diversity_support_jaccard.
            repulsion_reference:  Controls the repulsion reference pool /
                                  candidate representation when
                                  ``ellipsoid_augment=True``. Either
                                  ``"map_only"`` (MAPs vs MAPs) or
                                  ``"map_plus_ellipsoid"`` (ellipsoid cloud vs
                                  MAPs + accumulated ellipsoid samples,
                                  capped at 100 refs). Ignored when
                                  ``ellipsoid_augment=False``. Default:
                                  ``"map_plus_ellipsoid"``.
            finetune_coordinate:  If True, for 1-swap proposals where swapped
                                  features are correlated (|corr| >= threshold),
                                  use fast 1D linear regression to fit only the
                                  new feature's weight instead of full refit.
            finetune_corr_threshold: Min |correlation| to use finetune (default 0.3).
            rashomon_loss_bound:  Optional absolute NLL threshold. If provided,
                                  overrides eps-based filtering and returns all
                                  models with nll <= rashomon_loss_bound.
            max_repulsion_refs:  Maximum number of collected models used when
                                  computing each repulsive diversity term.

        Returns:
            RashomonResult
        """
        burn_in = n_steps // 10 # 10% of the total steps as burn-in
        target_models = n_steps - burn_in
        eps = eps if eps is not None else self.eps
        t0 = time.perf_counter()
        self._mh_hessian_cache = {}
        self._rashomon_hessian_cache = {}
        self.n_steps = n_steps

        # 1. Initial support
        if init_support is None:
            init_support = self.find_initial_support(self.p_feat)
            # init_support = list(range(self.X.shape[1]))
        idx0 = sorted(init_support)
        theta0, proba0 = self.fit_logistic(
            self.X[:, idx0], self.y, self.sigma2, idx=idx0,
        )
        w_opt = np.zeros(self.p)
        w_opt[idx0] = theta0
        h0 = self._hessian_matrix(
            self.X[:, idx0], proba0, self.sigma2,
            theta=theta0, idx=idx0,
        )
        best_err = self.misclassification_error(proba0, self.y)
        # Rashomon bounds use the data + L2 objective only. Smoothness is only
        # used when scoring support sets through the Laplace/MH score.
        best_nll = self.total_nll(
            proba0, self.y, theta0, self.sigma2, idx=None,
        )
        nll_bound = (
            float(rashomon_loss_bound)
            if rashomon_loss_bound is not None
            else float(best_nll * (1.0 + eps))
        )
        print(f"Initial support: {len(init_support)} features, error: {best_err:.4f}, nll: {best_nll:.2f}")
        print(f"Time taken to find initial support and fit logistic model: {time.perf_counter() - t0} seconds")
        # 2. Build proposal
        corr = self.compute_correlation_scores(self.X, self.y)
        prop_map = {
            "swap": self.propose_swap,
            "multi_swap": partial(self.propose_multi_swap, n_swaps=n_swaps),
            "multi_swap_3": partial(self.propose_multi_swap, n_swaps=3),
            "add_remove": self.propose_add_or_remove,
            "swap_or_resize": self.propose_swap_or_resize,
            "weighted_swap": partial(self.propose_weighted_swap, corr_scores=corr),
            "promise_score_mixture": partial[tuple[Any, float, float] | tuple[Any, Any, Any]](
                self.propose_promise_score_mixture,
                feature_scores=corr,
                swap_prob=0.55,
                add_prob=0.30,
                remove_prob=0.15,
                temperature=0.2,
            ),
            "feature_group": self.propose_feature_group_swap,
            "mixture": self.make_mixture_proposal([
                self.propose_swap_or_resize,
                partial(self.propose_multi_swap, n_swaps=2),
                partial(self.propose_multi_swap, n_swaps=3),
            ]),
            "adaptive_resize": AdaptiveResizeProposer(
                init_swap_prob=0.33,
                init_add_prob=0.33,
                init_remove_prob=0.34,
                ema_alpha=0.05,
                temperature=1.0,
                min_prob=0.05,
                adapt_every=10,
                freeze_after=max(1, self.n_steps//10),
            ),
            "random": None,
        }
        if proposal not in prop_map:
            raise ValueError(
                f"Unknown proposal '{proposal}'. Options: {list(prop_map.keys())}"
            )

        # 3. Explore support sets (MH returns supports, scores, acc_rate, collected_w)
        support_to_w = {tuple(idx0): w_opt.copy()}
        support_to_hessian: Dict[tuple, np.ndarray] = {tuple(idx0): h0.copy()}
        log_alpha_history = None
        mh_skip = max(1, (n_steps - burn_in) // max(target_models, 1))
        mh_ellipsoid_by_support: Dict[Tuple[int, ...], Dict[str, np.ndarray]] = {}
        if proposal == "random":
            supports, _ = self.random_sample(len(init_support), target_models)
            acc_rate = float("nan")
            mh_collected_w: Optional[List[np.ndarray]] = None
            n_finetune, n_full = 0, 0
        elif mh_variant == "repulsive":
            (
                supports,
                _,
                acc_rate,
                mh_collected_w,
                log_alpha_history,
                n_finetune,
                n_full,
                mh_ellipsoid_by_support,
            ) = self.mh_sample_repulsive(
                init_support, n_steps, burn_in, mh_skip, beta,
                prop_map[proposal], repulsion_weight,
                diversity_fn=diversity_fn,
                ellipsoid_augment=ellipsoid_augment,
                ellipsoid_n_samples=ellipsoid_n_samples,
                repulsion_reference=repulsion_reference,
                rashomon_nll_bound=nll_bound,
                finetune_coordinate=finetune_coordinate,
                finetune_corr_threshold=finetune_corr_threshold,
                max_repulsion_refs=max_repulsion_refs,
            )
            for i, S in enumerate(supports):
                if i < len(mh_collected_w):
                    support_to_w[tuple(sorted(S))] = mh_collected_w[i]
            support_to_hessian.update(self._mh_hessian_cache)
        else:
            supports, _, acc_rate, mh_collected_w, n_finetune, n_full = self.mh_sample(
                init_support, n_steps, burn_in, mh_skip, beta,
                prop_map[proposal],
                finetune_coordinate=finetune_coordinate,
                finetune_corr_threshold=finetune_corr_threshold,
            )
            for i, S in enumerate(supports):
                if i < len(mh_collected_w):
                    support_to_w[tuple(sorted(S))] = mh_collected_w[i]
            support_to_hessian.update(self._mh_hessian_cache)
        supports.append(init_support)
        print(
            f"Explored {len(supports)} unique support sets "
            f"(accept rate: {acc_rate:.2%})"
        )

        # 4. Build models from stored MAPs (skip refit when we have theta from MH)
        all_w, all_err, all_nll, all_supp = [], [], [], []
        center_info = []  # (theta, idx, nll, eigvals, eigvecs) per support set
        for S in tqdm(supports, desc="Fitting models"):
            if not S:
                continue
            idx = sorted(S)
            key = tuple(idx)
            if key in support_to_w:
                w = support_to_w[key]
                theta = w[idx]
                proba = 1.0 / (1.0 + np.exp(-np.clip(self.X[:, idx] @ theta, -20, 20)))
            else:
                try:
                    theta, proba = self.fit_logistic(
                        self.X[:, idx], self.y, self.sigma2, idx=idx,
                    )
                except Exception:
                    print(f"Error fitting logistic model for support set {idx}")
                    continue
                w = np.zeros(self.p)
                w[idx] = theta
            err = self.misclassification_error(proba, self.y)
            # Post-MH filtering and ellipsoid radii use data + L2 NLL only.
            nll = self.total_nll(proba, self.y, theta, self.sigma2, idx=None)
            if nll < best_nll:
                best_nll = nll
                best_err = err
                w_opt = w.copy()

            all_w.append(w)
            all_err.append(err)
            all_nll.append(nll)
            all_supp.append(S)
            hessian_s = support_to_hessian.get(key)
            if hessian_s is None:
                hessian_s = self._hessian_matrix(
                    self.X[:, idx], proba, self.sigma2,
                    theta=theta, idx=idx,
                )
                support_to_hessian[key] = hessian_s
            self._rashomon_hessian_cache[key] = hessian_s.copy()
            if ellipsoid_augment:
                eigvals, eigvecs = self._hessian_eig_from_matrix(hessian_s)
                center_info.append((theta, idx, nll, eigvals, eigvecs))

        ellipsoid_by_support: Dict[tuple, List[np.ndarray]] = {}
        if ellipsoid_augment and center_info:
            nll_bound = (
                float(rashomon_loss_bound)
                if rashomon_loss_bound is not None
                else float(best_nll * (1.0 + eps))
            )
            n_ell_accepted = 0
            n_reused = 0
            n_redrawn = 0
            for theta_c, idx_c, nll_c, eigvals_c, eigvecs_c in tqdm(center_info,
                                               desc="Ellipsoid augmentation"):
                supp_key = tuple(sorted(idx_c))
                cached = mh_ellipsoid_by_support.get(supp_key)
                if cached is not None and cached["w_samples"].shape[0] > 0:
                    # Reuse the cloud we drew at MH accept time (Option 2).
                    w_samples_full = cached["w_samples"]
                    errs = cached["errors"]
                    nlls_ell = cached["nlls"]
                    n_reused += 1
                    ellipsoid_by_support.setdefault(supp_key, [])
                    for i in range(w_samples_full.shape[0]):
                        w = w_samples_full[i]
                        all_w.append(w)
                        all_err.append(float(errs[i]))
                        all_nll.append(float(nlls_ell[i]))
                        all_supp.append(list(idx_c))
                        ellipsoid_by_support[supp_key].append(w.copy())
                        n_ell_accepted += 1
                else:
                    ub = max(float(nll_bound) - float(nll_c), 1e-6)
                    X_S = self.X[:, idx_c]
                    theta_samples, errs, nlls_ell = self._sample_ellipsoid(
                        theta_c, X_S, self.y, self.sigma2,
                        ellipsoid_n_samples, ub,
                        eigvals=eigvals_c, eigvecs=eigvecs_c,
                    )
                    n_redrawn += 1
                    ellipsoid_by_support.setdefault(supp_key, [])
                    for ts, te, nl in zip(theta_samples, errs, nlls_ell):
                        w = np.zeros(self.p)
                        w[idx_c] = ts
                        all_w.append(w)
                        all_err.append(te)
                        all_nll.append(nl)
                        all_supp.append(list(idx_c))
                        ellipsoid_by_support[supp_key].append(w.copy())
                        n_ell_accepted += 1
            print(
                f"Ellipsoid augmentation: {n_ell_accepted} additional candidates "
                f"from {len(center_info)} support sets "
                f"(reused {n_reused} MH clouds, redrew {n_redrawn})"
            )

        # 5. Filter by Rashomon bound.
        nlls_arr = np.array(all_nll, dtype=np.float64)
        if nlls_arr.size == 0:
            mask = np.array([], dtype=bool)
            errs = np.array([], dtype=np.float64)
        else:
            best_nll = float(np.min(nlls_arr))
            if rashomon_loss_bound is not None:
                nll_bound = float(rashomon_loss_bound)
            else:
                nll_bound = float(best_nll * (1 + eps))
            mask = nlls_arr <= nll_bound
            errs = np.array(all_err)
        bound = float(np.max(errs[mask])) if mask.any() else best_err

        w_out = np.array(all_w)[mask] if mask.any() else np.zeros((0, self.p))
        e_out = errs[mask]
        s_out = [s for s, m in zip(all_supp, mask) if m]

        w_mcmc_out = None
        w_ellipsoid_out = None
        ellipsoid_by_support_filtered: Dict[tuple, List[np.ndarray]] = {}
        if ellipsoid_augment and center_info:
            n_mcmc = len(center_info)
            mask_mcmc = mask[:n_mcmc]
            mask_ell = mask[n_mcmc:]
            if mask_mcmc.any():
                w_mcmc_out = np.array(all_w[:n_mcmc])[mask_mcmc]
            if mask_ell.any():
                w_ellipsoid_out = np.array(all_w[n_mcmc:])[mask_ell]
            # Filter ellipsoid_by_support by Rashomon bound (samples at indices n_mcmc+)
            ell_idx = n_mcmc
            for supp_key, samples in ellipsoid_by_support.items():
                kept = []
                for _ in samples:
                    if ell_idx < len(mask) and mask[ell_idx]:
                        kept.append(all_w[ell_idx])
                    ell_idx += 1
                if kept:
                    ellipsoid_by_support_filtered[supp_key] = kept

        rt = time.perf_counter() - t0
        print(f"Best NLL: {best_nll:.2f}, NLL bound: {nll_bound:.2f}")
        print(f"Best error: {best_err:.4f}, Rashomon set: {mask.sum()}/{len(errs)} models, runtime: {rt:.1f}s")
        print(f"MH scoring: finetuned {n_finetune} times, full optimization {n_full} times")

        result = RashomonResult(
            w_models=w_out,
            w_opt=w_opt,
            support_sets=s_out,
            errors=e_out,
            best_error=best_err,
            error_bound=bound,
            eps=eps,
            n_explored=len(errs),
            n_accepted=int(mask.sum()),
            runtime=rt,
            acceptance_rate=acc_rate,
            w_mcmc=w_mcmc_out,
            w_ellipsoid=w_ellipsoid_out,
            ellipsoid_by_support=ellipsoid_by_support_filtered if ellipsoid_by_support_filtered else None,
            log_alpha_history=log_alpha_history,
            best_nll=best_nll,
            nlls=nlls_arr[mask],
            all_nlls=nlls_arr,
            nll_bound=nll_bound,
            feature_names=self.feature_names,
            feature_groups=self.feature_groups,
            X=self.X.copy(),
        )
        if result.feature_names is not None and result.feature_groups is not None:
            result.convert_gams_to_binned_representation()
        return result