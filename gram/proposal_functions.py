"""Support-set proposal functions for MCMC Rashomon sampling."""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr


class AdaptiveResizeProposer:
    """
    Adaptive mixture proposer over {swap, add, remove}.

    Move probabilities are adapted based on exponential moving averages of the
    log-posterior change produced by each move type. Intuitively, this adds
    "momentum": if removing features has recently been increasing the
    log-posterior quickly, ``remove`` gets sampled more; once the chain has
    stabilised around a support size (add/remove produce small or negative
    deltas), ``swap`` dominates.

    Use like any other proposal function: call ``proposer(S, K, score)`` to
    sample a proposal and ``proposer.observe(delta_score, accepted)`` after
    each MH step to feed the outcome back.

    NOTE on MH validity: adapting the kernel between steps breaks strict
    detailed balance. The acceptance ratio is still computed with the current
    step's kernel (``lq_f`` / ``lq_b`` reflect the present ``self._probs``),
    and setting ``freeze_after`` to a step count freezes the probabilities so
    the tail of the chain is an exact MH sampler. For a fully rigorous
    diminishing-adaptation variant, pass a decaying ``ema_alpha`` externally.
    """

    MOVES: Tuple[str, ...] = ("swap", "add", "remove")

    def __init__(
        self,
        init_swap_prob: float = 0.6,
        init_add_prob: float = 0.2,
        init_remove_prob: float = 0.2,
        ema_alpha: float = 0.05,
        temperature: float = 1.0,
        min_prob: float = 0.05,
        adapt_every: int = 50,
        freeze_after: Optional[int] = None,
        max_batch: Optional[int] = None,
    ):
        """
        Args:
            max_batch: Cap on the number of features added/removed per step.
                If ``None`` (default), ``add`` may insert up to ``K - |S|``
                features at once and ``remove`` may drop up to ``|S| - 1``
                features at once; the batch size ``m`` is sampled uniformly
                from ``{1, ..., m_max}`` and then a uniform ``m``-subset is
                chosen. If set to ``1``, behaviour reduces to the original
                single-feature add/remove.
        """
        w = np.array(
            [init_swap_prob, init_add_prob, init_remove_prob], dtype=np.float64
        )
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("init_*_prob must be non-negative with positive sum")
        self._probs: np.ndarray = w / w.sum()
        self._ema: np.ndarray = np.zeros(3, dtype=np.float64)
        self._counts: np.ndarray = np.zeros(3, dtype=np.int64)
        self.ema_alpha = float(ema_alpha)
        self.temperature = float(temperature)
        self.min_prob = float(min_prob)
        self.adapt_every = max(1, int(adapt_every))
        self.freeze_after = freeze_after
        if max_batch is not None and int(max_batch) < 1:
            raise ValueError("max_batch must be >= 1 or None")
        self.max_batch = 5 if max_batch is None else int(max_batch)
        self._n_obs = 0
        self.last_move: Optional[str] = None

    @staticmethod
    def _log_comb(n: int, k: int) -> float:
        """Log of the binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return float("-inf")
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
        )

    def _cap(self, m_max_feasible: int) -> int:
        """Apply the optional ``max_batch`` cap to a feasibility limit."""
        if self.max_batch is None:
            return int(m_max_feasible)
        return int(min(m_max_feasible, self.max_batch))

    @property
    def probs(self) -> Dict[str, float]:
        """Current (swap, add, remove) probabilities as a dict."""
        return {m: float(self._probs[i]) for i, m in enumerate(self.MOVES)}

    def _feasible_probs(self, S: set, K: int) -> np.ndarray:
        k = len(S)
        k_out = K - k
        feasible = np.array(
            [
                k > 0 and k_out > 0,  # swap
                k_out > 0,            # add
                k > 1,                # remove (keep at least one feature)
            ],
            dtype=bool,
        )
        probs = self._probs.copy()
        probs[~feasible] = 0.0
        s = probs.sum()
        if s <= 0:
            if not feasible.any():
                return np.zeros(3, dtype=np.float64)
            probs = feasible.astype(np.float64)
            s = probs.sum()
        return probs / s

    def __call__(self, S, K: int, score: float):
        S = set(S)
        probs = self._feasible_probs(S, K)
        if probs.sum() <= 0:
            self.last_move = None
            return S, 0.0, 0.0

        move_idx = int(np.random.choice(3, p=probs))
        move = self.MOVES[move_idx]
        self.last_move = move
        p_move_f = float(probs[move_idx])

        S_list = list(S)
        compl = [j for j in range(K) if j not in S]
        k = len(S_list)
        k_out = len(compl)

        if move == "swap":
            if k == 0 or k_out == 0:
                return S, 0.0, 0.0
            drop = S_list[np.random.randint(k)]
            add = compl[np.random.randint(k_out)]
            S_p = (S - {drop}) | {add}
            return S_p, 0.0, 0.0

        if move == "add":
            if k_out == 0:
                return S, 0.0, 0.0
            m_max_f = self._cap(k_out)  # forward batch-size upper bound
            m = int(np.random.randint(1, m_max_f + 1))
            add_idx = np.random.choice(compl, size=m, replace=False)
            S_p = S | {int(a) for a in add_idx}
            k_p = k + m
            probs_back = self._feasible_probs(S_p, K)
            p_remove_b = float(probs_back[self.MOVES.index("remove")])
            # Reverse move = remove ``m`` elements from S_p (size k_p).
            # Remove requires leaving >=1 feature => reverse batch range
            # is {1, ..., k_p - 1}, then capped by ``max_batch``.
            m_max_b = self._cap(k_p - 1)
            lq_f = (
                np.log(max(p_move_f, 1e-300))
                - np.log(m_max_f)
                - self._log_comb(k_out, m)
            )
            if m_max_b < m or p_remove_b <= 0.0:
                # Reverse kernel cannot produce the needed batch size;
                # forcing rejection keeps things safe under MH.
                lq_b = -np.inf
            else:
                lq_b = (
                    np.log(max(p_remove_b, 1e-300))
                    - np.log(m_max_b)
                    - self._log_comb(k_p, m)
                )
            return S_p, float(lq_f), float(lq_b)

        if k <= 1:
            return S, 0.0, 0.0
        m_max_f = self._cap(k - 1)  # keep at least one feature
        m = int(np.random.randint(1, m_max_f + 1))
        drop_idx = np.random.choice(S_list, size=m, replace=False)
        S_p = S - {int(d) for d in drop_idx}
        k_p = k - m
        k_out_p = k_out + m
        probs_back = self._feasible_probs(S_p, K)
        p_add_b = float(probs_back[self.MOVES.index("add")])
        # Reverse move = add ``m`` elements to S_p (complement size k_out_p).
        m_max_b = self._cap(k_out_p)
        lq_f = (
            np.log(max(p_move_f, 1e-300))
            - np.log(m_max_f)
            - self._log_comb(k, m)
        )
        if m_max_b < m or p_add_b <= 0.0:
            lq_b = -np.inf
        else:
            lq_b = (
                np.log(max(p_add_b, 1e-300))
                - np.log(m_max_b)
                - self._log_comb(k_out_p, m)
            )
        return S_p, float(lq_f), float(lq_b)

    def observe(self, delta_score: float, accepted: bool) -> None:
        """Feed the outcome of the most recent proposal back for adaptation.

        Args:
            delta_score: ``sc_proposed - sc_current`` in the Laplace score.
                Pass this regardless of whether the proposal was accepted;
                the gradient signal comes from the score change, not only
                accepted moves.
            accepted: Whether the proposal was accepted. Used only to keep
                per-move acceptance counts for diagnostics.
        """
        if self.last_move is None:
            return
        i = self.MOVES.index(self.last_move)
        a = self.ema_alpha
        ds = float(delta_score) if np.isfinite(delta_score) else 0.0
        self._ema[i] = (1.0 - a) * self._ema[i] + a * ds
        self._counts[i] += 1 if accepted else 0
        self._n_obs += 1

        if self.freeze_after is not None and self._n_obs > self.freeze_after:
            return
        if self._n_obs % self.adapt_every != 0:
            return

        z = self._ema / max(self.temperature, 1e-8)
        z = z - np.max(z)
        w = np.exp(z)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return
        w = w / s
        w = np.maximum(w, self.min_prob)
        w = w / w.sum()
        self._probs = w



def propose_swap(S, K, score):
    """Drop one feature, add one from complement. Symmetric; fixed support size."""
    S_list = list(S)
    compl = [j for j in range(K) if j not in S]
    if not S_list or not compl:
        return S, 0.0, 0.0
    drop = S_list[np.random.randint(len(S_list))]
    add = compl[np.random.randint(len(compl))]
    return (S - {drop}) | {add}, 0.0, 0.0

def propose_multi_swap(S, K, score, n_swaps=2):
    """Swap n_swaps features simultaneously. Symmetric; fixed support size."""
    S_list = list(S)
    compl = [j for j in range(K) if j not in S]
    n = min(n_swaps, len(S_list), len(compl))
    if n == 0:
        return S, 0.0, 0.0
    drops = set(np.random.choice(S_list, n, replace=False))
    adds = set(np.random.choice(compl, n, replace=False))
    return (S - drops) | adds, 0.0, 0.0

def propose_add_or_remove(S, K, score, add_prob=0.5):
    """Add or remove a single feature. Asymmetric; changes support size."""
    compl = [j for j in range(K) if j not in S]
    if np.random.random() < add_prob and compl:
        add = compl[np.random.randint(len(compl))]
        S_prop = S | {add}
        lq_f = -np.log(len(compl))
        lq_b = -np.log(len(S_prop))
    elif len(S) > 1:
        S_list = list(S)
        drop = S_list[np.random.randint(len(S_list))]
        S_prop = S - {drop}
        lq_f = -np.log(len(S_list))
        lq_b = -np.log(len(compl) + 1)
    else:
        return S, 0.0, 0.0
    return S_prop, lq_f, lq_b

def propose_weighted_swap(S, K, score, corr_scores=None,
                          add_power=1.0, drop_power=1.0):
    """Swap weighted by feature-target correlations. Asymmetric."""
    _e = 1e-12
    S_list = list(S)
    compl = [j for j in range(K) if j not in S]
    if not S_list or not compl:
        return S, 0.0, 0.0
    if corr_scores is None:
        return propose_swap(S, K, score)

    s_in = corr_scores[S_list]
    w_drop = (s_in + _e) ** (-drop_power)
    w_drop /= w_drop.sum()
    di = np.random.choice(len(S_list), p=w_drop)
    drop = S_list[di]

    s_out = corr_scores[compl]
    w_add = (s_out + _e) ** add_power
    w_add /= w_add.sum()
    ai = np.random.choice(len(compl), p=w_add)
    add = compl[ai]

    lq_f = np.log(w_drop[di]) + np.log(w_add[ai])

    S_prop = (S - {drop}) | {add}
    Sp_list = list(S_prop)
    compl_b = [j for j in range(K) if j not in S_prop]
    s_in_b = corr_scores[Sp_list]
    w_drop_b = (s_in_b + _e) ** (-drop_power)
    w_drop_b /= w_drop_b.sum()
    s_out_b = corr_scores[compl_b]
    w_add_b = (s_out_b + _e) ** add_power
    w_add_b /= w_add_b.sum()
    lq_b = (np.log(w_drop_b[Sp_list.index(add)])
            + np.log(w_add_b[compl_b.index(drop)]))

    return S_prop, lq_f, lq_b

def propose_swap_or_resize(S, K, score, swap_prob=0.6):
    """
    With probability swap_prob: swap one feature (fixed size).
    With probability (1-swap_prob)/2: add a random feature.
    With probability (1-swap_prob)/2: remove a random feature.

    Correctly computes the MH proposal ratio for the mixed move.
    """
    S_list = list(S)
    compl = [j for j in range(K) if j not in S]
    k = len(S_list)
    k_out = len(compl)

    u = np.random.random()

    grow_prob = (1 - swap_prob) / 2
    shrink_prob = (1 - swap_prob) / 2

    if u < swap_prob:
        # --- swap ---
        if not S_list or not compl:
            return S, 0.0, 0.0
        drop = S_list[np.random.randint(k)]
        add = compl[np.random.randint(k_out)]
        S_prop = (S - {drop}) | {add}
        # q_fwd = swap_prob * 1/k * 1/k_out
        # q_bwd = swap_prob * 1/k * 1/k_out  (same sizes after swap)
        return S_prop, 0.0, 0.0

    elif u < swap_prob + grow_prob:
        # --- add a feature ---
        if not compl:
            return S, 0.0, 0.0
        add = compl[np.random.randint(k_out)]
        S_prop = S | {add}
        # q_fwd = grow_prob * 1/k_out
        # q_bwd = shrink_prob * 1/(k+1)
        lq_f = np.log(grow_prob) - np.log(k_out)
        lq_b = np.log(shrink_prob) - np.log(k + 1)
        return S_prop, lq_f, lq_b
    else:
        # --- remove a feature ---
        if k <= 1:
            return S, 0.0, 0.0
        drop = S_list[np.random.randint(k)]
        S_prop = S - {drop}
        # q_fwd = shrink_prob * 1/k
        # q_bwd = grow_prob * 1/(k_out+1)
        lq_f = np.log(shrink_prob) - np.log(k)
        lq_b = np.log(grow_prob) - np.log(k_out + 1)
        return S_prop, lq_f, lq_b

def _softmax_probs(values: np.ndarray, temperature: float = 0.25, invert: bool = False) -> np.ndarray:
    """Stable softmax probabilities; optionally invert to prefer smaller values."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return v
    t = max(float(temperature), 1e-8)
    if invert:
        z = -v / t
    else:
        z = v / t
    z = z - np.max(z)
    p = np.exp(z)
    p_sum = p.sum()
    if not np.isfinite(p_sum) or p_sum <= 0:
        return np.ones_like(p) / len(p)
    return p / p_sum

def propose_promise_score_mixture(
    S,
    K,
    score,
    feature_scores=None,
    swap_prob: float = 0.55,
    add_prob: float = 0.30,
    remove_prob: float = 0.15,
    temperature: float = 0.25,
):
    """
    Mixture proposal over {swap, add, remove} with score-guided sampling.

    - Adds prefer high-score features outside support (softmax on scores).
    - Removes prefer low-score features inside support (softmax on -scores).
    - Swap combines one remove + one add with the same biases.

    Returns (S_prop, log q_forward, log q_backward).
    """
    if feature_scores is None:
        return propose_swap_or_resize(
            S, K, score, swap_prob=swap_prob
        )

    S_list = list(S)
    compl = [j for j in range(K) if j not in S]
    k = len(S_list)
    k_out = len(compl)
    if k == 0:
        return S, 0.0, 0.0

    raw_move_probs = {
        "swap": float(swap_prob),
        "add": float(add_prob),
        "remove": float(remove_prob),
    }
    feasible = {
        "swap": (k > 0 and k_out > 0),
        "add": (k_out > 0),
        "remove": (k > 1),  # keep at least one active feature
    }
    move_names = [m for m in ("swap", "add", "remove") if feasible[m] and raw_move_probs[m] > 0]
    if not move_names:
        return S, 0.0, 0.0
    move_weights = np.array([raw_move_probs[m] for m in move_names], dtype=np.float64)
    move_probs = move_weights / move_weights.sum()

    move = move_names[np.random.choice(len(move_names), p=move_probs)]
    p_move_f = move_probs[move_names.index(move)]

    if move == "swap":
        s_in = np.asarray(feature_scores[S_list], dtype=np.float64)
        w_drop = _softmax_probs(
            s_in, temperature=temperature, invert=True
        )
        di = int(np.random.choice(len(S_list), p=w_drop))
        drop = S_list[di]

        s_out = np.asarray(feature_scores[compl], dtype=np.float64)
        w_add = _softmax_probs(
            s_out, temperature=temperature, invert=False
        )
        ai = int(np.random.choice(len(compl), p=w_add))
        add = compl[ai]

        S_prop = (S - {drop}) | {add}
        lq_f = np.log(p_move_f) + np.log(w_drop[di]) + np.log(w_add[ai])

        Sp_list = list(S_prop)
        compl_b = [j for j in range(K) if j not in S_prop]
        s_in_b = np.asarray(feature_scores[Sp_list], dtype=np.float64)
        w_drop_b = _softmax_probs(
            s_in_b, temperature=temperature, invert=True
        )
        s_out_b = np.asarray(feature_scores[compl_b], dtype=np.float64)
        w_add_b = _softmax_probs(
            s_out_b, temperature=temperature, invert=False
        )
        lq_b = (
            np.log(p_move_f)
            + np.log(w_drop_b[Sp_list.index(add)])
            + np.log(w_add_b[compl_b.index(drop)])
        )
        return S_prop, float(lq_f), float(lq_b)

    if move == "add":
        s_out = np.asarray(feature_scores[compl], dtype=np.float64)
        w_add = _softmax_probs(
            s_out, temperature=temperature, invert=False
        )
        ai = int(np.random.choice(len(compl), p=w_add))
        add = compl[ai]
        S_prop = S | {add}
        lq_f = np.log(p_move_f) + np.log(w_add[ai])

        Sp_list = list(S_prop)
        s_in_b = np.asarray(feature_scores[Sp_list], dtype=np.float64)
        w_drop_b = _softmax_probs(
            s_in_b, temperature=temperature, invert=True
        )
        remove_idx = Sp_list.index(add)
        p_remove_b = move_probs[move_names.index("remove")]
        lq_b = np.log(p_remove_b) + np.log(w_drop_b[remove_idx])
        return S_prop, float(lq_f), float(lq_b)

    s_in = np.asarray(feature_scores[S_list], dtype=np.float64)
    w_drop = _softmax_probs(
        s_in, temperature=temperature, invert=True
    )
    di = int(np.random.choice(len(S_list), p=w_drop))
    drop = S_list[di]
    S_prop = S - {drop}
    lq_f = np.log(p_move_f) + np.log(w_drop[di])

    compl_b = [j for j in range(K) if j not in S_prop]
    s_out_b = np.asarray(feature_scores[compl_b], dtype=np.float64)
    w_add_b = _softmax_probs(
        s_out_b, temperature=temperature, invert=False
    )
    add_idx = compl_b.index(drop)
    p_add_b = move_probs[move_names.index("add")]
    lq_b = np.log(p_add_b) + np.log(w_add_b[add_idx])
    return S_prop, float(lq_f), float(lq_b)

def propose_feature_group_swap(S, K, score, feature_groups=None, fallback_proposal=propose_swap):
    """Swap all columns of one feature group for another. Symmetric."""
    if feature_groups is None:
        return fallback_proposal(S, K, score)
    g_in = [(n, i) for n, i in feature_groups.items() if set(i) & S]
    g_out = [(n, i) for n, i in feature_groups.items()
             if not (set(i) & S)]
    if not g_in or not g_out:
        return S, 0.0, 0.0
    _, d_idx = g_in[np.random.randint(len(g_in))]
    _, a_idx = g_out[np.random.randint(len(g_out))]
    return (S - set(d_idx)) | set(a_idx), 0.0, 0.0

def make_mixture_proposal(proposal_fns, weights=None):
    """Create a proposal that randomly selects from multiple proposals."""
    if weights is None:
        weights = np.ones(len(proposal_fns))
    w = np.array(weights, dtype=float)
    w /= w.sum()

    def _propose(S, K, score):
        idx = np.random.choice(len(proposal_fns), p=w)
        return proposal_fns[idx](S, K, score)
    return _propose

def compute_correlation_scores(X, y):
    """Absolute Pearson correlation of each feature with y."""
    out = np.zeros(X.shape[1])
    # A constant column has no correlation to report: scipy warns and returns
    # NaN, which the isfinite check below turns into a score of 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(X.shape[1]):
            r, _ = pearsonr(X[:, j], y)
            out[j] = np.abs(r) if np.isfinite(r) else 0.0
    return out
