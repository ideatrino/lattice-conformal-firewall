r"""
Pluggable verdict models - a stronger scorer behind the same interface.

The orchestrator only needs a scorer with `log_odds(x)` (and optionally
`posterior` / `log_odds_contributions`).  The GaussianVerdict in bayesian.py is
a single diagonal Gaussian per class, which smears across *multimodal* traffic:
when "benign" has several distinct shapes (interactive + bulk + DNS ...), one
Gaussian centres on the average and treats the gaps between modes as normal -
exactly where an attacker can hide.

GaussianMixtureVerdict fits a K-component diagonal mixture per class by EM, so it
represents each mode separately.  Crucially, swapping it in changes only the
*score*: the conformal layer calibrates on whatever scores it is given, so the
alpha false-positive guarantee transfers to the stronger model unchanged - the
guarantee is model-agnostic.  This module demonstrates that.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def _diag_log_gauss(X, mu, var):
    """log N(x; mu, diag(var)) for each row of X. X:(n,d) mu:(d,) var:(d,)."""
    return -0.5 * (((X - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(axis=1)


@dataclass
class _DiagGMM:
    weights: np.ndarray      # (K,)
    means: np.ndarray        # (K, d)
    vars: np.ndarray         # (K, d)

    def score(self, X) -> np.ndarray:
        """log p(x) under the mixture, per row."""
        X = np.atleast_2d(X)
        comp = np.stack([np.log(self.weights[k]) +
                         _diag_log_gauss(X, self.means[k], self.vars[k])
                         for k in range(len(self.weights))], axis=1)  # (n,K)
        return _logsumexp(comp, axis=1)

    @staticmethod
    def fit(X, K, iters=80, restarts=3, reg=1e-4, seed=0) -> "_DiagGMM":
        X = np.atleast_2d(X).astype(float)
        n, d = X.shape
        gvar = X.var(0) + reg
        best, best_ll = None, -np.inf
        for r in range(restarts):
            rng = np.random.default_rng(seed + r)
            means = X[rng.choice(n, size=min(K, n), replace=False)].copy()
            if means.shape[0] < K:                 # pad if n < K
                means = np.vstack([means, X[rng.integers(0, n, K - means.shape[0])]])
            vars_ = np.tile(gvar, (K, 1))
            weights = np.full(K, 1.0 / K)
            prev_ll = -np.inf
            for _ in range(iters):
                logcomp = np.stack([np.log(weights[k]) +
                                    _diag_log_gauss(X, means[k], vars_[k])
                                    for k in range(K)], axis=1)      # (n,K)
                ll = _logsumexp(logcomp, axis=1)                     # (n,)
                total_ll = ll.sum()
                logresp = logcomp - ll[:, None]
                resp = np.exp(logresp)                               # (n,K)
                Nk = resp.sum(0) + 1e-9
                weights = Nk / n
                means = (resp.T @ X) / Nk[:, None]
                for k in range(K):
                    diff2 = (X - means[k]) ** 2
                    vars_[k] = (resp[:, k] @ diff2) / Nk[k] + reg
                if abs(total_ll - prev_ll) < 1e-4 * abs(prev_ll + 1e-9):
                    break
                prev_ll = total_ll
            if total_ll > best_ll:
                best_ll, best = total_ll, _DiagGMM(weights.copy(), means.copy(),
                                                   vars_.copy())
        return best


@dataclass
class GaussianMixtureVerdict:
    """Per-class diagonal-GMM generative model + Bayes posterior. Drop-in for
    GaussianVerdict: exposes log_odds(x) and posterior(x)."""
    gmm_b: _DiagGMM
    gmm_m: _DiagGMM
    prior_m: float
    K: int = 2

    @staticmethod
    def fit(Xb, Xm, prior_m, K=2, seed=0) -> "GaussianMixtureVerdict":
        return GaussianMixtureVerdict(
            gmm_b=_DiagGMM.fit(Xb, K, seed=seed),
            gmm_m=_DiagGMM.fit(Xm, K, seed=seed + 100),
            prior_m=float(prior_m), K=K)

    def benign_logdensity(self, x) -> float:
        return float(self.gmm_b.score(np.atleast_2d(x))[0])

    def nonconformity(self, x) -> float:
        """One-class novelty score: how unlike benign x is (higher = stranger)."""
        return -self.benign_logdensity(x)

    def log_odds(self, x) -> float:
        lm = float(self.gmm_m.score(np.atleast_2d(x))[0]) + np.log(self.prior_m)
        lb = float(self.gmm_b.score(np.atleast_2d(x))[0]) + np.log(1 - self.prior_m)
        return lm - lb

    def posterior(self, x) -> float:
        z = self.log_odds(x)
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))
