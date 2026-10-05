"""
Layer 3 - Bayesian verdict and sequential testing.

Instead of a binary allow/deny, we maintain a posterior probability that a
flow is malicious and update it as evidence arrives.

  * GaussianVerdict:   P(malicious | x) via a diagonal-Gaussian likelihood
                       for each class and Bayes' rule.
  * SPRT:              Wald's Sequential Probability Ratio Test - accumulate
                       the per-packet log-likelihood ratio and decide only
                       when the evidence crosses a threshold, minimising the
                       expected number of packets for given error rates.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


_LOG2PI = np.log(2.0 * np.pi)


@dataclass
class GaussianVerdict:
    """Diagonal-Gaussian generative model per class + Bayes posterior."""
    mu_b: np.ndarray        # benign mean
    var_b: np.ndarray       # benign variance
    mu_m: np.ndarray        # malicious mean
    var_m: np.ndarray       # malicious variance
    prior_m: float          # base rate P(malicious)

    @staticmethod
    def fit(Xb: np.ndarray, Xm: np.ndarray, prior_m: float,
            eps: float = 1e-6) -> "GaussianVerdict":
        return GaussianVerdict(
            mu_b=Xb.mean(0), var_b=Xb.var(0) + eps,
            mu_m=Xm.mean(0), var_m=Xm.var(0) + eps,
            prior_m=float(prior_m),
        )

    def _log_lik(self, x, mu, var) -> float:
        return float(-0.5 * (((x - mu) ** 2) / var + np.log(var) + _LOG2PI).sum())

    def log_odds(self, x: np.ndarray) -> float:
        """log P(x|mal)P(mal) - log P(x|ben)P(ben).  >0 favours malicious."""
        lm = self._log_lik(x, self.mu_m, self.var_m) + np.log(self.prior_m)
        lb = self._log_lik(x, self.mu_b, self.var_b) + np.log(1 - self.prior_m)
        return lm - lb

    def benign_logdensity(self, x: np.ndarray) -> float:
        return self._log_lik(x, self.mu_b, self.var_b)

    def nonconformity(self, x: np.ndarray) -> float:
        """One-class novelty score: how unlike benign x is (higher = stranger)."""
        return -self._log_lik(x, self.mu_b, self.var_b)

    def log_odds_contributions(self, x: np.ndarray):
        """
        Exact additive decomposition of the log-odds across features.
        For a diagonal Gaussian the class log-likelihoods are sums over features,
        so z(x) = sum_i c_i + prior_term, where c_i is feature i's signed push
        toward 'malicious' (the -0.5*log(2pi) cancels between classes). Returns
        (contributions[n], prior_term); their sum equals log_odds(x) exactly.
        """
        c_m = -0.5 * (((x - self.mu_m) ** 2) / self.var_m + np.log(self.var_m))
        c_b = -0.5 * (((x - self.mu_b) ** 2) / self.var_b + np.log(self.var_b))
        contributions = c_m - c_b
        prior_term = float(np.log(self.prior_m) - np.log(1 - self.prior_m))
        return contributions, prior_term

    def posterior(self, x: np.ndarray) -> float:
        """P(malicious | x) in [0,1], numerically stable via the logit."""
        z = self.log_odds(x)
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = np.exp(z)
        return float(ez / (1.0 + ez))


@dataclass
class SPRT:
    """
    Wald SPRT on a stream of per-packet log-likelihood ratios.

    Decision thresholds for target error rates (alpha = false alarm,
    beta = missed detection):
        A = log((1 - beta) / alpha)     # cross upward  -> declare malicious
        B = log(beta / (1 - alpha))     # cross down    -> declare benign
    """
    alpha: float = 0.01
    beta: float = 0.01

    def __post_init__(self):
        self.A = np.log((1 - self.beta) / self.alpha)
        self.B = np.log(self.beta / (1 - self.alpha))
        self.llr = 0.0
        self.n = 0

    def update(self, per_packet_llr: float) -> str:
        """Feed one packet's LLR. Returns 'malicious' | 'benign' | 'continue'."""
        self.llr += per_packet_llr
        self.n += 1
        if self.llr >= self.A:
            return "malicious"
        if self.llr <= self.B:
            return "benign"
        return "continue"

    def reset(self):
        self.llr = 0.0
        self.n = 0
