r"""
Layer 5, hardened - conformal prediction under distribution shift.

The plain split-conformal guarantee (conformal.py) assumes the benign traffic
is *exchangeable*.  A deliberately drifting adversary - or ordinary concept
drift - breaks that, and the realised false-positive rate wanders away from the
target alpha.  Two standard repairs are implemented here.

1. AdaptiveConformalGate - Adaptive Conformal Inference (ACI, Gibbs & Candes
   2021).  It keeps a time-varying level alpha_t and nudges it with online
   feedback so the long-run error rate tracks the target under *arbitrary*
   distribution shift, with a deterministic regret bound (no exchangeability
   needed):

       err_t      = 1{ benign flow t was blocked }
       alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

   Theorem (Gibbs & Candes).  For any sequence whatsoever,
       | (1/T) sum_t err_t  -  alpha_target |  <=  (alpha_1 + gamma) / (gamma T),
   so the empirical FPR converges to alpha_target as T grows, regardless of
   how the distribution moves.

2. WeightedConformalGate - covariate-shift conformal (Tibshirani et al. 2019).
   When benign test flows are drawn from a shifted density, reweight the
   calibration scores by the likelihood ratio w(x)=p_test(x)/p_cal(x); the
   coverage guarantee is restored exactly when the weights are correct.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

from .conformal import ConformalGate


@dataclass
class AdaptiveConformalGate:
    """ACI wrapper around a fixed calibration set of benign scores."""
    cal_scores: np.ndarray
    alpha_target: float = 0.01
    gamma: float = 0.02
    alpha_t: float = field(init=False)
    history: List[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.cal_scores = np.sort(np.asarray(self.cal_scores, float))
        self.alpha_t = float(self.alpha_target)

    def p_value(self, score: float) -> float:
        n = self.cal_scores.size
        ge = n - int(np.searchsorted(self.cal_scores, score, side="left"))
        return (1 + ge) / (n + 1)

    def blocks(self, score: float) -> bool:
        """Decision at the *current* adaptive level (no state change)."""
        return self.p_value(score) < self.alpha_t

    def observe(self, score: float, is_benign: bool) -> bool:
        """
        Make a decision AND update alpha_t from feedback.  `is_benign` is the
        ground-truth label that arrives after the fact (e.g. analyst triage);
        alpha_t is only moved on confirmed-benign flows, which is where the
        false-positive rate is defined.
        Returns the block decision taken at this step.
        """
        decided_block = self.blocks(score)
        if is_benign:
            err = 1.0 if decided_block else 0.0
            # ACI update. alpha_t is deliberately NOT clamped to [0,1]: letting it
            # drift slightly below 0 ("block nothing") or above 1 ("block all") is
            # what keeps the telescoping identity exact, which is what gives the
            # long-run average-error bound under arbitrary shift. A wide clip only
            # guards against numerical runaway.
            self.alpha_t = float(
                np.clip(self.alpha_t + self.gamma * (self.alpha_target - err),
                        -2.0, 2.0))
            self.history.append(err)
        return decided_block

    def realized_fpr(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0


@dataclass
class MondrianConformalGate:
    r"""
    Class-conditional (Mondrian) conformal prediction.

    Plain conformal controls the FPR *marginally* - averaged over all benign
    traffic.  That can hide a subpopulation being systematically over-blocked
    (e.g. DNS flows blocked at 20% while the overall rate is 1%).  Mondrian
    conformal partitions calibration by a group key g(x) and ranks a test point
    only against its own group, so the guarantee holds *within every group*:

        P( p(x) < alpha | group(x) = k )  <=  alpha   for each k,

    under exchangeability within the group.  Unknown or tiny groups fall back to
    the pooled calibration set, which keeps the marginal guarantee.
    """
    groups: dict                      # group key -> sorted score array
    pooled: np.ndarray                # all scores, sorted (fallback)
    alpha: float = 0.01
    min_group: int = 50               # below this, use the pooled set

    @staticmethod
    def fit(scores, group_keys, alpha: float = 0.01, min_group: int = 50):
        scores = np.asarray(scores, float)
        groups: dict = {}
        for g in set(group_keys):
            s = np.sort(scores[np.asarray(group_keys) == g])
            groups[g] = s
        return MondrianConformalGate(groups, np.sort(scores), float(alpha),
                                     min_group)

    def _cal_for(self, group) -> np.ndarray:
        s = self.groups.get(group)
        if s is None or s.size < self.min_group:
            return self.pooled
        return s

    def p_value(self, score: float, group) -> float:
        cal = self._cal_for(group)
        n = cal.size
        ge = n - int(np.searchsorted(cal, score, side="left"))
        return (1 + ge) / (n + 1)

    def blocks(self, score: float, group) -> bool:
        return self.p_value(score, group) < self.alpha


@dataclass
class WeightedConformalGate:
    """Covariate-shift conformal: calibration scores carry importance weights."""
    cal_scores: np.ndarray
    cal_weights: np.ndarray
    alpha: float = 0.01

    def __post_init__(self):
        s = np.asarray(self.cal_scores, float)
        w = np.asarray(self.cal_weights, float)
        order = np.argsort(s)
        self.cal_scores = s[order]
        self.cal_weights = w[order]
        self.cal_weights /= self.cal_weights.sum()

    def p_value(self, score: float, test_weight: float = 1.0) -> float:
        # weighted fraction of calibration scores >= score, plus the test point
        ge_mask = self.cal_scores >= score
        num = self.cal_weights[ge_mask].sum() + test_weight
        den = self.cal_weights.sum() + test_weight
        return float(num / den)

    def blocks(self, score: float, test_weight: float = 1.0) -> bool:
        return self.p_value(score, test_weight) < self.alpha
