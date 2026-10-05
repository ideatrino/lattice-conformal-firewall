"""
Layer 5 - The conformal guarantee.

This is the headline property.  We wrap any real-valued nonconformity score
s(x) (here: the Bayesian log-odds of maliciousness) in split-conformal
prediction.  Calibrate on held-out *benign* flows, then for a test flow
compute a p-value

        p(x) = ( 1 + #{ i : s(cal_i) >= s(x) } ) / ( n + 1 )

and block iff p(x) < alpha.

Guarantee
---------
If benign flows are exchangeable, then for a genuinely benign test flow

        P( p(x) < alpha )  <=  alpha .

So alpha is a *dial on the false-positive rate* with a proof behind it:
set alpha = 0.001 and at most ~1 in 1000 legitimate flows is wrongly
blocked over the long run - a guarantee no rule-based firewall can offer.
(Proof sketch in SPEC.md: the rank of s(x) among the n+1 scores is uniform
on {1..n+1} under exchangeability, so P(p <= k/(n+1)) = k/(n+1).)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ConformalGate:
    cal_scores: np.ndarray       # nonconformity scores on benign calibration set
    alpha: float                 # target false-positive rate

    @staticmethod
    def fit(cal_scores: np.ndarray, alpha: float = 0.01) -> "ConformalGate":
        return ConformalGate(np.sort(np.asarray(cal_scores, float)), float(alpha))

    def p_value(self, score: float) -> float:
        n = self.cal_scores.size
        # number of calibration scores >= test score
        ge = n - int(np.searchsorted(self.cal_scores, score, side="left"))
        return (1 + ge) / (n + 1)

    def blocks(self, score: float) -> bool:
        return self.p_value(score) < self.alpha

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)
