"""
Layer 4 - Information-theoretic anomaly gate.

We keep a baseline distribution Q of "normal" payload bytes.  For a live
flow with empirical byte distribution P we measure the Kullback-Leibler
divergence

        D(P || Q) = sum_a P(a) log( P(a) / Q(a) )

and flag the flow when it sits too far from baseline.  Because KL works on
the *shape* of traffic rather than its content, it survives encryption:
exfiltration, compression and C2 beacons all push payload entropy toward
uniform, which Q (a benign, skewed distribution) assigns low probability.

Threshold
---------
By Sanov's theorem, the probability that n benign samples produce an
empirical P with D(P||Q) >= t decays like exp(-n t) (up to a polynomial
factor).  So a threshold t_eps ~ (1/n) log(1/eps) controls the false-alarm
rate.  In practice we also expose a *calibrated* threshold: the (1-eps)
quantile of D(P||Q) measured on held-out benign flows, which makes no
distributional assumption.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p.astype(float)
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / p.size)
    q = q.astype(float) + eps
    q = q / q.sum()
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / q[nz])).sum())


@dataclass
class KLGate:
    baseline: np.ndarray        # Q, length-256 probability vector
    threshold: float            # t: flag if D(P||Q) > threshold

    @staticmethod
    def fit(benign_hists: np.ndarray, eps: float = 0.01) -> "KLGate":
        """
        benign_hists: (N, 256) counts for benign calibration flows.
        Sets Q to the pooled benign distribution and the threshold to the
        (1-eps) empirical quantile of per-flow KL to Q (distribution-free).
        """
        q = benign_hists.sum(0).astype(float)
        q = q / q.sum()
        kls = np.array([kl_divergence(h, q) for h in benign_hists])
        thr = float(np.quantile(kls, 1 - eps))
        return KLGate(baseline=q, threshold=thr)

    def score(self, hist: np.ndarray) -> float:
        return kl_divergence(hist, self.baseline)

    def trips(self, hist: np.ndarray) -> bool:
        return self.score(hist) > self.threshold
