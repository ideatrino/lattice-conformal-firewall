r"""
Concept-drift detection - knowing when to recalibrate.

ACI (adaptive.py) keeps the error rate on target *continuously*, but a sudden,
large shift still deserves an explicit signal: raise an alert, log it, and
trigger recalibration of the Gaussian model / conformal set.

PageHinkley is the classic sequential change-point test.  It accumulates how far
each observation sits above a reference mean (minus a tolerance delta) and fires
when that accumulation, measured from its running minimum, exceeds lambda:

    U_T   = sum_{t<=T} (z_t - delta)          z_t = (x_t - mu0) / sigma0
    PH_T  = U_T - min_{t<=T} U_T              (upward drift)
    alarm   iff  PH_T > lambda

Observations are standardised against a benign reference (mu0, sigma0) so the
tuning is scale-free: delta is measured in standard deviations (how big a mean
shift to ignore) and lambda trades detection delay against false alarms.  Under
no change, U_T drifts downward by delta each step, so PH_T stays bounded; a true
upward shift larger than delta overcomes that drift and fires.  State is O(1).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PageHinkley:
    delta: float = 0.5          # ignored change, in sigma units (half the shift to catch)
    lam: float = 30.0           # alarm threshold (lambda): higher = fewer false alarms
    mu0: float = 0.0            # reference mean
    sigma0: float = 1.0         # reference std
    direction: str = "both"     # "up" | "down" | "both"

    n: int = field(default=0, init=False)
    u_up: float = field(default=0.0, init=False)
    min_up: float = field(default=0.0, init=False)
    u_dn: float = field(default=0.0, init=False)
    max_dn: float = field(default=0.0, init=False)

    @staticmethod
    def fit(reference, delta: float = 0.5, lam: float = 30.0,
            direction: str = "both") -> "PageHinkley":
        ref = np.asarray(reference, float)
        return PageHinkley(delta=delta, lam=lam, mu0=float(ref.mean()),
                           sigma0=float(ref.std() + 1e-9), direction=direction)

    def update(self, x: float) -> bool:
        """Feed one observation; return True on a drift alarm (and reset)."""
        z = (x - self.mu0) / self.sigma0
        self.n += 1
        self.u_up += z - self.delta
        self.min_up = min(self.min_up, self.u_up)
        self.u_dn += z + self.delta
        self.max_dn = max(self.max_dn, self.u_dn)

        ph_up = self.u_up - self.min_up
        ph_dn = self.max_dn - self.u_dn
        fired = ((self.direction in ("up", "both") and ph_up > self.lam) or
                 (self.direction in ("down", "both") and ph_dn > self.lam))
        if fired:
            self._reset_stats()
        return fired

    def _reset_stats(self):
        self.n = 0
        self.u_up = self.min_up = 0.0
        self.u_dn = self.max_dn = 0.0
