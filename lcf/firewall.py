"""
The Lattice-Conformal Firewall - orchestrator.

Combines all five layers into one decision with a number attached to it:

  1. features   : flow  -> x in R^n  + byte histogram
  2. lattice    : first-match policy -> ALLOW (trust) | DENY (hard block)
                  | INSPECT (default: fall through to statistics)
  3. bayesian   : posterior P(malicious | x), used as the nonconformity score
  4. anomaly    : KL(P || Q) gate on payload byte distribution
  5. conformal  : p-value with a provable long-run false-positive bound

Policy verdicts are hard overrides (a proven-consistent whitelist/blacklist);
everything else is judged statistically, and every statistical BLOCK carries
the alpha-level guarantee from Layer 5.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .features import Flow, extract, FEATURE_NAMES
from .lattice import Policy, ALLOW, DENY, INSPECT
from .bayesian import GaussianVerdict
from .anomaly import KLGate
from .conformal import ConformalGate


@dataclass
class Verdict:
    decision: str            # "ALLOW" | "BLOCK"
    layer: str               # which layer decided
    reason: str
    posterior: Optional[float] = None
    kl: Optional[float] = None
    p_value: Optional[float] = None
    rule_index: int = -1
    drivers: Optional[list] = None   # [(feature, signed_contribution), ...]

    def explain(self, k: int = 3) -> str:
        if not self.drivers:
            return self.reason
        top = self.drivers[:k]
        parts = ", ".join(f"{n} ({'+' if v >= 0 else ''}{v:.2f})" for n, v in top)
        return f"{self.reason}; driven by {parts}"


class LatticeConformalFirewall:
    def __init__(self, policy: Policy, verdict: GaussianVerdict,
                 kl_gate: KLGate, conformal: ConformalGate):
        self.policy = policy
        self.verdict = verdict
        self.kl_gate = kl_gate
        self.conformal = conformal

    def decide(self, flow: Flow) -> Verdict:
        # Layer 1
        x = extract(flow)

        # Layer 2 - proven-consistent hard policy
        action, idx = self.policy.decide(flow.header)
        if action == ALLOW:
            return Verdict("ALLOW", "lattice", "matched trusted allow rule",
                           rule_index=idx)
        if action == DENY:
            return Verdict("BLOCK", "lattice", "matched explicit deny rule",
                           rule_index=idx)

        # Layers 3-5 - statistical path (action == INSPECT)
        post = self.verdict.posterior(x)
        score = self.verdict.log_odds(x)          # nonconformity score
        kl = self.kl_gate.score(flow.byte_hist)
        p = self.conformal.p_value(score)

        if hasattr(self.verdict, "log_odds_contributions"):
            contribs, _prior = self.verdict.log_odds_contributions(x)
            drivers = sorted(zip(FEATURE_NAMES, (float(c) for c in contribs)),
                             key=lambda kv: -abs(kv[1]))
        else:
            drivers = None            # scorer offers no additive attribution

        if self.kl_gate.trips(flow.byte_hist):
            return Verdict("BLOCK", "anomaly",
                           f"payload divergence {kl:.3f} > "
                           f"{self.kl_gate.threshold:.3f}",
                           posterior=post, kl=kl, p_value=p, drivers=drivers)

        if self.conformal.blocks(score):
            return Verdict("BLOCK", "conformal",
                           f"conformal p={p:.4f} < alpha={self.conformal.alpha}",
                           posterior=post, kl=kl, p_value=p, drivers=drivers)

        return Verdict("ALLOW", "conformal",
                       f"within tolerance (p={p:.4f} >= alpha)",
                       posterior=post, kl=kl, p_value=p, drivers=drivers)
