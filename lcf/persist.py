"""
Persistence - save and load a trained firewall as portable JSON.

Captures every learned parameter: the policy (as boxes), the Gaussian verdict
model, the KL baseline + threshold, and the conformal calibration scores, so a
model trained offline can be shipped to a sensor and loaded without retraining.
"""

from __future__ import annotations
import json
import numpy as np

from .lattice import Box, Rule, Policy
from .bayesian import GaussianVerdict
from .anomaly import KLGate
from .conformal import ConformalGate
from .firewall import LatticeConformalFirewall


def save_firewall(fw: LatticeConformalFirewall, path: str):
    doc = {
        "version": 1,
        "policy": [
            {"bounds": [list(b) for b in r.box.bounds],
             "action": r.action, "name": r.name}
            for r in fw.policy.rules
        ],
        "verdict": {
            "mu_b": fw.verdict.mu_b.tolist(), "var_b": fw.verdict.var_b.tolist(),
            "mu_m": fw.verdict.mu_m.tolist(), "var_m": fw.verdict.var_m.tolist(),
            "prior_m": fw.verdict.prior_m,
        },
        "kl_gate": {
            "baseline": fw.kl_gate.baseline.tolist(),
            "threshold": fw.kl_gate.threshold,
        },
        "conformal": {
            "cal_scores": fw.conformal.cal_scores.tolist(),
            "alpha": fw.conformal.alpha,
        },
    }
    with open(path, "w") as fh:
        json.dump(doc, fh)


def load_firewall(path: str) -> LatticeConformalFirewall:
    with open(path) as fh:
        doc = json.load(fh)
    rules = [
        Rule(Box(tuple(tuple(b) for b in r["bounds"])), r["action"], r["name"])
        for r in doc["policy"]
    ]
    v = doc["verdict"]
    verdict = GaussianVerdict(
        mu_b=np.array(v["mu_b"]), var_b=np.array(v["var_b"]),
        mu_m=np.array(v["mu_m"]), var_m=np.array(v["var_m"]),
        prior_m=v["prior_m"])
    k = doc["kl_gate"]
    kl_gate = KLGate(baseline=np.array(k["baseline"]), threshold=k["threshold"])
    c = doc["conformal"]
    conformal = ConformalGate(cal_scores=np.sort(np.array(c["cal_scores"])),
                              alpha=c["alpha"])
    return LatticeConformalFirewall(Policy(rules), verdict, kl_gate, conformal)
