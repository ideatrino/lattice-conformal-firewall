"""
Evaluation harness - ROC / AUC / precision-recall, pure numpy.

Scores flows with the Bayesian log-odds and measures ranking quality
independent of any threshold (AUC), plus the precision/recall and the actual
operating point the conformal alpha selects.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .features import extract
from .bayesian import GaussianVerdict


def scores_labels(verdict: GaussianVerdict, benign, malicious
                  ) -> Tuple[np.ndarray, np.ndarray]:
    sb = np.array([verdict.log_odds(extract(f)) for f in benign])
    sm = np.array([verdict.log_odds(extract(f)) for f in malicious])
    scores = np.concatenate([sb, sm])
    labels = np.concatenate([np.zeros(sb.size), np.ones(sm.size)])
    return scores, labels


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact AUC = P(score_pos > score_neg) via the Mann-Whitney statistic."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_ranks(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray):
    thr = np.unique(scores)[::-1]
    P, N = (labels == 1).sum(), (labels == 0).sum()
    tpr, fpr = [0.0], [0.0]
    for t in thr:
        pred = scores >= t
        tpr.append(float((pred & (labels == 1)).sum() / P))
        fpr.append(float((pred & (labels == 0)).sum() / N))
    return np.array(fpr), np.array(tpr)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    recall = tp / max(1, y.sum())
    rec = np.concatenate([[0], recall])
    return float((np.diff(rec) * precision).sum())


def report(verdict: GaussianVerdict, benign, malicious, alpha: float = 0.01,
           cal_scores: np.ndarray = None) -> dict:
    scores, labels = scores_labels(verdict, benign, malicious)
    a = auc(scores, labels)
    ap = average_precision(scores, labels)
    fpr, tpr = roc_curve(scores, labels)
    out = {"auc": a, "ap": ap, "fpr": fpr, "tpr": tpr}
    print(f"AUC = {a:.4f}   average precision = {ap:.4f}   "
          f"(n_benign={len(benign)}, n_malicious={len(malicious)})")
    print(f"{'ROC: FPR':>10} | {'TPR':>6}")
    print("-" * 21)
    for target in (0.001, 0.01, 0.05, 0.1):
        i = int(np.searchsorted(fpr, target))
        i = min(i, len(fpr) - 1)
        print(f"{fpr[i]:>10.4f} | {tpr[i]:>6.3f}")
    if cal_scores is not None:
        from .conformal import ConformalGate
        gate = ConformalGate.fit(cal_scores, alpha=alpha)
        op = np.mean([gate.blocks(s) for s in scores[labels == 1]])
        opf = np.mean([gate.blocks(s) for s in scores[labels == 0]])
        print(f"conformal operating point (alpha={alpha}): "
              f"TPR={op:.3f}  FPR={opf:.3f}")
    return out


def auc_from_vectors(verdict, Xb, Xm) -> Tuple[float, np.ndarray, np.ndarray]:
    sb = np.array([verdict.log_odds(x) for x in Xb])
    sm = np.array([verdict.log_odds(x) for x in Xm])
    s = np.concatenate([sb, sm])
    y = np.concatenate([np.zeros(sb.size), np.ones(sm.size)])
    return auc(s, y), *roc_curve(s, y)


if __name__ == "__main__":
    from .synth import train_firewall, gen_flow
    rng = np.random.default_rng(0)
    fw, cal = train_firewall(rng)
    benign = [gen_flow(0, rng) for _ in range(2000)]
    malicious = [gen_flow(1, rng) for _ in range(1000)] + \
                [gen_flow(2, rng) for _ in range(1000)]
    print("LCF evaluation - clean (loud + stealthy malicious vs benign)\n")
    report(fw.verdict, benign, malicious, alpha=0.01, cal_scores=cal)

    # Harder case: a weak attacker whose features sit most of the way toward
    # benign (70% blend + noise) - the classes now overlap, so AUC < 1 and the
    # ROC shows a real trade-off. Demonstrates the harness is not degenerate.
    print("\nLCF evaluation - hard (weak attacker, overlapping distributions)\n")
    Xb = np.array([extract(f) for f in benign])
    mu_b = Xb.mean(0)
    Xm_weak = []
    for _ in range(2000):
        xm = extract(gen_flow(1, rng))
        blend = 0.7 * mu_b + 0.3 * xm + rng.normal(0, 0.15 * np.abs(mu_b))
        Xm_weak.append(blend)
    a, fpr, tpr = auc_from_vectors(fw.verdict, list(Xb), Xm_weak)
    print(f"AUC = {a:.4f}  (weak attacker)")
    for target in (0.01, 0.05, 0.1, 0.2):
        i = min(int(np.searchsorted(fpr, target)), len(fpr) - 1)
        print(f"  at FPR {fpr[i]:.3f}  ->  TPR {tpr[i]:.3f}")
