#!/usr/bin/env python3
"""
LCF v3 demonstration - the maximal layer.

  G. Volumetric: cross-flow DoS flood and port-scan detection
  H. Mondrian conformal: per-group FPR guarantee vs marginal over-blocking
  I. Drift: Page-Hinkley flags a distribution shift for recalibration
  J. Explainability: every verdict names the features that drove it
  K. Evaluation: AUC on separable and overlapping (weak-attacker) traffic
"""
import numpy as np
from lcf import (train_firewall, gen_flow, extract, ConformalGate,
                 MondrianConformalGate, VolumetricMonitor, PageHinkley, evaluate,
                 GaussianVerdict, GaussianMixtureVerdict)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def part_G_volumetric():
    section("G.  Volumetric layer - DoS flood & port scan (cross-flow)")
    rng = np.random.default_rng(0)
    benign = [{"src_ip": int(rng.integers(0, 2**32)),
               "dst_ip": int(rng.integers(0, 2**32)),
               "src_port": int(rng.integers(1024, 65535)),
               "dst_port": 80, "proto": 6} for _ in range(4000)]
    mon = VolumetricMonitor.fit(benign, window=500)
    print(f"calibrated entropy floors: dst>={mon.dst_entropy_floor:.2f} "
          f"src>={mon.src_entropy_floor:.2f} bits")
    flood = [{"src_ip": int(rng.integers(0, 2**32)), "dst_ip": 0x0A0A0A0A,
              "src_port": 1, "dst_port": 80, "proto": 6} for _ in range(500)]
    for h in flood:
        for a in mon.observe(h):
            print(f"  ALERT {a.kind:5} -> {a.detail}")
    scan = [{"src_ip": 0xC0A80001, "dst_ip": int(rng.integers(0, 2**32)),
             "src_port": 2, "dst_port": int(rng.integers(1, 65535)), "proto": 6}
            for _ in range(500)]
    for h in scan:
        for a in mon.observe(h):
            print(f"  ALERT {a.kind:5} -> {a.detail}")


def part_H_mondrian():
    section("H.  Mondrian conformal - per-group guarantee (alpha = 0.05)")
    rng = np.random.default_rng(11)
    gA, gB = rng.normal(0, 1, 3000), rng.normal(2, 1, 3000)
    scores = np.concatenate([gA, gB]); keys = np.array(["A"] * 3000 + ["B"] * 3000)
    marg = ConformalGate.fit(scores, alpha=0.05)
    mond = MondrianConformalGate.fit(scores, keys, alpha=0.05)
    tA, tB = rng.normal(0, 1, 4000), rng.normal(2, 1, 4000)
    print(f"              group A    group B")
    print(f"  marginal    {np.mean([marg.blocks(s) for s in tA]):.3f}      "
          f"{np.mean([marg.blocks(s) for s in tB]):.3f}   <- group B over-blocked ~2x")
    print(f"  mondrian    {np.mean([mond.blocks(s,'A') for s in tA]):.3f}      "
          f"{np.mean([mond.blocks(s,'B') for s in tB]):.3f}   <- both held at target")


def part_I_drift():
    section("I.  Drift detection - Page-Hinkley (recalibration trigger)")
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 1000)
    ph = PageHinkley.fit(ref, delta=0.5, lam=30)
    fa = sum(ph.update(x) for x in rng.normal(0, 1, 8000))
    ph = PageHinkley.fit(ref, delta=0.5, lam=30)
    stream = np.concatenate([rng.normal(0, 1, 1500), rng.normal(1.5, 1, 1500)])
    at = next((t for t, x in enumerate(stream) if ph.update(x)), None)
    print(f"  false alarms on 8000 stationary samples : {fa}")
    print(f"  shift injected at t=1500 -> detected at t={at} (delay {at-1500})")


def part_J_explain():
    section("J.  Explainability - why a flow was blocked")
    rng = np.random.default_rng(3)
    fw, _ = train_firewall(rng)
    for label, name in [(1, "loud"), (2, "stealthy")]:
        v = fw.decide(gen_flow(label, rng))
        print(f"  {name:9} {v.decision} via {v.layer}: {v.explain()}")


def part_K_eval():
    section("K.  Evaluation - AUC / ROC")
    rng = np.random.default_rng(0)
    fw, cal = train_firewall(rng)
    benign = [gen_flow(0, rng) for _ in range(1500)]
    mal = [gen_flow(1, rng) for _ in range(750)] + [gen_flow(2, rng) for _ in range(750)]
    s, y = evaluate.scores_labels(fw.verdict, benign, mal)
    print(f"  clean traffic  : AUC = {evaluate.auc(s, y):.4f}")
    Xb = [extract(f) for f in benign]; mu = np.mean(Xb, 0)
    weak = [0.7 * mu + 0.3 * extract(gen_flow(1, rng)) +
            rng.normal(0, 0.15 * np.abs(mu)) for _ in range(1500)]
    a, _, _ = evaluate.auc_from_vectors(fw.verdict, Xb, weak)
    print(f"  weak attacker  : AUC = {a:.4f}  (overlapping distributions)")


def part_L_models():
    section("L.  Pluggable scorer - GMM beats single Gaussian, guarantee transfers")
    rng = np.random.default_rng(0)

    def benign(n):                       # bimodal 'normal': two distinct shapes
        mode = rng.random(n) < 0.5
        size = np.where(mode, rng.normal(100, 15, n), rng.normal(900, 15, n))
        return np.column_stack([size, rng.normal(3, 0.3, n), rng.normal(0.05, 0.01, n)])

    def attack(n):                        # novel attack hiding in the valley (~500)
        return np.column_stack([rng.normal(500, 40, n), rng.normal(3, 0.3, n),
                                rng.normal(0.05, 0.01, n)])

    Xb, Xm = benign(4000), attack(4000)
    cal, tb, tm = benign(2000), benign(4000), attack(4000)
    g = GaussianVerdict.fit(Xb, Xm, prior_m=0.5)
    gm = GaussianMixtureVerdict.fit(Xb, Xm, prior_m=0.5, K=2)
    print("benign feature is bimodal (~100 and ~900); novel attack sits at ~500\n")
    print(f"  {'model':16} | {'AUC (novelty)':>13} | conformal FPR @ alpha=0.05")
    print("  " + "-" * 56)
    for name, v in [("single Gaussian", g), ("GMM (K=2)", gm)]:
        nb = np.array([v.nonconformity(x) for x in tb])
        nm = np.array([v.nonconformity(x) for x in tm])
        s = np.concatenate([nb, nm])
        y = np.concatenate([np.zeros(len(nb)), np.ones(len(nm))])
        gate = ConformalGate.fit(np.array([v.nonconformity(x) for x in cal]),
                                 alpha=0.05)
        fpr = np.mean([gate.blocks(x) for x in nb])
        print(f"  {name:16} | {evaluate.auc(s, y):>13.3f} | {fpr:.3f}")
    print("\n  the single Gaussian scores below 0.5 (the valley looks 'normal' to it);")
    print("  the GMM captures both modes - and the alpha guarantee holds for both.")


if __name__ == "__main__":
    print("LATTICE-CONFORMAL FIREWALL v3 - maximal layer demonstration")
    part_G_volumetric()
    part_H_mondrian()
    part_I_drift()
    part_J_explain()
    part_K_eval()
    part_L_models()
    print("\ndone.")
