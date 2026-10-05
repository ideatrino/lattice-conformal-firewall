#!/usr/bin/env python3
"""
Lattice-Conformal Firewall - runnable, self-verifying demonstration.

It builds synthetic benign/malicious traffic, trains all five layers, then:
  A. shows the rule lattice catching a shadowed rule and a conflict, and
     reducing a redundant policy to canonical form;
  B. empirically verifies the conformal false-positive guarantee FPR <= alpha
     across a sweep of alpha (averaged over independent trials), against the
     exact finite-sample target floor(alpha*(n+1))/(n+1);
  C. reports full-system detection rate and false-positive rate;
  D. demonstrates the SPRT deciding a streaming flow early.
"""

import numpy as np
from collections import Counter

from lcf import (
    Flow, extract, Policy, Rule, ALLOW, DENY,
    GaussianVerdict, SPRT, KLGate, ConformalGate,
    LatticeConformalFirewall,
)

# byte distributions used to synthesise payloads (shape, not content)
_P_BENIGN = np.ones(256) * 0.2
_P_BENIGN[32:127] += 4.0
_P_BENIGN[[9, 10, 32, 101]] += [2.0, 2.0, 6.0, 3.0]
_P_BENIGN /= _P_BENIGN.sum()
_P_UNIFORM = np.ones(256) / 256.0


def gen_flow(label, rng):
    header = {
        "src_ip": int(rng.integers(0, 2**32)),
        "dst_ip": int(rng.integers(0, 2**32)),
        "src_port": int(rng.integers(1024, 65535)),
        "dst_port": int(rng.choice([80, 443, 22, 53, 8080, 4444])),
        "proto": 6,
    }
    if label == 0:   # benign: structured payload, regular timing, download-heavy
        n = int(max(3, rng.normal(40, 10)))
        sizes = rng.normal(520, 140, n).clip(40, 1500)
        iats = np.abs(rng.normal(0.05, 0.02, max(1, n - 1)))
        hist = rng.multinomial(int(sizes.sum()), _P_BENIGN).astype(float)
        down, up = sizes.sum() * rng.uniform(1.2, 3.0), sizes.sum()
    elif label == 1:  # loud malicious: near-uniform payload (encrypted/exfil)
        n = int(max(3, rng.normal(30, 12)))
        sizes = rng.normal(300, 260, n).clip(40, 1500)
        iats = np.abs(rng.normal(0.12, 0.09, max(1, n - 1)))
        hist = rng.multinomial(int(sizes.sum()), _P_UNIFORM).astype(float)
        down, up = sizes.sum(), sizes.sum() * rng.uniform(1.5, 4.0)
    else:             # stealthy: benign-looking payload (KL gate misses) but
                      # off-manifold flow shape - big uploads, few huge packets
        n = int(max(3, rng.normal(35, 8)))
        sizes = rng.normal(1420, 60, n).clip(40, 1500)
        iats = np.abs(rng.normal(0.05, 0.02, max(1, n - 1)))
        hist = rng.multinomial(int(sizes.sum()), _P_BENIGN).astype(float)  # looks benign
        down, up = sizes.sum() * 0.02, sizes.sum() * 8.0                   # heavy exfil
    return Flow(header, sizes, iats, hist, down, up)


def feats(flows):
    return np.vstack([extract(f) for f in flows])


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# --------------------------------------------------------------------------
def part_A_lattice():
    section("A.  Rule lattice - provable policy analysis (Layer 2)")
    rules = [
        Rule.make(ALLOW, "allow-web-subnet", dst_ip="10.0.0.0/24", dst_port=443),
        Rule.make(DENY,  "block-host-443",   dst_ip="10.0.0.5",    dst_port=443),   # shadowed
        Rule.make(ALLOW, "allow-ssh-admin",  src_ip="192.168.1.0/24", dst_port=22),
        Rule.make(DENY,  "block-ssh-all",    dst_port=22),                           # conflict
        Rule.make(ALLOW, "allow-web-a",      dst_ip="10.0.1.0/25",   dst_port=443),
        Rule.make(ALLOW, "allow-web-b",      dst_ip="10.0.1.128/25", dst_port=443),  # mergeable
    ]
    pol = Policy(rules)
    print(f"raw policy: {len(pol)} rules")
    for a in pol.analyze():
        print(f"  [{a.kind:14}] {a.detail}")
    canon = pol.canonical()
    print(f"\ncanonical normal form: {len(canon)} rules "
          f"(dead/redundant dropped, adjacent merged, policy-equivalent)")
    for r in canon.rules:
        print(f"  {r.action:7} {r.box}")


def train_once(rng, n_train=1500, n_cal=2000):
    tr_b = feats([gen_flow(0, rng) for _ in range(n_train)])
    tr_m = feats([gen_flow(1, rng) for _ in range(n_train)])
    cal = [gen_flow(0, rng) for _ in range(n_cal)]
    verdict = GaussianVerdict.fit(tr_b, tr_m, prior_m=0.05)
    kl_gate = KLGate.fit(np.vstack([f.byte_hist for f in cal]), eps=0.01)
    cal_scores = np.array([verdict.log_odds(extract(f)) for f in cal])
    return verdict, kl_gate, cal_scores


def part_B_guarantee():
    section("B.  Conformal guarantee - empirical FPR <= alpha (Layer 5)")
    alphas = [0.20, 0.10, 0.05, 0.02, 0.01]
    trials, n_test = 30, 4000
    acc = {a: [] for a in alphas}
    n_cal = 2000
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        verdict, _, cal_scores = train_once(rng, n_cal=n_cal)
        gate = ConformalGate.fit(cal_scores, alpha=0.01)
        test_scores = np.array(
            [verdict.log_odds(extract(gen_flow(0, rng))) for _ in range(n_test)])
        pvals = np.array([gate.p_value(s) for s in test_scores])
        for a in alphas:
            acc[a].append(float(np.mean(pvals < a)))

    print(f"{'alpha':>7} | {'target floor(a(n+1))/(n+1)':>26} | "
          f"{'mean emp. FPR':>14} | holds?")
    print("-" * 72)
    all_ok = True
    for a in alphas:
        target = np.floor(a * (n_cal + 1)) / (n_cal + 1)   # exact marginal FPR
        mean_fpr = float(np.mean(acc[a]))
        se = float(np.std(acc[a]) / np.sqrt(trials))
        ok = mean_fpr <= a + 3 * se + 1e-9
        all_ok &= ok
        print(f"{a:>7.3f} | {target:>26.5f} | {mean_fpr:>14.5f} | "
              f"{'yes' if ok else 'NO':>5}")
    print("-" * 72)
    print("guarantee held: empirical FPR tracks the theoretical target <= alpha"
          if all_ok else "guarantee VIOLATED")


def part_C_full_system():
    section("C.  Full firewall - detection rate & false-positive rate")
    rng = np.random.default_rng(42)
    verdict, kl_gate, cal_scores = train_once(rng)
    fw = LatticeConformalFirewall(
        Policy([]), verdict, kl_gate, ConformalGate.fit(cal_scores, alpha=0.01))

    test_b = [gen_flow(0, rng) for _ in range(3000)]
    test_loud = [gen_flow(1, rng) for _ in range(1500)]
    test_stealth = [gen_flow(2, rng) for _ in range(1500)]
    vb = [fw.decide(f) for f in test_b]
    v_loud = [fw.decide(f) for f in test_loud]
    v_stealth = [fw.decide(f) for f in test_stealth]

    fpr = np.mean([v.decision == "BLOCK" for v in vb])
    tpr_loud = np.mean([v.decision == "BLOCK" for v in v_loud])
    tpr_stealth = np.mean([v.decision == "BLOCK" for v in v_stealth])
    print(f"detection rate, loud (encrypted exfil) : {tpr_loud:6.2%}")
    print(f"detection rate, stealthy (benign-looking payload) : {tpr_stealth:6.2%}")
    print(f"false-positive rate on benign          : {fpr:6.2%}  "
          f"(conformal alpha=1% + KL gate eps=1%)")
    print("\nblock attribution shows the layers are complementary:")
    print("  loud     ->", dict(Counter(v.layer for v in v_loud if v.decision == "BLOCK")))
    print("  stealthy ->", dict(Counter(v.layer for v in v_stealth if v.decision == "BLOCK")))
    print("\nexample verdicts:")
    for tag, v in [("loud", v_loud[0]), ("stealth", v_stealth[0]),
                   ("benign", vb[0]), ("benign", vb[1])]:
        print(f"  {tag:8} {v.decision:5} via {v.layer:9} | post={v.posterior:.3f} "
              f"kl={v.kl:.3f} p={v.p_value:.4f}")


def part_D_sprt():
    section("D.  Sequential test - decide a stream early (Layer 3, SPRT)")
    rng = np.random.default_rng(3)
    mu_b, mu_m, sigma = 520.0, 300.0, 180.0

    def llr(size):
        return (-0.5 * ((size - mu_m) / sigma) ** 2) - (-0.5 * ((size - mu_b) / sigma) ** 2)

    for label, mu in [("benign", mu_b), ("malicious", mu_m)]:
        ns, last = [], "continue"
        for _ in range(300):
            sprt, last = SPRT(alpha=0.01, beta=0.01), "continue"
            for _ in range(500):
                last = sprt.update(llr(rng.normal(mu, sigma)))
                if last != "continue":
                    break
            ns.append(sprt.n)
        print(f"{label:10}: last decision = {last:9} | "
              f"avg packets to decide = {np.mean(ns):5.1f}")


if __name__ == "__main__":
    print("LATTICE-CONFORMAL FIREWALL - demonstration")
    part_A_lattice()
    part_B_guarantee()
    part_C_full_system()
    part_D_sprt()
    print("\ndone.")
