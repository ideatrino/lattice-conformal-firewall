#!/usr/bin/env python3
"""
LCF v2 demonstration - real ingestion, persistence, and drift robustness.

  E. synthesise a mixed-traffic .pcap, train through the ingestion pipeline,
     save the model, reload it, and replay the capture packet-by-packet;
  F. adaptive conformal (ACI) vs static conformal under an abrupt
     distribution shift - ACI's long-run error returns to the target alpha.
"""
import os
import tempfile
from collections import Counter
import numpy as np

from lcf import (train_firewall, gen_pcap_packets, write_pcap, save_firewall,
                 load_firewall, Engine, extract, gen_flow,
                 ConformalGate, AdaptiveConformalGate)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def part_E_ingestion():
    section("E.  Real ingestion: pcap -> flows -> verdicts (save/reload model)")
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as d:
        pcap = os.path.join(d, "mix.pcap")
        model = os.path.join(d, "model.json")
        packets, labels = gen_pcap_packets(rng, n_benign=20, n_loud=8, n_stealth=8)
        write_pcap(pcap, packets)
        print(f"captured {len(packets)} packets / "
              f"{labels.count(0)} benign, {labels.count(1)} loud, "
              f"{labels.count(2)} stealthy flows")

        fw, _ = train_firewall(rng, via_pcap=True, n_train=800, n_cal=1500,
                               alpha=0.01)
        save_firewall(fw, model)
        fw = load_firewall(model)                     # ship + reload
        print(f"model trained, saved ({os.path.getsize(model)} bytes), reloaded")

        verdicts = Engine(fw).run_pcap(pcap)
        n_block = sum(v.verdict.decision == "BLOCK" for v in verdicts)
        print(f"\nreplayed {len(verdicts)} flows parsed from raw packets: "
              f"{n_block} blocked, {len(verdicts) - n_block} allowed")
        print("block attribution:",
              dict(Counter(v.verdict.layer for v in verdicts
                           if v.verdict.decision == "BLOCK")))
        print("\nsample (parsed straight from the wire format):")
        for v in verdicts[:4]:
            print("  " + v.line())


def part_F_drift():
    section("F.  Drift robustness: adaptive vs static conformal (Layer 5+)")
    rng = np.random.default_rng(5)
    fw, cal = train_firewall(rng, n_train=800, n_cal=1500)
    alpha, gamma, T = 0.05, 0.1, 4000
    cal_std = float(np.std(cal))

    static = ConformalGate.fit(cal, alpha=alpha)
    adaptive = AdaptiveConformalGate(cal, alpha_target=alpha, gamma=gamma)
    se, ae = [], []
    for t in range(T):
        off = 0.0 if t < T // 2 else 0.8 * cal_std    # adversary shifts at T/2
        s = fw.verdict.log_odds(extract(gen_flow(0, rng))) + off
        se.append(static.blocks(s))
        ae.append(adaptive.observe(s, is_benign=True))
    se, ae = np.array(se), np.array(ae)
    bound = (alpha + gamma) / (gamma * T)
    print(f"benign traffic, target false-positive rate alpha = {alpha}")
    print(f"  before shift : static {se[:T//2].mean():.3f}   "
          f"adaptive {ae[:T//2].mean():.3f}")
    print(f"  after  shift : static {se[T//2:].mean():.3f}   "
          f"adaptive {ae[T//2:].mean():.3f}   (transient)")
    print(f"  long-run avg : static {se.mean():.3f}   adaptive {ae.mean():.3f}")
    print(f"  ACI guarantee |avg - alpha| <= (alpha+gamma)/(gamma*T) = {bound:.4f}"
          f"  ->  measured {abs(ae.mean()-alpha):.4f}")
    print("static conformal drifts with the attacker; ACI returns to target.")


if __name__ == "__main__":
    print("LATTICE-CONFORMAL FIREWALL v2 - ingestion & drift demonstration")
    part_E_ingestion()
    part_F_drift()
    print("\ndone.")
