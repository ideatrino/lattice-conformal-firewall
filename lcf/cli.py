"""
Command-line interface for the Lattice-Conformal Firewall.

    lcf train    --out model.json [--alpha A] [--benign b.pcap --malicious m.pcap]
    lcf analyze  --policy rules.json
    lcf run      --model model.json --pcap traffic.pcap
    lcf live     --model model.json [--queue N]     (Linux, root, NFQUEUE)
    lcf demo-pcap --out traffic.pcap                 (synthesise a capture)

A policy file is JSON: a list of {action, name, <field>: spec} objects, e.g.
    [{"action":"ALLOW","name":"web","dst_ip":"10.0.0.0/24","dst_port":443}]
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
import numpy as np

from .lattice import Policy, Rule
from .persist import save_firewall, load_firewall
from .synth import train_firewall, gen_pcap_packets, feats, gen_flow
from .features import extract
from .bayesian import GaussianVerdict
from .anomaly import KLGate
from .conformal import ConformalGate
from .firewall import LatticeConformalFirewall
from .engine import Engine, print_verdict
from .pcap import write_pcap, pcap_to_flows


def _load_policy(path: str) -> Policy:
    with open(path) as fh:
        spec = json.load(fh)
    rules = []
    for r in spec:
        kw = {k: v for k, v in r.items() if k not in ("action", "name")}
        rules.append(Rule.make(r["action"], r.get("name", ""), **kw))
    return Policy(rules)


def cmd_train(a):
    rng = np.random.default_rng(a.seed)
    policy = _load_policy(a.policy) if a.policy else None
    if a.benign and a.malicious:
        fb = pcap_to_flows(a.benign)
        fm = pcap_to_flows(a.malicious)
        Xb, Xm = feats([f for _, f in fb]), feats([f for _, f in fm])
        verdict = GaussianVerdict.fit(Xb, Xm, prior_m=a.prior)
        kl_gate = KLGate.fit(np.vstack([f.byte_hist for _, f in fb]), eps=a.kl_eps)
        cal_scores = np.array([verdict.log_odds(f) for f in Xb])
        fw = LatticeConformalFirewall(
            policy or Policy([]), verdict, kl_gate,
            ConformalGate.fit(cal_scores, alpha=a.alpha))
        src = f"{len(fb)} benign + {len(fm)} malicious flows from pcaps"
    else:
        fw, _ = train_firewall(rng, policy=policy, alpha=a.alpha, kl_eps=a.kl_eps,
                               prior_m=a.prior, via_pcap=True,
                               n_train=800, n_cal=1500)
        src = "synthetic traffic (via ingestion pipeline)"
    save_firewall(fw, a.out)
    print(f"trained on {src}; alpha={a.alpha}; saved to {a.out}")


def cmd_analyze(a):
    pol = _load_policy(a.policy)
    print(f"policy: {len(pol)} rules")
    anomalies = pol.analyze()
    if not anomalies:
        print("  no anomalies found")
    for an in anomalies:
        print(f"  [{an.kind:14}] {an.detail}")
    canon = pol.canonical()
    print(f"canonical normal form: {len(canon)} rules")
    for r in canon.rules:
        print(f"  {r.action:7} {r.box}")


def cmd_run(a):
    fw = load_firewall(a.model)
    eng = Engine(fw)
    verdicts = eng.run_pcap(a.pcap, on_verdict=print_verdict if a.verbose else None)
    n_block = sum(v.verdict.decision == "BLOCK" for v in verdicts)
    print(f"\n{len(verdicts)} flows | {n_block} blocked | "
          f"{len(verdicts)-n_block} allowed")
    print("block attribution:",
          dict(Counter(v.verdict.layer for v in verdicts
                       if v.verdict.decision == "BLOCK")))


def cmd_live(a):
    fw = load_firewall(a.model)
    Engine(fw).run_nfqueue(queue_num=a.queue, min_pkts=a.min_pkts)


def cmd_demo_pcap(a):
    rng = np.random.default_rng(a.seed)
    packets, labels = gen_pcap_packets(rng)
    write_pcap(a.out, packets)
    print(f"wrote {len(packets)} packets "
          f"({labels.count(0)} benign, {labels.count(1)} loud, "
          f"{labels.count(2)} stealthy flows) to {a.out}")


def build_parser():
    p = argparse.ArgumentParser(prog="lcf",
                                description="Lattice-Conformal Firewall")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and save a model")
    t.add_argument("--out", required=True)
    t.add_argument("--alpha", type=float, default=0.01)
    t.add_argument("--kl-eps", dest="kl_eps", type=float, default=0.01)
    t.add_argument("--prior", type=float, default=0.05)
    t.add_argument("--policy")
    t.add_argument("--benign")
    t.add_argument("--malicious")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    an = sub.add_parser("analyze", help="check a policy for anomalies")
    an.add_argument("--policy", required=True)
    an.set_defaults(func=cmd_analyze)

    r = sub.add_parser("run", help="replay a pcap through the firewall")
    r.add_argument("--model", required=True)
    r.add_argument("--pcap", required=True)
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    lv = sub.add_parser("live", help="inline NFQUEUE mode (Linux/root)")
    lv.add_argument("--model", required=True)
    lv.add_argument("--queue", type=int, default=0)
    lv.add_argument("--min-pkts", dest="min_pkts", type=int, default=4)
    lv.set_defaults(func=cmd_live)

    d = sub.add_parser("demo-pcap", help="synthesise a mixed-traffic pcap")
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo_pcap)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
