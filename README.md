# Lattice-Conformal Firewall (LCF) · v3

[![CI](https://github.com/ideatrino/lattice-conformal-firewall/actions/workflows/ci.yml/badge.svg)](https://github.com/ideatrino/lattice-conformal-firewall/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-18%2F18-brightgreen)

An original, mathematically grounded firewall. Instead of a list of hand-written
rules emitting a binary allow/deny, every flow becomes a point in a measurable
space, gets a calibrated probability, and receives a decision that carries a
**provable error bound** — now ingesting real packet captures, hardened against
distribution shift, and runnable as a tool.

| Layer | File | What it adds | Shortcoming it fixes |
|---|---|---|---|
| 0 Volumetric | `volumetric.py` | Count-Min + entropy: DoS flood & port-scan detection | per-flow design is blind to cross-flow attacks |
| 1 Features | `features.py`, `pcap.py` | flow → point in R^n + byte histogram; real pcap/Ethernet/IP ingestion | signatures see only surface patterns |
| 2 Lattice | `lattice.py` | provably consistent policy (shadow/conflict detection, canonical form) | rule explosion, ordering bugs, silent conflicts |
| 3 Bayesian | `bayesian.py`, `models.py` | posterior + Wald SPRT + exact attribution; pluggable GMM scorer | binary verdicts; unexplained blocks; multimodal traffic |
| 4 Anomaly | `anomaly.py` | KL-divergence gate on payload shape | DPI breaks under encryption |
| 5 Conformal | `conformal.py` | false-positive rate ≤ α, guaranteed | FPR is otherwise unknown and uncontrolled |
| 5+ Adaptive | `adaptive.py` | ACI (drift) + Mondrian (per-group) + weighted conformal | marginal bound hides over-blocked subgroups; shift |

Plus drift detection (`drift.py`), an evaluation harness (`evaluate.py`), a live
engine (`engine.py`), model persistence (`persist.py`), and a CLI (`cli.py`).
See **SPEC.md** for the formal specification and every proof.

## Install

```bash
pip install -e .           # numpy only; adds the `lcf` command
# optional inline mode on Linux: pip install -e ".[live]"  (needs root)
```

## Run the demos and tests

```bash
python3 demo.py            # parts A–D: lattice, conformal guarantee, detection, SPRT
python3 demo_live.py       # parts E–F: real pcap ingestion, drift robustness
python3 demo_max.py        # parts G–L: volumetric, Mondrian, drift, explain, AUC, GMM
python3 -m lcf.evaluate    # ROC / AUC / precision-recall
python3 tests/test_lcf.py  # 18 tests, ~7 s
```

## Use the CLI

```bash
lcf demo-pcap --out traffic.pcap             # synthesise a mixed capture
lcf train     --out model.json --alpha 0.01  # train (synthetic, via ingestion)
lcf train     --out model.json --benign b.pcap --malicious m.pcap   # from real pcaps
lcf run       --model model.json --pcap traffic.pcap --verbose       # replay + verdicts
lcf analyze   --policy examples/policy.json  # shadow/conflict/canonical analysis
lcf live      --model model.json --queue 0   # inline NFQUEUE (Linux, root)
```

## Use the library

```python
from lcf import (train_firewall, save_firewall, load_firewall, Engine,
                 Policy, Rule, ALLOW, DENY, AdaptiveConformalGate)
import numpy as np

fw, cal = train_firewall(np.random.default_rng(0), via_pcap=True)
save_firewall(fw, "model.json")
verdicts = Engine(load_firewall("model.json")).run_pcap("traffic.pcap")
```

The design is yours — "Lattice-Conformal Firewall" is a working name; rename it
however you like.
