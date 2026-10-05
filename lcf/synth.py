"""
Synthetic traffic + one-call training.

Shared by the demos, the CLI and the test-suite so there is a single source of
truth for how benign / loud-malicious / stealthy-malicious flows are shaped,
both as `Flow` objects and as real .pcap packets (for the ingestion path).

Labels: 0 = benign, 1 = loud malicious (encrypted exfil), 2 = stealthy.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from .features import Flow, extract
from .lattice import Policy
from .bayesian import GaussianVerdict
from .anomaly import KLGate
from .conformal import ConformalGate
from .firewall import LatticeConformalFirewall
from .pcap import build_frame, PROTO_TCP, FlowAssembler

_P_BENIGN = np.ones(256) * 0.2
_P_BENIGN[32:127] += 4.0
_P_BENIGN[[9, 10, 32, 101]] += [2.0, 2.0, 6.0, 3.0]
_P_BENIGN /= _P_BENIGN.sum()
_P_UNIFORM = np.ones(256) / 256.0

_DST_PORTS = [80, 443, 22, 53, 8080, 4444]


def _params(label, rng):
    if label == 0:
        n = int(max(3, rng.normal(40, 10)))
        sizes = rng.normal(520, 140, n).clip(40, 1400)
        iats = np.abs(rng.normal(0.05, 0.02, max(1, n - 1)))
        p = _P_BENIGN
        down, up = sizes.sum() * rng.uniform(1.2, 3.0), sizes.sum()
    elif label == 1:
        n = int(max(3, rng.normal(30, 12)))
        sizes = rng.normal(300, 260, n).clip(40, 1400)
        iats = np.abs(rng.normal(0.12, 0.09, max(1, n - 1)))
        p = _P_UNIFORM
        down, up = sizes.sum(), sizes.sum() * rng.uniform(1.5, 4.0)
    else:
        n = int(max(3, rng.normal(35, 8)))
        sizes = rng.normal(1380, 60, n).clip(40, 1400)
        iats = np.abs(rng.normal(0.05, 0.02, max(1, n - 1)))
        p = _P_BENIGN
        down, up = sizes.sum() * 0.02, sizes.sum() * 8.0
    return n, sizes, iats, p, down, up


def gen_flow(label, rng) -> Flow:
    n, sizes, iats, p, down, up = _params(label, rng)
    header = {
        "src_ip": int(rng.integers(0, 2**32)),
        "dst_ip": int(rng.integers(0, 2**32)),
        "src_port": int(rng.integers(1024, 65535)),
        "dst_port": int(rng.choice(_DST_PORTS)),
        "proto": PROTO_TCP,
    }
    hist = rng.multinomial(int(sizes.sum()), p).astype(float)
    return Flow(header, sizes, iats, hist, down, up)


def feats(flows) -> np.ndarray:
    return np.vstack([extract(f) for f in flows])


def _flow_packets(label, rng, t0=0.0):
    """Build one flow's worth of (ts, frame) packets with its 5-tuple."""
    n, sizes, iats, p, down, up = _params(label, rng)
    cli_ip = f"10.0.{int(rng.integers(0,255))}.{int(rng.integers(1,255))}"
    srv_ip = f"93.184.{int(rng.integers(0,255))}.{int(rng.integers(1,255))}"
    sport = int(rng.integers(1024, 65535))
    dport = int(rng.choice(_DST_PORTS))
    ratio = up / max(down, 1.0)
    p_up = float(np.clip(ratio / (ratio + 1.0), 0.1, 0.9))
    out = []
    tt = t0
    for k in range(n):
        if k > 0:
            tt += float(iats[min(k - 1, len(iats) - 1)])
        length = int(sizes[k])
        payload = rng.choice(256, size=length, p=p).astype(np.uint8).tobytes()
        if rng.random() < p_up:
            out.append((tt, build_frame(cli_ip, srv_ip, sport, dport, payload)))
        else:
            out.append((tt, build_frame(srv_ip, cli_ip, dport, sport, payload)))
    return out


def gen_flow_via_pcap(label, rng):
    """A Flow produced through the real ingestion pipeline (parse + assemble),
    so training and inference share one feature representation."""
    asm = FlowAssembler(idle_timeout=float("inf"))
    from .pcap import parse_frame, LINKTYPE_ETHERNET
    for ts, frame in _flow_packets(label, rng):
        pk = parse_frame(frame, LINKTYPE_ETHERNET)
        asm.add_packet(pk, ts)
    header, flow = asm.flush_all()[0]
    return flow


def train_firewall(rng, policy: Optional[Policy] = None, alpha: float = 0.01,
                   kl_eps: float = 0.01, prior_m: float = 0.05,
                   n_train: int = 1500, n_cal: int = 2000, via_pcap: bool = False,
                   model: str = "gaussian", K: int = 2):
    g = gen_flow_via_pcap if via_pcap else gen_flow
    tr_b = feats([g(0, rng) for _ in range(n_train)])
    tr_m = feats([g(1, rng) for _ in range(n_train)])
    cal = [g(0, rng) for _ in range(n_cal)]
    if model == "gmm":
        from .models import GaussianMixtureVerdict
        verdict = GaussianMixtureVerdict.fit(tr_b, tr_m, prior_m=prior_m, K=K)
    else:
        verdict = GaussianVerdict.fit(tr_b, tr_m, prior_m=prior_m)
    kl_gate = KLGate.fit(np.vstack([f.byte_hist for f in cal]), eps=kl_eps)
    cal_scores = np.array([verdict.log_odds(extract(f)) for f in cal])
    fw = LatticeConformalFirewall(policy or Policy([]), verdict, kl_gate,
                                  ConformalGate.fit(cal_scores, alpha=alpha))
    return fw, cal_scores


# --------------------------------------------------------------------------
# Emit a mixed-traffic .pcap whose flows match the parameters above, so the
# ingestion + engine path can be exercised end-to-end on real packet bytes.
# --------------------------------------------------------------------------
def gen_pcap_packets(rng, n_benign=20, n_loud=8, n_stealth=8
                     ) -> Tuple[List[Tuple[float, bytes]], List[int]]:
    packets: List[Tuple[float, bytes]] = []
    labels: List[int] = []
    t = 0.0
    plan = ([0] * n_benign) + ([1] * n_loud) + ([2] * n_stealth)
    rng.shuffle(plan)
    for label in plan:
        labels.append(label)
        flow_pkts = _flow_packets(label, rng, t0=t + float(rng.uniform(0, 0.5)))
        packets.extend(flow_pkts)
        t = flow_pkts[-1][0] + 1.0
    packets.sort(key=lambda x: x[0])
    return packets, labels
