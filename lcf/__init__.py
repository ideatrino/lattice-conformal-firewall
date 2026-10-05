"""
Lattice-Conformal Firewall (LCF)
================================

An original, mathematically grounded firewall architecture in five layers:

    Layer 1  features   - flow -> point in R^n            (features.py)
    Layer 2  lattice    - provably consistent policy      (lattice.py)
    Layer 3  bayesian   - posterior verdict + SPRT        (bayesian.py)
    Layer 4  anomaly    - KL-divergence gate              (anomaly.py)
    Layer 5  conformal  - calibrated false-positive bound (conformal.py)

The orchestrator is LatticeConformalFirewall in firewall.py.
See SPEC.md for the formal specification and demo.py for a runnable,
self-verifying demonstration.
"""

from .features import Flow, extract, FEATURE_NAMES, shannon_entropy
from .lattice import Box, Rule, Policy, Anomaly, ALLOW, DENY, INSPECT, FIELDS
from .bayesian import GaussianVerdict, SPRT
from .models import GaussianMixtureVerdict
from .anomaly import KLGate, kl_divergence
from .conformal import ConformalGate
from .adaptive import (AdaptiveConformalGate, WeightedConformalGate,
                       MondrianConformalGate)
from .volumetric import (CountMinSketch, VolumetricMonitor, VolumetricAlert)
from .drift import PageHinkley
from .firewall import LatticeConformalFirewall, Verdict
from .pcap import (read_pcap, parse_frame, pcap_to_flows, write_pcap,
                   build_frame, FlowAssembler, Packet, int_to_ip, ip_to_int)
from .persist import save_firewall, load_firewall
from .engine import Engine, FlowVerdict, print_verdict
from .synth import gen_flow, train_firewall, gen_pcap_packets
# note: `lcf.evaluate` is a submodule imported on demand (from lcf import evaluate);
# it is intentionally NOT eager-imported here so `python -m lcf.evaluate` runs clean.

__all__ = [
    "Flow", "extract", "FEATURE_NAMES", "shannon_entropy",
    "Box", "Rule", "Policy", "Anomaly", "ALLOW", "DENY", "INSPECT", "FIELDS",
    "GaussianVerdict", "GaussianMixtureVerdict", "SPRT",
    "KLGate", "kl_divergence",
    "ConformalGate", "AdaptiveConformalGate", "WeightedConformalGate",
    "MondrianConformalGate",
    "CountMinSketch", "VolumetricMonitor", "VolumetricAlert",
    "PageHinkley",
    "LatticeConformalFirewall", "Verdict",
    "read_pcap", "parse_frame", "pcap_to_flows", "write_pcap", "build_frame",
    "FlowAssembler", "Packet", "int_to_ip", "ip_to_int",
    "save_firewall", "load_firewall",
    "Engine", "FlowVerdict", "print_verdict",
    "gen_flow", "train_firewall", "gen_pcap_packets",
]
__version__ = "3.1.0"
