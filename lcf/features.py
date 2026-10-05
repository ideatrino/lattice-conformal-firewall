"""
Layer 1 - Flow as a point in measurable space.

A raw flow is turned into a fixed-length feature vector x in R^n plus a
256-bin payload byte histogram (used by the information-theoretic gate).
Everything downstream (probability, distance, divergence, geometry) is only
possible because the flow has been embedded in a metric space here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


# Order is fixed: it defines the axes of R^n.
FEATURE_NAMES: List[str] = [
    "mean_pkt_size",     # bytes
    "std_pkt_size",
    "mean_iat",          # inter-arrival time, seconds
    "std_iat",
    "payload_entropy",   # Shannon entropy of payload bytes, bits (0..8)
    "down_up_ratio",     # bytes received / bytes sent
    "duration",          # seconds
    "pkt_rate",          # packets / second
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class Flow:
    """Minimal raw-flow record.  `byte_hist` is a length-256 count vector."""
    header: Dict[str, int]          # src_ip, dst_ip, src_port, dst_port, proto
    pkt_sizes: np.ndarray
    iats: np.ndarray                # inter-arrival times between packets
    byte_hist: np.ndarray           # 256-bin histogram of payload bytes
    bytes_down: float
    bytes_up: float


def shannon_entropy(hist: np.ndarray) -> float:
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def extract(flow: Flow) -> np.ndarray:
    ps = np.asarray(flow.pkt_sizes, dtype=float)
    iat = np.asarray(flow.iats, dtype=float)
    duration = float(iat.sum()) if iat.size else 0.0
    n_pkts = ps.size
    x = np.array([
        ps.mean() if n_pkts else 0.0,
        ps.std() if n_pkts else 0.0,
        iat.mean() if iat.size else 0.0,
        iat.std() if iat.size else 0.0,
        shannon_entropy(flow.byte_hist),
        (flow.bytes_down + 1.0) / (flow.bytes_up + 1.0),
        duration,
        n_pkts / duration if duration > 0 else float(n_pkts),
    ], dtype=float)
    return x
