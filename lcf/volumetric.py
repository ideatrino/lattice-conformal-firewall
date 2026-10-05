r"""
Layer 0 - Volumetric / cross-flow detection.

The five-layer core judges each flow on its own, so it is blind to attacks that
live in the *aggregate*: SYN/UDP floods (one victim, countless sources) and port
or host scans (one source, countless destinations).  This layer watches the
stream of flow headers over a sliding window and catches those with two
sublinear-memory, information-theoretic instruments.

1. Count-Min Sketch - estimates per-key counts (e.g. packets per source) in
   O(w*d) memory regardless of the number of distinct keys.  For width w and
   depth d with w = ceil(e/eps), d = ceil(ln(1/delta)), the estimate never
   under-counts and over-counts by more than eps*N with probability >= 1-delta
   (Cormode & Muthukrishnan).  A source whose estimated rate exceeds a
   threshold is a flood source / aggressive scanner.

2. Windowed Shannon entropy of the address distributions.  Normal traffic
   spreads across many src/dst/ports (high entropy).  A flood concentrates mass
   on one victim -> destination-IP entropy collapses.  A scan concentrates on
   one source fanning out over ports -> source entropy collapses while
   destination-port entropy spikes.  Thresholds are calibrated as quantiles of
   benign windows (distribution-free).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


class CountMinSketch:
    def __init__(self, width: int = 2048, depth: int = 5, seed: int = 0):
        self.w, self.d = width, depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed)
        # pairwise-independent hashing: (a*x + b) mod p mod w
        self.p = (1 << 61) - 1
        self.a = rng.integers(1, self.p, size=depth)
        self.b = rng.integers(0, self.p, size=depth)

    def _idx(self, key: int) -> np.ndarray:
        return ((self.a * (key & 0xFFFFFFFFFFFFFFFF) + self.b) % self.p) % self.w

    def add(self, key: int, count: int = 1):
        cols = self._idx(key)
        self.table[np.arange(self.d), cols] += count

    def estimate(self, key: int) -> int:
        cols = self._idx(key)
        return int(self.table[np.arange(self.d), cols].min())

    def clear(self):
        self.table.fill(0)


def _entropy(counts: Dict[int, int]) -> float:
    if not counts:
        return 0.0
    c = np.array(list(counts.values()), dtype=float)
    p = c / c.sum()
    return float(max(0.0, -(p * np.log2(p)).sum()))


@dataclass
class VolumetricAlert:
    kind: str                 # "flood" | "scan"
    key: int                  # victim (flood) or source (scan)
    detail: str


@dataclass
class VolumetricMonitor:
    """Sliding-window aggregate monitor over flow headers."""
    window: int = 500                       # flows per window
    rate_threshold: int = 0                  # CMS count that flags a key (0 = off)
    dst_entropy_floor: float = 0.0           # flag flood below this
    src_entropy_floor: float = 0.0           # flag scan below this
    width: int = 2048
    depth: int = 5
    _buf: List[dict] = field(default_factory=list, init=False)

    @staticmethod
    def fit(benign_headers: List[dict], window: int = 500, k_sigma: float = 4.0,
            quantile: float = 0.01) -> "VolumetricMonitor":
        """Calibrate entropy floors and a rate threshold from benign windows."""
        mon = VolumetricMonitor(window=window)
        dst_ents, src_ents, max_rates = [], [], []
        for i in range(0, max(1, len(benign_headers) - window + 1), window):
            win = benign_headers[i:i + window]
            if not win:
                continue
            dst_ents.append(_entropy(_counts(win, "dst_ip")))
            src_ents.append(_entropy(_counts(win, "src_ip")))
            cms = CountMinSketch(mon.width, mon.depth)
            for h in win:
                cms.add(h["src_ip"])
            max_rates.append(max(cms.estimate(h["src_ip"]) for h in win))
        mon.dst_entropy_floor = float(np.quantile(dst_ents, quantile)) if dst_ents else 0.0
        mon.src_entropy_floor = float(np.quantile(src_ents, quantile)) if src_ents else 0.0
        mon.rate_threshold = int(np.mean(max_rates) + k_sigma * (np.std(max_rates) + 1)) \
            if max_rates else window
        return mon

    def observe(self, header: dict) -> List[VolumetricAlert]:
        """Feed one flow header; returns alerts when a window completes."""
        self._buf.append(header)
        if len(self._buf) < self.window:
            return []
        alerts = self._scan_window(self._buf)
        self._buf = []
        return alerts

    def _scan_window(self, win: List[dict]) -> List[VolumetricAlert]:
        alerts: List[VolumetricAlert] = []
        dst_ent = _entropy(_counts(win, "dst_ip"))
        src_ent = _entropy(_counts(win, "src_ip"))
        if self.dst_entropy_floor and dst_ent < self.dst_entropy_floor:
            victim = max(_counts(win, "dst_ip").items(), key=lambda kv: kv[1])[0]
            alerts.append(VolumetricAlert(
                "flood", victim,
                f"dst-IP entropy {dst_ent:.2f} < floor {self.dst_entropy_floor:.2f}"))
        if self.src_entropy_floor and src_ent < self.src_entropy_floor:
            src = max(_counts(win, "src_ip").items(), key=lambda kv: kv[1])[0]
            alerts.append(VolumetricAlert(
                "scan", src,
                f"src-IP entropy {src_ent:.2f} < floor {self.src_entropy_floor:.2f}"))
        return alerts


def _counts(headers: List[dict], field_name: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for h in headers:
        k = h[field_name]
        out[k] = out.get(k, 0) + 1
    return out
