r"""
Layer 2 - The rule lattice.

A firewall policy is modelled as elements of a bounded lattice of
hyperrectangles ("boxes") over the packet-header space.  This gives us
*provable* answers to questions existing firewalls can only guess at:

    * Is this rule dead (shadowed) ?           -> decidable
    * Do these two rules contradict ?          -> decidable
    * Is this rule redundant ?                 -> decidable
    * What is the canonical minimal policy ?   -> computable

The lattice
-----------
Let F_1 .. F_D be the header fields, each a finite integer range [0, M_i].
A *box* is a product of per-field intervals  b = X_i [lo_i, hi_i].

Over the set B of all boxes (plus a formal empty box bottom):

    meet   b1 /\ b2  = per-field interval intersection        (closed in B)
    join   b1 \/ b2  = smallest box containing both           (closed in B)
    bottom = empty                      top = whole space

(B, subseteq, /\, \/) is a bounded lattice: meet/join are idempotent,
commutative, associative and satisfy absorption (see SPEC.md for the proof).
Subsumption  b1 subseteq b2  is the lattice order.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import ipaddress


# --------------------------------------------------------------------------
# Header schema.  Each field is an inclusive integer range [0, max].
# IPv4 addresses are flattened to 32-bit integers so CIDRs become intervals.
# --------------------------------------------------------------------------
SCHEMA: Dict[str, int] = {
    "src_ip":   2**32 - 1,
    "dst_ip":   2**32 - 1,
    "src_port": 65535,
    "dst_port": 65535,
    "proto":    255,          # IP protocol number (6=TCP, 17=UDP, ...)
}
FIELDS: Tuple[str, ...] = tuple(SCHEMA.keys())

ALLOW, DENY, INSPECT = "ALLOW", "DENY", "INSPECT"


def _ip_interval(spec) -> Tuple[int, int]:
    """Accept an int, a dotted IP, a CIDR, or '*' -> (lo, hi) 32-bit interval."""
    if spec in ("*", None):
        return (0, 2**32 - 1)
    if isinstance(spec, int):
        return (spec, spec)
    if "/" in spec:
        net = ipaddress.ip_network(spec, strict=False)
        return (int(net.network_address), int(net.broadcast_address))
    a = int(ipaddress.ip_address(spec))
    return (a, a)


def _port_interval(spec, maxv) -> Tuple[int, int]:
    if spec in ("*", None):
        return (0, maxv)
    if isinstance(spec, int):
        return (spec, spec)
    if isinstance(spec, (tuple, list)):
        return (int(spec[0]), int(spec[1]))
    if "-" in str(spec):
        lo, hi = str(spec).split("-")
        return (int(lo), int(hi))
    return (int(spec), int(spec))


@dataclass(frozen=True)
class Box:
    """A hyperrectangle in header space: field -> (lo, hi) inclusive."""
    bounds: Tuple[Tuple[int, int], ...]   # one (lo,hi) per FIELDS, in order

    # ---- construction -----------------------------------------------------
    @staticmethod
    def from_spec(**kw) -> "Box":
        b = []
        for f in FIELDS:
            spec = kw.get(f, "*")
            if f in ("src_ip", "dst_ip"):
                b.append(_ip_interval(spec))
            else:
                b.append(_port_interval(spec, SCHEMA[f]))
        return Box(tuple(b))

    @staticmethod
    def full() -> "Box":
        return Box(tuple((0, SCHEMA[f]) for f in FIELDS))

    # ---- lattice primitives ----------------------------------------------
    def is_empty(self) -> bool:
        return any(lo > hi for lo, hi in self.bounds)

    def meet(self, other: "Box") -> Optional["Box"]:
        """Greatest lower bound: intersection.  None if empty (bottom)."""
        nb = tuple((max(a[0], b[0]), min(a[1], b[1]))
                   for a, b in zip(self.bounds, other.bounds))
        b = Box(nb)
        return None if b.is_empty() else b

    def join(self, other: "Box") -> "Box":
        """Least upper bound: smallest enclosing box (convex closure)."""
        nb = tuple((min(a[0], b[0]), max(a[1], b[1]))
                   for a, b in zip(self.bounds, other.bounds))
        return Box(nb)

    def subseteq(self, other: "Box") -> bool:
        """Lattice order: self is contained in other."""
        return all(b[0] <= a[0] and a[1] <= b[1]
                   for a, b in zip(self.bounds, other.bounds))

    def overlaps(self, other: "Box") -> bool:
        return self.meet(other) is not None

    def contains_point(self, pkt: Dict[str, int]) -> bool:
        for f, (lo, hi) in zip(FIELDS, self.bounds):
            v = pkt[f]
            if not (lo <= v <= hi):
                return False
        return True

    def size(self) -> int:
        """Cardinality of the region (number of header tuples it covers)."""
        n = 1
        for lo, hi in self.bounds:
            n *= (hi - lo + 1)
        return n

    def __repr__(self) -> str:
        parts = []
        for f, (lo, hi) in zip(FIELDS, self.bounds):
            full = (lo, hi) == (0, SCHEMA[f])
            parts.append(f"{f}=*" if full else f"{f}=[{lo},{hi}]")
        return "Box(" + ", ".join(parts) + ")"


@dataclass
class Rule:
    box: Box
    action: str
    name: str = ""

    @staticmethod
    def make(action: str, name: str = "", **kw) -> "Rule":
        return Rule(Box.from_spec(**kw), action, name)


# --------------------------------------------------------------------------
# Anomaly taxonomy (shadowing / redundancy / correlation) and normal form.
# Policies are first-match-wins, like every real firewall.
# --------------------------------------------------------------------------
@dataclass
class Anomaly:
    kind: str            # "shadowing" | "redundancy" | "correlation" | "generalization"
    i: int               # index of the affected (usually later) rule
    j: int               # index of the earlier rule involved
    detail: str


class Policy:
    def __init__(self, rules: List[Rule]):
        self.rules = list(rules)

    # first-match evaluation ----------------------------------------------
    def decide(self, pkt: Dict[str, int]) -> Tuple[str, int]:
        for idx, r in enumerate(self.rules):
            if r.box.contains_point(pkt):
                return r.action, idx
        return INSPECT, -1          # default: hand to the statistical pipeline

    # ---- the four classical inter-rule anomalies -------------------------
    def analyze(self) -> List[Anomaly]:
        out: List[Anomaly] = []
        for i in range(len(self.rules)):
            ri = self.rules[i]
            for j in range(i):
                rj = self.rules[j]
                if not ri.box.overlaps(rj.box):
                    continue
                sub = ri.box.subseteq(rj.box)
                sup = rj.box.subseteq(ri.box)
                same = ri.action == rj.action
                if sub and not same:
                    out.append(Anomaly(
                        "shadowing", i, j,
                        f"'{ri.name or i}' is fully covered by earlier "
                        f"'{rj.name or j}' with opposite action; it can never fire."))
                elif sub and same:
                    out.append(Anomaly(
                        "redundancy", i, j,
                        f"'{ri.name or i}' is covered by earlier '{rj.name or j}' "
                        f"with the same action; removing it changes nothing."))
                elif sup and not same:
                    out.append(Anomaly(
                        "generalization", i, j,
                        f"'{ri.name or i}' is more general than earlier "
                        f"'{rj.name or j}' and disagrees; order-dependent."))
                elif not sub and not sup and not same:
                    out.append(Anomaly(
                        "correlation", i, j,
                        f"'{ri.name or i}' and '{rj.name or j}' partially overlap "
                        f"with conflicting actions; verdict depends on ordering."))
        return out

    # ---- canonical normal form ------------------------------------------
    def canonical(self) -> "Policy":
        """
        Deterministic reduction:
          1. drop shadowed rules  (dead code - they never match)
          2. drop redundant rules (same-action cover by an earlier rule)
          3. merge same-action boxes that differ in exactly one field and
             are adjacent/overlapping there, to a fixpoint.
        The result is policy-equivalent on every packet.
        """
        anomalies = self.analyze()
        drop = {a.i for a in anomalies if a.kind in ("shadowing", "redundancy")}
        kept = [r for k, r in enumerate(self.rules) if k not in drop]
        kept = self._merge_fixpoint(kept)
        return Policy(kept)

    @staticmethod
    def _mergeable(a: Box, b: Box) -> Optional[Box]:
        """If a,b differ in exactly one field and touch/overlap there, return
        the merged box; else None."""
        diff = [k for k in range(len(FIELDS)) if a.bounds[k] != b.bounds[k]]
        if len(diff) == 0:
            return a                      # identical
        if len(diff) != 1:
            return None
        k = diff[0]
        (alo, ahi), (blo, bhi) = a.bounds[k], b.bounds[k]
        # adjacent or overlapping intervals merge into one
        if max(alo, blo) <= min(ahi, bhi) + 1:
            merged = list(a.bounds)
            merged[k] = (min(alo, blo), max(ahi, bhi))
            return Box(tuple(merged))
        return None

    def _merge_fixpoint(self, rules: List[Rule]) -> List[Rule]:
        rules = list(rules)
        changed = True
        while changed:
            changed = False
            for i in range(len(rules)):
                for j in range(i + 1, len(rules)):
                    if rules[i].action != rules[j].action:
                        continue
                    m = self._mergeable(rules[i].box, rules[j].box)
                    if m is not None:
                        rules[i] = Rule(m, rules[i].action,
                                        rules[i].name or rules[j].name)
                        del rules[j]
                        changed = True
                        break
                if changed:
                    break
        return rules

    def __len__(self):
        return len(self.rules)
