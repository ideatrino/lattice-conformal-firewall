"""
Live engine - run the firewall over captured or real-time traffic.

Two sources:
  * run_pcap(path)      - replay a capture file (fully supported & tested).
  * run_nfqueue(...)    - inline on a Linux host via netfilter NFQUEUE
                          (requires root + the `netfilterqueue` package; the
                          import is guarded so the module loads without it).

Decisions are made per flow.  In inline mode a per-5-tuple verdict cache is
built once a flow has accumulated `min_pkts` packets; earlier packets of a new
flow are passed (fail-open) until the first verdict, then the cached verdict is
applied to the rest of the flow.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .features import Flow
from .firewall import LatticeConformalFirewall, Verdict
from .pcap import (read_pcap, parse_frame, FlowAssembler, Packet, _flow_key,
                   int_to_ip)


@dataclass
class FlowVerdict:
    header: dict
    verdict: Verdict

    def line(self) -> str:
        h = self.header
        return (f"{self.verdict.decision:5} [{self.verdict.layer:9}] "
                f"{int_to_ip(h['src_ip'])}:{h['src_port']} -> "
                f"{int_to_ip(h['dst_ip'])}:{h['dst_port']} "
                f"proto={h['proto']} | {self.verdict.reason}")


class Engine:
    def __init__(self, firewall: LatticeConformalFirewall,
                 idle_timeout: float = 15.0):
        self.fw = firewall
        self.idle_timeout = idle_timeout

    # ---- replay (tested path) -------------------------------------------
    def run_pcap(self, path: str,
                 on_verdict: Optional[Callable[[FlowVerdict], None]] = None
                 ) -> List[FlowVerdict]:
        asm = FlowAssembler(self.idle_timeout)
        out: List[FlowVerdict] = []

        def emit(header, flow):
            fv = FlowVerdict(header, self.fw.decide(flow))
            out.append(fv)
            if on_verdict:
                on_verdict(fv)

        for ts, raw, lt in read_pcap(path):
            p = parse_frame(raw, lt)
            if p is None:
                continue
            asm.add_packet(p, ts)
            for header, flow in asm.flush_idle(ts):
                emit(header, flow)
        for header, flow in asm.flush_all():
            emit(header, flow)
        return out

    # ---- inline on a live interface (guarded; needs root) ---------------
    def run_nfqueue(self, queue_num: int = 0, min_pkts: int = 4):  # pragma: no cover
        try:
            from netfilterqueue import NetfilterQueue  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "run_nfqueue needs the 'netfilterqueue' package and root. "
                "Set up an iptables NFQUEUE rule, e.g.:\n"
                "  iptables -I FORWARD -j NFQUEUE --queue-num 0\n"
                f"(import failed: {e})")

        from .pcap import LINKTYPE_RAW
        asm = FlowAssembler(self.idle_timeout)
        cache: Dict[tuple, str] = {}
        counts: Dict[tuple, int] = {}

        def handle(pkt):  # netfilterqueue passes raw IP packets
            raw = pkt.get_payload()
            parsed = parse_frame(raw, LINKTYPE_RAW)
            if parsed is None:
                pkt.accept()
                return
            key = _flow_key(parsed)
            asm.add_packet(parsed, parsed.ts)
            counts[key] = counts.get(key, 0) + 1
            if key not in cache and counts[key] >= min_pkts:
                acc = asm._acc.get(key)
                if acc is not None:
                    cache[key] = self.fw.decide(acc.to_flow()).decision
            decision = cache.get(key, "ALLOW")   # fail-open before first verdict
            pkt.drop() if decision == "BLOCK" else pkt.accept()

        nfq = NetfilterQueue()
        nfq.bind(queue_num, handle)
        try:
            nfq.run()
        finally:
            nfq.unbind()


def print_verdict(fv: FlowVerdict):
    print(fv.line())
