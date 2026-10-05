"""
Real-traffic ingestion - pure-Python libpcap reader/writer and flow assembly.

No external dependencies (no scapy/dpkt).  Parses classic .pcap files
(microsecond and nanosecond magic, both byte orders), decodes
Ethernet / IPv4 / TCP|UDP, and assembles packets into bidirectional flows
that become `lcf.features.Flow` objects for the firewall.

A matching writer (`write_pcap`, `build_frame`) lets us synthesise valid
capture files for tests and the replay engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple
import struct
import socket
import numpy as np

from .features import Flow

# ---- link / ether / ip constants ----------------------------------------
LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_RAW_ALT = (12, 14)          # some stacks use these for raw IP
ETHERTYPE_IP = 0x0800
PROTO_TCP = 6
PROTO_UDP = 17

_MAGICS = {
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),   # big-endian, microsecond
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),   # little-endian, microsecond
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),   # big-endian, nanosecond
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),   # little-endian, nanosecond
}


@dataclass
class Packet:
    ts: float
    src_ip: int
    dst_ip: int
    src_port: int
    dst_port: int
    proto: int
    payload: bytes


def ip_to_int(s: str) -> int:
    return struct.unpack("!I", socket.inet_aton(s))[0]


def int_to_ip(i: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", i))


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
def read_pcap(path: str) -> Iterator[Tuple[float, bytes, int]]:
    """Yield (timestamp_seconds, raw_frame_bytes, linktype)."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic not in _MAGICS:
            raise ValueError(f"not a supported pcap file (magic {magic!r})")
        endian, ts_scale = _MAGICS[magic]
        # rest of 24-byte global header: ver(2+2) tz(4) sig(4) snap(4) net(4)
        rest = fh.read(20)
        (_vmaj, _vmin, _tz, _sig, _snap, linktype) = struct.unpack(
            endian + "HHiIII", rest)
        rec = endian + "IIII"
        while True:
            hdr = fh.read(16)
            if len(hdr) < 16:
                break
            ts_s, ts_frac, incl, _orig = struct.unpack(rec, hdr)
            data = fh.read(incl)
            if len(data) < incl:
                break
            yield (ts_s + ts_frac * ts_scale, data, linktype)


def parse_frame(raw: bytes, linktype: int) -> Optional[Packet]:
    """Decode one frame to a Packet, or None if not IPv4 TCP/UDP."""
    ip_off = None
    if linktype == LINKTYPE_ETHERNET:
        if len(raw) < 14:
            return None
        etype = struct.unpack("!H", raw[12:14])[0]
        if etype != ETHERTYPE_IP:
            return None
        ip_off = 14
    elif linktype in (LINKTYPE_RAW, *LINKTYPE_RAW_ALT):
        ip_off = 0
    elif linktype == LINKTYPE_NULL:
        ip_off = 4                      # 4-byte family header
    else:
        return None

    if len(raw) < ip_off + 20:
        return None
    vihl = raw[ip_off]
    if (vihl >> 4) != 4:                # IPv4 only
        return None
    ihl = (vihl & 0x0F) * 4
    proto = raw[ip_off + 9]
    src_ip, dst_ip = struct.unpack("!II", raw[ip_off + 12:ip_off + 20])
    l4 = ip_off + ihl
    if proto == PROTO_TCP:
        if len(raw) < l4 + 20:
            return None
        sport, dport = struct.unpack("!HH", raw[l4:l4 + 4])
        data_off = (raw[l4 + 12] >> 4) * 4
        payload = raw[l4 + data_off:]
    elif proto == PROTO_UDP:
        if len(raw) < l4 + 8:
            return None
        sport, dport = struct.unpack("!HH", raw[l4:l4 + 4])
        payload = raw[l4 + 8:]
    else:
        return None
    return Packet(0.0, src_ip, dst_ip, sport, dport, proto, payload)


# --------------------------------------------------------------------------
# Flow assembly (bidirectional, keyed on the unordered 5-tuple)
# --------------------------------------------------------------------------
def _flow_key(p: Packet):
    a = (p.src_ip, p.src_port)
    b = (p.dst_ip, p.dst_port)
    lo, hi = sorted((a, b))
    return (lo, hi, p.proto)


@dataclass
class _Acc:
    initiator: Tuple[int, int]          # (ip, port) of first packet's source
    header: Dict[str, int]
    ts: List[float] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    hist: np.ndarray = field(default_factory=lambda: np.zeros(256, float))
    bytes_up: float = 0.0               # initiator -> responder
    bytes_down: float = 0.0             # responder -> initiator
    last_ts: float = 0.0

    def add(self, p: Packet, ts: float):
        self.ts.append(ts)
        n = len(p.payload)
        self.sizes.append(n)
        if n:
            counts = np.bincount(np.frombuffer(p.payload, dtype=np.uint8),
                                 minlength=256)
            self.hist += counts
        if (p.src_ip, p.src_port) == self.initiator:
            self.bytes_up += n
        else:
            self.bytes_down += n
        self.last_ts = ts

    def to_flow(self) -> Flow:
        ts = np.array(self.ts, float)
        iats = np.diff(np.sort(ts)) if ts.size > 1 else np.array([0.0])
        sizes = np.array(self.sizes, float)
        if sizes.size == 0:
            sizes = np.array([0.0])
        return Flow(header=self.header, pkt_sizes=sizes, iats=iats,
                    byte_hist=self.hist.copy(),
                    bytes_down=self.bytes_down, bytes_up=self.bytes_up)


class FlowAssembler:
    """Incremental assembler; supports batch and timeout-based flushing."""

    def __init__(self, idle_timeout: float = 15.0):
        self.idle_timeout = idle_timeout
        self._acc: Dict[tuple, _Acc] = {}

    def add_packet(self, p: Packet, ts: float):
        k = _flow_key(p)
        acc = self._acc.get(k)
        if acc is None:
            header = {"src_ip": p.src_ip, "dst_ip": p.dst_ip,
                      "src_port": p.src_port, "dst_port": p.dst_port,
                      "proto": p.proto}
            acc = _Acc(initiator=(p.src_ip, p.src_port), header=header)
            self._acc[k] = acc
        acc.add(p, ts)

    def flush_idle(self, now: float) -> List[Tuple[dict, Flow]]:
        """Return and remove flows idle longer than the timeout."""
        done = []
        for k, acc in list(self._acc.items()):
            if now - acc.last_ts >= self.idle_timeout:
                done.append((acc.header, acc.to_flow()))
                del self._acc[k]
        return done

    def flush_all(self) -> List[Tuple[dict, Flow]]:
        out = [(a.header, a.to_flow()) for a in self._acc.values()]
        self._acc.clear()
        return out


def pcap_to_flows(path: str) -> List[Tuple[dict, Flow]]:
    asm = FlowAssembler(idle_timeout=float("inf"))
    for ts, raw, lt in read_pcap(path):
        p = parse_frame(raw, lt)
        if p is not None:
            asm.add_packet(p, ts)
    return asm.flush_all()


# --------------------------------------------------------------------------
# Writer - build valid Ethernet/IPv4/TCP|UDP frames and .pcap files
# --------------------------------------------------------------------------
def _ip_checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    s = sum(struct.unpack("!%dH" % (len(data) // 2), data))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def build_frame(src_ip, dst_ip, sport, dport, payload=b"", proto=PROTO_TCP,
                src_mac=b"\x02\x00\x00\x00\x00\x01",
                dst_mac=b"\x02\x00\x00\x00\x00\x02") -> bytes:
    """Construct a complete Ethernet frame carrying IPv4 + TCP/UDP + payload."""
    if isinstance(src_ip, str):
        src_ip = ip_to_int(src_ip)
    if isinstance(dst_ip, str):
        dst_ip = ip_to_int(dst_ip)
    if proto == PROTO_TCP:
        l4 = struct.pack("!HHIIBBHHH", sport, dport, 0, 0,
                         (5 << 4), 0x18, 65535, 0, 0) + payload
    else:
        length = 8 + len(payload)
        l4 = struct.pack("!HHHH", sport, dport, length, 0) + payload
    total = 20 + len(l4)
    ip = struct.pack("!BBHHHBBH", 0x45, 0, total, 0, 0, 64, proto, 0)
    ip += struct.pack("!II", src_ip, dst_ip)
    ip = ip[:10] + struct.pack("!H", _ip_checksum(ip)) + ip[12:]
    eth = dst_mac + src_mac + struct.pack("!H", ETHERTYPE_IP)
    return eth + ip + l4


def write_pcap(path: str, packets: List[Tuple[float, bytes]],
               linktype: int = LINKTYPE_ETHERNET):
    """packets = list of (timestamp_seconds, raw_frame_bytes)."""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
        for ts, raw in packets:
            sec = int(ts)
            usec = int(round((ts - sec) * 1e6))
            fh.write(struct.pack("<IIII", sec, usec, len(raw), len(raw)))
            fh.write(raw)
