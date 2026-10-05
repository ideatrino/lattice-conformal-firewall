"""
Zero-dependency test suite for the Lattice-Conformal Firewall.

Runs under pytest if installed, or standalone:  python3 tests/test_lcf.py
Every `test_*` function asserts; the standalone runner reports pass/fail.
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcf import (
    Box, Rule, Policy, ALLOW, DENY, INSPECT, FIELDS,
    GaussianVerdict, SPRT, KLGate, ConformalGate, AdaptiveConformalGate,
    MondrianConformalGate, CountMinSketch, VolumetricMonitor, PageHinkley,
    GaussianVerdict, GaussianMixtureVerdict,
    build_frame, write_pcap, pcap_to_flows, parse_frame, ip_to_int,
    save_firewall, load_firewall, Engine, extract, gen_flow, train_firewall,
    gen_pcap_packets, evaluate,
)
from lcf.pcap import LINKTYPE_ETHERNET


# ---- Layer 2: lattice ----------------------------------------------------
def test_lattice_axioms():
    a = Box.from_spec(dst_ip="10.0.0.0/24", dst_port=443)
    b = Box.from_spec(dst_ip="10.0.0.0/16", dst_port="400-500")
    assert a.meet(a) == a and a.join(a) == a                 # idempotence
    assert a.meet(b) == b.meet(a) and a.join(b) == b.join(a)  # commutativity
    assert a.meet(a.join(b)) == a                             # absorption
    assert a.join(a.meet(b)) == a
    assert a.subseteq(b) and not b.subseteq(a)                # order


def test_lattice_shadow_and_conflict():
    pol = Policy([
        Rule.make(ALLOW, "web", dst_ip="10.0.0.0/24", dst_port=443),
        Rule.make(DENY,  "host", dst_ip="10.0.0.5", dst_port=443),   # shadowed
        Rule.make(ALLOW, "ssh-admin", src_ip="192.168.1.0/24", dst_port=22),
        Rule.make(DENY,  "ssh-all", dst_port=22),                     # conflict
    ])
    kinds = {a.kind for a in pol.analyze()}
    assert "shadowing" in kinds
    assert "generalization" in kinds or "correlation" in kinds


def test_canonical_equivalence():
    rng = np.random.default_rng(0)
    pol = Policy([
        Rule.make(ALLOW, "a", dst_ip="10.0.1.0/25", dst_port=443),
        Rule.make(ALLOW, "b", dst_ip="10.0.1.128/25", dst_port=443),
        Rule.make(DENY,  "d", dst_port=22),
    ])
    canon = pol.canonical()
    assert len(canon) < len(pol)                              # merged
    for _ in range(500):                                      # same decisions
        pkt = {f: int(rng.integers(0, 70000)) for f in FIELDS}
        pkt["dst_ip"] = int(rng.integers(ip_to_int("10.0.0.0"),
                                         ip_to_int("10.0.3.0")))
        assert pol.decide(pkt)[0] == canon.decide(pkt)[0]


# ---- Layer 5: conformal coverage ----------------------------------------
def test_conformal_coverage():
    for alpha in (0.1, 0.05, 0.02):
        fprs = []
        for t in range(8):
            rng = np.random.default_rng(100 + t)
            cal = rng.standard_normal(1500)
            gate = ConformalGate.fit(cal, alpha=alpha)
            test = rng.standard_normal(4000)
            fprs.append(np.mean([gate.blocks(s) for s in test]))
        assert np.mean(fprs) <= alpha + 0.01                  # <= alpha (+ MC slack)


# ---- Layer 4 / 3 ---------------------------------------------------------
def test_kl_gate_separates():
    rng = np.random.default_rng(1)
    benign = np.vstack([gen_flow(0, rng).byte_hist for _ in range(300)])
    gate = KLGate.fit(benign, eps=0.01)
    assert gate.score(gen_flow(0, rng).byte_hist) < gate.threshold * 3
    assert gate.score(gen_flow(1, rng).byte_hist) > gate.threshold * 5


def test_sprt_decides():
    rng = np.random.default_rng(2)
    mu_b, mu_m, s = 520.0, 300.0, 180.0
    llr = lambda x: (-0.5*((x-mu_m)/s)**2) - (-0.5*((x-mu_b)/s)**2)
    sprt = SPRT(0.01, 0.01)
    d = "continue"
    for _ in range(500):
        d = sprt.update(llr(rng.normal(mu_m, s)))
        if d != "continue":
            break
    assert d == "malicious" and sprt.n < 100


# ---- Layer 1 ingestion ---------------------------------------------------
def test_pcap_roundtrip():
    frame = build_frame("10.0.0.1", "8.8.8.8", 1234, 53, b"hello-dns")
    pk = parse_frame(frame, LINKTYPE_ETHERNET)
    assert pk.src_ip == ip_to_int("10.0.0.1") and pk.dst_port == 53
    assert pk.payload == b"hello-dns"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.pcap")
        write_pcap(p, [(0.1, frame), (0.2, build_frame(
            "8.8.8.8", "10.0.0.1", 53, 1234, b"reply"))])
        flows = pcap_to_flows(p)
        assert len(flows) == 1                                # bidirectional merge
        assert flows[0][1].pkt_sizes.size == 2


# ---- persistence ---------------------------------------------------------
def test_persistence_roundtrip():
    rng = np.random.default_rng(3)
    fw, _ = train_firewall(rng, n_train=400, n_cal=600)
    samples = [gen_flow(0, rng) for _ in range(20)] + \
              [gen_flow(1, rng) for _ in range(20)]
    before = [fw.decide(f).decision for f in samples]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        save_firewall(fw, p)
        fw2 = load_firewall(p)
    after = [fw2.decide(f).decision for f in samples]
    assert before == after


# ---- engine over a synthesised capture -----------------------------------
def test_engine_pcap():
    rng = np.random.default_rng(4)
    fw, _ = train_firewall(rng, via_pcap=True, n_train=400, n_cal=800)
    pkts, labels = gen_pcap_packets(rng, n_benign=15, n_loud=6, n_stealth=6)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mix.pcap")
        write_pcap(p, pkts)
        verdicts = Engine(fw).run_pcap(p)
    n_block = sum(v.verdict.decision == "BLOCK" for v in verdicts)
    assert len(verdicts) == 27
    assert n_block >= 10                                       # most malicious caught


# ---- adaptive conformal under distribution shift -------------------------
def test_adaptive_beats_static_under_drift():
    """Abrupt distribution shift at T/2 (adversary switches tactics). ACI's
    long-run average error returns to alpha; static conformal does not."""
    rng = np.random.default_rng(5)
    fw, cal = train_firewall(rng, n_train=800, n_cal=1500)
    alpha, gamma = 0.05, 0.1
    cal_std = float(np.std(cal))

    static = ConformalGate.fit(cal, alpha=alpha)
    adaptive = AdaptiveConformalGate(cal, alpha_target=alpha, gamma=gamma)

    T = 4000
    static_err, adapt_err = [], []
    for t in range(T):
        off = 0.0 if t < T // 2 else 0.8 * cal_std           # shift in score space
        s = fw.verdict.log_odds(extract(gen_flow(0, rng))) + off
        static_err.append(1.0 if static.blocks(s) else 0.0)
        adapt_err.append(1.0 if adaptive.observe(s, is_benign=True) else 0.0)

    static_fpr = float(np.mean(static_err))
    adapt_fpr = float(np.mean(adapt_err))
    bound = (alpha + gamma) / (gamma * T)                    # Gibbs-Candes
    assert abs(adapt_fpr - alpha) <= bound + 0.01            # ACI meets its bound
    assert abs(adapt_fpr - alpha) < abs(static_fpr - alpha)  # and beats static
    return static_fpr, adapt_fpr


# ---- property-based lattice verification (randomised axioms) --------------
def test_lattice_property_axioms():
    rng = np.random.default_rng(7)

    def rand_box():
        kw = {}
        for f in ("src_port", "dst_port"):
            lo = int(rng.integers(0, 60000)); hi = lo + int(rng.integers(0, 5000))
            kw[f] = (lo, hi)
        return Box.from_spec(**kw)

    for _ in range(3000):
        a, b, c = rand_box(), rand_box(), rand_box()
        # idempotence, commutativity, associativity, absorption
        assert a.meet(a) == a and a.join(a) == a
        assert a.join(b) == b.join(a)
        m1 = a.meet(b); m2 = b.meet(a)
        assert m1 == m2
        assert a.join(b.join(c)) == a.join(b).join(c)
        assert a.meet(a.join(b)) == a
        assert a.join(a.meet(b) or a) == a
        # order consistency: a <= b  <=>  a.meet(b) == a
        if a.subseteq(b):
            assert a.meet(b) == a and a.join(b) == b


def test_mondrian_per_group_coverage():
    rng = np.random.default_rng(11)
    gA = rng.normal(0, 1, 3000); gB = rng.normal(2, 1, 3000)
    scores = np.concatenate([gA, gB])
    keys = np.array(["A"] * 3000 + ["B"] * 3000)
    alpha = 0.05
    marg = ConformalGate.fit(scores, alpha=alpha)
    mond = MondrianConformalGate.fit(scores, keys, alpha=alpha)
    tA, tB = rng.normal(0, 1, 4000), rng.normal(2, 1, 4000)
    marg_B = np.mean([marg.blocks(s) for s in tB])
    mond_A = np.mean([mond.blocks(s, "A") for s in tA])
    mond_B = np.mean([mond.blocks(s, "B") for s in tB])
    assert marg_B > 1.5 * alpha                    # marginal over-blocks group B (~2x)
    assert mond_A <= alpha + 0.02                  # mondrian holds per group
    assert mond_B <= alpha + 0.02


def test_countmin_never_undercounts():
    rng = np.random.default_rng(0)
    cms = CountMinSketch(2048, 5)
    truth = {}
    for _ in range(20000):
        k = int(rng.integers(0, 400)); cms.add(k); truth[k] = truth.get(k, 0) + 1
    for k, v in truth.items():
        assert cms.estimate(k) >= v                # one-sided error guarantee


def test_volumetric_flood_and_scan():
    rng = np.random.default_rng(1)
    benign = [{"src_ip": int(rng.integers(0, 2**32)),
               "dst_ip": int(rng.integers(0, 2**32)),
               "src_port": int(rng.integers(1024, 65535)),
               "dst_port": 80, "proto": 6} for _ in range(4000)]
    mon = VolumetricMonitor.fit(benign, window=500)
    victim = 0x0A0A0A0A
    flood = [{"src_ip": int(rng.integers(0, 2**32)), "dst_ip": victim,
              "src_port": 1, "dst_port": 80, "proto": 6} for _ in range(500)]
    kinds = {a.kind for h in flood for a in mon.observe(h)}
    assert "flood" in kinds
    scanner = 0xC0A80001
    scan = [{"src_ip": scanner, "dst_ip": int(rng.integers(0, 2**32)),
             "src_port": 2, "dst_port": int(rng.integers(1, 65535)),
             "proto": 6} for _ in range(500)]
    kinds = {a.kind for h in scan for a in mon.observe(h)}
    assert "scan" in kinds


def test_pagehinkley_detects_shift_no_false_alarm():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 1000)
    ph = PageHinkley.fit(ref, delta=0.5, lam=30)
    assert sum(ph.update(x) for x in rng.normal(0, 1, 8000)) == 0   # stationary
    ph = PageHinkley.fit(ref, delta=0.5, lam=30)
    stream = np.concatenate([rng.normal(0, 1, 1500), rng.normal(1.5, 1, 1500)])
    at = next((t for t, x in enumerate(stream) if ph.update(x)), None)
    assert at is not None and at >= 1500 and at < 1700         # detected, after shift


def test_explainability_exact_and_pointed():
    rng = np.random.default_rng(3)
    fw, _ = train_firewall(rng, n_train=600, n_cal=800)
    x = extract(gen_flow(2, rng))                              # stealthy: big packets
    contribs, prior = fw.verdict.log_odds_contributions(x)
    assert np.isclose(contribs.sum() + prior, fw.verdict.log_odds(x))  # exact
    v = fw.decide(gen_flow(2, rng))
    assert v.drivers is not None and len(v.drivers) == 8
    assert v.drivers[0][0] == "mean_pkt_size"                  # dominant driver


def test_auc_perfect_on_separable():
    rng = np.random.default_rng(4)
    fw, _ = train_firewall(rng, n_train=600, n_cal=800)
    benign = [gen_flow(0, rng) for _ in range(500)]
    malicious = [gen_flow(1, rng) for _ in range(500)]
    s, y = evaluate.scores_labels(fw.verdict, benign, malicious)
    assert evaluate.auc(s, y) > 0.99


def test_gmm_beats_gaussian_on_multimodal():
    """On bimodal benign traffic with an attack in the valley, the single
    Gaussian fails (AUC < 0.5) while the GMM succeeds - and the conformal FPR
    guarantee holds for both scorers (model-agnostic)."""
    rng = np.random.default_rng(0)

    def benign(n):
        mode = rng.random(n) < 0.5
        size = np.where(mode, rng.normal(100, 15, n), rng.normal(900, 15, n))
        return np.column_stack([size, rng.normal(3, 0.3, n),
                                rng.normal(0.05, 0.01, n)])

    def attack(n):
        return np.column_stack([rng.normal(500, 40, n), rng.normal(3, 0.3, n),
                                rng.normal(0.05, 0.01, n)])

    Xb, Xm = benign(3000), attack(3000)
    cal, tb, tm = benign(1500), benign(3000), attack(3000)
    g = GaussianVerdict.fit(Xb, Xm, prior_m=0.5)
    gm = GaussianMixtureVerdict.fit(Xb, Xm, prior_m=0.5, K=2)

    def auc_fpr(v):
        nb = np.array([v.nonconformity(x) for x in tb])
        nm = np.array([v.nonconformity(x) for x in tm])
        s = np.concatenate([nb, nm])
        y = np.concatenate([np.zeros(len(nb)), np.ones(len(nm))])
        gate = ConformalGate.fit(np.array([v.nonconformity(x) for x in cal]),
                                 alpha=0.05)
        return evaluate.auc(s, y), float(np.mean([gate.blocks(x) for x in nb]))

    auc_g, fpr_g = auc_fpr(g)
    auc_gm, fpr_gm = auc_fpr(gm)
    assert auc_g < 0.5                       # single Gaussian fails on multimodal
    assert auc_gm > 0.95                      # GMM succeeds
    assert fpr_g <= 0.07 and fpr_gm <= 0.07   # guarantee holds for BOTH scorers


# --------------------------------------------------------------------------
def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("running LCF test suite")
    ok = _run_standalone()
    sys.exit(0 if ok else 1)
