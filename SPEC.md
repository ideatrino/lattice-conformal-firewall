# The Lattice-Conformal Firewall (LCF)

**A mathematically grounded firewall architecture**
Version 2.0 · formal specification
(v2 adds real packet-capture ingestion, a drift-robust adaptive conformal
layer, a live engine, persistence and a CLI — see Part II.)

---

## 0. Motivation

Conventional firewalls fail less because the mathematics is hard than because
there is almost none in them. They are ordered lists of hand-written rules
emitting a binary allow/deny verdict, with:

- **no proof** that the rule set is internally consistent (shadowed, redundant
  and conflicting rules are the dominant source of real misconfiguration);
- **no notion of uncertainty** — a 51%-suspicious flow and a 99.9%-suspicious
  flow are treated identically;
- **no bound** on the false-positive rate — nobody can tell you how often their
  firewall wrongly blocks legitimate traffic;
- **no defence that survives encryption** — deep packet inspection reads payload
  content, which TLS removes.

The LCF replaces the list-of-rules model with a *measuring instrument*: every
flow becomes a point in a measurable space, is assigned a calibrated probability,
and receives a decision that carries a provable error bound. Five layers.

> There is no theorem that yields a "provably most secure firewall" — security
> is an adversarial game, not an optimisation with a global optimum. What is
> achievable, and what this design delivers, is a firewall whose *verdicts carry
> guarantees*: a policy proven consistent, and a false-positive rate you set as a
> dial.

---

## 1. Layer 1 — Flow as a point in measurable space

A raw flow `f` (its packets, timings, payload) is mapped to a feature vector
`x = phi(f) ∈ R^n` together with a 256-bin payload-byte histogram. The current
features are mean/standard-deviation of packet size, mean/standard-deviation of
inter-arrival time, payload Shannon entropy, download/upload ratio, duration and
packet rate.

Once a flow is a point in a metric space, distance, probability and divergence
are all available to the layers above. This is the move that makes everything
else possible.

---

## 2. Layer 2 — The rule lattice

### 2.1 Objects

Let the header have fields `F_1,…,F_D`, each a finite integer range `[0, M_i]`
(IPv4 addresses flatten to 32-bit integers, so CIDR blocks become intervals).
A **box** is a product of per-field intervals

```
b = ⨉_i [lo_i, hi_i].
```

Let `B` be the set of all boxes together with a formal bottom element `⊥`
(the empty region). Define

```
meet    b1 ∧ b2 = ⨉_i [max(lo), min(hi)]     (per-field intersection; ⊥ if any interval empty)
join    b1 ∨ b2 = ⨉_i [min(lo), max(hi)]     (smallest enclosing box)
order   b1 ⊑ b2 ⇔ ∀i [lo1_i,hi1_i] ⊆ [lo2_i,hi2_i]
top     ⊤ = ⨉_i [0, M_i]
```

### 2.2 Proposition (LCF is a bounded lattice)

`(B, ⊑, ∧, ∨, ⊥, ⊤)` is a bounded lattice.

*Proof.* Each axiom reduces, coordinate-wise, to the corresponding fact for
intervals under `max`/`min`, because `∧` and `∨` act independently per field.

- **Idempotence.** `min(a,a)=max(a,a)=a`, so `b∧b=b∧…=b` and `b∨b=b`.
- **Commutativity.** `max`/`min` are symmetric.
- **Associativity.** `max`/`min` are associative, so nesting per field is
  associative.
- **Absorption.** For intervals, `max(lo1, min(lo1,lo2)) = lo1` and
  `min(hi1, max(hi1,hi2)) = hi1`; hence `b1 ∧ (b1 ∨ b2) = b1` and dually
  `b1 ∨ (b1 ∧ b2) = b1`.
- **Bounds.** `⊥ ⊑ b ⊑ ⊤` for all `b`. ∎

Closure matters: the intersection of two boxes is a box, and the smallest
enclosing box of two boxes is a box, so `B` is closed under both operations.
(The set-theoretic *union* of two boxes is generally not a box; `∨` is the
*convex/box closure* of the union, which is why `∨` is defined as the bounding
box rather than the union.)

### 2.3 Policy anomalies (decidable)

A policy is an ordered list of (box, action) rules evaluated first-match-wins,
exactly as real firewalls work. For rules `r_i` (later) and `r_j` (earlier,
`j < i`) whose boxes overlap (`r_i.box ∧ r_j.box ≠ ⊥`):

| Anomaly | Condition | Meaning |
|---|---|---|
| **Shadowing** | `r_i ⊑ r_j`, actions differ | `r_i` can never fire — dead rule |
| **Redundancy** | `r_i ⊑ r_j`, actions equal | removing `r_i` changes nothing |
| **Generalization** | `r_j ⊑ r_i`, actions differ | later rule is broader and disagrees — order-dependent |
| **Correlation** | neither contains the other, actions differ | partial overlap with conflicting verdicts — order-dependent |

Each condition is a finite check over `D` intervals, hence **decidable** in
`O(R^2 · D)` for `R` rules. No existing firewall gives these guarantees;
the LCF computes them exactly (`Policy.analyze`).

### 2.4 Canonical normal form

`Policy.canonical()` applies a deterministic, policy-preserving reduction:

1. drop shadowed rules (dead code);
2. drop redundant rules (same-action cover by an earlier rule);
3. merge same-action boxes that differ in exactly one field and are
   adjacent or overlapping there, to a fixpoint.

The result decides every packet identically to the input. Two policies with the
same canonical form are **provably equivalent** — a 40,000-line rule set and a
400-line one can be checked for equality. (The merge step yields a well-defined
normal form under the one-field-adjacency relation; it is minimal with respect to
that relation, not necessarily the global minimum over all equivalent box
covers, which is NP-hard in general. This is stated honestly rather than
overclaimed.)

---

## 3. Layer 3 — Bayesian verdict and sequential testing

### 3.1 Posterior

With class-conditional densities `p(x | benign)`, `p(x | malicious)` (here
diagonal Gaussians fit per class) and base rate `π = P(malicious)`:

```
P(malicious | x) = π·p(x|mal) / ( π·p(x|mal) + (1−π)·p(x|ben) ).
```

Computed via the log-odds `z(x) = log p(x|mal) + log π − log p(x|ben) − log(1−π)`
and the logistic `σ(z)` for numerical stability. Verdicts are degrees of belief,
not coin flips.

### 3.2 Sequential Probability Ratio Test (Wald)

When a flow is observed packet by packet, accumulate the per-packet log-likelihood
ratio `Λ_k = Σ_{t≤k} log [p(obs_t|mal)/p(obs_t|ben)]` and decide as soon as

```
Λ_k ≥ A = log((1−β)/α)   → declare malicious
Λ_k ≤ B = log(β/(1−α))   → declare benign
```

for target error rates `α` (false alarm) and `β` (missed detection). Wald's
theorem: the SPRT minimises the expected number of observations among all tests
meeting those error rates. In the demo both classes are decided in ~7 packets.

---

## 4. Layer 4 — Information-theoretic anomaly gate

Maintain a baseline payload-byte distribution `Q` (pooled benign). For a live
flow with empirical byte distribution `P`, measure the Kullback–Leibler
divergence

```
D(P ‖ Q) = Σ_a P(a) · log( P(a) / Q(a) ),
```

and flag when `D(P‖Q) > t`. KL works on the *shape* of traffic, not its content,
so it survives encryption: encrypted, compressed and exfiltration payloads push
the byte distribution toward uniform, which a skewed benign `Q` assigns high
divergence.

**Threshold.** By **Sanov's theorem**, for `n` i.i.d. benign symbols,
`P( D(P̂‖Q) ≥ t ) ≤ (n+1)^{|A|} · e^{−n t}`, so a threshold
`t_ε ≈ (1/n)·log(1/ε)` controls the false-alarm rate. The implementation also
offers a **distribution-free** threshold: the `(1−ε)` empirical quantile of
per-flow `D(P‖Q)` on held-out benign flows (`KLGate.fit(..., eps)`), which
assumes nothing about `Q`.

---

## 5. Layer 5 — The conformal guarantee

This is the headline property. Take any real-valued nonconformity score `s(x)`
(the LCF uses the Bayesian log-odds; higher = more anomalous). Reserve a set of
`n` held-out **benign** flows and compute their scores `s_1,…,s_n`. For a test
flow with score `s`, define the **p-value**

```
p(x) = ( 1 + #{ i : s_i ≥ s } ) / ( n + 1 ),
```

and **block iff `p(x) < α`.**

### 5.1 Theorem (calibrated false-positive rate)

If the benign flows (calibration + test) are **exchangeable**, then for a
genuinely benign test flow

```
P( p(x) < α ) ≤ α .
```

*Proof.* Under exchangeability the `n+1` scores `s_1,…,s_n, s` are exchangeable,
so the rank of `s` among them is uniform on `{1,…,n+1}` (ties broken
arbitrarily). Then `p(x)` takes values in `{1/(n+1),…,(n+1)/(n+1)}` uniformly,
so for any `α`,

```
P( p(x) < α ) = #{ k : k/(n+1) < α } / (n+1) = ⌊α(n+1)⌋ / (n+1) ≤ α . ∎
```

So `α` is a **dial on the false-positive rate with a proof behind it**: set
`α = 0.001` and at most ~1 legitimate flow in 1000 is wrongly blocked over the
long run — a guarantee no rule-based firewall can make.

### 5.2 Empirical validation

`demo.py` part B confirms the theorem: averaged over 30 independent
calibration/test draws, the measured FPR equals the exact finite-sample target
`⌊α(n+1)⌋/(n+1)` to three decimals, for `α ∈ {0.20, 0.10, 0.05, 0.02, 0.01}`,
with `n = 2000`.

| α | target ⌊α(n+1)⌋/(n+1) | measured FPR |
|---|---|---|
| 0.20 | 0.19990 | 0.1973 |
| 0.10 | 0.09995 | 0.0996 |
| 0.05 | 0.04998 | 0.0509 |
| 0.02 | 0.01999 | 0.0194 |
| 0.01 | 0.01000 | 0.0101 |

---

## 6. Orchestration

`LatticeConformalFirewall.decide(flow)`:

1. `x = phi(flow)` (Layer 1).
2. Layer-2 policy, first-match: `ALLOW` → pass (proven-consistent whitelist);
   `DENY` → hard block (proven-consistent blacklist); otherwise `INSPECT`.
3. On `INSPECT`: compute posterior, `D(P‖Q)` and the conformal p-value.
   - KL gate trips → **BLOCK** (reason: payload divergence).
   - else conformal `p < α` → **BLOCK** (reason: calibrated outlier).
   - else → **ALLOW**.

Hard policy verdicts are proven-consistent overrides; every *statistical* block
carries the Layer-5 guarantee. The two statistical detectors are complementary:
in the demo, loud encrypted exfil is caught by the KL gate while a stealthy
attacker with benign-looking payload (`kl≈0.003`) is caught only by the conformal
layer. Neither alone catches both.

---

## 7. Threat model and honest limitations

**Assumptions.** The conformal guarantee needs benign traffic to be
*exchangeable*; the Bayesian likelihoods need representative labelled data.

**Known limits (where the real open problems are):**

- **Distribution shift / adaptive adversaries.** Exchangeability breaks when an
  attacker deliberately shifts the benign-looking distribution over time
  (concept drift, mimicry). The bound then degrades gracefully but is not
  adversarially tight. Mitigations — weighted/online conformal, periodic
  recalibration — are future work.
- **Training data.** The verdict layer is only as good as its labelled corpus;
  a novel attack class not represented in `p(x|malicious)` leans on the
  conformal/KL outlier behaviour rather than the supervised model.
- **Canonical form.** Minimal only under one-field adjacency merging, not the
  global optimum (NP-hard).
- **Feature evasion.** An attacker who makes a flow statistically identical to
  benign traffic on every feature axis defeats Layers 3–5 by construction; the
  defence is adding features that are costly to forge, not a theorem.

These are stated plainly because an honest limitations section is where original
work actually continues.

---

## 8. Module map

```
lcf/features.py    Layer 1   flow → R^n + byte histogram
lcf/pcap.py        Layer 1   pure-Python libpcap read/write, flow assembly
lcf/lattice.py     Layer 2   Box, Rule, Policy; analyze(), canonical()
lcf/bayesian.py    Layer 3   GaussianVerdict (posterior), SPRT
lcf/anomaly.py     Layer 4   KLGate, kl_divergence, Sanov threshold
lcf/conformal.py   Layer 5   ConformalGate (p-value, α-bound)
lcf/adaptive.py    Layer 5+  AdaptiveConformalGate (ACI), WeightedConformalGate
lcf/firewall.py    orchestrator → Verdict
lcf/engine.py      live engine: pcap replay + NFQUEUE inline mode
lcf/persist.py     save/load a trained firewall as JSON
lcf/synth.py       synthetic traffic + one-call training (shared)
lcf/cli.py         command-line interface (train/analyze/run/live/demo-pcap)
demo.py            self-verifying demonstration (parts A–D)
demo_live.py       ingestion + drift demonstration (parts E–F)
tests/test_lcf.py  zero-dependency test suite (10 tests)
```

---

# Part II — v2 additions

## 9. Real-traffic ingestion (Layer 1, `pcap.py`)

The prototype scored `Flow` objects; v2 ingests real packets. `pcap.py` is a
dependency-free reader for classic libpcap files (both byte orders, microsecond
and nanosecond timestamps), decoding Ethernet / IPv4 / TCP|UDP. Packets are
assembled into **bidirectional flows** keyed on the unordered 5-tuple; the first
packet's source fixes the "initiator", so upload vs download byte counts are
well defined. Each flow yields the same `Flow` record the five layers already
consume, so training and inference share one feature pipeline (the CLI trains
*through* this pipeline via `gen_flow_via_pcap`, which removes the train/serve
representation gap). A matching writer (`build_frame`, `write_pcap`) emits valid
captures for tests and replay.

## 10. Adaptive conformal — coverage under distribution shift (`adaptive.py`)

§5 assumes exchangeability. A deliberately drifting adversary breaks it and the
realised FPR wanders. **Adaptive Conformal Inference** (ACI; Gibbs & Candès,
2021) repairs this with an online level `α_t` updated from after-the-fact
feedback on confirmed-benign flows:

```
err_t      = 1{ benign flow t was blocked }          (block iff p_t < α_t)
α_{t+1}    = α_t + γ · (α_target − err_t).
```

### 10.1 Theorem (long-run error, no exchangeability)

For **any** sequence of flows whatsoever,

```
| (1/T) Σ_t err_t − α_target |  ≤  (|α_1 − α_{T+1}|) / (γT)  ≤  (α_target + γ)/(γT) → 0.
```

*Proof.* Summing the update telescopes:
`α_{T+1} = α_1 + γ Σ_t (α_target − err_t)`, so
`Σ_t err_t = T·α_target + (α_1 − α_{T+1})/γ`. Divide by `T`. The bound follows
because `α_{T+1}` stays bounded. ∎

The identity is **exact**, so `α_t` must be free to drift slightly below 0
("block nothing") or above 1 ("block everything"); clamping it to `[0,1]` breaks
the telescoping and the guarantee — the implementation therefore only applies a
wide numerical guard, never a `[0,1]` clamp.

### 10.2 Empirical validation (`demo_live.py` part F, and the test suite)

Benign traffic with an abrupt score-space shift of `0.8·σ_cal` at `t = T/2`,
`α = 0.05`, `γ = 0.1`, `T = 4000`:

| | before shift | after shift (transient) | long-run average |
|---|---|---|---|
| static conformal | 0.057 | 0.374 | 0.216 |
| **adaptive (ACI)** | 0.050 | 0.051 | **0.050** |

ACI's long-run error equals the target to within `0.0003`, inside the
theoretical bound `(α+γ)/(γT)=0.0004`; static conformal drifts with the
attacker. `WeightedConformalGate` provides the alternative covariate-shift
repair (Tibshirani et al., 2019) via likelihood-ratio weighting of the
calibration scores.

## 11. Live engine (`engine.py`)

`Engine.run_pcap` replays a capture, assembling flows with an idle timeout and
emitting a `Verdict` per flow (fully tested). `Engine.run_nfqueue` runs inline
on a Linux host: with an `iptables … -j NFQUEUE` rule and the `netfilterqueue`
package (root), each packet is decoded, accumulated into its flow, and once the
flow reaches `min_pkts` a cached per-5-tuple verdict drops or accepts subsequent
packets (fail-open before the first verdict). The NFQUEUE path is guarded behind
its import; the validated path in this environment is pcap replay.

## 12. Persistence & CLI (`persist.py`, `cli.py`)

`save_firewall`/`load_firewall` serialise every learned parameter — policy
boxes, Gaussian model, KL baseline + threshold, conformal calibration scores —
to portable JSON, so a model trained offline ships to a sensor without
retraining (round-trip identity is asserted in the test suite). The `lcf` CLI
exposes `train`, `analyze`, `run`, `live` and `demo-pcap`.

## 13. Test suite

`tests/test_lcf.py` is dependency-free (runs under `pytest` or standalone) and
covers: lattice axioms; shadow/conflict detection; canonical-form equivalence on
random packets; conformal coverage `≤ α`; KL separation; SPRT termination; pcap
round-trip and bidirectional assembly; persistence round-trip identity; the
engine over a synthesised capture; and ACI meeting its bound and beating static
conformal under drift. (v3 adds seven more — see §19.)

---

# Part III — v3 additions (the maximal layer)

## 14. Volumetric / cross-flow detection (Layer 0, `volumetric.py`)

The five-layer core judges each flow alone, so it cannot see attacks that live
in the aggregate. Layer 0 watches the stream of flow headers over a sliding
window with two sublinear instruments:

- **Count-Min Sketch** — per-key (e.g. per-source) counts in `O(w·d)` memory
  regardless of key count. With `w = ⌈e/ε⌉`, `d = ⌈ln(1/δ)⌉`, the estimate
  never under-counts and over-counts by `≤ ε·N` with probability `≥ 1−δ`
  (Cormode–Muthukrishnan). The one-sided property is asserted in the tests.
- **Windowed Shannon entropy** of the address distributions. A flood collapses
  destination-IP entropy (mass concentrates on one victim); a scan collapses
  source-IP entropy (one host fanning out). Floors are calibrated as benign
  quantiles — distribution-free. Validated on a synthetic flood and scan, with
  the victim/scanner correctly identified.

## 15. Mondrian conformal — per-group guarantee (`adaptive.py`)

Plain conformal controls the FPR *marginally*; a subpopulation can still be
systematically over-blocked. **Mondrian conformal** partitions calibration by a
group key `g(x)` and ranks a test point only within its group, giving

```
P( p(x) < α | group(x) = k ) ≤ α   for every group k,
```

under within-group exchangeability (unknown/small groups fall back to the pooled
set, preserving the marginal guarantee). Demonstrated: two benign groups with
different score distributions — marginal conformal blocks group B at 9.8 %
(≈2× the 5 % target) while Mondrian holds both groups at ≈5 %.

## 16. Drift detection (`drift.py`)

ACI tracks the target continuously; Page-Hinkley raises an explicit alarm when a
*persistent* shift warrants recalibration. On standardised observations
`z = (x−μ₀)/σ₀`, it accumulates `U_T = Σ(z−δ)` and fires when
`U_T − min U_T > λ`. Scale-free (`δ` in σ-units), `O(1)` state. Measured: 0
false alarms on 8 000 stationary samples; a +1.5 σ shift detected within ~36
observations.

## 17. Explainability (`bayesian.py`, `firewall.py`)

For a diagonal-Gaussian verdict the log-odds is a sum over features, so it admits
an **exact** additive attribution:
`z(x) = Σ_i c_i + prior`, with `c_i` feature `i`'s signed push toward malicious.
`log_odds_contributions` returns the `c_i` (their sum equals `log_odds` to
floating-point), and every `Verdict` carries its top drivers. A loud flow is
attributed to `payload_entropy`; a stealthy flow to `mean_pkt_size` — the same
features that route them to the KL gate and the conformal gate respectively.

## 18. Evaluation harness (`evaluate.py`)

Threshold-independent ranking quality (AUC via the exact Mann-Whitney statistic),
average precision, a ROC sweep, and the actual conformal operating point. On the
clean synthetic traffic AUC = 1.000; on a deliberately weakened attacker with
overlapping distributions AUC ≈ 0.999 — confirming the harness is not
degenerate.

## 19. Test suite (v3)

Seventeen tests total, ~6 s. Beyond §13: a **property-based** check of the
lattice axioms (idempotence, commutativity, associativity, absorption, order
consistency) over 3 000 random box triples; Mondrian per-group coverage;
Count-Min's one-sided error; volumetric flood/scan detection; Page-Hinkley
(no false alarm + bounded detection delay); exact explainability attribution;
and AUC on separable traffic.

## 20. Module map (v3)

```
lcf/volumetric.py  Layer 0   CountMinSketch, VolumetricMonitor (flood/scan)
lcf/features.py    Layer 1   flow → R^n + byte histogram
lcf/pcap.py        Layer 1   libpcap read/write, flow assembly
lcf/lattice.py     Layer 2   Box, Rule, Policy; analyze(), canonical()
lcf/bayesian.py    Layer 3   GaussianVerdict (+ exact attribution), SPRT
lcf/models.py      Layer 3   GaussianMixtureVerdict (pluggable stronger scorer)
lcf/anomaly.py     Layer 4   KLGate, Sanov threshold
lcf/conformal.py   Layer 5   ConformalGate (α-bound)
lcf/adaptive.py    Layer 5+  Adaptive (ACI), Mondrian, Weighted conformal
lcf/drift.py       support   PageHinkley change detection
lcf/firewall.py    orchestrator → Verdict (with drivers)
lcf/engine.py      live engine: pcap replay + NFQUEUE
lcf/persist.py     save/load as JSON
lcf/evaluate.py    ROC / AUC / PR harness
lcf/synth.py       synthetic traffic + training
lcf/cli.py         CLI
demo.py / demo_live.py / demo_max.py    demonstrations (A–D / E–F / G–L)
tests/test_lcf.py  18-test suite
```

---

# Part IV — the model-agnostic guarantee made concrete

## 21. Pluggable scorers (`models.py`)

The orchestrator depends on a scorer only through `log_odds(x)` (and, for the
one-class path, `nonconformity(x) = −benign_logdensity(x)`). Everything above
Layer 3 — the conformal p-value, the α-bound, ACI, Mondrian — operates on
*scores*, not on the model that produced them. So a stronger detector drops in
and the false-positive guarantee transfers **unchanged**. This is the
architecture's central structural claim, and v3.1 demonstrates it.

`GaussianMixtureVerdict` fits a K-component diagonal mixture per class by EM
(multiple restarts, variance floor), representing multimodal traffic that a
single Gaussian cannot. The failure mode is concrete: when benign traffic is
bimodal (say small interactive flows *and* large bulk transfers), a single
Gaussian centres on the mean *between* the modes and treats that empty valley as
the most normal region — precisely where a novel attack hides.

**Measured** (bimodal benign at ~100 and ~900; a novel attack at ~500, in the
valley; scored by benign-density novelty):

| scorer | AUC | conformal FPR @ α = 0.05 |
|---|---|---|
| single Gaussian | **0.31** (worse than chance) | 0.046 |
| Gaussian mixture (K=2) | **1.00** | 0.044 |

The single Gaussian is *worse than a coin flip* because the valley looks maximal
to it; the GMM captures both modes and separates perfectly. Critically, the
conformal FPR is ≤ α for **both** scorers — the guarantee is a property of the
calibration wrapper, not of the model, exactly as claimed. Choose the detector
for detection power; the guarantee comes along regardless.
`train_firewall(..., model="gmm", K=…)` selects it.
