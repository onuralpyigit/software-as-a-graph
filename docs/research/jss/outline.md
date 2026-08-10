# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

> **`draft.md` is the authoritative manuscript text.** This outline is a structural map of it,
> regenerated from the current draft. Where the two disagree, `draft.md` wins — and the figures below
> are reproduced from it rather than maintained independently, because the previous version of this
> file drifted a full revision behind and contradicted the manuscript on its headline result.

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS)
* **Target Topic:** *AI for Reliability and Dependability Analysis in Complex ICT Systems*
* **Review model:** double-anonymised (`[Anon-A]` withheld; repo paths and tool names scrubbed)

## Highlights

* Typed multigraph model predicts pub-sub cascading failure before deployment.
* Heterogeneous GNN leads a training-free QoS-weighted centrality on critical-set F1.
* Equal dimension weights outrank calibrated ones; RMAV repositioned as attribution.
* Edge criticality is measured by simulated removal, not inferred from node labels.
* SaG runs as a blocking CI/CD gate in well under a minute, even at 500+ components.

## Keywords

publish–subscribe middleware; architectural dependability; cascading failure; heterogeneous graph
neural networks; static system analysis; pre-deployment verification; quality attributes; CI/CD
quality gate.

---

## Headline figures (as reported in the draft)

| Quantity | Value | Source |
|---|---|---|
| In-distribution mean $\rho$, HGL vs Topo-QoS | 0.730 vs 0.595 | Table 18 |
| Paired Wilcoxon, HGL vs Topo-QoS ($n=7$) | $p = 0.375$, **n.s.** | Table 19 |
| Paired Wilcoxon, HGL vs Topo-BL | $p = 0.016$, significant | Table 19 |
| LOSO mean $\rho$, HGL vs Topo-QoS | 0.608 vs 0.521 *(artifact not retained)* | Table 20 |
| LOSO $F_1@K$, HGL vs Topo-QoS | 0.465 vs 0.308 | Table 20 |
| In-domain k-fold, HGL vs Topo-QoS | 0.666 vs 0.492 | §8.1 |
| QoS encoding (RQ3) | null in all three regimes | §8.3 |
| AHP shrinkage, $\lambda=0$ vs $\lambda=0.70$ | 0.292 vs 0.181, monotone decline | Table 21 |
| Normalisation, rank vs min–max / z-score | 0.181 vs 0.318 ($+0.137$) | §8.3 |
| Threshold sweep span | 0.230 (0.001 at $t=0$ → 0.231 at $t=1$) | Table 22 |
| Oracle agreement, $I^*$ vs $I_{\text{comp}}$ | $\rho = 0.394$, Jaccard 0.286 | §7.5 |
| Oracle agreement, $I_{\text{dyn}}$ vs $I^*$ | $\rho = 0.765$, min 0.548 | §7.5 |
| Remediation acceptance | 162 / 332 (48.8%), 7 scenarios | Table 13 |
| $\Delta$SRI range | $+0.0025$ to $+0.0158$ | Table 13 |
| Edge removal, `av_system` | 4 of 50 candidates non-zero | §8.2 |
| Library blast hypothesis | not confirmed (165 libraries) | §5.4 |
| Real-world $\rho$ (Autoware / Cloud / Train-Ticket) | 0.696 / 0.778 / 0.759 | Table 23 |
| Real-world gate | 5 of 15 checks fail; all three fail SPOF-F1 | Table 23 |
| Corpus | 1,545 synthetic + 225 real-world = 1,770 components | §7.1 |
| CI/CD gate runtime | <2 s tiny → ~40 s xlarge; precision = recall = 1.0 | §8.4 |

**Two standing caveats that travel with these numbers.** The ranking half of RQ1 is not established:
it fails the paired test in-distribution and its LOSO artifact was not retained (§8.1, §9.2). And
$I_{\text{comp}}$-backed results (§5.4, §5.5, §6.7) do not transfer to $I^*$-backed tables (§8.1),
because the two oracles agree only at $\rho = 0.394$.

---

## Section map

### 1. Introduction
Motivation (pub-sub decoupling obscures failure paths; hardening is cheapest pre-deployment;
sustainability framing) · the Architecture–Code Gap and the two sub-problems (quality attribution,
failure-impact analysis) · three gaps in prior work (SCA is topology-blind; chaos engineering needs a
deployed system; untyped centrality collapses typed semantics) · **RQ1–RQ5** · eight contributions ·
relationship to prior work `[Anon-A]` · organization.

RQ1 *where* learning pays · RQ2 what typing exposes · RQ3 QoS encoding · RQ4 CI/CD feasibility ·
**RQ5 transfer to independently-sourced architectures** (§8.5).

### 2. Related Work
Pub-sub dependability [1–3, 44, 45] · SCA vs SSA [29–31] · continuous pre-deployment gating
[18–21] · structural criticality [4–6, 36–39] · learning-based criticality [7–14, 40–43, 53] ·
quality attributes and multi-criteria scoring [15–17, 32–35] · architectural remediation and
anti-patterns [22–28] · positioning.

### 3. The SaG Model
Formal object $G = (V, E, \tau_V, \tau_E, w_E, w_V)$ over five node types and six structural edge
types (Tables 1–2) · QoS-aware edge and vertex weights (Tables 3–4) · the six `DEPENDS_ON` projection
rules, sequential cascade vs simultaneous blast (Table 5) · SCA metric ingestion (`cm_*`) · the two
graph views and four layer projections (Table 6) · running example (Figure 2).

### 4. Multi-Dimensional Quality Attribution
Four dimensions and definitions D1–D4 (Table 7) · RMAV formulas (Table 8) · composite $Q(v)$, the
three weighting paths, the QoS-profile adaptation, the Quality-in-Use transformation matrix — stated,
not validated (Table 9) · adaptive box-plot classification · determinism and the independence
guarantee · worked attribution (Table 10) · relationship criticality (Table 11), **defined but not
validated**, and why no common edge population exists.

### 5. Failure-Impact Analysis
**Three** simulation oracles — $I^*$ (`FaultInjector`), $I_{\text{comp}}$ (`FailureSimulator`),
$I_{\text{dyn}}$ (`MessageFlowSimulator`) — and what each can support · the two predictors, HGL and
$HGL\text{-}QoS$ reported separately (Figure 6) · the independence guarantee · **§5.4 the
shared-library blast mechanism: a negative result** · **§5.5 stratified correlation: no Simpson's
paradox** (Figure 3), with three scoping conditions.

### 6. Prescriptive Remediation and CI/CD Gating
Generate→Verify · four operators (Table 12) · triggering on blast radius, not $Q(v)$ · the
acceptance criterion $\Delta I > \kappa\sigma_{\text{seed}}$ at every threshold · three independence
invariants · the delta-aware gate, waiver register, three-tier exit codes · **§6.7 what remediation
actually yields** (Table 13): 48.8% acceptance, yield concentrated where a fan-out bottleneck
exists, improvements real but small, singletons not shown to compose.

### 7. Experimental Setup
Seven synthetic scenarios + three real-world graphs (§7.1) · predictors and baselines (Table 14) ·
evaluation metrics, the one-contract correction, "absent is not zero", gate thresholds by topology
class (Table 15) · in-distribution / LOSO / k-fold protocols and five seeds · the three oracles,
their agreement and the label-coverage bounds (Table 16) · **§7.6 model configuration and
hardware** (Table 17).

### 8. Results
**8.1 RQ1** — Table 18 (with bootstrap CIs and held-out $n$), Table 19 (paired Wilcoxon), Table 20
(LOSO) plus the artifact-retention caveat; resolves as a scope condition, with set identification the
defensible half. **8.2 RQ2** — heterogeneity pays in generalisation not fit; edge criticality
measured by removal; the reproducibility hazard in ordering; the two negative results restated with
their oracle caveat. **8.3 RQ3 and robustness** — QoS null in all regimes; shrinkage (Table 21,
Figure 4); normalisation; propagation threshold (Table 22, Figure 5). **8.4 RQ4** — gate runtime and
detection efficacy. **8.5 RQ5** — three real-world graphs (Table 23), the failing gate, the
tie-breaking artifact behind $F_1@K = 1.000$, and four scoping conditions.

### 9. Discussion, Threats to Validity, Conclusion
Interpretation (five findings) · threats: construct validity and the unmeasured link ② to
Quality-in-Use, the weak oracle agreement, two corrected instrument defects, **uneven artifact
retention**, the unlabelled third of each system, the label noise ceiling; internal validity as
*view* independence not source independence; external validity as the weakest dimension; conclusion
validity · six limitations ordered by how much they would change the claims · conclusion.

### References
53 numbered entries + `[Anon-A]`, followed by the Declarations block (CRediT, competing interest,
funding, data availability, generative-AI use).

---

## Known remaining gaps

1. **Table 20's artifact is not retained** — re-running the LOSO sweep under the final apparatus is
   the first item of outstanding work, and is what would settle RQ1's ranking half.
2. ~~Figures 1–6 are captioned placeholders~~ — resolved: all six are rendered and embedded in the
   LaTeX conversion (see next section).
3. Author block, CRediT roles, funding and the archive link are stubs pending de-anonymisation — see
   `latex/title_page.tex` and `latex/sections/declarations.tex`.
4. The manuscript remains long (~26k words, 23 tables); §4.7 and §3.4 are the compression candidates
   if the editor asks.

---

## LaTeX conversion

The submission-ready Elsevier `elsarticle` LaTeX source lives in **[`latex/`](latex/)** — that folder,
not this outline or `draft.md`, is what gets built and zipped for Editorial Manager. See
[`latex/README.md`](latex/README.md) for the build, the figure-numbering mapping (draft.md's own
figure labels are not in physical reading order — its "Figure 6" appears before its "Figure 3" — so
the printed number in the compiled PDF differs from four of the six captions' own label text; every
`\ref{}` in the LaTeX resolves correctly regardless), and what remains a placeholder pending
de-anonymisation.
