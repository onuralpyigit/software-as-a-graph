# Graph Neural Networks for Reliability and Dependability Analysis in Complex Distributed Systems based on Publish–Subscribe Architecture

> **`draft.md` is the authoritative manuscript text.** This outline is a structural map of it,
> regenerated from the current draft. Where the two disagree, `draft.md` wins.
>
> **This is the condensed submission draft.** `draft.md` was cut from ~30,100 words (23 tables, 6
> figures) to ~16,800 words (13 tables, 4 figures) to fit JSS's ≤36-single-column-page guidance,
> refocused on the paper's graph-learning-and-dependability claim (Reliability, Maintainability,
> Availability, Vulnerability attribution is now a baseline the results are read against, not a
> co-equal contribution). Nothing was deleted outright: the pre-condensation text and every cut
> subsection are preserved verbatim in **[`../thesis/`](../thesis/)** for reuse in thesis chapters —
> see that folder's `README.md` for the section-by-section map of what moved where and why.

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS)
* **Target Topic:** *AI for Reliability and Dependability Analysis in Complex ICT Systems*
* **Review model:** double-anonymised (`[Anon-A]` withheld; repo paths and tool names scrubbed)

## Highlights

* Typed multigraph model predicts pub-sub cascading failure before deployment.
* Heterogeneous GNN leads a training-free QoS-weighted centrality on critical-set F1.
* Equal dimension weights outrank calibrated ones; RMAV repositioned as an interpretable baseline.
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
| In-distribution mean $\rho$, HGL vs Topo-QoS | 0.730 vs 0.595 | Table 8 |
| Paired Wilcoxon, HGL vs Topo-QoS ($n=7$) | $p = 0.375$, **n.s.** | Table 9 |
| Paired Wilcoxon, HGL vs Topo-BL | $p = 0.016$, significant | Table 9 |
| LOSO mean $\rho$, HGL vs Topo-QoS | 0.608 vs 0.521 *(artifact not retained)* | Table 10 |
| LOSO $F_1@K$, HGL vs Topo-QoS | 0.465 vs 0.308 | Table 10 |
| In-domain k-fold, HGL vs Topo-QoS | 0.666 vs 0.492 | §8.2 |
| QoS encoding (RQ3) | null in all three regimes | §8.3 |
| AHP shrinkage, $\lambda=0$ vs $\lambda=0.70$ | 0.292 vs 0.181, monotone decline | Table 11 |
| Normalisation, rank vs min–max / z-score | 0.181 vs 0.318 ($+0.137$) | §8.3 |
| Threshold sweep span | 0.230 (0.001 at $t=0$ → 0.231 at $t=1$) | Table 12 |
| Oracle agreement, $I^*$ vs $I_{\text{comp}}$ | $\rho = 0.394$, Jaccard 0.286 | §7.4 |
| Oracle agreement, $I_{\text{dyn}}$ vs $I^*$ | $\rho = 0.765$, min 0.548 | §7.4 |
| Remediation acceptance | 162 / 332 (48.8%), 7 scenarios | §6.1, §8.4 |
| $\Delta$SRI range | $+0.0025$ to $+0.0158$ | §6.1 |
| Edge removal, `av_system` | 4 of 50 candidates non-zero | §8.2 |
| Library blast hypothesis | not confirmed (165 libraries) | §5.4 |
| Real-world $\rho$ (Autoware / Cloud / Train-Ticket) | 0.688 / 0.778 / 0.759 | Table 13 |
| Real-world gate | 5 of 15 checks fail; all three fail SPOF-F1 | Table 13 |
| Corpus | 1,545 synthetic + 225 real-world = 1,770 components | §7.1 |
| CI/CD gate runtime | <2 s tiny → ~27 s enterprise; catalog precision 0.24–0.40 | §8.4 |

**Two standing caveats that travel with these numbers.** The ranking half of RQ1 is not established:
it fails the paired test in-distribution and its LOSO artifact was not retained (§8.1, §9.2). And
$I_{\text{comp}}$-backed results (§5.4) do not transfer to $I^*$-backed tables (§8.1), because the
two oracles agree only at $\rho = 0.394$.

---

## Section map

### 1. Introduction
Motivation (pub-sub decoupling obscures failure paths; hardening is cheapest pre-deployment;
sustainability framing, condensed to two sentences) · the Architecture–Code Gap and the two
sub-problems (quality attribution, failure-impact analysis), merged with the three-gaps survey ·
**RQ1–RQ5** · six contributions (down from eight — the harness-defect account is no longer listed as
a standalone contribution; it survives as a paragraph in §9.2) · relationship to prior work `[Anon-A]`
and organization, merged into one closing subsection.

RQ1 *where* learning pays · RQ2 what typing exposes · RQ3 QoS encoding · RQ4 CI/CD feasibility ·
**RQ5 transfer to independently-sourced architectures** (§8.5).

### 2. Related Work
No longer subdivided into eight numbered subsections — four themed paragraphs, all 53 citations
retained: pub-sub dependability and runtime/chaos approaches · SCA vs. SSA and continuous
pre-deployment gating · structural and learning-based criticality · quality attributes, remediation,
and positioning.

### 3. The SaG Model
Formal object $G = (V, E, \tau_V, \tau_E, w_E, w_V)$ over five node types and six structural edge
types (**Table 1**, merging the former separate node- and edge-type tables) · QoS-aware weights
stated in prose rather than as separate tables (full formulas in the replication package) · the six
`DEPENDS_ON` projection rules, sequential cascade vs simultaneous blast (**Table 2**) · the two graph
views, four layer projections (one sentence, not a table), and the running example (Figure 2). The
former §3.4 (SCA metric ingestion) is now one sentence in §3.1; the `cm_*` metric list moved to
[`../thesis/material/model_details.md`](../thesis/material/model_details.md).

### 4. Interpretable Attribution as a Baseline
**Retitled** from "Multi-Dimensional Quality Attribution" and cut from ~4,840 words / 5 tables to
~900 words / 1 table (**Table 3**, merging the former dimension-question and RMAV-formula tables).
Kept: the four RMAV dimensions and what each answers, the composite $Q(v)$, the consequence-not-risk
and relative-not-absolute definitions (D4 is load-bearing — it is what forbids pooling scores across
systems and is cited by §5.4 and §8.5), box-plot classification, determinism, and the headline
robustness result (equal weights outrank the calibrated AHP weighting, §8.3). Cut to
[`../thesis/material/rmav_attribution.md`](../thesis/material/rmav_attribution.md): the full RMAV
formula derivation, the three-weighting-paths account, the Quality-in-Use transformation matrix, and
the worked attribution example. Cut entirely to
[`../thesis/material/relationship_criticality.md`](../thesis/material/relationship_criticality.md):
relationship (edge-level) criticality, which the draft's own §8.2 states is not validated by the
edge-removal measurement — removing it costs no evidence.

### 5. Failure-Impact Analysis
**Three** simulation oracles — $I^*$ (`FaultInjector`), $I_{\text{comp}}$ (`FailureSimulator`),
$I_{\text{dyn}}$ (`MessageFlowSimulator`) — kept in full, this is core dependability content · the two
predictors, HGL and $HGL\text{-}QoS$ (Figure 3, the HGT attention case study) · the independence
guarantee · **§5.4** now folds the former §5.4 (shared-library blast, negative result) and §5.5
(stratified correlation, no Simpson's paradox) into one subsection, one paragraph each — both were
restated nearly in full in the old §8.2, so this removes duplication rather than a finding; full
original treatment (including the dropped Figure 3-of-6, pooled-vs-per-type correlation) is in
[`../thesis/material/oracles_and_labels.md`](../thesis/material/oracles_and_labels.md).

### 6. Prescriptive Remediation and CI/CD Quality Gating
Cut from seven subsections to two. **§6.1** merges Generate→Verify, the four operators (**Table 4**),
the acceptance criterion $\Delta I > \kappa\sigma_{\text{seed}}$, the independence invariants, and the
remediation-yield result (48.8% acceptance, yield concentrated where a fan-out bottleneck exists,
$\Delta$SRI $+0.0025$ to $+0.0158$ — now two sentences instead of a full subsection with Table 13).
**§6.2** is the CI/CD gate: exit-code protocol, absolute-not-delta-aware limitation. Full original
treatment, including the yield table, is in
[`../thesis/material/remediation_and_gating.md`](../thesis/material/remediation_and_gating.md).

### 7. Experimental Setup
Seven synthetic scenarios + three real-world graphs (§7.1) · predictors, baselines, and evaluation
metrics merged into one subsection (§7.2, **Table 5**) · protocols (§7.3) · the three oracles, their
agreement and the label-coverage bounds, condensed (§7.4, **Table 6**) · model configuration and
hardware (§7.5, **Table 7**). The former topology-class gate-threshold table (old Table 15) is cut to
inline threshold values quoted where §8.5 needs them.

### 8. Results
**8.1 RQ1** — Table 8 (with bootstrap CIs and held-out $n$), Table 9 (paired Wilcoxon), Table 10
(LOSO) plus the artifact-retention caveat; resolves as a scope condition, with set identification the
defensible half. **8.2 RQ2** — heterogeneity pays in generalisation not fit; edge criticality
measured by removal; the two negative results from §5.4 restated with their oracle caveat, without
re-deriving the full figures. **8.3 RQ3 and robustness** — QoS null in all regimes; shrinkage
(Table 11, Figure 4); normalisation; propagation threshold (Table 12; the corresponding figure was
cut, the table carries the same numbers). **8.4 RQ4** — gate runtime and detection efficacy, now
including the remediation-yield summary absorbed from the old §6.7. **8.5 RQ5** — three real-world
graphs (Table 13), the failing gate, the tie-breaking artifact behind $F_1@K = 1.000$, and four
scoping conditions.

### 9. Discussion, Threats to Validity, Conclusion
Interpretation (five findings, condensed; the CI/CD-gate finding now correctly states the gate is
*absolute, not delta-aware* — an earlier revision's §9.1 and §9.4 both mis-described it as
delta-aware, which contradicted §1.3/§6.2/§9.3, and both are fixed here) · threats: construct validity
and the unmeasured link ② to Quality-in-Use, the weak oracle agreement, the six instrument defects
(condensed to one paragraph naming only the two that touched reported figures — full six-defect
account, including how each was found, is in
[`../thesis/material/threats_and_instrument_defects.md`](../thesis/material/threats_and_instrument_defects.md)),
uneven artifact retention, the unlabelled third of each system, the label noise ceiling · internal
validity as *view* independence not source independence · external validity as the weakest dimension
· conclusion validity · limitations ordered by how much they would change the claims · conclusion.

### References
53 numbered entries + `[Anon-A]`, unchanged, followed by the Declarations block (CRediT, competing
interest, funding, data availability, generative-AI use) with table numbers updated to match the
condensed draft.

---

## Outcome of the condensation pass

1. **Length: 30,106 → ~16,800 words** (body ~15,300, references 1,280, declarations 220), 23 → 13
   tables, 6 → 4 figures. Estimated at 32–35 single-column pages once typeset, comfortably inside
   JSS's ≤36-page guidance — to be confirmed against a real build once the LaTeX conversion catches
   up (see below).
2. **Table 20's artifact (now Table 10) is still not retained** — re-running the LOSO sweep under the
   final apparatus remains the first item of outstanding work, and is what would settle RQ1's ranking
   half. This did not change in the condensation.
3. Author block, CRediT roles, funding and the archive link are stubs pending de-anonymisation — see
   `latex/title_page.tex` and `latex/sections/declarations.tex`.
4. **Everything the condensation removed is preserved**, not discarded: a byte-identical snapshot of
   the pre-condensation draft and every cut subsection, organised by likely thesis chapter, is in
   [`../thesis/`](../thesis/README.md).

---

## LaTeX conversion

The submission-ready Elsevier `elsarticle` LaTeX source lives in **[`latex/`](latex/)**. **It has not
yet been re-converted from the condensed `draft.md` above** — `latex/sections/*.tex` still reflect the
pre-condensation, 23-table/6-figure version, and the previously-recorded 75-page build (under the
double-spaced `review` class option) describes that older text, not this one. See
[`latex/README.md`](latex/README.md) for the build instructions and what remains a placeholder
pending de-anonymisation; re-converting the sections against this draft, and switching to a
single-spaced class option to check the real page count, is the next step before this manuscript is
submission-ready.
