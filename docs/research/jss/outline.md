# Software-as-a-Graph: Heterogeneous Graph Learning for Pre-Deployment Reliability and Dependability Analysis of Complex Distributed Systems

> **[`latex/`](latex/) is the authoritative manuscript**; [`draft.md`](draft.md) is a maintained Markdown
> rendering of the same text, kept in sync after each revision round (see
> [`latex/README.md`](latex/README.md#L3-L12)). This outline is a section-by-section reading map of the
> pair — what each subsection argues, what backs it, what caveats travel with it, and where a reviewer is
> likeliest to push. Where any two disagree, `latex/` wins.
>
> **Regenerated after the predictive-pathway revision.** The previous version of this file was written
> against the pre-revision structure and numbers and should not be consulted; it is in git history.

* **Target Journal:** Journal of Systems and Software (JSS) — Elsevier
* **Target Venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems" (VSI:AI4MSS); submission deadline 30 September 2026
* **Target Topics:** *AI for Reliability and Dependability Analysis in Complex ICT Systems* (primary); *Explainable, Interpretable, and Robust AI in Performance Analysis* (secondary, now carried by §4's attention analysis plus §5's explanation layer); *AI for Automated Performance Tasks* (RQ5/§7.5)
* **Review model:** double-anonymised (`[Anon-A]` withheld; author identity confined to `title_page.tex`)
* **Scale:** ~18,100 words in `latex/sections/`, 8 sections, 16 tables, 5 figures, 56 cited references + `[Anon-A]`
* **Build:** 37 pages under `[preprint,3p]`, zero LaTeX errors, zero undefined references or citations

---

## The thesis, and what changed

The manuscript is now organised around **one primary contribution with a supporting layer**, rather than
two co-equal pathways:

> SaG is a pre-deployment cascading-failure **predictor** — a relation-specific Heterogeneous Graph
> Transformer that forecasts blast radii from Architecture-as-Code alone — made trustworthy and
> actionable by a standards-grounded **explanation layer**.

**Structural changes in this revision.** §4 and §5 swapped: the HGT predictive pathway is now §4 and the
RM attribution model is now §5, retitled *The Explanation Layer: Standards-Grounded Criticality
Attribution*. The shared typed node-feature representation moved out of the old §4.2 into
**§3.4**, since both pathways consume it and leaving it inside the explanation layer would have made §4
forward-reference §5. Within §4 the model now leads and the oracle apparatus follows (§4.1 architecture,
§4.2 heads and loss, §4.3 oracles, §4.4 independence guarantee). §7 gained **RQ5**, which adopts the
previously orphaned scalability subsection, and §8 gained **§8.2 Performance and Sustainability
Implications**, which fulfils a forward reference §1.6 had been making to a section that did not exist.
Figure 1 and its ASCII counterpart in `draft.md` now draw the predictive pathway first.

**What did not change: any number, or the strength of any claim.** The `p = 0.64` boundary on the ranking
comparison is stated in the abstract, in §7.1, in §8.1 and in the conclusion, exactly as before.

---

## Highlights

*(verbatim from [`latex/highlights.tex`](latex/highlights.tex) — 5 bullets, longest 85 chars)*

* Software-as-a-Graph models pub-sub and microservice systems as typed multigraphs.
* Heterogeneous Graph Transformers forecast cascading failure impact before deployment.
* Relation-specific typing decides cross-architecture transfer; untyped learning fails.
* Inference costs 44 ms per system, making per-commit architectural gating practical.
* An ISO-grounded explanation layer turns predicted criticality into concrete fixes.

## Keywords

Graph representation learning; heterogeneous graph neural networks; publish–subscribe middleware;
distributed systems dependability; cascading failures; static system analysis; architectural quality
models; explainable AI; CI/CD quality gates.

---

## Headline figures

| Quantity | Value | Section |
|---|---|---|
| In-distribution mean $\rho$ — HGL / GL / Topo-QoS / Topo-BL | 0.725 / 0.609 / 0.586 / 0.209 | §7.1 (Table 6) |
| In-distribution mean $\rho$ — HGL-QoS / GL-QoS | 0.653 / 0.439 | §7.1 (Table 6) |
| Paired Wilcoxon, HGL vs Topo-BL ($n=7$) | $\Delta\rho = +0.516$, $p = 0.0156$, **significant** | §7.1 (Table 7) |
| Paired Wilcoxon, HGL vs GL-QoS | $\Delta\rho = +0.286$, $p = 0.0312$, **significant** | §7.1 (Table 7) |
| Paired Wilcoxon, **HGL vs Topo-QoS** | $\Delta\rho = +0.139$, $p = 0.469$, **n.s.** | §7.1 (Table 7) |
| Paired Wilcoxon, HGL-QoS vs HGL | $\Delta\rho = -0.072$, $p = 0.078$, n.s. (marginal) | §7.1 (Table 7) |
| **LOSO mean $\rho$** — HGL-QoS / Topo-QoS / HGL / GL-QoS / Topo-BL / RM / GL | **0.608** / 0.571 / 0.439 / 0.363 / 0.301 / 0.195 / 0.086 | §7.1 (Table 8) |
| LOSO $F_1@K$ — HGL-QoS / Topo-QoS / GL | 0.414 / 0.380 / 0.237 | §7.1 (Table 8) |
| **HGL-QoS vs Topo-QoS (LOSO)** | $+0.037$, 5/8 folds, $p = 0.64$, **n.s.** | §7.1 |
| Heterogeneity gap, LOSO (HGL − GL) | $+0.353$, 8/8 folds, $p = 0.0078$ | §7.2 |
| Heterogeneity gap, LOSO (HGL-QoS − GL-QoS) | $+0.246$, 8/8 folds, $p = 0.0078$ | §7.2 |
| QoS encoding, in-dist vs LOSO | $-0.072$ vs $+0.169$ (7/8, $p = 0.0156$) — **sign flips by protocol** | §7.3 |
| Edge removal, `av_system` | 4 of 50 candidates non-zero, all library channels | §7.2 |
| AHP shrinkage, $\lambda = 0 \to$ raw | $0.348 \to 0.232$; uniform prior wins 7/7, $p = 0.0156$ | §7.3 (Table 11) |
| Morris screening, most/least influential | $\lambda$, $r_\alpha$ dominate; $p$, $\gamma$ least | §7.3 (Table 12) |
| Dirichlet simplex, 100 draws | mean $\rho = 0.243$ (shipped 0.252), sd 0.006, $\tau = 0.899$ | §7.3 |
| Oracle agreement, $I_{\text{dyn}}$ vs $I^*$ | $\rho = 0.883$ (min 0.756) | §4.3, §7.3 (Table 13) |
| Oracle agreement, top-$K$ Jaccard | 0.31–0.42 vs 0.111 chance; labeler self-agreement floor 0.44 | §7.3 |
| QoS convergence null | $\Delta\rho = -0.009$ — reported, not omitted | §7.3 |
| Availability defect, $\sigma_A$ before → after | $[0.0145, 0.0398] \to [0.0254, 0.0530]$ | §7.3 |
| RM pooled vs stratified | pooled 0.028 outside per-type range $[0.14, 0.50]$ (Simpson's) | §7.3, §8.3 |
| Tier changes under pooling | 62.8% of 1,619 components; 19.0% cross CRITICAL/HIGH | §7.3 |
| Real-world $\rho$ (Autoware / Cloud / Train-Ticket) | 0.688 / 0.778 / 0.759 | §7.4 (Table 14) |
| **Inference cost** | 44 ms at 2,000 components vs 23.8 s analysis; 12× → 545× | §7.5 (Table 15) |
| Whole CI gate | 0.02 s (29 components) → 27.4 s (520 components) | §7.5 |
| Corpus | 1,770 components, 10 architectures (7 synthetic + 3 real-world), 6 domains | §6.1 (Table 4) |

### Three standing caveats that travel with every figure above

1. **RQ1's ranking margin over the untrained baseline is not established.** HGL-QoS beats Topo-QoS on the
   point estimate but fails the paired test ($+0.037$, 5/8 folds, $p = 0.64$). The defensible predictive
   claims are the **typing gap** ($+0.353$, 8/8, $p = 0.0078$), the **QoS-encoding gap** ($+0.169$, 7/8,
   $p = 0.0156$), the **critical-set gain** ($F_1@K$ 0.414 vs 0.380), the **real-world transfer**, and the
   **cost profile**. Any sentence reading "graph learning outperforms structural baselines" without one of
   those qualifiers overclaims against the paper's own Table 7/8. §7.1 insight 4, §8.1 and §8.5 all state
   this explicitly — keep it that way.
2. **Results are oracle-scoped and do not transfer between oracles.** Top-$K$ Jaccard across oracle pairs
   is 0.31–0.42. Findings measured against $I_{\text{comp}}$ (§7.2's edge removal, §7.3's anti-pattern
   catalog) are **not** evidence for claims measured against $I^*$ (Tables 6–8, 14).
3. **Nothing here is an energy measurement.** §8.2 claims a cost profile in wall-clock seconds and an
   argument from avoided compute. It states in its own text that no power, energy or carbon figure was
   measured, and §8.3 carries a matching threat. Do not let a later edit upgrade that to a joules claim.

---

## Section-by-section map

| § | Title | What it establishes | Where a reviewer pushes |
|---|---|---|---|
| **1.1** | Motivation | Pub-sub decoupling creates a visibility barrier; failures are either sequential cascades or simultaneous blast radii; pre-deployment is exactly when no telemetry exists | "Why not just wait for staging telemetry?" — answered by the CI/CD cost argument in §1.3.1 and §7.5 |
| **1.2** | Problem Statement: the Architecture–Code Gap | **Predictive forecasting is the primary task; the explanation layer is what a rank alone cannot say.** Separation is architectural — no shared parameters, coupling term off by default | The two-task split now reads as one thesis; if a reviewer wants them merged, §4.2's $\lambda_{\text{RM}}$ ablation is the evidence they are separable |
| **1.3** | The SaG Approach | Four pipeline stages: typed multigraph → QoS projection → **HGT predictor** → **explanation layer** | — |
| **1.3.1** | Rationale for graph learning vs direct simulation | Unmeasured components (30–47%), variance reduction plus CI/CD-viable cost, diagnostic explainability | "Just run the simulator" — rebutted on labels, noise, and 44 ms vs minutes-to-hours |
| **1.4** | Research questions | RQ1 efficacy, RQ2 typing, RQ3 QoS/calibration/sensitivity, RQ4 real-world, **RQ5 cost** | RQ3's scope was widened to match what §7.3 actually delivers |
| **1.5** | Key contributions + **prior-work disclosure** | Predictor first, then typed model, explanation layer, evaluation. Discloses the conference version and enumerates what is new | Editorial-desk item: the CFP requires this for conference extensions |
| **1.6** | Paper organization | Now promises what §8 actually contains | — |
| **2.1–2.4** | Related work | Dependability/chaos engineering; SCA vs SSA; **quality models and AHP (2.3)**; **network science and graph representation learning, ending on the heterogeneous-GNN gap and a graph-XAI paragraph (2.4)** | §2 now ends on the gap this paper fills rather than on quality models |
| **3.1** | Formal multigraph | 5 entity types, 6 structural edge types (Table 1) | — |
| **3.2** | QoS-aware weights, `DEPENDS_ON` projection | $w(t)$ with $(\beta,\alpha,\psi) = (0.75, 0.15, 0.10)$; QoS sub-weights $(0.24, 0.62, 0.14)$, CR ≈ 0.016; six projection rules (Table 2) | Sub-weights are AHP-consistent and regression-pinned; sensitivity in §7.3 |
| **3.3** | Dual graph views | $G_{\text{structural}}$ (oracles only) vs $G_{\text{analysis}}$ (predictors only) — the independence guarantee's substrate | — |
| **3.4** | **Typed node feature representation** *(moved here from the old §4.2)* | 19–23 dims per entity type, consumed by **both** pathways | Its placement is now what lets §4 precede §5 without a forward reference |
| **4** | **Graph Learning for Failure-Impact Prediction** — the primary pathway | | |
| 4.1 | HGT architecture | 3 layers, $D = 64$, 4 heads; 16-D edge encoding (7 QoS dims); type-specific projection, relational attention, bidirectional passing | Hyperparameters are stated inline; a training-protocol table was considered and dropped on page budget |
| 4.2 | Multi-task heads, dimension-masked loss | R/M/composite/edge heads; $m = [1,0]$ mask; $\lambda_{\text{RM}} = 0$ headline, 0.1 as ablation | The mask is why no maintainability correlation is reported |
| 4.3 | Ground-truth simulation oracles | Four component-level + one relationship-level; $I^*$ declared primary | Miscount fixed: the text says four and §8.3 agrees |
| 4.4 | Input–label independence guarantee | Features from $G_{\text{analysis}}$, labels from $G_{\text{structural}}$ | Enforced by `tests/test_independence_guarantee.py` |
| **5** | **The Explanation Layer** — standards-grounded attribution | | |
| 5.1 | ISO/IEC grounding | $D_1$/$D_2$ criticality; RM decomposition into FT / A / M (Table 3) | Opens by stating it consumes the predictor's Top-$K$ via triage, and is not a ranking model |
| 5.2 | Composite quality score | AHP hierarchy; $r_\alpha = 0.36$, $(q_R, q_M) = (0.80, 0.20)$ | Constants are a re-parameterisation of the retired 4-D AHP vector, not independently tuned |
| **6.1** | Corpus | 1,770 components, 10 architectures, byte-identical regeneration, **plus an explicit note on the ATM case study's status** | ATM is an 8th LOSO fold deliberately outside Table 4 — now stated rather than implied |
| **6.2** | Baselines and predictors | Proposed variants listed first; RM included as a non-competing reference; **evaluation substrate** scope condition | The GL/HGL substrate asymmetry is declared here and repeated in §7.2 |
| **6.3** | Metrics and protocols | $\rho$, $\tau$, $F_1@K$; single stated population; in-distribution / LOSO / real-world | — |
| **7.1** | RQ1 | Positive case first (best configuration, critical set, QoS transfer), **boundary as insight 4** | The boundary is stated, not buried |
| **7.2** | RQ2 | $+0.353$ typing gap, 8/8 folds — the paper's strongest result — with the substrate scope condition attached | The scope condition is load-bearing; do not drop it |
| **7.3** | RQ3 | QoS encoding, topic weights, AHP shrinkage, Morris + Dirichlet joint screening, oracle convergent validity, thresholds, the Availability defect, anti-patterns, stratification, attention | Condensed in this revision: all tables, numbers and verdicts kept, narration tightened |
| **7.4** | RQ4 | Zero-shot transfer, $\rho = 0.688$–$0.778$ | Gain over degree is $+0.014$ on one of three |
| **7.5** | **RQ5** | 44 ms inference vs 23.8 s analysis; 12× → 545×; whole gate 0.02–27.4 s; training ~45 min per variant | Nothing above 2,000 components was timed; CPU only |
| **8.1** | Discussion | What the predictor is for / where the boundary is / what the explanation layer adds | — |
| **8.2** | **Performance and sustainability** | The learned model is the cheapest stage; sustainability is an avoided-compute argument, explicitly unmeasured | This is the special issue's thinnest theme for this paper, and the section says so |
| **8.3** | Threats to validity | Construct (oracles), internal (normalisation defect, QoS-flat confound), external, conclusion (Simpson's, degenerate substrate), **unmeasured sustainability** | Unusually candid; two self-reported instrument defects changed conclusions |
| **8.4** | Limitations and future work | Safety/security, HIL validation, **direct energy attribution**, prescriptive synthesis | — |
| **8.5** | Conclusion | Leads with the predictive result, restates the boundary | — |

---

## Table and figure inventory

16 tables, 5 figures, each owned by exactly one subsection. Useful as a checksum against renumbering.
LaTeX numbers automatically; `draft.md` numbers by hand and keeps **Table 0** plus its own **Figure 2**.

| draft.md | LaTeX label | Owning subsection | Content |
|:--:|---|---|---|
| Table 0 | `tab:paradigm_comparison` | §2.4 | Dependability-analysis paradigms compared |
| Table 1 | `tab:1` | §3.1 | Entity and structural edge types |
| Table 2 | `tab:2` | §3.2 | Six `DEPENDS_ON` projection rules |
| Table 3 | `tab:3` | §5.1 | RM quality decomposition |
| Table 4 | `tab:4` | §6.1 | Evaluation corpus (10 architectures) |
| Table 5 | `tab:genparams` | §6.1 | Generative parameters (7 synthetic scenarios) |
| Table 6 | `tab:5` | §7.1 | In-distribution held-out $\rho$ |
| Table 7 | `tab:6` | §7.1 | Paired Wilcoxon tests |
| Table 8 | `tab:7` | §7.1 | Inductive LOSO, 8 folds |
| Table 9 | `tab:8d` | §7.3 | HGL-QoS vs HGL under both protocols |
| Table 10 | `tab:8b` | §7.3 | Topic-weight coefficient sensitivity |
| Table 11 | `tab:8` | §7.3 | AHP shrinkage $\lambda$ sweep |
| Table 12 | `tab:8e` | §7.3 | Morris elementary-effects screening |
| Table 13 | `tab:8c` | §7.3 | Inter-oracle agreement |
| Table 14 | `tab:9` | §7.4 | Real-world architecture validation |
| Table 15 | `tab:scale` | §7.5 | Per-stage inference cost |
| Figure 1 | `fig:1` | §1.3 | End-to-end pipeline (predictive pathway drawn first) |
| — | `fig:2` | §3.3 | Running example — LaTeX only |
| Figure 2 | — | §4.1 | HGT layer stack (ASCII) — `draft.md` only |
| Figure 3 | `fig:5` | §7.1 | Results at a glance |
| Figure 4 | `fig:4` | §7.3 | AHP shrinkage sensitivity |
| Figure 5 | `fig:3` | §7.3 | HGT attention case study |

## RQ traceability

| RQ | Question (abbreviated) | Primary section | Primary evidence | Standing caveat |
|:--:|---|---|---|---|
| RQ1 | How accurately does typed learning predict impact and identify the critical set vs non-learning baselines? | §7.1 | Tables 6–8, Figure 3 | Ranking margin over Topo-QoS **n.s.** ($p = 0.64$); the defensible halves are $F_1@K$ and the LOSO transfer |
| RQ2 | Does typing beat homogeneous models, and does it hold out of distribution? | §7.2 | $+0.353$ (8/8, $p = 0.0078$); 4/50 edge removal | Substrate asymmetry (§6.2) means part of the gap is the wider node set; edge-removal result is $I_{\text{comp}}$-scoped |
| RQ3 | How do QoS encoding, weighting calibration, oracle choice and thresholds affect accuracy and explainability? | §7.3 | Tables 9–13, Figures 4–5 | Scope now matches the subsection's actual content |
| RQ4 | Does it transfer zero-shot to real open-source architectures? | §7.4 | Table 14 | Gain over degree is $+0.014$ on one of three; simulated failures, not production outages |
| RQ5 | What does the analysis cost at CI/CD time, and which stage dominates? | §7.5 | Table 15 | Nothing above 2,000 components timed; single-threaded CPU; no energy measurement |

---

## Outstanding work

1. **Page budget.** The manuscript builds to **37 pages** under `[preprint,3p]` against JSS's ≤36
   single-column guidance. §7.3 was condensed by ~800 words in this revision and everything else is at
   word parity with the pre-revision text; the residual page is attributable to **§8.2**, verified by a
   controlled build (removing §8.2 alone returns the manuscript to 36 pages). Either justify the length in
   the cover letter as an extended conference version, shorten §8.2 further, or move §7.3's weighting
   sweeps to supplementary material.
2. **De-anonymise before submission**: [`latex/title_page.tex`](latex/title_page.tex) (names, affiliations,
   corresponding-author email) and [`latex/sections/declarations.tex`](latex/sections/declarations.tex)
   (CRediT roles, funding, generative-AI declaration). The manuscript body must stay anonymous.
3. **Prune uncited `refs.bib` entries** — 73 entries against 56 cited. Uncited entries do not render, so
   this is hygiene rather than a defect. The same 10 entries are uncited in `draft.md`'s hand-numbered list.
4. **Consider a training-protocol table for §4.1.** Dropped from this revision on page budget. A reviewer
   of an AI paper will plausibly ask for optimiser, epochs, early-stopping and seed detail beyond the
   architecture parameters already stated inline.
5. **Graphical abstract** — encouraged by the Guide for Authors, not required; not produced.

## Closed by this revision

*(previously listed as outstanding or as known inconsistencies; each verified against the current text)*

- ~~Prior-work disclosure missing; `[Anon-A]` uncited.~~ Restored as a §1.5 paragraph enumerating what is new.
- ~~Figure numbering broken; three of four figures cited but not rendered.~~ All 5 figures render and are
  `\ref`'d; `draft.md` carries a note recording its own numbering.
- ~~Orphan `.tex` files from the retired 9-section structure.~~ No longer present.
- ~~`0.521` vs `0.522` Topo-QoS LOSO discrepancy.~~ Superseded — Topo-QoS LOSO is now 0.571 on the
  Application population.
- ~~Abstract mixes HGL and HGL-QoS variants silently.~~ The abstract now names the variant for each figure.
- ~~Wrong cross-reference to the ATM case study (cited as §7.4).~~ Now points at §7.3, where the attention
  analysis lives, in both `draft.md` and the LaTeX.
- ~~ATM is an undocumented 8th scenario.~~ §6.1 now carries an explicit note on its status and exclusion
  from Table 4.
- ~~§5.1 announces three oracles then defines four.~~ The text says four; §8.3 agrees.
- ~~Highlights bullet 3 claims unsupported "significant gains".~~ Reworded to the transfer claim.
- ~~§5.2 names edge types (`CALLS`, `HOSTED_ON`) that do not exist.~~ Not present in the current text.
- ~~RQ3's stated scope under-covers §7.3.~~ RQ3 widened to name oracle validity, thresholds and calibration.
- ~~Simpson's-paradox paragraph mis-filed under the anti-pattern heading.~~ It sits under its own heading.
- ~~"Ten distinct system domains" vs "six declared domains".~~ §8.3 now says ten architectures across six domains.
- ~~§1.3.1's third justification argues for retired remediation machinery.~~ Rewritten around the measured cost profile.
- ~~`draft.md` and `latex/` had materially diverged.~~ Re-synced; numeric-token parity is now exact in both
  directions, and the check below is the gate.

## Keeping the two versions in sync

`draft.md` is regenerated from `latex/sections/*.tex` section by section, then re-seeded with the four
things it keeps and the LaTeX does not: **Table 0**, its **ASCII pipeline diagram** and **ASCII HGT layer
stack**, its **figure-numbering note**, and repo paths the LaTeX scrubs for anonymity. Its citation
markers are hand-numbered `[N]`, so a new `\cite` key needs a matching entry appended to its reference
list. The gate, which must come back empty in both directions:

```bash
cd docs/research/jss
comm -3 <(grep -o '0\.[0-9]\{3\}' draft.md | sort -u) \
        <(cat latex/sections/*.tex | grep -o '0\.[0-9]\{3\}' | sort -u)
```

## Where the condensed material went

`draft.md` was cut from ~30,100 words to its current length across two condensation passes, and this
revision condensed §7.3 further. Nothing was deleted outright: the pre-condensation text is preserved at
[`../thesis/jss_draft_full.md`](../thesis/jss_draft_full.md) (a frozen snapshot, still using the retired
"RMAV" terminology), with topic-organised extracts under
[`../thesis/material/`](../thesis/material/) — `model_details.md`, `rm_attribution.md`,
`relationship_criticality.md`, `oracles_and_labels.md`, `remediation_and_gating.md`,
`threats_and_instrument_defects.md`, and `why_not_simulate.md`. The prescriptive-remediation work removed
in the second pass now lives in its own manuscript under [`../ause/`](../ause/).
