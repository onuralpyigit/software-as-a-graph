# Software-as-a-Graph — JSS submission reading map

> **[`latex/`](latex/) is the authoritative manuscript.** [`draft.md`](draft.md) is now *generated*
> from it by [`reproduce/render_draft_md.py`](../../../reproduce/render_draft_md.py), taking its
> section, table, figure and citation numbers from the compiled `.aux`/`.bbl`, so the two cannot
> disagree unless the generator has not been re-run. This file is a section-by-section map of the
> pair — what each part argues, what backs it, and where a reviewer will push.
>
> **Regenerated after the corpus re-derivation (PR #56) and the consistency pass that followed.**
> Every earlier version of this file is stale: it carried the pre-re-derivation numbers
> (ρ = 0.608/0.571, eight folds, p = 0.64, 43.7 ms) and a "double-anonymised" review model that is
> simply wrong — JSS is single-anonymised.

* **Target journal:** Journal of Systems and Software (Elsevier)
* **Target venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of
  Modern Software Systems" (VSI:AI4MSS); deadline 30 September 2026
* **Target topics:** *AI for Reliability and Dependability Analysis in Complex ICT Systems*
  (primary); *Explainable, Interpretable, and Robust AI* (secondary, §5 and §7.3); *AI for Automated
  Performance Tasks* (RQ5, §7.5)
* **Review model:** single-anonymised (confirmed against the Elsevier Guide for Authors, September 2026); authors are named in the manuscript, `title_page.tex` uploaded separately
* **Scale:** 43 pages, 9 sections, 12 tables, 4 figures, 91 references, plus a 7-page supplement
  (S1–S7)
* **Build:** zero LaTeX errors, zero undefined references or citations; 177 reported table figures
  reconcile against their artifacts via `reproduce/reconcile_manuscript.py`

---

## The thesis

> SaG is a pre-deployment cascading-failure **predictor** — a relation-specific Heterogeneous Graph
> Transformer that forecasts blast radii from Architecture-as-Code alone — paired with a
> standards-grounded **explanation layer**, and evaluated to the point where its own boundaries are
> the contribution.

The paper's distinguishing feature is how much of it is negative. Four claims an earlier version made
are withdrawn in the text rather than quietly dropped: superiority over a training-free QoS-weighted
centrality baseline, zero-shot transfer to real systems, a label-free confidence signal, and the
computational-efficiency argument. Reviewers should be able to find each withdrawal stated plainly;
if an edit ever softens one back into a claim, that is a regression.

---

## Headline figures

| Quantity | Value | Where |
|---|---|---|
| In-distribution mean ρ — Topo / Topo-QoS / GAT-N / GAT-N-QoS / HGT / HGT-QoS | 0.166 / 0.629 / **0.691** / 0.653 / 0.624 / 0.630 | §7.1 (Table 6) |
| **LOSO mean ρ** — Topo / Topo-QoS / GAT-N / GAT-N-QoS / HGT / HGT-QoS / RM | 0.250 / 0.568 / 0.493 / 0.581 / 0.640 / **0.695** / 0.133 | §7.1 (Table 8) |
| LOSO F₁@K — same order | 0.306 / 0.353 / 0.417 / 0.474 / 0.466 / **0.507** / 0.258 | §7.1 (Table 8) |
| **Typing, LOSO** (HGT-QoS vs GAT-N-QoS) | **+0.114**, 11/12, p = 0.0122, CI [+0.048, +0.170] | §7.2 |
| Typing, unweighted pair (HGT vs GAT-N) | +0.147, 11/12, p = 0.0010 | §7.2 |
| Typing, **in-distribution** | −0.023, 3/7, p = 0.813 — no benefit | §7.2 |
| QoS edge encoding, LOSO | +0.054, 11/12, p = 0.0093 | §7.3.1 |
| QoS encoding, **active stratum** | +0.025, p = 0.151 — gain does not survive | §7.1.2, §7.3.1 |
| **vs Topo-QoS (LOSO)** | +0.127, 9/12, p = 0.077; **+0.078 without the ATM fold** | §7.1 |
| vs Topo-QoS, critical set | F₁@K +0.154, 8/12, p = 0.034 | §7.1 |
| Active-stratum retention — Topo-QoS / HGT-QoS | 32% / 59% | §7.1.2 (Table 9) |
| Real-world zero-shot, full population / active stratum | 0.680 / **+0.160**, negative on 2 of 5 | §7.4.1 (Table 11) |
| Real-world training-free references — RM / Topo | 0.516 / 0.511 (Topo-QoS not computable) | §7.4.1, §8.4 |
| Oracle agreement — I_dyn·I\* / I_comp·I\* / I_comp·I_dyn | 0.620 / 0.395 / 0.366 (12 folds) | §7.3 (Table 10) |
| Stratified vs pooled RM ρ | 0.515 / 0.183 / 0.149 vs pooled **0.057** (Simpson's) | §7.3.6 |
| Label-noise ceiling | test–retest 0.880–1.000, median 0.979 | §7.1 |
| **Cost** — HGT forward vs structural analysis at 2,000 components | 56.2 ms vs 239.34 s (**4,259×**) | §7.5 (Table 12) |
| Gate vs its own oracle | gate 0.04–82.7 s, oracle 0.14–7.2 s → **gate ~11× dearer** | §7.5.1 |
| AHP shrinkage, uniform → raw | 0.262 → 0.166 (elicited weights are anti-predictive) | Supp. S1 |
| Morris screening, only load-bearing constants | r_α (μ\* 0.144), λ (0.117); other eight ≤ 0.023 | Supp. S1 |
| Corpus | 2,812 components, 17 architectures (12 synthetic + 5 real-world) | §6.1 (Table 4) |

### Four standing caveats that travel with every figure above

1. **Superiority over the untrained baseline is not established.** The +0.127 margin fails its test
   (p = 0.077) and rests on one fold; §7.1 insight 1 says so, and §8.1 recommends `Topo-QoS` as the
   defensible default for a team that wants a ranking and nothing more. Any sentence reading "graph
   learning outperforms structural baselines" overclaims against the paper's own Table 8.
2. **Typing is an inductive bias, not capacity.** It wins 11/12 out-of-distribution and loses
   in-distribution. Stating the first without the second inverts the finding.
3. **Results are oracle-scoped.** Top-K Jaccard across oracle pairs is 0.24–0.49. Anything measured
   against I_comp (Supp. S6, S7) is not evidence for claims measured against I\* (Tables 6–9, 11).
4. **Nothing here is an energy measurement, and the gate is not cheap.** §7.5.1 and §8.2 state that
   the static gate costs about eleven times its own simulation oracle. Do not let a later edit
   upgrade the surviving infrastructure-avoidance argument back into an efficiency or joules claim.

---

## Section-by-section map

| § | Title | What it establishes | Where a reviewer pushes |
|---|---|---|---|
| **1.1** | Motivation | Pub-sub decoupling creates a visibility barrier; pre-deployment is exactly when no telemetry exists. Sustainability framed as *infrastructure* avoidance, with the cost caveat stated up front | "Then why is your gate slower than your simulator?" — answered in §7.5.1 and §8.2, and conceded |
| **1.2** | Problem statement | Prediction is primary; the explanation layer is what a rank cannot say. Separation is architectural — no shared parameters | If a reviewer wants them merged, §4.2's λ_RM ablation is the evidence they are separable |
| **1.3** | The SaG approach | Four pipeline stages; Figure 1 | — |
| **1.4** | RQ1–RQ5 | Efficacy, typing, QoS/robustness, real-world, cost | RQ5 now asks how the gate compares to simulation, and answers unfavourably |
| **1.5** | Contributions + prior-work disclosure | Discloses the conference version and enumerates what is new | Editorial-desk item: required for conference extensions |
| **2** | Related work | Architecture-based reliability prediction, AADL error annex, architecture recovery, data-driven microservice RCA, green SE | "Why no percolation baseline?" — §2.4 concedes this explicitly as a gap in the baseline set |
| **3** | The typed multigraph | Five entity types, six projection rules, dual graph views, typed node features | Rule 6 symmetry; §3.2 justifies it and bounds its influence to zero on labels |
| **4** | The HGT predictor | Architecture, 16-D edge encoding, multi-task heads, four oracles, input–label independence | §4.4 concedes the task is recovering a closed-form functional of the same graph — which is why a centrality baseline is competitive |
| **5** | The explanation layer | ISO/IEC 25010 decomposition, RM composite | §5.2 concedes three of five AHP matrices are rank-one and their CRs uninformative (Supp. S4) |
| **6** | Experimental setup | Corpus (Table 4), which subset backs which analysis (Table 5), baselines, metrics, protocols, pre-registration | §6.3 concedes model selection is made on the training distribution — the protocol's main weakness, also in §8.4 |
| **7.1** | RQ1 | In-distribution and LOSO ranking; label-noise ceiling; the active-stratum re-scoring that halves the baselines' apparent accuracy | Insight 1 declines to claim a win the bootstrap interval would have supported |
| **7.2** | RQ2 | The typing result, and the regime contrast that interprets it | The single lost fold (ATM) is reported rather than excluded |
| **7.3** | RQ3 | QoS ablation and its narrowing, oracle convergent validity, stratification | Sensitivity sweeps live in Supp. S1–S3 |
| **7.4** | RQ4 | **Reported as a negative result.** Full-population 0.680 collapses to +0.160 on the active stratum and inverts on both microservice call trees | The configuration for this run differs from Table 8's; §7.4.1 states it and declines to claim zero-shot purity |
| **7.5** | RQ5 | Cost profile; the gate is dearer than its oracle | The withdrawal is the finding |
| **8** | Discussion, threats, limitations | When to use which engine; the withdrawn fallback gate; sustainability restated honestly; four limitations | §8.4's four paragraphs are the paper's own strongest critique |
| **9** | Conclusion | What is established, what is not | — |

## Supplement (S1–S7)

Parameter sensitivity (S1, with Figure S1), zero-inflation bounds on oracle agreement (S2), domain
weighting and thresholds (S3), the AHP matrices and their rank-one diagnostic (S4), generative
parameters of the corpus (S5), the anti-pattern detection benchmark (S6), and the explanation
layer's real-world evaluation against I_comp (S7). Cross-references from the supplement into the
body are literal text, never `\ref` — the two documents do not share an `.aux`.
