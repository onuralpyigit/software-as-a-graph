# Advisor Review Revision Matrix & Tracker

**Journal:** Journal of Systems and Software (JSS)  
**Article Title (Base Draft):** *Software-as-a-Graph: Predicting Cascading-Failure Impact in Publish–Subscribe Systems Before Deployment with QoS-Aware Graphs and Hybrid Learning*  
**Advisor / Co-author:** Prof. Feza Buzluca  
**Corresponding Author / Candidate:** Ibrahim Onuralp Yigit  
**Base Review Commit:** [`c2d2f792`](https://github.com/onuralpyigit/software-as-a-graph/commit/c2d2f7927e89e524dfbd3a1267834e4c85ad6a53) (Git tag: `v4-advisor-review-submitted`)  
**Base Artifact Dimensions:** 22 pages, 11 sections, 780 numbered lines (`\linenumbers`), 90 references  
**Current Advanced HEAD:** [`bac0ce60`](https://github.com/onuralpyigit/software-as-a-graph/commit/bac0ce60) (29 pages, Amendments 7–14)  
**Target Integration Branch:** `revision/advisor-integration`  

---

## 1. Revision Workflow Protocol

1. **Ingestion & Logging**:
   - As soon as the advisor sends his major and minor comments (via marked PDF, email, or meeting notes), each point is entered into Section 3 below with a unique ID (`ADV-MAJ-XX` or `ADV-MIN-XX`).
   - Line numbers from the advisor's feedback map 1:1 to lines 1–780 of the 22-page base manuscript.
2. **Context Resolution (`scripts/advisor_line_mapper.py`)**:
   - Run `python3 scripts/advisor_line_mapper.py --line <LINE_NUM>` to identify:
     - The corresponding TeX file and line in the base version (`c2d2f792`).
     - The corresponding context and diff in current `HEAD`.
     - Relevant amendments or empirical artifacts that address the comment.
3. **Harmonization & Decision Criteria**:
   - **Retain v4 Base vs Adopt HEAD**:
     - *Narrative & Style*: If the advisor requests textual clarifications or phrasing refinements, apply them directly to the base text.
     - *Section 5 (ISO/IEC 25010 Quality Model)*: If the advisor values Section 5, retain it in the main body (pages 10–11). If the advisor agrees the paper should focus strictly on graph learning, cleanly relocate Section 5 to the supplement (`supplementary.tex`).
     - *Baselines (Table 7)*: Retain `Topo` and `Topo-QoS` in the main body table as reviewed, while referencing the supplementary analyses for other heuristics.
     - *Queue-Flow Surrogate ($I_{\text{dyn}}$)*: If the advisor asks about dynamic fidelity or runtime latency, weave in the Amendment 11 GBM surrogate finding ($0.799$ vs $0.706$).
4. **Verification & Audit**:
   - Run `python3 reproduce/reconcile_manuscript.py --profile=advisor-revision` to ensure all numerical claims match the artifact bundle.
   - Build clean PDF and generate `manuscript_diff.pdf` (latexdiff) for the advisor's final sign-off.

---

## 2. Section & Line-Number Reference Map (Base v4 vs. HEAD)

| v4 Page | v4 Lines | Section in v4 Manuscript | Primary TeX File in `c2d2f792` | Corresponding State in `HEAD` | Key Differences & Intersections |
|:---:|:---:|:---|:---|:---|:---|
| **p. 1** | 1–36 | Title, Abstract, Keywords, §1.1 Motivation | `sections/abstract.tex`, `sec1_introduction.tex` | Refocused on dependency graphs | Abstract revised to 248 words; title narrowed. |
| **p. 2** | 37–57 | §1.2 SaG, §1.3 RQs, §1.4 Contributions (item 1) | `sec1_introduction.tex` | RQs reframed around learning | RQ2/RQ3 updated; 6 contributions reordered. |
| **p. 3** | 58–104 | §1.4 Contributions (items 2–5), §2.1 Dependability, §2.2 SCA/SSA | `sec1_introduction.tex`, `sec2_related_work.tex` | Related work condensed | Focus sharpened on GNNs in software engineering. |
| **p. 4** | 105–144 | §2.3 Quality Models, §2.4 Graph Learning, §3.1 Multigraph | `sec2_related_work.tex`, `sec3_sag_model.tex` | Model updated with query rules | Added formal query for afferent coupling. |
| **p. 5** | — | Figure 1 (Architecture Pipeline), Table 1 (Entities/Edges) | `figures/Figure_1.pdf`, `sec3_sag_model.tex` | Figure 1 relabeled | "Training-free baseline" used instead of centrality. |
| **p. 6** | 145–172 | Table 2 (Notation), §3.2 QoS Weights, §3.2.1 DEPENDS_ON, Table 3 | `sec3_sag_model.tex` | Table 2 & 3 identical | Rules 1–6 preserved with full mathematical definitions. |
| **p. 7** | 173–194 | Figure 2 (Toy Example), §3.3 Dual Graph Views, §3.4 Node Encoding | `figures/Figure_2.pdf`, `sec3_sag_model.tex` | Figure 2 preserved; §3.4 intact | 18-D base block + 19–25-D typed blocks verified. |
| **p. 8** | 195–219 | §4 Ranking Engines, Figure 3 (Engines/Ground Truth), §4.1 HGT, §4.1.1 QoS Edge (16-D) | `figures/Figure_3.pdf`, `sec4_failure_impact_prediction.tex` | Figure 3 relabeled | Engine architecture identical; caption clarified. |
| **p. 9** | 220–257 | §4.2 Loss Eq (5–6), Table 4 (Oracles), Primary target $I^*$, Further oracles | `sec4_failure_impact_prediction.tex` | Table 4 updated with $I_{\text{dyn}}$ cost | Added circularity boundaries and $I_{\text{dyn}}$ cost notes. |
| **p. 10** | 258–301 | §4.4 Input–Label Independence, §5 Explanation Layer, §5.1 ISO Standards, §5.2 Q Score, §5.3 Counterfactuals | `sec4_failure_impact_prediction.tex`, `sec5_explanation_layer.tex` | §5 moved to supplement S.35 | **Major divergence point**: §5 is in body in v4, supplement in HEAD. |
| **p. 11** | — | Figure 4 (Explanation Layer Diagram), §6 Setup, §6.1 Corpus, Table 5 | `figures/Figure_4.pdf`, `sec6_experimental_setup.tex` | Fig 4 moved with §5; Table 5 intact | Corpus counts (2,812 nodes across 17 architectures) identical. |
| **p. 12** | 302–364 | §6.2 Predictors, Table 6 (Taxonomy), Eq (7), §6.3 Metrics & Statistics | `sec6_experimental_setup.tex` | Table 6 streamlined | Topo-QoS sole baseline in HEAD; Topo in v4 Table 6. |
| **p. 13** | 365–392 | §7 Results, §7.1 RQ1 Engines vs Baselines, Table 7 (Main LOSO Results) | `sec7_results.tex` | Table 7 rebuilt | v4 has 6 rows (Topo..Hybrid-GAT); HEAD adds $I_{\text{dyn}}$ GBM. |
| **p. 14** | — | Figure 5 (Main Results at a Glance: Panels A, B, C) | `figures/Figure_5.pdf` | Fig 5 panel titles updated | Relabeled baseline legend and panel titles. |
| **p. 15** | 393–440 | §7.1.1 Hybrids, §7.2 RQ2 What Learning Needs, Table 8 ($2\times2$ Matched) | `sec7_results.tex` | Table 8 identical | Cell means ($0.563, 0.548, 0.635, 0.622$) match byte-identically. |
| **p. 16** | 441–476 | §7.3 RQ3 Zero-Shot Transfer, Table 9, §7.4 RQ4 Analysis Cost | `sec7_results.tex` | Table 9 identical; Table 10 updated | Like-for-like counting vs oracle vs GNN added in HEAD. |
| **p. 17** | 477–505 | Table 10 (Latency), §8 Discussion, §8.1 Practical Consequences, Table 11 | `sec7_results.tex`, `sec8_discussion.tex` | Table 11 relabeled | Guidance matrix relabeled to "training-free baseline". |
| **p. 18** | 506–549 | §8.1 (cont), §8.2 Threats to Validity, §8.3 Limitations & Future Work | `sec8_discussion.tex` | Threats expanded | Added degree leak, seed drift, and selection rule records. |
| **p. 19** | 550–598 | §9 Conclusion, Declarations, CRediT, Funding, Data Availability, AI Statement | `sec9_conclusion.tex`, `sections/declarations.tex` | Conclusion refocused | Rewritten into 5 empirical findings on learning. |
| **pp. 20–22** | 599–780 | References [1]–[90] | `refs.bib` | References expanded to 110+ | All 90 original references present and verified. |

---

## 3. Advisor Comments Action Matrix

*To be populated upon receipt of the advisor's review.*

| Comment ID | Category | v4 Line(s) | v4 Section | Advisor Comment Summary | Intersection with HEAD Advances | Planned Action & Resolution | Status |
|:---|:---:|:---:|:---|:---|:---|:---|:---:|
| `ADV-MAJ-01` | Major | *TBD* | *TBD* | *Awaiting advisor input* | *TBD* | *Pending* | Open |
| `ADV-MAJ-02` | Major | *TBD* | *TBD* | *Awaiting advisor input* | *TBD* | *Pending* | Open |
| `ADV-MIN-01` | Minor | *TBD* | *TBD* | *Awaiting advisor input* | *TBD* | *Pending* | Open |
| `ADV-MIN-02` | Minor | *TBD* | *TBD* | *Awaiting advisor input* | *TBD* | *Pending* | Open |

---

## 4. Verification Checkpoints

- [ ] **Checkpoint 1: Line Mapping**: All advisor comments mapped to exact TeX lines via `scripts/advisor_line_mapper.py`.
- [ ] **Checkpoint 2: Content Revision**: Text revisions implemented and reviewed against advisor's directives on `revision/advisor-integration`.
- [ ] **Checkpoint 3: Figures & Tables**: All figures and tables reconciled; cross-references (`xr-hyper` S-*) verified.
- [ ] **Checkpoint 4: Reconciler Check**: `python3 reproduce/reconcile_manuscript.py --profile=advisor-revision` exits code 0 with 1,554 matched figures.
- [ ] **Checkpoint 5: LaTeX Compilation**: Clean compilation of `manuscript.pdf` (0 errors, 0 undefined citations/labels).
- [ ] **Checkpoint 6: Latexdiff Package**: Generated `manuscript_diff.pdf` highlighting changes for advisor review.
