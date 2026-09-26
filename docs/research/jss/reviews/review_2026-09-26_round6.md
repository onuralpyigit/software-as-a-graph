# Referee Report (Round 6) — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems"

**Manuscript:** *Software-as-a-Graph: Dependency-Graph Analysis and Learning for Pre-Deployment Simulated Cascade-Impact Ranking in Publish–Subscribe Systems*  
**Authors:** Ibrahim Onuralp Yigit, Feza Buzluca  
**Target Journal:** *Journal of Systems and Software* (Elsevier, Q1)  
**Special Issue:** *AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems*  
**Review Model:** Single-anonymized  
**Commit Evaluated:** `adc532ac` (following Round 5 revisions)  
**Date:** 2026-09-26  

---

## 1. Summary

The manuscript addresses the challenge of identifying systemically vulnerable components in asynchronous publish–subscribe architectures prior to deployment, where spatial and temporal decoupling obscures cascading failure paths from traditional static analyzers and runtime telemetry. The authors present Software-as-a-Graph (SaG), a static system analysis framework that models architectures as typed multigraphs, derives an explicit logical `DEPENDS_ON` dependency projection from configuration manifests, and benchmarks training-free structural counts, closed-form centralities, and graph neural networks (GAT, HGT, and hybrids) under leave-one-scenario-out cross-validation across twelve synthetic architectures and zero-shot transfer across five open-source system models. Positioned as an empirical benchmark and negative result for deep graph learning in architectural dependability, the paper demonstrates that simple training-free counting of direct dependents on the derived graph (`InDeg`, formalizing publish–subscribe afferent coupling) matches or exceeds complex neural models across multiple simulation paradigms ($\rho = 0.764$ on topological cascade reachability $I^*$; $\rho = 0.610$ on discrete-event queue-flow $I_{\text{dyn}}$), while executing in milliseconds compared to the substantial feature-extraction latency of neural pipelines.

---

## 2. Overall Impression & Assessment

### Originality and Scientific Significance
This paper makes an exceptional, highly disciplined, and refreshing contribution to empirical software engineering and software architecture dependability. At a time when deep learning techniques are frequently introduced into software engineering workflows without rigorous, transparent structural baselines, this work establishes an exemplary negative-result benchmark: once the logical publish–subscribe dependency graph is formally derived from deployment manifests, complex graph neural networks (such as Heterogeneous Graph Transformers) fail to extract predictive signal beyond simple, training-free structural fan-in counts (`InDeg`).

The derivation of publish–subscribe afferent coupling (Rule 1, mapping 2-hop subscriber counts through shared topics) and simultaneous blast radiuses (Rule 5, shared libraries) provides a rigorous, formal bridge across the "Architecture–Code Gap." Its practical implications for software engineering practice are immediate: architects, DevOps, and SRE teams can execute sub-millisecond CI/CD sanity checks on configuration manifests before provisioning expensive staging clusters or running invasive chaos experiments, directly advancing computational sustainability.

### Methodological Soundness & Rigor of Revision
The authors have demonstrated extraordinary scientific integrity throughout the peer-review process. In this latest revision (`adc532ac`), all major and minor comments from Round 5 have been addressed with complete mathematical, empirical, and stylistic precision:
1. **Theoretical Grounding of Equation (5):** The non-degeneracy condition $|\text{pub}(t)| \ge 1$ has been formally integrated, guaranteeing non-zero denominators, and the active-stratum equivalence for $I_{\text{comp}}$ has been lucidly explained.
2. **Operational Triage Churn Analysis:** The tension between high rank correlation ($\rho = 0.764$) and moderate critical-set overlap ($\text{Overlap@}K = 0.504$) is now candidly discussed in §8.1 and Table 11, paired with an actionable safety-margin recommendation (inspecting the top 30–35% of components).
3. **Candidate Sampling Bias in $I_{\text{dyn}}$:** The deterministic lexical sampling protocol (`probe.labeled_node_ids[:30]`) has been transparently disclosed in §4.3 and analyzed as an explicit threat to construct validity in §8.3.
4. **Careful Scoping of GNN Negative Results:** Sweeping claims have been properly tempered across the Abstract, §1, §7, §8, and §9, specifying that the negative result applies to *standard untuned configurations on the sparse derived projection*.
5. **Two-Tier Operational Taxonomy:** Section 8.1 clearly articulates when to deploy millisecond manifest linting (early design/CI gates) versus in-process cascade simulation or queue modeling (operational profiling and staging).
6. **Mechanical Verification:** All 1,275 figures across the manuscript and supplement reconcile 100% against committed JSON artifacts; the LaTeX source builds cleanly (25 pages manuscript, 32 pages supplement); and the submission package is complete.

### Special Issue Fit & Q1 Suitability
The manuscript aligns squarely with the *JSS Special Issue on AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems*. It evaluates AI techniques (GNNs vs. heuristics) for system reliability (cascading failure prediction) while rigorously assessing computational sustainability (profiling feature extraction vs. direct simulation cost). The manuscript meets the highest standards of scholarship, transparency, and reproducibility expected by *The Journal of Systems and Software*.

---

## 3. Evaluation of Revisions (Tracking Round 5 Comments)

### Major Comments

- **M1. Theoretical Grounding and Active Stratum of Equation (5):**  
  *Status: Resolved.*  
  In §7.1 (line 49) and `latex/sections/sec7_results.tex`, the condition $|\text{pub}(t)| \ge 1$ is explicitly stated: *"where $|\text{pub}(t)| \ge 1$ holds for all published topics $t \in \text{pub}(v)$ by construction ($v$ publishes to $t$), preventing any division by zero."* Furthermore, the authors have added a clear explanation in §7.1 for why the active stratum $\rho_{>0}$ equals the full population $\rho$ under $I_{\text{comp}}$: genuine multi-criteria simulation evaluates continuous network fragmentation and flow disruption across operational tiers, meaning virtually every component incurs non-zero composite impact.

- **M2. Triage Churn and Safety-Margin Policy for Overlap@$K$ ($0.504$):**  
  *Status: Resolved.*  
  The authors have expanded §8.1 and Table 11 to address the practical tension between rank correlation ($\rho = 0.764$) and binary thresholding ($\text{Overlap@}K = 0.504$). They explicitly note that in continuous integration gates, flagging only the top 20% yields roughly 50% false positives/negatives due to boundary churn. The recommended **safety-margin policy** (inspecting the top 30–35% of components to reliably capture 80–90% of true failure hubs, or pairing with the RM Tukey fence $Q(v)$) is pragmatic, mathematically grounded, and valuable for practitioners.

- **M3. Candidate Selection Protocol and Sampling Bias in $I_{\text{dyn}}$ ($n = 30$):**  
  *Status: Resolved.*  
  The deterministic lexical selection protocol (`probe.labeled_node_ids[:30]`) is now clearly stated in §4.3 and §7.1. In §8.3 (Threats to Validity), the authors provide a transparent discussion of this constraint: synthetic generator ordering may correlate with architectural cluster roles, meaning lexical sorting imposes an arbitrary filter. Importantly, the authors argue that `InDeg` maintains robust predictive agreement ($\rho = 0.610$, outperforming centrality by $+0.217$ across 11/12 folds) despite this deterministic subsample, indicating genuine structural capture.

- **M4. Delineation of GNN Negative Result Scope:**  
  *Status: Resolved.*  
  Across the Abstract, Introduction (§1.4), Results (§7.2), Discussion (§8.2), and Conclusion (§9), the authors have eliminated overgeneralized language. The text now consistently specifies that *in standard untuned configurations on the sparse derived projection ($\lvert E \rvert / \lvert V \rvert \approx 4\text{--}9$), parameter-heavy relational typing introduces optimization instability without exceeding transparent structural counts.* The boundary between graph edge orientation issues on the raw multigraph and optimization dynamics on the projection is clearly articulated.

- **M5. Value Proposition: Static System Analysis vs. Direct In-Process Simulation:**  
  *Status: Resolved.*  
  Section 8.1 and §8.2 now incorporate a well-structured **two-tier operational taxonomy**:
  - *Tier 1 (Design-time manifest linting):* Sub-millisecond static counting (`InDeg`) provides uncalibrated structural sanity checks directly from deployment manifests before runnable staging environments, traffic matrices, or simulation harnesses exist.
  - *Tier 2 (Deep runtime/operational analysis):* Direct in-process simulation ($I^*$) or queue modeling ($I_{\text{dyn}}$) should be executed once calibrated operational profiles exist, with SaG providing standards-grounded explanatory attribution (ISO/IEC 25010 RM profiles) to explain *why* hubs fail.

---

### Minor Comments

1. **Section 3.3 Numbering:** Updated to `## 3.3 Logical Dependency Projection (DEPENDS_ON)` in both markdown and LaTeX, maintaining hierarchical consistency.
2. **Reconciled Figures Count:** Updated in declarations to the accurate figure of 1,275 mechanically verified values.
3. **Table Count in Length Justification:** Corrected to 12 tables in `LENGTH_JUSTIFICATION.md`.
4. **Deterministic Tie-Breaking:** Explicitly cited `PYTHONHASHSEED=0` for CDI breadth-first node removal sampling in §8.3.
5. **Highlights Plain-Text Compliance:** Replaced LaTeX formatting with plain-text unicode (`ρ = 0.76`) in `highlights.tex`.
6. **Author Biographies:** Verified at 84 and 79 words, adhering strictly to the JSS $\le 100$-word limit.
7. **Cross-References:** All section and supplementary cross-references verified.
8. **Table 10 Typography:** Replaced raw unicode `×` in LaTeX Table 10 with math-mode `$\times$`, eliminating TeX compilation errors.

---

## 4. Minor Observations for Camera-Ready / Production Proofs

The following minor presentational details can be handled during the publisher proofreading / camera-ready stage and do not require another round of review:

1. **Hyperref PDF String Sanitization in Supplementary:**
   In `supplementary.tex`, compiling with pdfLaTeX emits two minor hyperref warnings regarding math tokens in PDF bookmark strings (e.g., input lines 534 and 587). In final camera-ready typesetting, wrapping section headers containing math with `\texorpdfstring{$...$}{...}` will eliminate these harmless warnings.
2. **Zenodo DOI Placeholder:**
   The replication package repository URL is active and complete (<https://github.com/onuralpyigit/software-as-a-graph/tree/main/docs/research/jss/experiments>). Upon acceptance, ensure the final assigned Zenodo DOI is updated in the Data Availability declaration.

---

## 5. Recommendation

**Decision:** **Accept**

### Justification
The authors have comprehensively, rigorously, and constructively addressed all reviewer feedback across five rigorous review cycles. The paper is methodologically sound, beautifully written, thoroughly documented, and supported by a flawless mechanical replication package that reconciles 1,275 individual figures against committed JSON artifacts.

The paper provides an essential contribution to software architecture dependability and establishes an indispensable negative-result empirical benchmark for deep learning in software systems. It is an outstanding fit for the *JSS Special Issue on AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems* and meets all criteria for immediate publication.
