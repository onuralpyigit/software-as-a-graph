# Referee Report (Round 5) — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems"

**Manuscript:** *Software-as-a-Graph: Dependency-Graph Analysis and Learning for Pre-Deployment Simulated Cascade-Impact Ranking in Publish–Subscribe Systems*  
**Authors:** Ibrahim Onuralp Yigit, Feza Buzluca  
**Target Journal:** *Journal of Systems and Software* (Elsevier, Q1)  
**Special Issue:** *AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems*  
**Review Model:** Single-anonymized  
**Commit Evaluated:** `d0d19c1b` (following round-4 revisions)  
**Date:** 2026-09-26  

---

## 1. Summary

The manuscript addresses the critical problem of identifying systemically vulnerable components in asynchronous publish–subscribe architectures prior to deployment, where spatial and temporal decoupling obscures cascading failure paths from traditional static code analyzers and runtime telemetry. The authors propose Software-as-a-Graph (SaG), a static system analysis framework that ingests configuration manifests into a typed multigraph, derives an explicit logical `DEPENDS_ON` dependency projection, and benchmarks training-free structural counts, closed-form centralities, and graph neural networks (GAT, HGT, and hybrids) under leave-one-scenario-out cross-validation across twelve synthetic architectures and zero-shot transfer across five open-source system models. Positioned as an empirical benchmark and negative result for deep graph learning in architectural dependability, the paper demonstrates that simple training-free counting of direct dependents on the derived graph (`InDeg`, formalizing publish–subscribe afferent coupling) matches or exceeds complex neural models across multiple simulation paradigms ($\rho = 0.764$ on topological cascade reachability $I^*$; $\rho = 0.610$ on discrete-event queue-flow $I_{\text{dyn}}$), while executing in milliseconds compared to the substantial feature-extraction latency of neural pipelines.

---

## 2. Overall Impression & Assessment

### Originality and Significance
This manuscript makes an important, refreshing, and methodologically disciplined contribution to the software engineering literature. In an era where deep learning architectures are frequently applied to software engineering tasks without transparent baselines, this study establishes an empirical negative-result benchmark: once the logical publish–subscribe dependency graph is formally derived from deployment manifests, complex graph neural networks (such as Heterogeneous Graph Transformers) fail to extract predictive signal beyond simple, training-free structural fan-in counts (`InDeg`). 

The conceptual formulation of publish–subscribe afferent coupling (Rule 1, mapping 2-hop subscriber counts through shared topics) and simultaneous blast radiuses (Rule 5, shared libraries) provides a genuine theoretical and practical bridge across the "Architecture–Code Gap." The practical significance is clear: architects and DevOps engineers can execute millisecond CI/CD sanity checks on configuration manifests before provisioning staging environments or executing heavy cluster-level chaos engineering experiments, contributing directly to computational sustainability.

### Methodological Soundness & Evolution
The authors have demonstrated exceptional scientific integrity in responding to earlier rounds of review. In this latest revision (`d0d19c1b`), the critical methodological flaws flagged in Round 4 have been systematically resolved:
1. **Genuine Multi-Criteria Failure Simulation ($I_{\text{comp}}$):** The previous misattribution of SaG's explanation score $Q(v)$ as an evaluation oracle has been rectified; $I_{\text{comp}}$ was recomputed exhaustively across all 1,321 application components using the true multi-criteria `FailureSimulator` (evaluating reachability, network fragmentation, throughput drop, and flow disruption).
2. **Provenance and Mechanical Reconciliation:** The unverified `GAT-P-QoS` row in Table 7 has been purged; 1,275 table values across the manuscript and supplement now mechanically reconcile against committed JSON artifacts in `data/benchmarks/`; and the analytic first-order ceiling ($0.808$ / $0.631$) exactly reproduces the underlying harness.
3. **QoS Attribution Controls:** The confounding between QoS weights and structural degree has been explicitly reported using registered controls (`qos_attribution_controls.json`), demonstrating that topic permutation incurs no significant loss ($\Delta = -0.006, p = 0.733$) and confirming that graph topology drives the predictive power.
4. **Tone and Caveat Integrity:** Overstatements regarding "pre-registration" and "orders of magnitude" have been corrected to strictly factual terms, and essential caveats regarding hardware drift, Wilcoxon anti-conservatism under fold dependence, and single-modeller bias have been candidly restored.

### Special Issue Fit & Q1 Suitability
The manuscript aligns squarely with the *JSS Special Issue on AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems*. It evaluates AI techniques (GNNs vs. heuristics) for system reliability (cascading failure prediction) while rigorously assessing computational sustainability (profiling feature extraction vs. direct simulation cost). The manuscript meets the high quality and evidentiary standards expected by JSS. A few remaining methodological subtleties and reporting discrepancies require attention before final publication.

---

## 3. Major Comments

### M1. Candidate Subsampling and Lexical Selection Bias in $I_{\text{dyn}}$ ($n = 30$)
The evaluation of rankers against the independent discrete-event queue-flow simulator ($I_{\text{dyn}}$) in Table 7 is the study's primary defense against circularity with the reachability oracle $I^*$. The authors have appropriately disclosed in the text and caption that $I_{\text{dyn}}$ is evaluated on an $n = 30$ candidate application sample per fold. 

However, inspecting `reproduce/convergent_validity.py` (line 108) and `saag/simulation/message_flow_simulator.py` reveals that the 30 candidates are selected via `probe.labeled_node_ids[:max_candidates]`, where `labeled_node_ids` is sorted alphabetically by string identifier: `['A0', 'A1', 'A10', 'A100', 'A101', ...]`.
- In large synthetic topologies such as Enterprise ($|V_{\text{app}}| = 300$) or Smart City ($|V_{\text{app}}| = 200$), taking the first 30 strings means that components with IDs `A2` through `A9` are evaluated alongside a tight cluster of `A10`–`A19` and `A100`–`A109`, while large swathes of the system (`A20`–`A99`, `A200`+) are completely unobserved.
- In synthetic topology generators, component generation sequences frequently correlate with structural roles (e.g., initial seed services vs. late-generated edge adapters). Lexical sorting imposes a non-random, deterministic sampling filter that may under-represent peripheral services.
- **Action Required:** The authors must either:
  1. Demonstrate via a quick sensitivity check that a uniformly seeded random sample (or an evaluation across all applications for the smaller folds where it is computationally feasible) yields rank correlations statistically indistinguishable from the reported $\rho = 0.610$; or
  2. Clearly state this lexical selection mechanism in §4.3 and discuss the potential sampling bias as an explicit threat to construct/external validity in §8.3.

### M2. Triage Disconnect: High Spearman Correlation ($\rho = 0.764$) vs. Moderate Critical-Set Overlap (Overlap@$K = 0.504$)
The paper strongly advocates for `InDeg` as a lightweight CI/CD gate for pre-deployment architectural triage. However, there is a substantial practical tension between the headline rank correlation and the set-identification metric:
- In Table 6, `InDeg` achieves an impressive $\rho = 0.764$ over the full population and $\rho_{>0} = 0.516$ on active components, but its Overlap@$K$ ($K = 0.20\,|V_{\text{app}}|$) is only $0.504$. For `Reach`, Overlap@$K$ is even lower at $0.344$.
- In industrial continuous integration, software architects rarely inspect a continuous ranking across hundreds of microservices; they configure automated gating rules that flag components in the top 10% or 20% severity tier for mandatory review or fail the build if unmitigated critical nodes are introduced.
- An Overlap@$K$ of ~0.50 means that approximately half of the predicted top-tier critical components are false positives, and conversely, half of the true top-tier failure hubs are false negatives.
- **Action Required:** In Section 8.1 and the narrative surrounding Table 11, the authors should temper the recommendation of `InDeg` for automated gating by explicitly addressing this threshold churn. The authors should discuss whether `InDeg` should be paired with a safety buffer (e.g., inspecting the top 30% to capture the true top 20%) or combined with the explanation layer’s Tukey fence ($Q(v)$) to prevent high false-alarm rates in automated CI pipelines.

### M3. Boundary Conditions of the Negative Result on Graph Neural Networks
The paper's positioning as an empirical negative result for deep graph learning is compelling, but the manuscript must ensure the boundary conditions are sharply delineated so as not to overclaim:
1. **Structural Edge Inversion on the Raw Multigraph:** As insightfully explained in §8.2, native publish–subscribe relations point away from Applications (`App -> Topic`, `Broker -> Topic`, `App -> Host`). Consequently, standard forward GAT/HGT message passing delivered zero messages to scored applications, reducing them to node-level MLPs over precomputed centralities. Thus, the poor performance of GNNs on the raw multigraph was partly an artifact of graph edge orientation rather than an intrinsic failure of graph learning.
2. **Untuned HGT on the Projection:** On the derived dependency projection where message passing reaches applications, `GAT-P-QoS` converged smoothly to $\rho = 0.748$ (matching `InDeg`), whereas `HGT-P-QoS` failed to train stably (seed spread $0.208$, mean $\rho = 0.514$). However, as the authors note, this was an untuned configuration (width 100, default hyperparameters).
- **Action Required:** Ensure that the abstract, contributions (§1.4), and conclusion (§9) do not generalize this as an absolute failure of relational transformers. The text should consistently specify that *in standard untuned configurations on sparse derived dependency projections, parameter-heavy relational typing introduces optimization instability without exceeding transparent structural counts.*

### M4. Value Proposition: Static System Analysis (SSA) vs. Direct In-Process Simulation
Section 7.4 (Table 10) presents a very honest and crucial measurement: feature extraction for neural engines takes median $5.6\times$ ($2.0\text{--}17.7\times$) longer than running direct in-process cascade simulation ($I^*$) on the raw graph.
- This creates an obvious pragmatic dilemma for practitioners: if direct simulation takes only a fraction of a second and requires no feature engineering or graph projection, why bother with static analysis models at all?
- The authors address this in §8.1 by arguing that static analysis: (1) operates before runtime/simulation parameters are calibrated; (2) provides ISO/IEC 25010 root-cause attribution; and (3) correlates with independent queue-flow dynamics.
- However, `InDeg` itself relies on the same topological connectivity that drives the first wave of $I^*$, and the ISO/IEC remediation layer remains an unvalidated proposal (§8.4).
- **Action Required:** Clarify the practical taxonomy in Section 8.1:
  - If a team has calibrated operational profiles and simulation models, direct in-process simulation ($I^*$) or queue modeling ($I_{\text{dyn}}$) is preferable.
  - The distinct value of SaG's static approach lies specifically in **millisecond manifest linting via `InDeg`** during early design/commit stages where no simulator has been built or calibrated, and where the goal is instantaneous architectural feedback.

---

## 4. Minor Comments

1. **Missing Subsection Numbering in Section 3:**
   In `manuscript.md` (line 120) and `sections/sec3_sag_model.md` (line 62), the heading is written as:
   `## Logical Dependency Projection (`DEPENDS_ON`)`
   It lacks the subsection number `3.3`. It should be updated to `## 3.3 Logical Dependency Projection (`DEPENDS_ON`)` to match the cross-references in §1.2, §1.4, and §6.2.
2. **Reconciled Figures Count Discrepancy:**
   In `declarations.md` (line 5) and `manuscript.md` (line 500), the Data Availability statement states:
   *"mechanically verifying 511 reported figures in the manuscript and supplement against the JSON artifacts."*
   However, running `python3 reproduce/reconcile_manuscript.py` now reports:
   *"Reconciled 1275 table figures against committed artifacts (1 check(s) skipped). OK — 1275 figures match their artifacts."*
   The text in the Data Availability statement should be updated to reflect the true number of reconciled figures (1,275).
3. **Table Count in Length Justification:**
   In `latex/LENGTH_JUSTIFICATION.md` (line 14), the text states: `(10 tables, 5 figures)`. In fact, the manuscript body contains 12 tables (Table 1 through Table 12). Please correct this count.
4. **Analytic First-Order Approximation Equation (5):**
   In Section 7.1, equation (5) defines:
   $$\hat{I}^*_1(v) = \sum_{t \in \text{pub}(v)} \frac{|\text{sub}(t)|}{|\text{pub}(t)|}$$
   Please ensure that it is explicitly noted that topics with $|\text{pub}(t)| = 0$ are either impossible by definition of $\text{pub}(v)$ or safely handled, preventing any formal ambiguity regarding division by zero.
5. **Table 7 Active Stratum Equality for $I_{\text{comp}}$:**
   In Table 7, for $I_{\text{comp}}$, `Mean \rho` and `\rho_{>0}` are identical across all rankers (e.g., 0.650 / 0.650 for `InDeg`, 0.702 / 0.702 for `Topo-QoS`). The authors should add a brief explanatory note in the text clarifying that because genuine multi-criteria simulation evaluates continuous fragmentation and flow disruption, virtually all components experience non-zero impact, making the active stratum coincide with the total population.
6. **Highlights Plain-Text Compliance:**
   In `latex/highlights.tex`, Bullet 2 contains LaTeX math formatting: `$\rho = 0.76$`. In Elsevier’s editorial system, highlights are submitted as plain text fields. Replace `$\rho = 0.76$` with unicode `ρ = 0.76`.
7. **Cross-References Check:**
   In §6.2, verify that all references to node property dimensions accurately point to §3.5 and Supplementary §S11.
8. **Vitae and Compliance Check:**
   The author biographies in `latex/vitae.tex` (84 and 87 words) adhere strictly to the JSS Guide for Authors limit of 100 words per author. The CRediT authorship statement and Declaration of Generative AI are present, complete, and properly placed.

---

## 5. Recommendation

**Decision:** **Minor Revision**

### Justification
The authors have done an exemplary job addressing the extensive feedback from previous review rounds. The paper has evolved from an overclaimed GNN advocacy piece into a rigorous, honest, and highly valuable empirical benchmark and negative result for graph learning in software architecture dependability. 

The central methodological concerns from Round 4 have been resolved: the circular $I_{\text{comp}}$ oracle was replaced with genuine multi-criteria failure simulations, the unbacked GAT row in Table 7 was removed, all figures are mechanically reconciled against released JSON artifacts, and the framing around afferent coupling and simple structural baselines is academically mature.

The remaining issues (clarifying the lexical subsampling in $I_{\text{dyn}}$, addressing the Overlap@$K$ triage gap, refining the wording on GNN boundary conditions, and correcting minor numbering and figure-count discrepancies) are straightforward and do not require major architectural re-experimentation. Upon addressing these minor points, this paper will be an outstanding, methodologically exemplary contribution to the *Journal of Systems and Software*.
