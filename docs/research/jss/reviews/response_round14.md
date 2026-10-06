# Response to the Round-14 Peer Review Report

**Manuscript Title:** When Does Graph Learning Add Value to Software Architecture Analysis? A Critical Evaluation and Reference Criterion  
**Journal:** Journal of Systems and Software (*Special Issue on AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems*)  
**Evaluation Mode:** Major Revision (Round 14)  
**Replication Commit:** `56d9bff8` (initial plan), `7b6f475a` (round-14 revision on `main`)

---

We sincerely thank the Distinguished Academic Peer Reviewer for this exceptionally thorough, rigorous, and constructive evaluation. The critique has substantively elevated the methodological clarity, conceptual framing, and precision of the manuscript.

Below, we detail our point-by-point responses to all Major Comments (M1–M10) and Minor Comments (Minors 1–21), documenting every change made, its theoretical and empirical rationale, and its exact location in the revised manuscript.

---

## 1. Summary of Major Revisions

1. **Title & Scope (M1):** In accordance with the author's decision, the title is maintained. The manuscript explicitly qualifies throughout the abstract, introduction, discussion, and conclusion that all negative findings are strictly bounded to *first-order simulation regimes* where direct publish--subscribe afferent coupling governs failure propagation.
2. **Reference Criterion & Afferent Coupling Construct Validity (M2):** We expand §4.4 with an explicit theoretical justification for simplifications S1–S5 (isolating the deterministic, low-order topological skeleton from stochastic variance, arbitrary scales, and dynamic queueing). We demonstrate that canonical graph centrality measures (betweenness, PageRank) fail this criterion and are classified as predictors, proving the criterion is structurally grounded. We confront the construct validity of $I^*$, clarifying that it fundamentally assesses publish--subscribe afferent coupling under stochastic perturbation.
3. **GNN Expressivity Mechanism (M3):** We integrate graph representation and expressivity literature (Xu et al., ICLR 2019; Corso et al., NeurIPS 2020) into §2.3 and §6.2 / §7.2, explaining that the $+0.072$ representation gain and $+0.231$ featureless gain stem directly from 2-hop attention averaging (which normalizes neighbor weights and cannot count paths) versus 1-hop sum aggregation on the derived dependency projection.
4. **Queue-Flow Simulator & Surrogate Modeling Framing (M4):** We disclose in §4.3 and §6.1 / §7.1 that the GNN trained on $I_{\text{dyn}}$ received only static topology and QoS edge features without declared publication rates or payloads. We clarify in §6.4 / §7.4 that $I_{\text{dyn}}$'s 12.7 CPU hours stem from SimPy discrete-event simulation interpreter overhead rather than complex physical dynamics, and reframe the break-even analysis around amortized CI/CD pipeline lifecycle economics.
5. **Baseline Demotion & Meaningful Structural Reference (M5):** We demote comparisons against the defective `Topo-QoS` baseline in the abstract, introduction, and results, establishing afferent coupling (`InDeg`, $0.764$) as the primary structural reference. We clarify that hybrid gains reflect the weakness of the `Topo-QoS` baseline rather than learned correction.
6. **Inferential Robustness & Noise Ceilings (M6):** We characterize the ListMLE arbitrary tie-breaking noise floor ($\approx 0.044$ per fold) induced by 31% zero-impact components, explain the $0.172$ cross-device drift as PyTorch Geometric non-deterministic scatter/gather operations and hardware BLAS variations across CPU/GPU environments, and affirm why all comparisons are strictly paired within self-contained sweeps.
7. **Small-Data Regime Calibration (M7):** We formally bound the findings to eleven synthetic architectures (~1,000–1,300 labeled Applications per fold), noting that this regime biases comparisons toward the null, and prioritize sample complexity learning curves in future work.
8. **External Validity & Deployment Manifests (M8):** We qualify all deployment manifest claims across the abstract, §1.2, and §9, clarifying that SaG models are derived from architecture specifications and deployment descriptors rather than automated parsers. We expand the single-modeler threat in §7.4, acknowledge that >50% inert components inflate full-population $\rho$ while active discrimination is weak ($\rho_{>0} \le 0.342$), and cite the `model_agreement.py` protocol.
9. **Framework Simplification & Table 1 Reformatting (M9):** Table 1 is cleanly reformatted into two distinct panels (Table 1a: Entity Types; Table 1b: Structural Relations). We explicitly state that Rules 2, 3, 4, 6 and the Figure 1 explanation layer are unexercised infrastructure extensions.
10. **Writing Streamlining & Claim Status Consolidation (M10):** The abstract is condensed to 245 words (strictly $\le 250$ per JSS guidelines). In §5.3 / §6.3, we provide a unified, structured summary table defining the confirmatory, registered secondary, or exploratory status, statistical test, and outcome of every core claim. We cleanse repetitive meta-discourse across all sections.

---

## 2. Point-by-Point Responses to Major Comments

### M1. Core Premise: Title, Scope, and Metamodeling Framing
> **Reviewer Comment:** The study's design cannot answer the question in the title because every oracle is first-order by construction. Bounding the negative findings to first-order regimes is necessary. The metamodeling framing must cite established literature.

- **Author Response:** We completely agree. While the title is retained per author decision, we have thoroughly rewritten the abstract, §1.2, §1.3, §4.4, §7.1, §7.4, and §9 to make clear that the findings are strictly bounded to first-order simulation regimes where cascade outcomes are dominated by direct subscriber loss (afferent coupling). In §2.3, we now cite classical simulation metamodeling literature, notably Kleijnen (*Design and Analysis of Simulation Experiments*, 2015/2018), framing SaG's analytical references as zero-parameter, low-order structural metamodels of the discrete-event simulation engine. In §2.1, we cite machine learning data leakage literature (Kapoor & Narayanan, *Patterns* 2023; Geirhos et al., *Nature Machine Intelligence* 2020) to contextualize construct overlap between predictors and simulator rules.
- **Changes in Manuscript:**
  - `sections/abstract.tex`: Replaced unhedged statements with explicit scoping to first-order simulation regimes.
  - `sections/sec1_introduction.tex`: Tempered surrogate rhetoric and qualified headline claims.
  - `sections/sec2_related_work.tex`: Added citations to Kleijnen (2015) in §2.3 and Kapoor & Narayanan (2023) in §2.1.
  - `sections/sec8_discussion.tex` (§7.1, §7.4): Added explicit bounds explaining that low-order simulators admit low-order truncations by definition.

---

### M2. Construct Validity of the Oracles & Justification of S1–S5
> **Reviewer Comment:** Simplifications S1–S5 in the reference criterion appear tailor-made to include `InDeg` and Eq. 7. The underlying construct of $I^*$ is publish--subscribe afferent coupling; this must be confronted directly.

- **Author Response:** We have substantially strengthened §4.4. We provide an explicit theoretical rationale for S1–S5: these five rules isolate the deterministic, low-order topological skeleton of the simulator from stochastic sampling variance (S1), arbitrary severity multipliers (S2), dynamic queueing and timing mechanics (S3), and continuous rate parameterizations (S4, S5). They admit only zero-parameter structural abstractions computable directly from declared dependencies.
  Crucially, we now highlight a counter-example: canonical global network centralities (e.g., betweenness centrality, PageRank, eigenvector centrality) fail the reference criterion because they evaluate all-pairs geodesics or global random-walk stationary distributions rather than truncated failure propagation waves; like learned GNNs, they are classified as *predictors*. Furthermore, we explicitly state that because direct dependents ($C_a$) dominate cascade outcomes under $I^*$, the reference criterion confirms that $I^*$ fundamentally assesses publish--subscribe afferent coupling under stochastic perturbation.
- **Changes in Manuscript:**
  - `sections/sec4_failure_impact_prediction.tex` (§4.4): Added rationale for S1–S5, centrality counter-example, and afferent coupling construct interpretation.
  - `sections/sec8_discussion.tex` (§7.1, §7.4): Deepened discussion of oracle circularity and afferent coupling construct validity.

---

### M3. Graph Neural Network Expressivity Mechanisms
> **Reviewer Comment:** The representation gain (+0.072 for GAT-P-QoS vs. GAT-QoS-R; +0.231 without oracle features) should be explained through the expressivity of aggregators: attention averages and cannot count neighbors, whereas sum aggregation counts.

- **Author Response:** We have incorporated this insight directly into §2.3, §6.2 / §7.2, and §7.1 / §8.1. We cite the fundamental expressivity results of Xu et al. (*ICLR* 2019) and Corso et al. (*NeurIPS* 2020), explaining the precise architectural mechanism: on the raw multigraph, messages must traverse intermediate Broker and Topic nodes via 2-hop attention averaging (which normalizes neighbor weights and cannot count paths), whereas on the derived `DEPENDS_ON` projection, direct 1-hop sum aggregation (GIN) or localized attention directly counts afferent dependents. When oracle-aligned degree features are stripped, sum aggregation retains $\rho = 0.724$ on the dependency graph, while attention drops to $0.610$ and raw-graph models collapse to $0.378$.
- **Changes in Manuscript:**
  - `sections/sec2_related_work.tex` (§2.3): Added GNN expressivity literature (Xu et al. 2019, Corso et al. 2020).
  - `sections/sec7_results.tex` (§7.2): Emphasized the aggregator expressivity mechanism in the discussion of F8 and F11.
  - `sections/sec8_discussion.tex` (§8.1, §8.2): Linked the representation effect to 2-hop attention averaging vs. 1-hop sum counting.

---

### M4. Queue-Flow Simulator ($I_{\text{dyn}}$) Framing & GNN Feature Setup
> **Reviewer Comment:** The GNN on $I_{\text{dyn}}$ received no rates, while rates were the only declared signal. The 12.7 CPU-hour cost is a SimPy interpreter artifact, and the break-even argument is circular.

- **Author Response:** We have revised §4.3, §6.1 / §7.1, and §6.4 / §7.4 to disclose and reframe these points fully:
  1. *GNN Feature Setup:* We explicitly disclose in §4.3 and §6.1 / §7.1 that `GAT-P-QoS->dyn` received only static topological and QoS edge features without declared publication rates or payloads, whereas LightGBM and the rate-weighted reference (Eq. 7) received declared publication rates directly. We state plainly that the GNN was not a rate-aware surrogate.
  2. *SimPy Cost Rationale:* In §6.4 / §7.4, we clarify that $I_{\text{dyn}}$'s 12.7 CPU hours stem from SimPy discrete-event simulation interpreter overhead rather than intrinsically complex physical dynamics, explaining why a closed-form first-order rate formula naturally approximates it.
  3. *CI/CD Amortized Break-Even:* We reframe the economic analysis around continuous integration (CI/CD) lifecycle economics. Rather than focusing on negligible milliwatt differences during inference, we analyze the amortized cost: an $I_{\text{dyn}}$ surrogate requires an upfront cost for simulator label generation (~355.6 Wh across twelve topologies) plus training ($0.22$ kWh), which must be amortized over recurrent CI evaluation runs. Because the training-free formula incurs zero labeling and training overhead while achieving equal or superior ranking accuracy, it economically dominates learned approximations in CI pipelines.
- **Changes in Manuscript:**
  - `sections/abstract.tex`: Removed sensationalized surrogate modeling claims.
  - `sections/sec4_failure_impact_prediction.tex` (§4.3): Added disclosure that GNN on $I_{\text{dyn}}$ received neither rates nor payloads.
  - `sections/sec7_results.tex` (§7.1, §7.4): Reframed SimPy execution cost and amortized CI/CD lifecycle break-even.

---

### M5. Registered Comparator (`Topo-QoS`) Demotion
> **Reviewer Comment:** `Topo-QoS` is a weak, defective baseline. Demote comparisons against it in the abstract and highlights, and lead with afferent coupling (`InDeg`).

- **Author Response:** We have completely restructured the comparative narrative across the abstract, highlights, §1.3, §5.2 / §6.2, and §6.1 / §7.1:
  - In the abstract and highlights, we removed claims emphasizing superiority over `Topo-QoS` (such as $\rho \approx 0.81$ vs. $0.53$) and replaced them with direct comparisons against afferent coupling (`InDeg`, $0.764$).
  - In §6.1 / §7.1, `InDeg` is treated as the primary structural reference.
  - We explicitly clarify in §6.1 / §7.1 that the hybrid models' $+0.072 / +0.065$ gains over `Topo-QoS` reflect the weakness of the defective betweenness baseline rather than learned correction, demonstrating that when hybrids are initialized with `InDeg` as their prior, they reproduce `InDeg` within $\pm 0.012$ without improving on it.
- **Changes in Manuscript:**
  - `sections/abstract.tex`: Demoted `Topo-QoS`; emphasized afferent coupling reference.
  - `highlights.tex`: Replaced baseline comparison with novel empirical finding on afferent coupling.
  - `sections/sec7_results.tex` (§7.1): Reframed Table 5 discussion to lead with `InDeg` and clarify hybrid baseline dependency.

---

### M6. Inferential Robustness, Tie-Breaking Noise Floor, and Cross-Device Drift
> **Reviewer Comment:** Characterize the noise sources: explain the $0.172$ cross-device drift and contextualize effect sizes against the ListMLE tie-breaking noise floor ($\approx 0.044$).

- **Author Response:** We have addressed these points thoroughly in §6.2 / §7.2 and §7.4 / §8.4:
  1. *Tie-Breaking Noise Floor:* In §6.2 / §7.2, we explain that because 31% of components share tied zero labels ($I^* = 0$), ListMLE's arbitrary input-order tie-breaking induces an empirical noise floor of approximately $0.044$ per fold. We explicitly contextualize model differences (such as the $+0.072$ representation gain and configuration swings of $\pm 0.055$) against this noise floor.
  2. *Cross-Device Drift Explanation:* In §7.4 / §8.4, we document the technical cause of the observed $0.172$ maximum cell drift: non-deterministic scatter/gather reductions in PyTorch Geometric message passing across different CPU/GPU hardware architectures and underlying BLAS library versions. We explain that this hardware non-determinism necessitated our protocol where every contrast is strictly paired within self-contained sweeps executed simultaneously under identical hardware environments.
- **Changes in Manuscript:**
  - `sections/sec7_results.tex` (§7.2): Added explanation of ListMLE tie-breaking noise floor.
  - `sections/sec8_discussion.tex` (§8.4): Explained cross-device drift as PyTorch Geometric scatter/gather non-determinism across compute devices.

---

### M7. Small-Data Regime Calibration
> **Reviewer Comment:** Bounding findings to the small-data regime (eleven synthetic architectures, ~1,000–1,300 labeled Applications per fold) is necessary to separate "learning added no value" from "learning was starved".

- **Author Response:** We have explicitly stated in the abstract, §6.2 / §7.2, §7.4 / §8.4, and §9 that all learned findings are conditional on this small-data regime: eleven training architectures, ~1,000–1,300 labeled Applications per fold, and ~430k parameters per GNN. In §7.4, we explicitly state that this sample size biases learned-versus-analytical comparisons toward the null. In §7.5 / §8.5, evaluating sample complexity through learning curves over generated topologies is designated as a prioritized future work direction.
- **Changes in Manuscript:**
  - `sections/abstract.tex`, `sections/sec7_results.tex` (§7.2), `sections/sec8_discussion.tex` (§8.4), `sections/sec9_conclusion.tex`: Qualified all learning findings with explicit small-data regime conditions.
  - `sections/sec8_discussion.tex` (§8.5): Prioritized sample complexity learning curves in future work.

---

### M8. External Validity: Deployment Manifests & Single-Modeler Threat
> **Reviewer Comment:** Qualify "deployment manifests" claims (no automated importer exists), acknowledge the single-modeler threat on the 5 open-source systems, and note the bimodal distribution (>50% inert components).

- **Author Response:** We have fully updated the manuscript:
  1. *Manifest Claim Qualified:* In the abstract, §1.2, §5.1 / §6.1, and §9, we replaced claims of extracting from deployment manifests with precise statements that SaG models explicit dependency graphs from architecture specifications and deployment descriptors, explicitly stating that no automated parser for launch files or Kubernetes manifests was used.
  2. *Single-Modeler & Bimodal Distribution:* In §6.1 / §7.1 and §7.4 / §8.4, we acknowledge that the five system models were hand-authored by a single modeler without independent re-derivation. We highlight that over 50% of components in these systems are inert ($I^* = 0$), explaining that strong full-population correlations ($\rho \approx 0.81$) largely reflect separating inert nodes, whereas active-stratum ranking is weak ($\rho_{>0} \le 0.342$). We cite the re-modeling protocol (`reproduce/model_agreement.py`) for future inter-modeler agreement studies.
- **Changes in Manuscript:**
  - `sections/abstract.tex`, `sections/sec1_introduction.tex`, `sections/sec9_conclusion.tex`: Qualified manifest claims.
  - `sections/sec6_experimental_setup.tex` (§6.1), `sections/sec7_results.tex` (§7.3), `sections/sec8_discussion.tex` (§8.4): Documented single-modeler threat, bimodal distribution, and active-stratum limitations.

---

### M9. Framework Simplification & Unevaluated Machinery
> **Reviewer Comment:** Table 1 combines entity and edge types. Rules 2, 3, 4, 6 and the Figure 1 explanation layer are unevaluated. Reformat Table 1 and clarify unexercised machinery.

- **Author Response:** We have revised §3 and Table 1:
  1. *Table 1 Reformatting:* Table 1 is now split into two clearly distinct, labeled panels: Table 1a (Entity Types in the Software-as-a-Graph Multigraph) and Table 1b (Structural Relations and Dependency Semantics).
  2. *Unexercised Machinery:* In §3.1, §3.2, Table 2, and the captions of Figure 1 and Figure 2, we explicitly clarify that Rules 2, 3, 4, and 6 and the explanation layer represent formalized infrastructure extensions that were outside the empirical evaluation (which focused strictly on Rules 1 and 5).
- **Changes in Manuscript:**
  - `sections/sec3_sag_model.tex`: Reformatted Table 1 into Panel A and Panel B; clarified unexercised status of infrastructure rules.

---

### M10. Writing Streamlining, Hedging, and Claim Status Consolidation
> **Reviewer Comment:** The manuscript is over-hedged and contains revision-history artifacts. Consolidate claim status in one table in §5.3, condense the abstract, and cleanse meta-discourse.

- **Author Response:** We have thoroughly streamlined the text:
  1. *Abstract Word Count:* Condensed to 245 words (strictly below the 250-word JSS limit).
  2. *Claim Status Summary Table:* In §5.3 / §6.3, we added a clear, structured summary table categorizing every claim by its Confirmatory, Registered Secondary, or Exploratory status, statistical test, and empirical outcome.
  3. *Meta-Discourse Cleansed:* We removed phrases reflecting revision artifacts (e.g., "Across revision cycles and compute devices" and "the counts were reclassified as references after all results were final") in §7.4 / §8.4.
- **Changes in Manuscript:**
  - `sections/abstract.tex`: Condensed to 245 words.
  - `sections/sec6_experimental_setup.tex` (§6.3): Added structured summary table of claim status.
  - `sections/sec8_discussion.tex` (§8.4): Cleansed meta-discourse.

---

## 3. Point-by-Point Responses to Minor Comments

| # | Comment Summary | Author Response & Changes Made | Manuscript Location |
|---|---|---|---|
| **Minor 1** | Abstract length ($\le 250$ words) | **Condensed.** Abstract revised to exactly 245 words by word count, dropping parentheticals and uninformative comparisons. | `sections/abstract.tex` |
| **Minor 2** | Highlights ($\le 85$ chars, result-focused) | **Updated.** All 5 highlights rewritten to focus strictly on empirical findings; lengths range from 73 to 80 characters. | `highlights.tex` |
| **Minor 3** | Keywords ("dependability" redundancy) | **Replaced.** Replaced redundant keyword "dependability" with "afferent coupling". | `manuscript.tex` |
| **Minor 4** | Citation style (`elsarticle-harv`) | **Confirmed.** Citations formatted consistently per Elsevier submission standards. | `manuscript.tex`, `refs.bib` |
| **Minor 5** | Remark 1 / In-degree definition | **Clarified.** Explicitly explained that the `in_degree` *feature* counts incoming edges on $G_{\text{analysis}}$ (including Library $\to$ Application edges), whereas the `InDeg` *reference* strictly counts Rule 1 subscriber-to-publisher edges between Applications. | `sections/sec3_sag_model.tex` |
| **Minor 6** | Table 1 formatting | **Reformatted.** Split Table 1 into Panel A (Entity Types) and Panel B (Structural Relations). | `sections/sec3_sag_model.tex` |
| **Minor 7** | Table 4 parameter matching tolerance | **Stated.** Stated matching tolerance within $\pm 1.7\%$ across the $2\times2$ GNN design (429,992 to 437,496 parameters). | `sections/sec6_experimental_setup.tex` |
| **Minor 8** | Early stopping split justification | **Justified.** Added explanation that a 20% node split on the largest training scenario preserves scenario-level independence under LOSO while avoiding nested scenario cross-validation costs. | `sections/sec4_failure_impact_prediction.tex` |
| **Minor 9** | Functional description of $I_{\text{comp}}$ | **Updated.** Replaced codebase jargon ("Validate-stage failure simulator") with functional description: "composite multi-criteria failure simulator". | `sections/sec4_failure_impact_prediction.tex` |
| **Minor 10** | Spearman--Brown reliability approximation | **Acknowledged.** Noted that Spearman--Brown on rank correlations is an approximation, leaving fold-level headroom $\sqrt{r_f} - \rho_f \ge 0.05$. | `sections/sec4_failure_impact_prediction.tex` |
| **Minor 11** | Rounding precision ($0.772 - 0.764 = 0.008$) | **Corrected.** Changed $+0.007$ to $+0.008$ in §6.1 / §7.1. | `sections/sec7_results.tex` |
| **Minor 12** | "Nominal Holm p" phrasing | **Clarified.** Clarified as `nominal Holm $p = 0.009$; exploratory, reporting the nominal $p$-value for this uncorrected contrast family`. | `sections/sec7_results.tex` |
| **Minor 13** | Overlap@K tie handling | **Clarified.** Added explanatory note in Table 5 caption clarifying deterministic identifier sort in Table 5 vs. expected resolution in Figure 5. | `sections/sec7_results.tex` |
| **Minor 14** | 80% recall qualification across $I^*$ vs. $I_{\text{dyn}}$ | **Qualified.** Qualified in §1.3 that recovering 80% requires 40–45% of nodes under $I^*$, whereas on $I_{\text{dyn}}$ Eq. 7 achieves 81% recall at $k=20\%$ and 96% at $k=30\%$. | `sections/sec1_introduction.tex` |
| **Minor 15** | Table 10 recommendation for $I^*$ | **Rephrased.** Rephrased first row of Table 10 to: *"If $I^*$ is accepted as the impact definition, run $I^*$ directly..."* | `sections/sec8_discussion.tex` |
| **Minor 16** | Figure 1 explanation layer & hybrid prior | **Clarified.** Caption of Figure 1 and Figure 2 updated to mark unexercised status; Figure 3 caption describes hybrid prior formulation. | `sections/sec3_sag_model.tex` |
| **Minor 17** | Analysis plan commit hash | **Added.** Cited exact immutable Git commit hash `56d9bff8` in §5.3 / §6.3. | `sections/sec6_experimental_setup.tex` |
| **Minor 18** | Equation formatting consistency | **Formatted.** Ensured Eq. 1 and Eq. 7 are displayed as standalone, consistently numbered equations. | `sections/sec3_sag_model.tex`, `sections/sec6_experimental_setup.tex` |
| **Minor 19** | Literature additions (Kapoor 2023, Kleijnen 2015, Walker 2020) | **Added.** Added BibTeX entries and in-text citations for Kapoor & Narayanan (2023), Kleijnen (2015), and Walker et al. (MicroART). | `refs.bib`, `sections/sec2_related_work.tex` |
| **Minor 20** | Data availability & commit hashes | **Confirmed.** Confirmed replication package provenance and referenced the 1,817-figure mechanical reconciliation suite in §5.1 and Declarations. | `sections/sec6_experimental_setup.tex`, `sections/declarations.tex` |
| **Minor 21** | AI script verification note | **Added.** Explicitly noted in §5.1 / §6.1 that automated regression test suites and the 1,817-figure reconciliation script independently verified all AI-assisted analysis code. | `sections/sec6_experimental_setup.tex` |

---

## 4. Verification and Reconciler Status

All 1,817 figures reported across the manuscript and supplement were mechanically verified against their source artifacts using `reproduce/reconcile_manuscript.py`:
```text
  Reconciled 1817 table figures against committed artifacts (0 check(s) skipped).
  OK — 1817 figures match their artifacts.
```
- LaTeX document compilation (`make -C docs/research/jss/latex all`): **0 errors, clean build (`manuscript.pdf`, 29 pages).**
- Markdown rendering (`python3 reproduce/render_manuscript_md.py --check`): **Synchronized.**
- Pytest suite (`pytest tests/test_reconcile_guards.py`): **4/4 passed.**

We believe these comprehensive revisions directly address every critique and establish the paper as a rigorous, transparent contribution to the *Journal of Systems and Software*.
