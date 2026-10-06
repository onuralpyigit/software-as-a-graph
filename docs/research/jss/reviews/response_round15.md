# Response to Peer Reviewers (Round 15 Revision)

**Manuscript Title:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?*  
**Journal:** *Journal of Systems and Software* (Elsevier, Q1)  
**Special Issue:** *AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems*  
**Recommendation:** Minor Revision  

---

We sincerely thank the Reviewer for their thorough, rigorous, and constructive re-evaluation of our revised manuscript. We are gratified by the positive assessment of our methodological rigor, the utility of the order-$k$ reference criterion, the transparency of our negative findings, and the quality of the replication package. 

Below, we detail our point-by-point responses and the corresponding amendments incorporated into the LaTeX source (`docs/research/jss/latex/`), the compiled PDF (`manuscript.pdf`), and the synchronized Markdown representations (`manuscript.md` and `sections/*.md`).

---

## 1. Response to Major Comments

### Major Comment 1: External Validity Boundaries and the "First-Order" Scope
> *The authors have commendably qualified the title, abstract, and discussion (§7.2) to specify that the findings apply to first-order simulation regimes where direct subscriber starvation and broker queue fills dominate. However, the manuscript must maintain vigilant precision in §7.1 and §7.5 so that readers do not overgeneralize this negative result to mean "GNNs are universally unsuited for distributed systems failure prediction." The authors should ensure that the final discussion explicitly re-emphasizes that in non-first-order regimes (e.g., dynamic multi-hop backpressure loops, cascading retry storms, or Byzantine faults), the higher-order multi-hop expressive capacity of GNNs remains theoretically unrefuted and an open empirical question.*

**Response:**  
We completely agree. We have expanded §7.1 and §7.5 (Future Work item 3) to explicitly prevent overgeneralization:
1. In **§7.1 (Section 8.1 / lines 8–11 of `sec8_discussion.tex`)**, we added:
   > *"Importantly, this negative result should not be misconstrued as an assertion that graph neural networks lack predictive utility across all distributed systems dependability tasks. In regimes characterized by non-first-order dynamics—such as cyclic backpressure loops, cascading retry storms, buffer bloat, or Byzantine failures—multi-hop message-passing GNNs possess expressive capacity that simple local degree counts cannot reproduce; whether GNNs outperform analytical surrogates in such non-first-order regimes remains an open empirical question."*
2. In **§7.5 (Section 8.4 / Future Work item 3)**, we added:
   > *"In such regimes, higher-order multi-hop GNN message passing may prove essential to capture complex cyclic or non-linear failure dynamics that defeat low-order topological truncations."*

---

### Major Comment 2: Stylized Systems and Active-Stratum Discrimination (RQ3)
> *The zero-shot evaluation on the five open-source systems (§6.3) is reported transparently: while global rank correlation is high ($\rho \approx 0.81$) due to the separation of inert components (>50% of the nodes) from active components, the ranking discrimination within the active stratum remains weak ($\rho_{>0} \le 0.342$). The authors must verify that in the abstract and conclusion, this limitation is not understated... the discussion in §7.3 should explicitly advise practitioners that neither learned models nor analytical baselines should be used as fine-grained prioritizers within active components without calibrated telemetry or operational failure histories.*

**Response:**  
We have incorporated explicit practitioner warnings into both the empirical presentation in §6.3 and the practical guidance in §7.3:
1. In **§6.3 (`sec7_results.tex`, lines 201–202)**:
   > *"Consequently, neither learned rankers nor analytical counts should be deployed as fine-grained prioritizers within the active component stratum without operational failure logs or runtime telemetry."*
2. In **§7.3 (`sec8_discussion.tex`, lines 26–27)**:
   > *"Practitioners should treat static structural rankers strictly as coarse-grained filters for isolating inert components, and avoid deploying them as fine-grained prioritizers within active components in unfamiliar systems unless supplemented by operational traces or failure histories."*

---

### Major Comment 3: Queue-Flow Simulator Metamodeling and Feature Parity
> *In §6.1, the comparison between the rate-weighted analytical approximation (Eq. 7, $\rho = 0.830$) and the learned GBDT/GAT models ($\rho = 0.799$) is now well contextualized with Kleijnen’s classic simulation metamodeling principles. The explanation that the learned models operated without explicit message rate features (relying solely on graph structure) is an essential disclosure that prevents misinterpreting the GNN's lower performance as algorithmic failure rather than an information-asymmetry effect. Ensure that Table 5 or its accompanying text explicitly highlights this feature asymmetry so that the comparison remains completely transparent.*

**Response:**  
We have made this feature distinction prominent in both Table 5 and the text:
1. In **Table 5 (`\caption{...}` of `\label{tab:independent_oracles}`)**:
   > *"Feature inputs differ: Eq.~(7) and GBM read declared rates, whereas GNNs receive topology and QoS properties without rate features."*
2. In **§6.1 (`sec7_results.tex`, lines 98–99)**:
   > *"Because this GNN received topological features and QoS profiles without explicit rate inputs, this deficit reflects feature asymmetry as well as modeling capacity; it did not fit its target and is not a credible learned approximation of $I_{\text{dyn}}$, while a GNN given declared rates was not run."*

---

## 2. Response to Minor Comments

### Minor Comment 1: Elsevier JSS Guide for Authors Compliance
> * **Abstract Length:** Ensure abstract satisfies $\le 250$ words.
> * **Highlights:** Ensure 5 bullets $\le 85$ characters each.
> * **Software and Data Citations:** References [102] and [103] formal bibliography integration.

**Response:**  
1. **Abstract Length:** The abstract has been tightened to **248 words** (verified by the exact counting script enforcing JSS word limits).
2. **Highlights:** All 5 highlights remain strictly between **76 and 80 characters** (maximum allowed: 85 characters).
3. **Software & Data Citations:** References [102] (Zenodo dataset) and [103] (GitHub software repository with commit hash `56d9bff8ae9583ee9df1d270f0650a3a7c3239e8`) are fully cited in the text and bibliography.

### Minor Comment 2: Mathematical Notation & Typographical Consistency
> * Check notation $w_V(a)$ and $w(t)$ consistency.
> * In §4.4 and §7.2, check that Simplifications S1–S5 match.
> * In §6.1 and Table 4, ensure all Holm-corrected $p$-values are consistently designated as $p_{\text{Holm}}$.

**Response:**  
1. **Node and Topic Weights:** Consistently designated as $w_V(a)$ for Application entity weight and $w(t)$ for Topic QoS weight in §3.1 and §3.5.
2. **Simplifications S1–S5:** Explicitly cross-referenced in §7.4 (`sec8_discussion.tex`, line 55) as defined in §4.4.
3. **$p$-Values:** Table 4, Table 5, Table 6, and accompanying prose clearly designate Holm–Bonferroni corrected values as $p_{\text{Holm}}$.

### Minor Comment 3: Figures and Diagram Clarity
> * In Figure 1, clarify implemented components from conceptual proposals.
> * In Figure 3, verify font readability.

**Response:**  
Figure 1 caption explicitly notes that the *Explanation Layer* is a conceptual proposal evaluated in Supplementary §S26, while the central ingestion, projection, and ranking pipelines are fully implemented. Figure 3 retains clean vector line art and legible typography across double-column layout.

---

## 3. Verification & Reconciliation Summary

1. **LaTeX Compilation:** Built cleanly via `make -C docs/research/jss/latex all` with **0 errors**, yielding `manuscript.pdf` (29 pages).
2. **Mechanical Reconciler:** `python3 reproduce/reconcile_manuscript.py` passed with **1,817 / 1,817 matched figures** across all tables and prose statements.
3. **Markdown Rendering:** `python3 reproduce/render_manuscript_md.py` regenerated `docs/research/jss/manuscript.md` and all 12 section files in `docs/research/jss/sections/*.md`, passing `--check` synchronization.
4. **Regression Tests:** `pytest tests/test_reconcile_guards.py` (4/4 passed) and full unit test suites run cleanly.
