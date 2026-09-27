# Pre-Submission Review: Refocus on Learning-Based Methods and the Dependency Graph

**Date:** 2026-09-27  
**Target Venue:** Journal of Systems and Software (JSS)  
**Evaluation Scope:** Complete revision evaluating the shift to learning-based methods (GNNs, hybrids, GBM surrogates) on derived dependency graphs, removal of centrality benchmarking from the title and core framing, consolidation of training-free baselines to the supplement, and full artifact reconciliation.

---

## 1. Executive Summary & Framing Transition

The manuscript has been systematically reframed from a comparative "centrality benchmarking" study into a focused investigation of **learning-based methods on derived dependency graphs** for predicting pre-deployment cascading-failure impact in event-driven systems.

### Core Strategic Changes
1. **Title & Headline Framing:**
   - *Previous:* "Software-as-a-Graph: Benchmarking Centrality and Graph Learning for Cascading-Failure Impact Prediction in Publish-Subscribe Systems"
   - *Current:* "Software-as-a-Graph: Learning Cascade-Impact Rankings on Derived Dependency Graphs for Publish--Subscribe Systems"
   - Benchmarking and centrality terms have been eliminated from the title, abstract, highlights, and conclusion.
2. **Baseline Consolidation:**
   - `Topo-QoS` is retained as the sole training-free baseline in the main text because it forms one side of the registered primary contrast (PREREGISTRATION.md) and provides the prior for both hybrid rankers.
   - Non-registered training-free heuristics (`Degree-raw`, `RevPR-raw`, `PR-raw`, and unweighted `Topo`) have been cleanly relocated to Supplementary Section S.36 (`\label{supp:baselines}`).
3. **Primary Learning Narrative:**
   - Section 1, Section 7 (RQ1), Section 8, and Section 9 now lead with the learned engines:
     - GNNs reading the derived dependency graph (`GAT-P-QoS` reaching $\rho = 0.748$) approach the reference level ($0.764$), whereas raw-multigraph message passing delivers no forward signal to ranked components.
     - The registered primary contrast was null ($\Delta\rho = +0.069$, $p = 0.27$); hybrids beat the baseline ($+0.103$, $+0.130$) but not their own base learners.
     - Where simulation is expensive ($I_{\text{dyn}}$, 12.7 CPU-hours), a learned gradient-boosted surrogate trained on simulation trajectories beats every closed-form approximation ($0.799$ vs $0.706$), providing a $19\times$ speedup over physical simulation.
     - Zero-shot transfer preserves macro rank ordering ($\rho \approx 0.81$), but within-cascade discrimination collapses on the active stratum ($\rho_{>0} \le 0.34$).
     - Inference feature extraction incurs $4.5$--$72\times$ the cost of reachability simulation, bounding where learning pays off.

---

## 2. Section-by-Section Review

### Front Matter
- **Title (`manuscript.tex`, `title_page.tex`):** Accurately reflects the dependency graph and learning focus.
- **Highlights (`highlights.tex`):** Exactly 5 bullet points, all $\le 85$ characters (max 85 chars, min 73 chars), strictly compliant with JSS Guide for Authors. Zero forbidden terms.
- **Abstract (`abstract.tex`):** Exactly 248 words (within the 250-word ceiling). Zero occurrences of `centrality` or `[Bb]enchmark`. All reported statistics match artifact values byte-identically.

### Section 1: Introduction
- Motivation clearly highlights the Architecture--Code Gap and frames the central question around what learning achieves on derived dependency graphs.
- SaG overview (Figure 1) establishes the three-stage pipeline (multigraph modeling $\to$ dependency graph derivation $\to$ learned ranking).
- Explicit distinction between ground-truth simulation and field outages is preserved.
- Four research questions (RQ1: Ranking accuracy; RQ2: What learning needs; RQ3: Transfer; RQ4: Cost) align with the revised learning narrative.
- Six contributions reordered to prioritize the learning evaluation and dependency graph representation.

### Section 2: Related Work
- Centrality discussion condensed into baseline context for distributed systems.
- Section balances architecture degradation analysis, graph neural networks in software engineering, and proxy modeling.

### Section 3: The Software-as-a-Graph Model
- Formal multigraph schema and derivation rules (Rules 1--6) clearly stated.
- Forward pointer added to RQ2 explaining why message flow on $G_{\text{structural}}$ fails to reach Applications and how the derived dependency graph solves this.
- Cross-references to Supplementary Section S.36 for raw-multigraph topological metrics.

### Section 4: Failure-Impact Prediction & Simulation Oracles
- Figure 3 relabeled from "closed-form centrality" to "training-free baseline".
- The three simulation oracles ($I^*$ reachability, $I_{\text{dyn}}$ queue-flow discrete event, $I_{\text{comp}}$ multi-criteria composite) and their respective circularity boundaries are rigorously operationalized.

### Section 6: Experimental Setup
- Table 2 (Predictor Taxonomy): Relabeled "Training-free baseline" with `Topo-QoS` as the sole entry; dependency-graph learners listed ahead of raw-multigraph learners; clean forward pointer to supplement for other baselines.
- Eq. (10) retains `Topo-QoS` with explicit disclosure of the articulation point implementation defect.

### Section 7: Empirical Results
- **§7.1 (RQ1: Ranking Accuracy of Learned Engines):** Reordered to lead with dependency-graph learners, followed by raw-multigraph learners and hybrids.
- **Table 3 (`tab:hybrid`):** Cleaned of `Degree-raw`, `RevPR-raw`, and `Topo`. `Topo-QoS` serves as the sole baseline row; learners on the dependency graph appear first.
- **Table 4 (`tab:independent_oracles`):** `Degree-raw` moved to supplement; block renamed "Training-free baseline"; `Topo-QoS` underlined for $I_{\text{comp}}$ ($0.702$); narrative frames learner collapse without prior and highlights the GBM queue-flow surrogate ($0.799$).
- **Table 5 (`tab:system_models_transfer`):** Unweighted `Topo` moved to supplement; block renamed "Training-free baseline"; narrative updated.
- **§7.2--§7.4 (RQ2--RQ4):** Factorial $2\times2$ matched controls, ablation of degree features, zero-shot transfer, and like-for-like cost profile remain rock solid.

### Section 8: Discussion & Threats to Validity
- Guidance table (Table 9) and regimes table (Table 10) relabeled to refer to "baseline" rather than "centrality" or "benchmarking".
- Discussion emphasizes practical recommendations: run cheap simulation or direct dependents when simulation is lightweight; deploy learned surrogates when simulation is heavy.
- Threats to validity (§8.3) retain full disclosure of construct circularity, seed sensitivity, and selection rule status.

### Section 9: Conclusion
- Completely rewritten into 5 structured empirical findings following the learning narrative.
- Zero matches for `centrality` or `[Bb]enchmark`.
- Ends with an open question on real-world production outage evaluation.

### Supplementary Material
- Added Section S.36 (`\label{supp:baselines}`) containing:
  - Unweighted `Topo` formal definition and defect disclosure.
  - Table S.44 (`tab:supp-moved-baselines`) reporting `Degree-raw`, `RevPR-raw`, and `Topo` across 12 LOSO folds.
  - Paragraph on Amendment 7 controls explaining the origin of closed-form gain.
  - F7 hybrid-vs-betweenness statistical test sentence.
  - Table S.45 (`tab:supp-moved-oracles`) reporting `Degree-raw` across the three simulation oracles.
  - Table S.46 (`tab:supp-moved-systems`) reporting `Topo` on the five system models.
- Amendment log (Table S.14) updated with the 2026-09-27 Reporting consolidation row.

---

## 3. Verification & Mechanical Reconciliation

1. **Reconciliation Tool (`reproduce/reconcile_manuscript.py`):**
   - Implemented automated focus verification via `check_learning_focus`:
     - Checks absence of `centralit|[Bb]enchmark` in title, abstract, highlights, and §9.
     - Checks absence of moved baseline rows (`Degree-raw`, `RevPR-raw`, `PR-raw`, `Topo`) in body tables.
     - Checks that `Topo-QoS` is the sole baseline in `tab:predictor_taxonomy`.
   - Mutation tested by temporarily injecting "centrality" into the abstract, confirming failure, and reverting.
   - **Result:** `Reconciled 1554 table figures against committed artifacts (0 check(s) skipped). OK — 1554 figures match their artifacts.`
2. **LaTeX Compilation:**
   - Both `manuscript.tex` (29 pages) and `supplementary.tex` (41 pages) build cleanly with 0 errors.
   - `grep -i "undefined" docs/research/jss/latex/*.log` reports **0 undefined references**.
   - Cross-document links via `xr-hyper` resolve bidirectionally between `manuscript.tex` and `supplementary.tex`.
3. **Markdown Sync & Link Integrity:**
   - `python reproduce/render_manuscript_md.py --check` confirms `manuscript.md` and `sections/*.md` are byte-up-to-date.
   - `python scripts/check_doc_links.py` confirms all 1080 relative documentation links and anchors resolve.
4. **Visual Figures:**
   - Figures 1, 3, 5, and the Graphical Abstract regenerated via `make -f reproduce/Makefile`.
   - Figure 5 updated to remove unweighted Topo from panel A, update panel titles, and relabel legend to baseline.
   - Graphical Abstract updated to relabel baseline and add the queue-flow surrogate result.
5. **Length Constraints:**
   - Manuscript: 29 pages (elsarticle single column, 101 bib entries), well within the 36-page limit.
   - `LENGTH_JUSTIFICATION.md` updated to document exact counts.

---

## 4. Final Assessment

The manuscript is coherent, methodologically rigorous, and completely reconciled against the underlying empirical artifacts. The refocusing on learning-based methods and dependency graphs is complete, with no lingering traces of the previous benchmarking framing in key sections, while strictly preserving every preregistered contrast and empirical number.

**Recommendation:** Proceed with submission to JSS.
