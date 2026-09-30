# Referee Report — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems", round 10

**Manuscript:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?*
**Material reviewed:** `manuscript.md` (32 pp. PDF, 11 tables, 5 figures, 75 references), `latex/highlights.tex`, JSS Guide for Authors. Supplementary material and the replication repository were consulted only to check specific claims; the assessment is of the manuscript as a stand-alone article.
**Date:** 2026-09-30 · **Recommendation:** Major Revision

---

## Summary

The paper introduces Software-as-a-Graph (SaG). SaG derives a typed `DEPENDS_ON` dependency graph from publish–subscribe deployment manifests and uses it to ask when graph learning improves pre-deployment ranking of Applications by cascading-failure impact. Analytical, hybrid and learned rankers (GAT, HGT, GINE, gradient-boosted trees) are compared under leave-one-scenario-out cross-validation on twelve synthetic architectures from one generator, against three simulators (reachability $I^*$, discrete-event queue flow $I_{\text{dyn}}$, weighted multi-criteria $I_{\text{comp}}$), and zero-shot on five hand-authored models of open-source systems. A post hoc "reference criterion" separates rankings that restate a simulator's rule from rankings that predict it. The main finding is negative: no learned model exceeds analytical rankings built on the same dependencies. Afferent coupling (0.764) is indistinguishable from the best GNN (0.748), and a rate-weighted first-order formula (0.830) exceeds a learned approximation of the queue-flow simulator (0.799). Learning helps only relative to a weak training-free baseline.

## Overall Impression & Assessment

**Originality.** Moderate. The question of whether learning adds value over dependency analysis is timely for this special issue. A careful negative result is within JSS's stated scope ("studies with negative results"). The reference criterion is the most original idea: asking whether a learned model does more than restate the simulator that labels it applies well beyond this domain. The dependency rules extend the authors' RASSE 2025 paper [32] incrementally. The learned-ranker architectures are off the shelf.

**Significance.** Limited in its current form. The headline result, that counting direct dependents recovers most of a reachability simulator, is close to true by construction (M1). None of the three oracles encodes an impact mechanism that first-order dependency structure fails to capture. The study therefore cannot identify any regime where learning adds value, and the paper says so ("a condition this corpus did not exercise"). The title asks *when* learning helps. The design can only answer "not here".

**Methodological soundness.** The paper is unusually transparent. It uses a version-controlled plan, discloses 15 amendments, applies Holm and omnibus corrections, runs Nadeau–Bengio tests, reports equivalence tests, uses matched-capacity factorial controls and ablations, and regenerates datasets byte-identically. That deserves credit. Several problems undercut the conclusions:
- there is no empirical ground truth;
- the registered baseline is weak and has a known defect;
- the reference criterion was introduced post hoc;
- a directionality confound runs through the "representation" claim;
- learned-model drift exceeds several of the reported effects;
- the headline recommendations rest on exploratory analyses added last.

**Suitability for a Q1 venue / this special issue.** The topic fits the special issue's reliability and sustainability themes. The sustainability evidence consists only of TDP × time estimates. The manuscript is also much longer and more repetitive than its findings warrant. With substantial revision (reframing, one real-system anchor, a fresh confirmation set, the directionality control, and condensation), this could become a solid JSS contribution.

---

## Major Comments

### M1. The design cannot answer the title question; the main finding is largely true by construction

$I^*$ is a breadth-first reachability cascade on the same pub-sub topology the rankers read. `InDeg` is exactly its first wave (Remark 1), and Eq. 6 is its first-order expansion. $I_{\text{dyn}}$ shares that topology and first-order effect: it agrees with $I^*$ at $\rho = 0.711$, and a rate-weighted truncation recovers it at 0.830. $I_{\text{comp}}$ is a declared linear combination whose fragmentation terms track degree and betweenness. So all three impact notions are well approximated by low-order functions of the dependency graph. Finding that "representation matters more than model complexity" is then an expected property of the benchmark rather than an empirical discovery about software systems.

The paper concedes that learning "can be expected to add value only where no analytical approximation aligned with the simulator is available, a condition this corpus did not exercise" (§8). That is the regime the title asks about.

**Requested:**
1. Add at least one oracle whose impact depends on mechanisms that first-order structure does not capture: heterogeneous service capacity, backpressure propagating upstream to publishers, retry amplification, or load-dependent thresholds. This would show that the benchmark *can* discriminate learned from analytical rankers. A benchmark in which no method could ever beat the closed-form truncation says little about learning.
2. If (1) is out of scope, retitle and reframe the paper explicitly as a negative-result and evaluation-methodology study. The current title promises a characterization ("when") that the evidence does not provide.

### M2. No empirical ground truth; every conclusion is conditional on simulator fidelity

The threats section is candid (§7.5). It even describes plausible real-world failure modes in which "physical failure impact decouples from structural graph degree". But the manuscript contains no measurement that bounds how far the simulators are from reality. For a special issue on reliability, one small real-system anchor would change the paper's standing. Train-Ticket [33] is cited, publicly deployable, and ships with catalogued faults. A ROS 2 or Kafka reference application would also serve.

**Requested:** Inject crash faults into each Application of at least one real deployment (roughly 20–40 services). Measure observed delivery loss and report $\rho(I^*, \text{observed})$ and $\rho(I_{\text{dyn}}, \text{observed})$. Even one system turns "conditional on simulation fidelity" into a quantified statement. If this is impossible, state in the abstract and title that the object of study is *simulator approximation*, not failure prediction.

### M3. The reference criterion is post hoc and asymmetric, and it removes the strongest comparators from the contrasts

**Timing.** The counts were reclassified as references "after all results were finalized" (§7.5).

**The criterion excludes the most relevant comparator.** `InDeg` is publish–subscribe afferent coupling [47–49], a practitioner metric that predates and is independent of any simulator. Excluding it from all contrasts leaves `Topo-QoS` as the only training-free "predictor". `Topo-QoS` is the weakest analytical ranker in the paper (M4).

**Asymmetry.** A learned model trained directly on $I^*$ labels is at least as tied to $I^*$ as a two-decade-old coupling metric. Yet the learned model counts as a "predictor" and the metric does not.

**Oracle relativity.** The criterion is oracle-relative and applied inconsistently. `Topo-QoS` is not a truncation of $I_{\text{comp}}$, yet it is treated "with the same caution as the references" there. That case needs a separate label; it does not follow from the stated definition.

**Requested:**
- Report afferent coupling as a legitimate practitioner baseline, with contrasts. The honest and stronger message is "no learned model beats afferent coupling". Keep the reference notion as an interpretive lens that quantifies circularity, not as a filter that removes comparators.
- Operationalize circularity quantitatively (for example, the share of oracle rank variance explained by its first wave, per fold) instead of by binary classification.
- State clearly in §4.4 that the criterion was introduced after the results.

### M4. The registered baseline is weak and defective; the hybrid "wins" are not evidence that learning adds value

**The defect.** `Topo-QoS` has an index-mismatch defect that zeroes its articulation term. Eq. 5 therefore does not describe what was run. Publishing a registered comparator with a known bug is unusual and damages confidence, even though the corrected score is slightly lower (0.533). Worse, both hybrids were trained on the *defective* prior.

**The baseline is dominated.** Unweighted betweenness (0.591), constant topic weights (0.595) and a raw published-topic count (0.731) all beat `Topo-QoS`. Weighting folds by $|V_{\text{app}}|$ raises `Topo-QoS` to 0.596, level with the raw-graph learners (0.595–0.606).

**The hybrid gains are hard to attribute to learning:**
- Neither hybrid differs from its own base learner (+0.035, $p = 0.73$; +0.048, $p = 0.30$).
- Those base learners do not differ from `Topo-QoS` either (+0.069, $p = 0.266$; +0.082, $p = 0.233$).
- The hybrid–baseline contrast is significant largely because the hybrid nests its comparator, which lowers paired variance.
- The contrasts were registered after the primary null was known.
- Both hybrids transfer worse zero-shot than their base learners.

Presenting this as "Learning adds measurable value" (§1.2, §7.2, §8, highlight 5) overstates it.

**Requested:**
- Fix the defect.
- Retrain the hybrids with the corrected prior, and with an `InDeg` prior (the archived `GAT-P+InDeg`).
- Report hybrid vs. base learner as the primary evidence for what the correction contributes.
- Temper the abstract and highlights accordingly.
- Reconcile explicitly with [32], which validated a closed-form betweenness–articulation score of this family against a reachability-loss simulation. Here that family is the weakest ranker; readers of both papers need to know why.

### M5. The "representation" effect is confounded with edge direction

Contribution 1 and §7.1 credit the +0.113 gain of `GAT-P-QoS` over `GAT-QoS` to explicit dependency derivation. But the raw-graph GAT is effectively an MLP: every structural edge points *away* from Applications, and deleting all edges leaves its outputs unchanged (§7.2). In PyG it is standard practice to add reverse edges (`ToUndirected` / `AddReverseEdges`) before message passing on heterogeneous or directed graphs. A 3-layer GAT on $G_{\text{structural}}$ with reverse edges can reach subscribers via Application ← Topic ← Subscriber. The present comparison contrasts "derived graph" with "graph on which no message reaches the scored node", not with a reasonable raw-graph learner. The `HGT-QoS-U` evidence concerns HGT only, a model the paper itself finds unstable.

**Requested:**
- Add a bidirectional raw-graph GAT (and GINE) at matched capacity, with and without degree features.
- Attribute the remaining gap to the derivation only after that control.
- Scale contribution 1 to the evidence on the analytical side. Remark 1 shows `InDeg` is a typed two-hop query computable without the projection. The derivation's only measured analytical value is Rule 5's +0.058 on `Reach`. The derived graph organizes signal that is already recoverable from the raw graph; it does not create it.

### M6. "Matches" claims contradict the paper's own equivalence results

§7.5 states: "Equivalence … is not established for any pair at ±0.05". The TOST for `GAT-P-QoS` vs. `InDeg` fails ($p = 0.17$), and the 90% CI of the per-fold difference is ±0.076. Nevertheless, the following all assert that the count "matches" the learned model:
- highlight 3 ("Counting direct dependents matches the best learned ranker");
- the abstract ("The learned model is matched by…");
- §1.2, §7.1 and §8 ("Counting direct dependents … matched the best learned model").

The phrase "statistically similar but not equivalent" (§6.1, Figure 4 caption) misreads the test: failing TOST does not establish non-equivalence.

**Requested:**
- Replace "matches" with "was not statistically distinguishable from (equivalence not established)" throughout.
- Report the minimum detectable effect at 12 folds, so readers can see the study is underpowered for equivalence at ±0.05.
- Treat "matching or exceeding" (Eq. 7 vs. GBM) the same way. There the data show *exceeding* (+0.031, 10/12 folds), but the contrast is exploratory and was added last.

### M7. Headline recommendations rest on post hoc analyses; a fresh confirmation set is cheap and should be run

Only two contrasts are confirmatory, and both are null. The paper's recommendations do not rest on them:
- Table 11's queue-flow recommendation (Eq. 7) and the abstract's 0.830 figure are *exploratory*, "added after all other results existed".
- The dependency-graph learners, hybrids, matched 2×2 and sensitivity arms were registered after the primary null was known.
- The registered nested selection was not applied to the published results, and the post hoc re-run of that selection does not reproduce the published scores even on folds where it chose the same hyperparameters.

The abstract does not tell the reader any of this.

The corpus generator is deterministic and configurable, so generating a new set of scenarios after freezing the analysis costs little. A new generator seed per domain, or two or three new domains, would serve. That set can test the key post hoc claims confirmatorily:
- Eq. 7 > GBM→dyn on $I_{\text{dyn}}$;
- `GAT-P-QoS` > `GAT-QoS`;
- Hybrid > `Topo-QoS` with the corrected prior;
- `InDeg` ≥ `GAT-P-QoS`.

**Requested:**
- Run such a confirmation set.
- State in the abstract that the primary registered contrast was null and that the headline findings are secondary or exploratory.
- Consider a mixed-effects analysis (fold as a random effect; seed nested in fold) instead of the patchwork of per-family Wilcoxon, Nadeau–Bengio and sign-flip tests.
- Drop bootstrap CIs over $n = 5$ systems (M9).

### M8. Learned-model results are less stable than the effects reported

§7.5 reports several instabilities:
- learned cells drifted by up to 0.172 across revision cycles and devices;
- single HGT seeds in the same fold differ by more than 1.0 in $\rho$;
- `HGT-P-QoS` has a within-fold seed SD of 0.254;
- the nested harness fails to reproduce published scores by up to 0.10 on folds where it chose the published hyperparameters.

Several reported effects are smaller than this drift: the primary contrast (+0.069), the QoS main effect (+0.073) and the typing effects (−0.014 / −0.036). Reporting every cell "from one named CPU sweep" pins what is reported, but it does not control the scientific uncertainty. It selects one realization of a noisy process.

**Requested:**
- Make training deterministic, or report results across at least three independent sweeps and include between-sweep variance in the intervals.
- State that learned-model differences smaller than the observed drift are not interpretable.
- The typing conclusion rests on one fragile HGT configuration, and its width is reported inconsistently: $D = 64$ in §4.1, "width 100" in §6.2 and §7.2, and identical parameter counts for all three HGT variants in Table 4. Either tune HGT-P-QoS properly or remove the typing claims for the dependency graph. Lv et al. (KDD 2021) found that heterogeneous GNNs often fail to beat a well-tuned GAT; this is directly relevant context.

### M9. External validity: one generator family and five single-author models

**Synthetic corpus.** All twelve LOSO folds come from one generator. The zero-shot models differ from them on exactly the properties that drive the metric: 51% vs. 31% inert Applications, and fan-in Gini 0.65 vs. 0.50.

**Zero-shot models.** All five were hand-authored by the first author with partly assumed brokers, QoS and code metrics. Two are RPC systems recast as pub-sub meshes.

**The RQ3 claim is weak.** It says learned models transfer better than the baseline (≈0.81 vs. 0.53). But:
- it is measured against the defective baseline (M4);
- it rests on $n = 5$ with no test;
- $\rho$ is inflated by the inert half;
- active-stratum agreement is weak ($\rho_{>0} \le 0.342$, CIs spanning zero for `InDeg`).

Bootstrapping over five units (126 distinct resamples) yields intervals that should not be printed as 95% CIs.

**Requested:**
- Extract at least some models mechanically: ROS 2 launch files for Autoware, docker-compose/Kubernetes manifests, or AsyncAPI documents. Or have a second modeler rebuild them and report inter-modeler agreement.
- Show per-system values in a small table instead of bootstrapped CIs.
- Report generator-vs-real structural statistics (degree distributions, inert fraction, fan-in concentration) to show what the synthetic corpus does and does not cover.

### M10. The cost and sustainability claims (RQ4) are weaker than their prominence in the paper

**Energy.** All energy figures are TDP × wall-time estimates. The i7-1370P exposes RAPL through Linux `powercap`, so measured package energy costs little effort. For a special issue with sustainability in its title, measured figures are expected, not a stated limitation.

**Cost of $I_{\text{dyn}}$.** The "12.7 CPU-hours" framing overstates the cost. $I_{\text{dyn}}$ is embarrassingly parallel over components and seeds (about 7 s per run). Per architecture it is minutes on a CI runner, and the corpus total (355.6 Wh estimated) is small in absolute terms.

**Feature cost.** The finding that features are the most expensive stage (4.5–72.5× one pass) is an artefact of the chosen feature set. The articulation/CDI phase accounts for 88–91% of it. A learned ranker on lean features (degree, Eq. 6/7 terms) would be cheap. The conclusion that "neural inference itself is negligible" supports this reading.

**Internal inconsistency.** §6.4 gives 11.1 s total (0.086 Wh) for the corpus-wide dependency count. Table 10 gives 0.4–15.6 ms per architecture, which sums to well under a second. Tier 1 of §7.4 then pairs "about 11 ms" with "0.086 Wh". 0.086 Wh is 28 W × 11.1 s, so these figures differ by three orders of magnitude.

**Requested:**
- Measure energy with RAPL.
- Report cost for a lean-feature learned ranker.
- Fix the count-cost inconsistency.
- Present $I_{\text{dyn}}$ cost per architecture and parallelized, alongside the corpus total.

### M11. Length, repetition and self-containedness

The manuscript is 32 pages with a 44-page supplement, and the same few numbers recur throughout:
- 0.748 appears 19 times, 0.764 eleven times, and "12.7 CPU-hours" twelve times;
- the findings are restated in the abstract, "Findings in brief" (§1.2), contributions (§1.4), each RQ's summary box, §7.1–7.4, Table 11 and §8.

Internal plan identifiers appear about 20 times and are never defined in the article: F1a, F2b, F3, F5, F6b, Family A/B/C, arm N, rule R5, "Amendment".

The text defers to "the replication repository" 18 times, including for material a reader needs to evaluate the method: notation, HGT layer equations, the feature schema, generator parameters and per-scenario composition. Repository URLs change; JSS articles should be self-contained, with extended material in the supplement.

Several unevaluated components take main-text space:
- the explanation layer and ISO/IEC 25010 mapping (§2.4, Figure 1);
- Rules 2–4 and 6;
- the AHP formalization of QoS weights, which the paper then shows to be inert;
- the CQP-based vertex weight $w_V$, which no reported result uses.

**Requested:** Remove "Findings in brief". Collapse the RQ summary boxes into one results overview. Define or remove every plan identifier. Move the unevaluated machinery to the supplement. Bring the method details the reader needs into the paper or its supplement. A 25–30% reduction is achievable without losing evidence.

### M12. Literature gaps

The related work (§2) is short for a Q1 journal and omits several directly relevant lines of work. The authors should verify and position against the following:

- **Architecture-level error-propagation analysis.**
  - Abdelmoez et al., "Error propagation in software architectures" (METRICS 2004);
  - Popic et al., "Error propagation in the reliability analysis of component-based systems" (ISSRE 2005);
  - Cortellessa & Grassi, "A modeling approach to analyze the impact of error propagation on reliability of component-based systems" (CBSE 2007);
  - Hiller, Jhumka & Suri, EPIC (IEEE TC 2004).

  These compute propagation probabilities from architectural descriptions, the closest prior art to the dependency rules.
- **Model-based failure-propagation analysis.** FPTN, HiP-HOPS, and the AADL Error Model Annex.
- **GNN evaluation pitfalls and simple-baseline results.**
  - Shchur et al., "Pitfalls of graph neural network evaluation" (2018);
  - Lv et al., "Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks" (KDD 2021), which parallels the typing null;
  - Huang et al., "Combining label propagation and simple models out-performs graph neural networks" (ICLR 2021);
  - Chen et al., "Can graph neural networks count substructures?" (NeurIPS 2020), for the counting argument in §6.2.
- **Learned GNN-based failure analysis in microservices,** e.g., Eadro (ICSE 2023) and DiagFusion, as the runtime counterparts to the pre-deployment setting.
- **Simulation metamodelling / surrogate modelling,** which is what the $I_{\text{dyn}}$ approximations are.
- **Statistical practice for comparing learners in SE:** Arcuri & Briand (ICSE 2011) and Demšar (JMLR 2006).

---

## Minor Comments

### Compliance with the Guide for Authors

1. **Keywords:** eight are given; the Guide allows 1–7. Drop one (e.g., "graph learning", which overlaps with "graph neural networks").
2. **Highlights:** five bullets, each ≤ 85 characters, so they are compliant. Highlight 3 ("matches") conflicts with M6, and highlight 5 ("Learning beats the training-free baseline") overstates M4.
3. **Abstract:** 246 words, just under the limit. It omits that the primary registered contrast was null and that the zero-shot result rests on five single-author models.
4. **Figure files:** Figure 4 is `Figure_5.png` and Figure 5 is `Figure_6.png`. The Guide asks for logical file naming that matches figure numbers.
5. **Funding:** use Elsevier's standard wording ("This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.").
6. **Generative-AI declaration:** Grammarly is a basic tool the Guide exempts. Keep it if desired, but make sure the model names are exact product names.
7. **Data availability:** link a tagged release or commit, not `tree/main`. Confirm that the Zenodo DOI resolves before submission.
8. **Submission package:** `LENGTH_JUSTIFICATION.md` is stale. It states 30 pages, 101 references, 13 tables, 6 figures and supplement S1–S35. The manuscript has 32 pages, 75 references, 11 tables and 5 figures, and cites up to §S39.

### Internal consistency

9. §7.2 says "Only `HGT-QoS` was run under the registered selection rule". §6.2 and §7.5 say it was applied to `HGT-QoS` and `GAT-P-QoS`.
10. §7.4 and Table 11 explain `Topo-QoS`'s lead on $I_{\text{comp}}$ by its "betweenness and articulation terms". The articulation term is zero because of the defect (§5.2).
11. **Table 6 underlining.** GBM-P-QoS→dyn's $I^*$ value (0.764) is underlined as best predictor, but `GAT-P-QoS` (0.772) is higher in the same column.
12. **Table 6 mixed statistics.** The table mixes seed-ensemble rows (GNNs) with per-seed-mean rows (→dyn models). With the ensemble, `GAT-P-QoS` is *above* `InDeg` (0.772 vs. 0.764); per seed it is below (0.748). The ordering therefore depends on the statistic. Use one statistic per table.
13. **"Secondary" is overloaded.** §1.3 and §6 call the unweighted contrast the "secondary" confirmatory contrast, which collides with the "registered secondary" tier. Rename it "co-primary".
14. **Remark 1 vs. §3.3.** Eq. 2 counts *direct* subscribers only. §3.3 says node properties follow up to three `USES` links, so library-mediated subscribers count as Rule 1 dependents. It then says the `-P` learners use Rule 1 "only for direct subscriptions". State which definition each ranker and feature uses.
15. §6.1 ("statistically similar but not equivalent within ±0.05") and the Figure 4 caption misstate what a failed TOST shows (see M6).

### Method details

16. **§4.2, ListMLE ties.** Tied labels (~31% of Applications at $I^* = 0$) are ordered by entity identifier. This injects an arbitrary, fixed ordering into the loss. Use a tie-aware listwise loss or re-randomize tie order each epoch, and report the sensitivity.
17. **§4.2, validation split.** Early stopping on a 20% node-level split within one training scenario is transductive: validation nodes share the graph with training nodes. Scenario-level validation is cleaner. Discuss why the nested harness, which used scenario-level validation, does not reproduce the published scores.
18. **§4.3, QoS in $I_{\text{dyn}}$.** The text says $I_{\text{dyn}}$ uses QoS contracts. Specify how, and report its QoS sensitivity as is done for $I^*$ ($\rho = 0.965$ with QoS scaling disabled). Otherwise the "QoS policies carry no signal" result (§3.2, contribution 3) cannot be separated from oracle design. The parenthetical in contribution 3 ("revealing that declared QoS contracts carry negligible predictive utility for cascade ranking") generalizes beyond what simulators that barely read QoS can show.
19. **§4.3, $I_{\text{comp}}$.** The sentence defining the components is grammatically mixed ("…RL, FR…, TL is Throughput Loss, and FD is Flow Disruption"). "Weights from one of three AHP matrices that encode a declared vector" and §3.2's "the framework's other AHP matrices also use a declared vector" are cryptic. Say plainly that the weights were chosen, not elicited, or drop the AHP framing for $I_{\text{comp}}$.
20. **§3.1, CQP.** How are code metrics (LOC, cyclomatic complexity, LCOM) generated for synthetic Applications? Since no reported result depends on $w_V$, consider removing it from the main text.
21. **§4.1, "depart-mode flag".** The term is undefined.
22. **§5.3.** PR-AUC, $F_1@\tau$ and nDCG@10 are relegated to the repository. For a critical-set use case, report at least PR-AUC in Table 5, as Table 9 already does.
23. **§5.3, fold weighting.** The $|V_{\text{app}}|$-weighted sensitivity erases the gap between the raw-graph learners and `Topo-QoS` (0.596 vs. 0.595–0.606). This belongs in §7.5's conclusion-validity paragraph.
24. **Figure 5, right panel.** The panel uses the exploratory $n = 30$ developmental subset (lexicographic, not random), but the exhaustive $N = 1{,}321$ labels exist. Redraw it on the full population. Also fix "Idynn = 30" in the caption.

### Presentation and typography

25. Figure 4 caption: "When win is held" → "When $w_{\text{in}}$ is held".
26. §7.5: "Evaluating SAG" → "SaG".
27. Table 1 combines entity types and edge types under a mid-table header row. Split it into two tables or add a clear separator. Table 2's four unexercised rules (†) can go to the supplement.
28. **Eq. 5.** Present the formula that was actually executed, and document the defect in a note, or fix the defect (M4).
29. §6.4: "17–176× less" → "17–176× cheaper".
30. §7.2 cites `results/engine_regimes.json` in the body. Move file paths to the supplement or the data-availability statement.
31. **"Architecture–Code Gap" (§1.1).** The term is coined and then never used again. Define it with a citation or drop it.
32. **§1.1 green-software motivation.** The claim that "pre-deployment static analysis is a greener option" is supported only by generic Green AI references [21–23]. Either substantiate it (M10) or shorten it.
33. **Reference [75].** The author field contains a LaTeX escape artefact ("\.I. O. Yigit") in the Markdown rendering. Check the BibTeX entry.

---

## Recommendation

**Major Revision.**

The manuscript is transparent, reproducible, and statistically more careful than most submissions in this area. A well-argued negative result about learning versus dependency analysis fits JSS and this special issue. It is not publishable as is, for four reasons:

1. **Framing (M1, M3).** The central question cannot be answered by a benchmark whose three oracles are all low-order functions of the dependency graph. The post hoc reference criterion then removes the strongest practitioner baseline from the contrasts.
2. **Evidence underlying the positive claims (M4, M5, M7).** "Learning adds value" rests on beating a defective, dominated baseline. "Representation matters" is confounded with edge direction. The headline recommendations are exploratory analyses added last, and a fresh confirmation set would be cheap to generate.
3. **Robustness and scope (M2, M8, M9).** There is no real-system anchor. Learned-model drift is of the same size as the effects. The zero-shot evidence rests on five single-author models.
4. **Presentation (M6, M10, M11).** "Matches" claims contradict the equivalence tests. The energy figures are estimates with an internal inconsistency. The text is heavily repetitive and not self-contained.

Each item can be fixed within one revision cycle. Most need CPU sweeps the authors' tooling already supports, plus one fault-injection study on a deployable reference system. I would be glad to see a revised version.
