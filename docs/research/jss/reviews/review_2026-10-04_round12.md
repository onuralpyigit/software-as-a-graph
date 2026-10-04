# Referee Report — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems", round 12 (submission version)

**Manuscript:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?*
**Material reviewed:** `manuscript.md` (35 pp. PDF, 12 tables, 5 figures, 87 references), `latex/highlights.tex`, `latex/supplementary_flat.tex` (S1–S40, 44 pp.), and the JSS Guide for Authors. The manuscript is assessed as a stand-alone first submission, without reference to earlier review rounds.
**Date:** 2026-10-04 · **Recommendation:** Major Revision

---

## Summary

The paper presents Software-as-a-Graph (SaG). SaG derives a typed `DEPENDS_ON` dependency graph from publish–subscribe deployment descriptions and asks when graph learning improves cascade-impact ranking of Applications beyond analytical rankings computed from the same dependencies. The method is evaluated under leave-one-scenario-out (LOSO) on twelve synthetic architectures, and zero-shot on five hand-authored models of open-source systems. It compares GAT/HGT/GIN learners, hybrids that correct a closed-form prior, and gradient-boosted surrogates against training-free rankings, scored on three simulators: reachability ($I^*$), queue-flow ($I_{\text{dyn}}$) and composite ($I_{\text{comp}}$). A "reference criterion" separates rankings that restate a simulator's rule from rankings that predict it. The main finding is a well-documented negative result. The registered primary contrast is null, and no learned model exceeds the simple dependency count ($\rho = 0.764$) or a rate-weighted first-order formula ($\rho = 0.830$ on $I_{\text{dyn}}$). The authors conclude that the dependency representation, not model complexity, carries the signal.

## Overall Impression & Assessment

**Originality.** Moderate. Deriving explicit dependencies from pub-sub topologies extends the authors' RASSE 2025 paper with Library and infrastructure rules. The most interesting idea is the restatement-versus-prediction distinction. In its current form, though, it is a post hoc definition rather than a method (M2), and related ideas exist under other names: shortcut learning, oracle-aligned heuristics, and the degree-versus-spreading results in network science. The comparison with matched model capacity and the reverse-edge control are careful, and they are the strongest original parts of the paper.

**Significance.** Potentially useful as a cautionary benchmark. Simulator-labelled evaluations of GNNs for architecture analysis should be checked against analytical approximations aligned with the simulator, and this paper shows why. JSS explicitly welcomes negative results. However, the paper cannot answer its own title question (M1). It shows that learning did not help in a regime where, by the authors' own admission, it could not have helped. A reader learns *that* learning is unnecessary for these three oracles, but not *when* it becomes necessary.

**Methodological soundness.** The transparency is exemplary: a version-controlled plan, disclosed deviations, Holm and Nadeau–Bengio corrections, active-stratum $\rho_{>0}$, mechanical reconciliation of 1,711 figures, and a Zenodo archive. The candor about defects (the zeroed articulation term, seed failures, cross-device drift) is unusual and welcome. But that candor also lists serious problems:

- the registered comparator has an implementation bug and is the weakest analytical ranker;
- every headline result is registered-secondary or exploratory and was designed after the primary null;
- the search harness does not reproduce the published scores;
- learned cells drift across devices by more than several of the effects reported;
- no label is tied to an observed failure.

Disclosure is necessary but does not by itself make the inferences sound.

**Suitability for a Q1 venue.** The topic fits the Special Issue, and the engineering and reproducibility are above the JSS norm. In its present form, though, the paper reads more as an audit log of a long internal investigation than as a focused contribution. It needs a sharper problem formulation, at least one condition in which the "when" question has a non-trivial answer, a confirmation step for the post hoc headline claims, and roughly a quarter less text.

---

## Major Comments

### M1. The design cannot answer the question in the title, and the learned-ranking task for $I^*$ has no use case

The title and RQ1–RQ2 ask *when* graph learning improves cascade-impact ranking. The paper's answer (§7.5, §8) is that "all three simulators are well approximated by low-order functions of the dependency graph, so the benchmark did not exercise a regime in which learning could exceed an analytical ranking aligned with the simulator." A "when" question requires variation in the condition that determines the answer. Here that condition is held constant at "a first-order analytical approximation exists", so the study can only report that learning is not needed under that condition.

There is a second problem. $I^*$ is a deterministic function of the same manifest, computable in 0.01–0.72 s (Table 11). The paper itself recommends "run $I^*$ directly" (Table 12, row 1). Training a 430k-parameter GNN to approximate a cheap function of its own input is not a pre-deployment problem anyone faces. The only setting where a learned or analytical approximation has a practical rationale is the expensive simulator, $I_{\text{dyn}}$ (12.7 CPU-h).

*Requested:*
- (a) Reframe the study around surrogate modelling of expensive simulators, with $I^*$ as a calibration case for the reference criterion rather than a prediction target.
- (b) Add at least one oracle with mechanisms that are not first-order, where Eqs. 6–7 are expected to fail: backpressure onto publishers, retry amplification, heterogeneous service capacity, or bounded-queue coupling across brokers. §7.6 item 4 already names this. It is the experiment that would let the paper answer "when". A positive *or* negative result there would be informative.
- (c) If neither (a) nor (b) is done, retitle the paper to reflect what it shows, for example "Dependency Representation, Not Model Complexity, Explains Simulator-Labelled Cascade-Impact Ranking in Publish–Subscribe Systems".

### M2. The reference criterion is post hoc, not operational, and asymmetrically applied; its novelty is overstated

The definition (§4.4), "a closed-form simplification of $O$'s own computation using the same inputs", is not decidable as stated. Almost any degree- or reach-based score is a "simplification" of almost any cascade simulator. Whether `Topo-QoS` is a "predictor" for $I^*$ but merely "term-aligned" for $I_{\text{comp}}$ is a judgement call. The authors adopted the criterion after seeing that the counts performed strongly (§4.4, §7.5), and the reclassification removed `InDeg` from all contrasts.

The criterion is also applied asymmetrically. Learned models *trained on $I^*$ labels*, with in-degree, fan-out-criticality and connectivity-degradation features (M3), are classed as "predictors". The untrained count is classed as a "reference". By any reasonable standard, a model fitted to the oracle is at least as circular as a count that happens to match the oracle's first wave. From a practitioner's viewpoint, afferent coupling is a legitimate pre-deployment predictor whose value depends on whether it predicts *real* failures. Whether it restates a simulator matters only to the evaluation methodology.

On novelty, the claim "we are not aware of prior work that separates these two cases explicitly" (§2.1) should engage with:
- the degree/k-shell-versus-spreading literature, where first-order neighbourhood counts are known to approximate low-transmissibility SIR impact (e.g., Kitsak et al., *Nature Physics* 2010);
- shortcut learning (Geirhos et al., *Nature Machine Intelligence* 2020);
- the simulation verification-and-validation literature (e.g., Sargent, *Journal of Simulation* 2013), where the circularity of a simulator used as ground truth is a standard concern.

*Requested:* operationalise the criterion as a decidable, a-priori test. One option: a ranking is an order-$k$ reference for $O$ if it equals $O$ truncated at propagation depth $k$ under stated simplifications, with the derivation given for each reference. Then report the share of each oracle's rank variance explained by its order-1 truncation as a *property of the oracle*. Apply the same accounting to the learned models, for example via partial correlation with the order-1 truncation, which Table 6 already does for $I_{\text{dyn}}$. Present contribution 2 as an evaluation guideline rather than a methodological contribution.

### M3. Several input features of the learned models compute what the oracle computes; the degree ablation does not remove them

The procedural input–label separation (§4.4) does not guarantee that the inputs carry no oracle-like information. Supplementary Table S12 lists, among the 18 base features every learner receives:
- **CDI**, "change in all-pairs reachability when entity $v$ is removed". This is a removal-and-reachability simulation on $G_{\text{analysis}}$, structurally close to $I^*$.
- **FOC**, "topic-modulated subscriber blast radius", which is close to Eq. 6.
- **MPCI**, afferent multi-topic coupling.
- Reverse PageRank and directed articulation.

Ablation F1 removes only `in_degree` and $w_{\text{in}}$ (and, in −deg$^*$, PageRank, closeness and eigenvector centrality). CDI, FOC and MPCI stay in every arm. So the observation that "learners approach the reference" and the zero-shot results (0.810–0.821 even without degree columns, Table 8) may be carried by these features rather than by message passing or the dependency representation.

*Requested:* a minimal-feature ablation for `GAT-P-QoS`, `GAT-QoS` and `GIN-P-QoS`, with (i) all oracle-like features removed (CDI, FOC, MPCI, RPR, both articulation flags, in-degree, $w_{\text{in}}$), and (ii) constant or random node features. This is the standard test of what a GNN learns from structure. If $+0.113$ / $+0.072$ (dependency graph vs. raw / reverse-edge) survive (ii), the representation claim becomes much stronger. If they do not, the claim must be restated. The training cost matches the existing F1 sweep.

### M4. Every headline result is post hoc relative to the primary null; a confirmation step is cheap and should be done now

§5.3 is explicit about this:
- the only confirmatory tests (HGT-QoS vs. Topo-QoS, HGT vs. Topo-QoS) are null;
- the paper's headlines (the dependency-graph gain, the reverse-edge control, the hybrid wins, the rate-weighted formula, the QoS null on $I_{\text{dyn}}$) were all specified after the primary null was known, across sixteen amendments;
- the formula recommended in the abstract, Table 12 and Tier 1 (Eq. 7) is labelled exploratory and was "added after all other results existed".

Holm correction within families defined after the fact does not control the error rate of this process. "Registered" here means committed to the authors' own repository, not a third-party registry (OSF, a Registered Report). The word should be qualified, for example as "pre-specified in a version-controlled plan, with commit timestamps".

The remedy is unusually cheap here, because the authors own the generator (§5.1) and $I^*$ labels cost seconds. *Requested:* freeze the analysis and generate a confirmation corpus of new scenarios: new seeds and, ideally, new domain configurations not used so far. Re-test the five or six claims the paper actually relies on there:
1. `GAT-P-QoS` vs. `GAT-QoS-R`;
2. `GAT-P-QoS` vs. `InDeg` (with TOST);
3. Eq. 7 vs. `GBM-P-QoS→dyn`, on an $I_{\text{dyn}}$-labelled subset if the full cost is prohibitive;
4. the hybrid vs. base-learner nulls;
5. the QoS-policy null on $I_{\text{dyn}}$.

§7.6 item 5 already proposes this ("a confirmation corpus generated after the analysis is frozen"). It belongs in this paper, not in future work.

Also in this vein: the obvious follow-up to the Eq. 7 result is a learned model that *receives Eq. 7 as an input* or is trained to correct it (stacking or residual learning on top of the formula). That is the most direct test of "does learning add value beyond the analytical approximation". For gradient boosting it costs seconds, and §7.2 and §7.6 defer it. Please run it.

### M5. The robustness of the learned-model evidence is weaker than the text implies, and one self-imposed rule contradicts the headline

- **The drift rule.** §7.5 states: "Across revision cycles and compute devices, learned cells drifted by up to 0.172 … Differences between learned rankers smaller than this drift should not be interpreted, whatever their nominal $p$-value." The central representation claims are differences between learned rankers: `GAT-P-QoS` vs. `GAT-QoS` (+0.113) and vs. `GAT-QoS-R` (+0.072). Under the 0.172 reading, the rule disqualifies the paper's main finding. If "this drift" means the ±0.041 on HGT-QoS, say so. Better, show directly that drift is a shared per-device shift that cancels in contrasts paired within a single sweep. For example, re-run the F8 contrast on a second device and report the paired difference on each.
- **Hyperparameters.** The registered nested selection was not applied (§4.2, §6.2). When applied later, it moves `GAT-P-QoS` by −0.054 and `HGT-QoS` by +0.055. These shifts are as large as the effects under discussion, and they *close most of the gap* between the two. On folds where the search chose the published configuration, the harness "does not reproduce the published scores (differences up to 0.10)". This needs an explanation, not just a disclosure: which part of the pipeline differs (validation split, early-stopping target, data loading)?
- **HGT instability.** `HGT-P-QoS` has a within-fold seed SD of 0.254, and 4 of 60 runs are anti-correlated with the oracle. That is an optimisation failure, not a property of relation typing. With one fixed width-100 configuration and no tuning, the HGT rows (and the typing conclusions drawn from them) carry little information. Either give HGT-P a tuning budget equal to GAT-P's, or remove typing-on-the-dependency-graph claims from the summaries.
- **Tie ordering by identifier.** In ListMLE, tied labels (≈31% of Applications at $I^*=0$) are ordered "by entity identifier" (§4.2). The $I_{\text{dyn}}$ subset check uses "the first 30 Applications per fold in lexicographic order" (§4.3), and Overlap@$K$ breaks ties "by node order" (§5.3). If generator identifiers correlate with role or creation order (publishers generated first, for example), this can inject label-correlated signal. Please report the rank correlation between identifier order and $I^*$ in each fold, or randomise.
- **Early stopping.** Early stopping uses a 20% node-level split of the *largest* training scenario, so the selection signal comes from a different scenario in the fold where that scenario is held out. Please state how this is handled and whether it biases that fold.

### M6. RQ2's typing and QoS conclusions are drawn mostly on a substrate where message passing is inert

The authors show that on $G_{\text{structural}}$ forward messages never reach Applications, that removing all edges does not change GAT's outputs, and that a graph-free GBM matches GAT-QoS (§7.2). The matched 2×2 (Table 7) is therefore a comparison of per-node MLPs with different parameterisations and node columns. It says nothing about relation typing or about QoS *edge* information.

Moreover, the 16-D edge vector of the "untyped" `GAT-QoS` contains a one-hot relation encoding (indices 2–8, §4.1). So on any substrate where messages do flow, the "untyped" arm receives type information through its edge features, and the typing factor is confounded.

On the dependency graph, the App–Lib projection has only two relation types, which leaves little room for typing to matter.

The summary of §6.2 ("When capacity is matched, relation typing has no effect") and Highlights/§1.2 should be restricted accordingly. *Requested:* move Table 7 to the supplement, or keep it only as the control that establishes the inert-substrate finding. If typing is to be tested at all, do it on a substrate with active message passing and more than two relation types, with the one-hot removed from the untyped arm.

### M7. External validity: no topology comes from a real artifact, and nothing anchors the oracles to observed failures

The paper motivates SaG as analysing "deployment manifests" (abstract, §1.1, contribution 1). Yet:
- all twelve LOSO scenarios come from one generator family;
- the five "open-source" models were hand-authored by the first author from documentation, with brokers, QoS, code metrics and hosts "partly assumed";
- two of the five are RPC systems re-expressed as pub-sub meshes;
- no second modeller re-derived any of them.

The manifest-to-graph pipeline is therefore never exercised on a real manifest. On the hand-authored models, `Reach` attains $\rho = 0.938$, which shows that $I^*$ collapses to reachability there. RQ3 then tests the authors' modelling style more than transfer.

*Requested (in order of value):*
1. Extract at least two system models automatically from real artifacts: ROS 2 launch files (cf. HAROS [53]; ROSDiscover, Timperley et al., ICSA 2022), or the docker-compose/Kubernetes/Helm manifests that EdgeX Foundry and Online Boutique publish. Compare them with the hand-authored versions as a partial substitute for a second modeller.
2. Run a small live sanity check of the oracle construct. For example, crash-fault injection on a ROS 2 demo graph, or on a Kafka/MQTT deployment of one system model, reporting the rank agreement between observed delivered-rate loss and $I^*$/$I_{\text{dyn}}$/`InDeg` over its Applications. Even $n \approx 20$–$40$ components would turn "all conclusions are conditional on simulation fidelity" from an untested caveat into a measured one.
3. If neither is feasible, move RQ3 to the supplement and say plainly that no real topology was analysed.

### M8. The registered comparator is defective and the weakest ranker, so the "learning adds value" findings carry little weight

`Topo-QoS` has an index bug that zeroes its articulation term (§5.2), and it is the weakest analytical ranker in the study. The only significant wins for learning (the hybrids, +0.103/+0.130, and the zero-shot 0.81 vs. 0.53) are against this comparator. The hybrids do not differ from their own base learners, and the zero-shot contrast has no test. Given $p(v)$ = `InDeg` as their prior, the hybrids reproduce `InDeg`. The abstract and §1.2 nonetheless give these results a full paragraph ("Learning adds measurable value only relative to a weak comparator…").

*Requested:*
- use the corrected `Topo-QoS-AP` (0.533) as the reported baseline in all main tables, with the defective value in a footnote;
- state the hybrid result as a null (no gain over the base learner) and drop it from the abstract;
- explain in §5.3 why afferent coupling, a classical metric [54–56], was not the registered comparator.

The admission that the RASSE 2025 score family is "the weakest analytical ranker" and that "the present results supersede that paper's implicit recommendation of it" is commendable. It should be stated once, plainly, in §1.4, as it already is.

### M9. Presentation: the paper is overlong, repetitive, and written in the vocabulary of its own audit trail

- **Repetition.** Headline numbers appear many times: 0.748 twenty times, 0.764 thirteen, 0.830 thirteen, 0.799 eleven, "12.7 CPU-hours" ten. "Findings in brief" (§1.2), the italic summaries of §6.1–6.4, §7.1–7.2, Table 12 and §8 each restate the same results.
- **Internal vocabulary.** "Amendment 16", "registered secondary" / "exploratory" (40 occurrences), "the articulation defect" (22 occurrences of "defect"), "published per-seed", "one CPU sweep", "re-ran every comparator to an exact match". This is process vocabulary from the project's history. JSS readers need the epistemic status of each claim stated once, in a table, not in nearly every sentence.
- **Unevaluated components in the main text.** The ISO/IEC 25010 explanation layer (in Figure 1), Rules 2, 3, 4 and 6, the 18-detector anti-pattern "gate" in Table 11, the AHP formalisation for $I_{\text{comp}}$ (whose matrix "was filled in to reproduce" a chosen vector), and the full QoS weighting machinery that §3.2 itself shows to be inert. Each should be cut or moved to the supplement.
- **Paragraph length.** Several paragraphs exceed 400 words with more than 25 numbers each (e.g., "Beyond the reachability oracle", §6.1; "Internal validity", §7.5).

A 25–30% reduction is achievable without losing evidence. The authors need not hit the 36-page ceiling just because they are under it.

---

## Minor Comments

1. **Abstract.** With each math span counted as one word it is exactly 250 words, the JSS maximum; trim for safety. "whereas on that graph the registered primary contrast…" is ambiguous (it refers to the raw multigraph). "The learned model is matched by counting direct dependents" reads as equivalence, which TOST rejects ($p = 0.17$). Use "is not distinguishable from".
2. **Highlights.** All five are within 85 characters. Highlight 3 ("matches the best learned ranker") contradicts the paper's own definition of "matches" (§5.3) for a general reader; use "is not outperformed by". Avoid "rho" and "pub-sub" in highlights. Highlight 5 ("Learning beats the training-free baseline") inherits M8.
3. **Keywords.** "Graph neural networks" and "graph learning" are redundant. Consider "surrogate modelling" or "afferent coupling".
4. **§1.2 structure.** "Findings in brief" sits under "The SaG Approach". Make it its own short subsection or fold it into §1.4.
5. **§1.4, contribution 1** claims infrastructure dependency rules, which are never exercised (§3.3, §7.5). Limit the claim to Rules 1 and 5.
6. **§3.5** says "18 normalized topological metrics" but lists 17. Directed articulation (S12, index 16) is missing.
7. **§3.3.** The `InDeg` reference counts direct subscriptions only, while the in-degree *feature* follows up to three `USES` links. Explain why the definitions differ and report how often they disagree.
8. **§4.1.** "Message passing runs over both $G_{\text{analysis}}$ and its transpose" conflicts with HGT-QoS running on $G_{\text{structural}}$. State, per model, which graph and which directions are used.
9. **§4.2.** Define the "reliability column" $I^*_R$. The masked maintainability head contributes nothing; remove it.
10. **§4.3.** Justify the $I^*$ QoS severity constants (×1.2/×1.15/×1.05), or note that they are arbitrary, as Table 6's QoS-off checks suggest they barely matter.
11. **§2.3 / §4.3.** If the $I_{\text{comp}}$ weights were declared rather than elicited, drop the AHP framing for them altogether. Invoking AHP and then disclaiming it adds nothing.
12. **Table 5** marks the `GAT-P-QoS` / `HGT-P-QoS` contrasts against `Topo-QoS` as ‡exploratory, whereas §1.3 and §5.3 list "the dependency-graph learners" as registered secondary. Reconcile.
13. **§6.1 / §7.2.** "`GAT-P-QoS→dyn` … is no better than the same GNN trained on $I^*$ ($-0.008$)". I could not reconcile this with Table 9 (GAT-P-QoS per-seed $I_{\text{dyn}}$ = 0.597, giving +0.001) or Table 6 (ensemble 0.615, giving −0.017). Name the comparator statistic.
14. **Table 6** mixes seed-ensemble rows and per-seed rows in one column, and the underline rule needs a parenthetical exception. Report one statistic per table.
15. **Table 12** cites the GNN approximation (0.598) as evidence, although that model received neither rates nor payloads (§7.2). Footnote it as input-mismatched or drop it.
16. **§6.4 / §7.4, energy.** TDP × single-thread wall time is crude, as §7.5 acknowledges. On the Linux/Intel machine used, RAPL is available through `powercap` at negligible effort. "0.086 mWh, 0.31 J" for an 11 ms computation is false precision.
17. **Overlap@K** breaks ties by node order. Since Figure 5 already gives tie-aware recall, consider replacing Overlap@K with a tie-aware metric in Tables 5 and 8.
18. **Effect sizes.** Arcuri & Briand [85] recommend standardised effect sizes (e.g., Vargha–Delaney $\hat{A}_{12}$). Add them for the decision-bearing contrasts, and list the "13 decision-bearing contrasts" of the omnibus correction in the main text rather than only in S26.
19. **Figures.** The body's Figures 4 and 5 are supplied as `Figure_5` and `Figure_6`, and `Figure_4` is a supplementary figure. The Guide asks for logical file naming. Figure 4 combines three panels. That is acceptable where they are directly related, but check legibility at single-column width and use a colour-vision-safe palette.
20. **References.** Many entries lack DOIs where one exists (e.g., [1], [7], [10], [26], [27], [34], [35], [72]). [74] is a preprint and should carry its arXiv DOI. [82] (SimPy) needs a version. Per the JSS software-citation guidance, cite the SaG software (version and persistent identifier) separately from the dataset [87]. A numeric style is acceptable at submission, but JSS house style is author–year.
21. **Repository link.** `tree/jss-submission/...` resolves through a mutable ref. Cite an immutable commit SHA or a Software Heritage identifier.
22. **Supplement.** Section titles such as "Amendment 12: Referee Analyses" and "Round-8 Analyses (Amendment 14)" refer to a review history that JSS reviewers cannot see. The HGT formulation (cited as S2.1) sits under "Parameter Sensitivity of the Explanation Layer". The heading "The elicited weights are anti-predictive, and we keep them anyway" is informal. At 44 pages and 40 sections, the supplement should be curated around the main-text claims.
23. **Spelling.** The main text uses US spelling, but the supplement has "Variance-Stabilised" (S17). The Guide asks for one variety throughout.
24. **§5.3.** The statement that weighting folds by $|V_{\text{app}}|$ lifts `Topo-QoS` to the level of the raw-graph learners deserves a sentence in §6.1, not only in §5.3 and §7.5. It shows how fragile the ordering is with twelve folds.
25. **Related work (optional additions):** metastable failures (Bronson et al., HotOS 2021; Huang et al., OSDI 2022) as the canonical example of impact decoupled from structure (§7.5); microservice dependency characterisation from production traces (Luo et al., SoCC 2021); and GNN node-importance estimation on heterogeneous graphs (GENI, Park et al., KDD 2019) as the closest learned analogue to HGT-QoS.

**Compliance with the Guide for Authors (checked):** title page with corresponding author ✓; abstract ≤ 250 words (borderline, item 1) ✓; 1–7 keywords ✓; 3–5 highlights ≤ 85 characters ✓; numbered sections ✓; CRediT ✓; competing interests, funding and data statement with a Zenodo DOI ✓; generative-AI declaration with the required heading, at the end ✓; length 35 single-column pages (< 36) ✓; vitae present in the LaTeX package ✓; figure file naming ✗ (item 19); DOIs incomplete ✗ (item 20).

---

## Recommendation

**Major Revision.**

The paper is honest, reproducible and carefully controlled in places, and it addresses a real methodological problem for the Special Issue: simulator-labelled evaluations of AI rankers can mistake restatement for skill. It is not acceptable in its present form, for four reasons:

1. As designed, it cannot answer the "when" question it poses, and its only prediction target where learning would be practically motivated ($I_{\text{dyn}}$) is studied with an input-mismatched GNN and without the obvious formula-plus-correction learner (M1, M4).
2. Every headline claim is post hoc relative to a null primary result, and the cheap confirmation corpus that would settle them is deferred to future work (M4).
3. Oracle-like input features and the self-declared drift threshold undermine the representation claim until the minimal-feature ablation and paired cross-device check are done (M3, M5).
4. No topology comes from a real artifact and no label is tied to an observed failure (M7).

Items M3, M4 (confirmation corpus plus Eq. 7 stacking), M5 and M8 are inexpensive given the existing harness. M1(b) and M7(1–2) take more effort and are what would lift the paper from a careful negative benchmark to a Q1 contribution. A revision that does M3–M5 and M8, prunes the text (M9), and either adds a non-first-order oracle or retitles honestly (M1) would be close to acceptable. If the confirmation corpus overturns the representation claim, the paper would still be publishable as a negative result, provided it is framed as one.
