# Referee Report (Round 3) — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems"

**Manuscript (second revision):** *Software-as-a-Graph: Dependency-Graph Analysis and Learning for Pre-Deployment Cascading-Failure Ranking in Publish–Subscribe Systems* (Yigit, Buzluca)
**Material reviewed:**
- [`manuscript.md`](../manuscript.md) and the compiled [`latex/manuscript.pdf`](../latex/manuscript.pdf) (26 pp.);
- [`latex/supplementary.tex`](../latex/supplementary.tex) (32 pp.), including the Amendment 7, 9 and 10 sections;
- [`PREREGISTRATION.md`](../PREREGISTRATION.md), Amendments 7–10 and the results log;
- [`latex/highlights.tex`](../latex/highlights.tex), [`latex/vitae.tex`](../latex/vitae.tex) and [`latex/LENGTH_JUSTIFICATION.md`](../latex/LENGTH_JUSTIFICATION.md);
- the code paths the central claims depend on: [`saag/simulation/fault_injector.py`](../../../../saag/simulation/fault_injector.py), [`saag/analysis/structural_analyzer.py`](../../../../saag/analysis/structural_analyzer.py) and [`saag/prediction/data_preparation.py`](../../../../saag/prediction/data_preparation.py).

**Checked against:** *Guide for Authors — Journal of Systems and Software* (PDF in this directory, retrieved 13 Sep 2026).
**Review model:** single-anonymised.
**Date:** 2026-09-26

---

## 1. Summary

The paper models publish–subscribe architectures as typed multigraphs, derives a `DEPENDS_ON` graph from them with six rules, and asks which rankers best order Applications by the cascade impact that a failure simulator (I\*) assigns to them.

The evaluation covers:
- **Rankers:** dependency counts, closed-form centrality, heterogeneous and homogeneous GNNs, and hybrids.
- **Protocols:** leave-one-scenario-out over twelve synthetic architectures from a single generator, and zero-shot on five hand-authored models of open-source systems.
- **Pre-registration:** a registered analysis plan with ten amendments.

The main finding is that counting a component's dependents on the derived graph (ρ = 0.764; transitive reach 0.938 on the system models) matches or beats every learned engine. The paper presents the dependency derivation as the decisive contribution and the learned and hybrid engines as secondary.

## 2. Overall Impression and Assessment

**Transparency and reproducibility are exemplary.** Very few submissions:
- register their analyses and publish negative decisions against their own thesis;
- regenerate their corpus byte-identically;
- reconcile hundreds of reported figures mechanically against artifacts.

The authors have again responded to the previous round with new registered experiments (Amendments 9 and 10), and the text is more candid than before. §8.3 now concedes that `InDeg`'s accuracy "partly measures how consistent SaG's dependency rules are with the oracle". JSS explicitly welcomes negative results and replication packages, and this work would be a credit to the Open Science track.

**The revision did not address the central problem, and it added headline claims the evidence does not carry.** My assessment against the four usual criteria:

- **Originality — low to moderate.**
  - The winning ranker is afferent coupling (fan-in). The supplement concedes that for an Application it *is* the raw 2-hop subscriber count (Supplementary section on Amendments 9 and 10).
  - The recommended engines use only Rule 1 and, for `Reach` on synthetic data, Rule 5. Rules 2, 3, 4 and 6 have no empirical role in any result.
  - What remains original is the careful benchmark and the negative result about learning. The "six-rule derivation" is not what is shown to matter.
- **Significance — not yet established.** Every accuracy figure is agreement with a simulator whose propagation graph is the same subscriber→publisher and app→library relation the counts read (see M1). By the authors' own measurement (§7.4), that simulator is also *cheaper* to run than the learned pipeline, from the same inputs (see M2). The practical value of predicting it is therefore unclear.
- **Methodological soundness — mixed.**
  - The statistics are careful and honestly caveated.
  - However, three causal attributions in the abstract are confounded or rest on strawman comparators (M3, M4, M5).
  - One registered conclusion is contradicted by the paper's own central thesis (M6).
- **Suitability for this Q1 special issue — weak as framed.** The recommended technique is a non-learned count, and the AI components are dominated. The paper never argues its fit to an *AI-techniques* special issue (M8).

In short, this is a rigorous, honest study whose headline is close to a property of its own benchmark. The one experiment that would break that circularity was requested in the previous round, is cheap to run with the released harness, and is again deferred to future work.

### Status of the previous round's major comments

| Round-2 item | Status in this revision |
|:---|:---|
| M1 — evidence independent of the oracle's construction (I_dyn, I_comp) | **Not addressed.** Deferred again (§8.3 "the next experiment"; §8.4 item 3; §9). |
| M2 — novelty relative to afferent coupling / AIS | **Partly.** Cited in §2.2 and §7.1. The main text still claims the derivation "makes the count predictive" on a strawman comparator (M3 below). |
| M3 — two scoring pipelines in the system-model comparison | **Explained, not fixed.** Table 6 still mixes harnesses (dagger note; S36 caption). |
| M4 — exploratory contrasts presented as headlines | **Partly.** Now labelled exploratory in §6.3, but they remain the abstract's lead claims, and the primary registered test's failure is absent from the abstract (M7). |
| M5 — weight of dominated learned-engine material | **Partly.** Table 11 drops them; §4 is unchanged. |
| M6 — QoS machinery with no demonstrated role | **Not addressed, and worsened.** A new QoS claim in the abstract is confounded (M4). |
| M7 — special-issue fit | **Not addressed.** |

---

## 3. Major Comments

### M1. The accuracy of the dependency counts is largely implied by how the oracle is built. The independent test is still missing.

The code settles what §4.4 and §8.3 leave qualitative:
- **The oracle builds the counts' graph itself.** `FaultInjector` constructs its own `DEPENDS_ON` graph, with one subscriber→publisher arc per shared topic plus app→library arcs (`fault_injector.py`, lines 273–282). Library failure propagates along these arcs (Phase A), and publisher failure propagates to subscribers through topic feed loss (Phase B).
- **I\*(v) is a first-order function of v's subscribers.** I\*(v) is the mean feed loss over subscribers (lines 55–57). In the first wave, the set of subscribers whose feed loss is non-zero is exactly the set `InDeg(v)` counts, and each loses the publisher share 1/|pub(t)| of the affected topics.
- **The counts are therefore the oracle's first-order and closure terms.** `InDeg` is the unweighted first-order term of I\*, and `Reach` is the support of its cascade closure.

Two consequences follow:
1. **Zero dependents means zero impact, by construction.** Any component with no dependents has I\* = 0, and 21–52% of every population is inert, so a large part of the full-population ρ is a definitional tie between two zero sets. The paper's own numbers show this on the system models: `InDeg` falls from 0.863 to ρ₍>0₎ = 0.321 on the active stratum. The equivalent LOSO figure for `InDeg` and `Reach` is not in the main text.
2. **The benchmark cannot separate "the derivation captures real failure paths" from "the derivation matches the simulator's propagation rule".** An analytic first-order approximation of I\* (publisher-share-weighted subscriber count) would, I expect, score higher still. That would demonstrate the point.

**Requested:**
- **(a) Score every ranker on I_dyn and I_comp.** The previous round asked for this. The harness exists, the counts are training-free, and I_dyn is already computed for all twelve folds at seed 42 (Supplementary S3.1). For the counts and `Topo-QoS` this costs minutes. Re-training the learned engines on I_dyn is desirable but not required for the counts' claim.
- **(b) Add the active stratum to Table 6.** Report ρ₍>0₎ for `InDeg`, `Reach` and `GAT-P-QoS` under LOSO beside the full-population figure.
- **(c) Add the analytic first-order I\* approximation as a reference row.** Label it as the ceiling of what "agreement with the simulator's own rule" can reach.
- **(d) Reword the claims.** Until (a) exists, the title's "Cascading-Failure Ranking", the abstract's "makes cascading-failure risk rankable before deployment", and §9's opening sentence should say *simulated* cascade impact under a reachability oracle.

If the counts still lead on I_dyn, the paper's main claim becomes substantive. If the learned engines lead there, the paper recovers its AI contribution. Either outcome is publishable. The current state is not.

### M2. Why predict an oracle that is cheaper than the predictor?

§7.4 reports two facts about the costs:
- Cold feature extraction takes 2–18× (median 5.6×) as long as the in-process five-seed cascade simulation.
- The simulation needs only the architecture model: G_structural, which is available at the same point in the lifecycle as G_analysis.

The paper concedes that "on raw CPU time, direct simulation is therefore faster wherever its parameters are available". The motivation (§1.1) is that runtime telemetry is unavailable before deployment, but I\* needs no telemetry. If I\* is the ground truth, a CI gate can simply run I\*. A learned or counted proxy for a cheap, deterministic function of the same input needs a stated purpose. Possible purposes include:
- the proxy is a stand-in for an expensive or unobservable true impact (then M1(a) is required to show it tracks that quantity);
- the proxy provides explanation that the simulator does not;
- the proxy supports incremental analysis.

Please state which one it is, and evidence it. "Avoids staging infrastructure" applies equally to the in-process simulator and is not a differentiator. The same argument weakens the sustainability framing (§8.1): the cheapest way to obtain I\* is to compute I\*.

### M3. "The derivation is what makes the count predictive" rests on a strawman comparator.

**The comparator.** Amendment 10 records, as an identity pinned by tests, that for every Application `InDeg` equals its raw 2-hop subscriber count, and that removing Rule 5 cannot change it. The main text does not state this identity; it appears only in the supplement. Instead, §7.1, §8.1, §9, the abstract ("far above counting raw connections, +0.565") and highlight 1 compare `InDeg` with *total raw degree*. That count mixes:
- `SUBSCRIBES_TO` edges, which make v a dependent rather than a dependency;
- `RUNS_ON` and `USES` edges, which are unrelated to v's consumers.

No one would propose it as a fan-in estimate. The honest comparator is the 2-hop subscriber count, one join on the raw multigraph with no rule engine required. That comparator is *identical* to `InDeg`.

**What the evidence supports:**
1. **`InDeg`.** For Applications, SaG's derivation adds nothing beyond a 2-hop topic join.
2. **`Reach`.** On the synthetic folds, Rule 5 adds +0.058 to transitive reach. On all five system models, Rule-1-only reach *equals* `Reach` (§7.3), so the 0.938 headline uses no rule beyond Rule 1.
3. **Rules 2, 3, 4 and 6.** They contribute to no reported accuracy figure (acknowledged in §8.4).

**Requested:**
- State the identity in the main text, in §6.2 or §7.1.
- Remove "+0.565 over raw connections" from the abstract and highlights, or pair it with the 2-hop count it equals.
- Restate Contribution 1 accordingly: the derivation makes pub-sub afferent coupling expressible, and Rule 5 adds a measured increment to transitive reach on synthetic data. Nothing more is shown.
- "Six rules" should not headline a contribution in which four rules are untested.

### M4. The "QoS helps learned engines" claim is confounded with the dependency count, and its registered test failed.

**The claim.** The abstract states: "Declared Quality-of-Service contracts help learned engines through three node-level coupling features." §7.2 locates the +0.073 QoS main effect in the node columns (w, w_in, w_out).

**The confound.** In the code, `qos_weight_in` is `dependency_weight_in`: the **sum of incoming `DEPENDS_ON` edge weights** on the analysis graph (`structural_analyzer.py`, lines 467–472; `data_preparation.py`, line 1438). This is a *weighted dependent count*, i.e. a weighted `InDeg`. The "QoS-off" arms zero exactly these columns (`data_preparation.py`, lines 781–782).

The QoS ablation therefore removes the single feature most closely aligned with the paper's strongest ranker. The following findings are all what one would expect if the columns work as a dependent count, not as QoS content:
- removing the columns costs 0.095, while adding the 16-D edge channel alone gives −0.023;
- the seed-stability effect follows the same columns;
- on the closed-form side, permuting QoS profiles changes nothing (−0.006, p = 0.73);
- unweighted projection betweenness (0.591) and constant topic weights (0.595) both *beat* QoS-weighted `Topo-QoS` (0.553), so QoS weighting slightly *hurts* closed-form ranking.

**The statistics.** The registered test of the QoS main effect in Table 8 is *not significant* after correction (p_Holm = 0.127). The supporting follow-up is exploratory.

**Requested:**
- **(a) A control that separates count from contract.** Keep the three columns but replace the weights with (i) 1 (unweighted in/out degree on the projection) and (ii) permuted QoS profiles, as was done for `Topo-QoS`.
- **(b) Until (a) is run,** drop the QoS sentence from the abstract and from Contribution 3.
- **(c) State in §3.2 that QoS weighting does not improve closed-form ranking** (−0.038 against unweighted betweenness on the same graph). Move Eqs. 2–4 and the AHP derivation to the supplement. The round-2 request (M6) stands, and the case for it is now stronger.

### M5. The learned engines were structurally unable to use the graph for most of the study, and the typing conclusion is drawn only from that regime.

§8.2 discloses that on the native multigraph every relation points away from Applications:
- the GATs "never deliver [a message] to the population they score";
- deleting every edge leaves their outputs unchanged;
- removing HGT's reverse pass also costs nothing (`HGT-QoS-U`).

The raw-multigraph "GNNs" are therefore per-node MLPs over precomputed centralities. That is a design defect in the engines the paper presents in §4 as SaG's learned engines, not a finding about graph learning. It has direct consequences:
1. **RQ2's typing conclusion is not supported.** "Relation-specific weights add nothing" (§7.2, §8.2, Table 8) is drawn in a regime where no message passing reaches the scored nodes. Typing cannot help when no relation is exercised. On the dependency graph, where messages do flow, the typed arm `HGT-P-QoS` failed to train (seed spread 0.208) and was reported untuned. Typing has therefore **never been tested in a regime where it could matter.** "No scenario class calls for a heterogeneous model" and "an untyped attention network suffices" should be withdrawn, or scoped to "on a substrate without incoming messages".
2. **The abstract's GNN claim omits the arm that failed.** "Graph neural networks gain +0.08 to +0.11" reports the three GAT arms and leaves out `HGT-P-QoS` (−0.107). Say "attention networks (GAT)", and report HGT's failure alongside.
3. **The learned engines underperform a feature they effectively contain.** Every learned engine and `GBM-Feat` already read `in_degree_centrality` (a 0.55–0.90 correlate of `InDeg`, Amendment 9) and w_in (weighted `InDeg`). Yet they score 0.13 below `InDeg` alone. A monotone learner given `InDeg` should not fall below it. This points to per-graph normalisation, the loss weighting, or overfitting to eleven graphs, and it deserves diagnosis. At minimum, report `GBM-Feat` with the projection `InDeg` column added.
4. **The learned engines were never tuned.** The loss coefficients were "set by judgment", and no hyperparameter was tuned, even by inner cross-validation. That is defensible as a fairness choice, but it caps what "learning adds nothing over counting" can mean. A modest, registered inner-CV tuning budget would make the negative result much stronger.

With this in view, §4.1–4.2 (HGT architecture, the 16-D edge encoding and the composite head) describe an engine that the paper's own diagnostics show did not use its graph. They should be condensed substantially, with the space given to M1 and M4.

### M6. The central thesis contradicts the paper's own results.

§1.2 states: "The central thesis is that the representation, not the choice of analyzer, is the decisive investment." On the *same* representation (the Application–Library `DEPENDS_ON` graph), the choice of analyzer moves ρ by:
- 0.21: `InDeg` 0.764 against `Topo-QoS` 0.553;
- 0.25: `InDeg` against `HGT-P-QoS` 0.514;
- 0.11: `InDeg` against `GAT-P` 0.653.

These analyzer effects are as large as the representation effects the paper reports (+0.075 to +0.113 for GNNs; +0.204 for centrality, which also changes the analyzer from application-layer to projection betweenness). The data support a narrower thesis: *on this oracle, counting dependents on a representation that exposes subscriber→publisher arcs is as good as anything tried.* Please restate the thesis accordingly in §1.2, §8.1 and §9.

### M7. Registration: the primary registered test failed, and the amendment chain erodes the confirmatory value.

**The primary test.** The registered primary contrast, `HGT-QoS` vs `Topo-QoS`, failed: +0.069, p = 0.266 (GPU sweep Holm 0.303). A pre-registered study must report its primary outcome prominently. The abstract does not mention it; it leads instead with exploratory families (Amendments 7, 9 and 10) and reports the hybrids, which are secondary registered contrasts against a comparator the count beats on 12 of 12 folds.

**The amendments.** Ten amendments were made in twenty days:
- Amendments 7, 9 and 10 were each registered within hours of the review or result that motivated them.
- They were written in the authors' own repository (Amendment 10 at 11:36, manuscript updated at 11:54 on the same day).
- Amendment 9 records that its comparators (`InDeg`, `Reach`) and its counterparts were already known.
- Amendment 10's two referee comparisons were identities known before registration.

The authors are admirably open about all this. Still, "registered before their runs" carries less evidential weight here than the abstract and §1.3 imply.

**Requested:**
- Report the primary registered outcome in the abstract.
- Use "registered" only for the confirmatory omnibus, and call Amendments 7, 9 and 10 "planned exploratory analyses".
- Timestamp any further amendment with a third party (e.g. OSF), as requested in round 2.

### M8. Evaluation scope, external validity and special-issue fit.

- **One generator.** All training and LOSO evidence comes from one generator whose QoS–topology coupling is itself "part of the mechanism" (R2′). The generator, the rules and the oracle were designed by the same team.
- **The system models.** They are hand-authored by one author (22–41 Applications). Two depart materially from their originals: a gRPC system modelled as a four-broker pub-sub mesh, and a discovery server modelled as a broker. Calling them "models of open-source systems" overstates them. "Models inspired by" is accurate for the two RPC-derived ones.
- **The recommendation rule.** "`InDeg` for familiar architectures, `Reach` for unfamiliar ones" (§8.1, §9, Table 11) is a post hoc rule chosen on five systems and rests on a cross-harness comparison (round-2 M3, unresolved). §8.2 itself states that no measured descriptor places an architecture in a regime before deployment, so a practitioner cannot tell which one applies.
- **Top-K identification.** On LOSO, `Reach`'s Overlap@K (0.341) is *below* both closed-form centralities (0.366, 0.388), and `InDeg` recovers only half of the true top-20% (0.506). A CI gate acts on a flagged set, not on ρ. The recommendation should be qualified by the identification metric, which tells a different story from the rank correlation.
- **Special-issue fit.** The manuscript never argues its fit to a special issue on *AI techniques*. The defensible framing is an empirical study with a negative result for graph learning on architecture-level impact ranking, supported by the literature on simple baselines in SE and ML. Examples:
  - Fu and Menzies, "Easy over hard", FSE 2017;
  - Ferrari Dacrema et al., RecSys 2019;
  - Errica et al., "A fair comparison of graph neural networks for graph classification", ICLR 2020.

  Please make that argument explicitly, or the editor may prefer the regular track.

### M9. Literature gaps that bear directly on the contribution.

§2 omits the closest prior work on each of the paper's three claims.

**Dependency-graph measures against simple counts.** Two directly comparable prior studies:
- Zimmermann and Nagappan, "Predicting defects using network analysis on dependency graphs", ICSE 2008;
- Premraj and Herzig, "Network versus code metrics to predict defects: a replication study", ESEM 2011, where simple metrics largely matched network measures.

These studies ask exactly the paper's question (does a sophisticated graph measure beat a simple count on a dependency graph?) and anticipate its answer.

**Architecture-level reliability risk.** Yacoub and Ammar, "A methodology for architecture-level reliability risk analysis", IEEE TSE 28(6), 2002. It combines component dependency graphs, coupling-based complexity and FMEA severity to rank components by risk before deployment. This is the closest methodological ancestor of SaG and should be positioned explicitly.

**Microservice dependency-graph localisation and fault injection.** Two examples:
- MicroRank (Yu et al., WWW 2021), PageRank on service dependency graphs;
- service-level fault injection testing (Meiklejohn et al., SoCC 2021), a pre-production alternative to cluster-level chaos engineering. It weakens §1.1's claim that chaos engineering "needs a provisioned cluster".

---

## 4. Minor Comments

**Text and consistency**
1. **§1.4 Contribution 3 and §9: the zero-shot range.** "ρ = 0.760–0.831" includes plain `GAT` (0.831), which appears in no main-text table. Table 9 and the §7.3 summary give 0.760–0.805. Use one range and name the engines.
2. **Table 6: Topo's grouping.** Topo is grouped under "Training-free, on the dependency graph", but Table 5 and §6.2 place it on the application-layer graph. Regroup or relabel.
3. **§6.3: "registered" versus "confirmatory".** The text speaks of thirteen "registered" and "confirmatory" contrasts, but also of "registered exploratory families". Define the two terms once and use them consistently (see M7).
4. **§3.4: the CDI complexity.** It is stated as O(|V|² + |V||E|), but the implementation uses a fixed-size breadth-first sample (round-2 minor 2, still open). The supplement's reproducibility finding (hash-salted tie order in that sample) belongs in §8.3, because it affects every learned result and the explanation layer.
5. **§4.3: the durability-aware rescaling.** "Substituting a durability-aware w(t) scaling moves it less still" should report a number and a supplement pointer.
6. **§5: the explanation layer.**
   - It is still listed as Contribution 4, but §8.4 states it has not been evaluated with developers or against injected faults.
   - Its Q(v) ranks at ρ ≈ 0.2–0.3 (S30).
   - §5.2 reports that a uniform prior beats the AHP weights (0.319 vs 0.200).

   Present it as a design proposal, not a contribution, or move it to the supplement.
7. **§7.1: "Label noise and inert components".** It says the active-stratum restriction "halves every predictor's correlation". Give the numbers for `InDeg` and `Reach` in the main text (M1(b)).
8. **§7.3: "the counts separate inert from active components almost perfectly".** This is true by construction (M1) and should be stated as such, not as a property of the systems.
9. **§8.1: the untyped GAT.** "The untyped `GAT` without QoS inputs transfers at 0.831 zero-shot". This plain `GAT` also passes no messages to Applications (§8.2), so it is an MLP. Say so.
10. **§8.2: overlapping analyses.** It mixes post hoc terciles defined with the labels, 96 uncorrected descriptor correlations and a recommendation ("the hybrid is therefore the safe default in distribution"). Since Table 11 no longer recommends the hybrids, this sentence contradicts §8.1. Remove it or align it.
11. **§7.4, Table 10: the forward-pass timings.** The 249-node row "still carries first-call warm-up" (26.5 ms > 16.4 ms at 499 nodes). Discard warm-up runs rather than footnoting them.
12. **Title.** "Dependency-Graph Analysis and Learning" is an improvement. "Cascading-Failure Ranking" should become "Simulated Cascade-Impact Ranking" until M1(a) is done.

**Figures and tables**

13. **Figures 1 and 3 predate the reframing** (round-2 minor 1, still open). They show "three ranking engines" and a "QoS-weighted `DEPENDS_ON` projection", with no dependency-count path, although the counts are now the headline instrument.
14. **Table 2 has an empty header row.** The four-column notation layout is hard to read. Use a two-column symbol/meaning table.
15. **Table 6 is too wide.** It has nine columns and mixes two harnesses in one column. Split LOSO and system-model results, and make the system-model column single-source (round-2 M3).
16. **§3.2 has an unnumbered sub-heading** ("Logical Dependency Projection"). The Guide asks for numbered subsections (3.2.1). The "#### Summary" headings in §7 are also unnumbered; consider italic lead-ins instead.

**Compliance with the Guide for Authors**

17. **Abstract.** 224 words (limit 250) ✓. It defines SaG but uses "ISO/IEC 25010" and ρ without expansion, which is acceptable. It should report the primary registered outcome (M7).
18. **Highlights.** 5 bullets, each ≤ 85 characters ✓. Three changes are needed:
    - write "ρ = 0.76", not "rho .76";
    - "12/12" needs a noun (folds);
    - highlight 5 promotes the hybrids, which Table 11 no longer recommends.
19. **Keywords.** Seven ✓. "Empirical study" was added as suggested.
20. **Length.** 26 single-column pages, within the 36-page guidance ✓. `LENGTH_JUSTIFICATION.md` is stale: it says 22 pages, 90 references and S1–S29, against 26 pages, 93 references and S1–S33. Update it or drop it, since no justification is required.
21. **References.**
    - Only 11 of 93 entries carry a DOI; the Guide asks for DOIs "where available".
    - Several conference entries lack page ranges, e.g. [47].
    - Reference [53] (IWSM/MetriKon, Shaker Verlag) should be checked for completeness, per the Guide's rule that every reference must correspond to a real, locatable source.
22. **Vitae.** `vitae.tex` still contains two unfilled `\vitaeTODO` placeholders (degrees, years, research interests). The Guide requires a biography of ≤ 100 words per author in an editable format.
23. **Data availability.**
    - The statement cites Zenodo DOI 10.5281/zenodo.14922108 and "511 reported figures". Round 2 was told 537. Confirm that the cited *version* DOI contains the Amendment 7, 9 and 10 artifacts. A record number of that age suggests a concept DOI or an older version.
    - The experiment-page link points to the tag `jss-submission-v4`. That tag is not present in the repository clone I reviewed, and Amendment 9 describes v4 as a manuscript version that "does not report its arms". Point the link at a tag that matches this version.
24. **Generative-AI declaration.** Present, correctly titled, placed before the references ✓. Because the replication scripts were AI-assisted, say whether AI tools were used to generate or check the reconciled figures, the adapters for the five system models, or the code that computes the labels.
25. **Graphical abstract.** Encouraged by the Guide but not provided; optional.
26. **Figures.** Supplied as separate files with logical names (`Figure_1` … `Figure_5`) ✓.

---

## 5. Recommendation

**Major Revision.**

**Justification.** This is a careful, unusually transparent study, and its central negative result about graph learning has real value. It is not acceptable in its current form, for four reasons.

1. **Circularity (M1, M2).** The headline accuracy is agreement with a simulator that builds the same subscriber→publisher and app→library graph the counts read. That simulator is cheaper to run than the predictor, from the same inputs. The one experiment that would make the finding independent of this construction (scoring the rankers on I_dyn and I_comp) was requested in the previous round, needs no training for the counts, and is again deferred.
2. **Unsupported attributions in the abstract (M3, M4).** "Derivation is decisive" rests on a total-degree strawman; the honest comparator is identical to `InDeg`. "QoS helps learned engines" rests on columns that are a weighted dependent count, and its registered test is not significant after correction.
3. **Conclusions drawn where the mechanism was inert (M5, M6).** The typing conclusion comes from engines that could not pass messages to the scored nodes. The typed arm on the working substrate failed to train, and the stated thesis contradicts the analyzer effects in the paper's own tables.
4. **Reporting of the registered study (M7).** The primary registered outcome is absent from the abstract.

**Conditions for a favourable next round:**
- **(i) M1(a).** Counts, `Topo-QoS` and at least `GAT-P-QoS` scored on I_dyn and I_comp, with the active stratum reported.
- **(ii) M4(a).** The count-versus-contract control on the QoS node columns.
- **(iii) Revised claims.** Abstract, highlights, Contributions 1–3, §1.2 and §9 restated to match M3–M7.
- **(iv) Condensed engines.** The raw-multigraph learned-engine exposition condensed, and the special-issue fit argued.

None of these requires new infrastructure. Given that this is the third round, I would advise the editor that if (i) is again deferred, the paper cannot establish its main claim and should be redirected to the regular JSS track as an empirical negative-result study.
