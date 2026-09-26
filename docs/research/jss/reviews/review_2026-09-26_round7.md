# Referee Report — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems"

**Manuscript:** *Software-as-a-Graph: Dependency-Graph Analysis and Learning for Pre-Deployment Simulated Cascade-Impact Ranking in Publish–Subscribe Systems*
**Material reviewed:** `manuscript.md` (generated from `latex/`, 25 pp. PDF), supplementary section list (S1–S33), `highlights.tex`, `vitae.tex`, `LENGTH_JUSTIFICATION.md`, JSS Guide for Authors. Table 7 values were spot-checked against `results/independent_oracle_evaluation.json` and match.
**Date:** 2026-09-26 (round 7)

---

## 1. Summary

The paper argues that publish–subscribe middleware hides the paths along which failures cascade. It proposes SaG, which turns deployment manifests into a typed multigraph and derives from it a `DEPENDS_ON` dependency graph. The central claim is that once this graph exists, simple training-free counts rank simulated cascade impact well: in-degree (`InDeg`, which equals pub-sub afferent coupling) and transitive reach (`Reach`). The paper also claims these counts match or beat graph neural networks.

The evaluation compares closed-form, count-based, GNN (GAT/HGT) and hybrid rankers. It uses leave-one-scenario-out (LOSO) evaluation over 12 generator-produced synthetic architectures and zero-shot evaluation on 5 hand-authored models inspired by open-source systems. Ground truth comes from a reachability simulator (`I*`). A discrete-event simulator (`I_dyn`) and a multi-criteria simulator (`I_comp`) serve as secondary oracles.

The main contribution is an empirical, largely negative result for AI techniques: `InDeg` reaches ρ = 0.764 and beats centrality on 12/12 folds, and dependency-graph GNNs only reach parity (0.748). A preregistered protocol, a reconciled artifact bundle and an ISO/IEC 25010 explanation layer complete the package.

## 2. Overall Impression and Assessment

**Strengths.** The paper is unusually transparent for this area:

- the analysis plan is preregistered and every amendment is logged;
- simulation inputs are kept separate from predictor inputs, and a CI test enforces it;
- the paper reports the null primary contrast (HGT-QoS vs. Topo-QoS, p = 0.266) instead of burying it;
- capacity-matched 2×2 controls separate typing from the QoS channel;
- it admits the anti-conservatism of Wilcoxon tests across dependent folds, the single-modeller threat and metric drift;
- a script mechanically reconciles the manuscript's numbers against the artifacts.

A result that simple counts match GNNs is worth publishing, and JSS explicitly welcomes negative results.

**Originality.** Modest. The quantity that wins, `InDeg`, is by the authors' own admission afferent coupling / Absolute Importance of the Service (AIS) / service fan-in [57–59]. For an Application it is also exactly the raw two-hop subscriber count (§6.2). What is new is (i) the claim that a formal derivation is needed to compute it, and (ii) the benchmark. Point (i) is weak (see M2). The derivation rules are also only partly exercised: Rules 2, 3, 4 and 6 are never evaluated.

**Significance.** Limited by one question the paper never answers: why predict `I*` at all? The simulator runs from the same manifests as the predictors, and it is faster than the SaG analysis (§7.4). The paper therefore shows that a proxy agrees with a simulator the practitioner could simply run. It does not show that either predicts real outages.

**Methodological soundness.** The statistics are careful, but construct validity is the weak point:

- `InDeg` is the first propagation wave of `I*`, and the paper's own "analytic ceiling" is a first-order expansion of the oracle;
- the "independent" oracle `I_dyn` is sampled on n = 30 lexicographically chosen nodes per fold, and its first-order effect is the same subscriber loss;
- the latency and "sustainability" claims in the abstract and highlights are not measured for the recommended method.

**Special-issue fit.** Borderline. The AI contribution is a set of learned engines that are not recommended (Table 11 drops them), plus one untuned HGT configuration. Sustainability is argued only from CPU wall-clock, not from energy or carbon. The paper would fit better as a regular JSS article or as an explicit negative-results benchmark, unless the AI and sustainability angles are strengthened.

**Q1 suitability.** The engineering and bookkeeping are publishable. The framing, the claim strength and several unsupported headline statements currently fall short of a Q1 standard.

## 3. Major Comments

**M1. The prediction target is computable from the predictors' own inputs, so the practical value of predicting it is unestablished.**
`I*` runs on `G_structural`, which is built from the same manifests SaG reads. Its QoS severity ladder also comes from the manifests. Table 10 and §7.4 show that direct simulation is 2.0–17.7× (median 5.6×) *cheaper* than SaG's analysis stage.

The "two-tier taxonomy" of §8.1 argues that in Tier 1 "simulation harnesses [are] uncalibrated or unrunnable". That holds for `I_dyn`, but not for `I*`, which needs no operational profile. In the setting the paper targets, the ground truth itself is therefore available, cheap and exact.

The authors must state what a practitioner gains from `InDeg` or a GNN over running `I*`. Two credible answers would be:

- (a) `I*` is itself a proxy for real outages, in which case some external validation is required (post-mortems, fault-injection on a running benchmark such as Train-Ticket or DeathStarBench, or published incident data);
- (b) the counts serve a different purpose (e.g., interpretability, or incremental evaluation under a changing manifest), in which case that purpose must be evaluated.

As it stands, RQ1–RQ3 measure agreement between two functions of the same graph.

**M2. The claim that the derivation "unlocks" or "makes computable" `InDeg` is not supported. The fair baseline is missing.**
§6.2 states that for Applications `InDeg` equals the raw two-hop subscriber count (pub → topic ← sub). That count is a trivial query on the *raw* multigraph, with no `DEPENDS_ON` projection needed. Yet the abstract, §1.2, contribution 1 and §9 attribute the 0.764 to "deriving this explicit dependency projection".

The only measured value of the derivation is Rule 5 adding +0.058 to `Reach` (Amendment 10). No such figure is given for `InDeg`.

Meanwhile the "closed-form centrality" baselines are:

- betweenness (Topo, 0.349), which theory does not suggest as a proxy for downstream impact;
- a Topo formula whose articulation term "reads zero for every node" because of an implementation defect (§6.2).

In-degree, reverse PageRank or reverse reachability on the raw multigraph (possibly with topic-mediated two-hop traversal) are the natural closed-form baselines. They are absent from Table 6.

Please add raw-graph two-hop fan-in, raw-graph reverse reachability and reverse PageRank as baselines, and restate the contribution in line with what they show. If raw two-hop fan-in equals `InDeg` exactly (as the paper itself states), the +0.211 is an advantage of *choosing the right metric*, not of the derivation.

**M3. The circularity is acknowledged but not resolved, and the abstract and conclusion claim it is resolved.**
§4.4 correctly states that the first wave of `I*` is exactly what `InDeg` counts. Eq. 5 (Σ |sub(t)| / |pub(t)|) is a first-order expansion of the oracle, and it reaches 0.808. This makes the synthetic ρ = 0.764 largely a check that the count reproduces the simulator's propagation rule. Yet the abstract says "without architectural circularity", and §9 says the results "confirm … genuine distributed message dynamics rather than reachability simulator artifacts".

The evidence offered for independence is `I_dyn`, and it does not bear that weight:

- (i) `I_dyn` runs on the same structural graph, and its first-order effect — surviving consumers losing a failed publisher's messages — is again subscriber loss;
- (ii) consistent with this, `I*` itself agrees with `I_dyn` at only 0.627, and `InDeg` (0.610) and Analytic `I*` (0.636) sit at the same level, so on `I_dyn` all first-order subscriber counts are indistinguishable from the reference oracle;
- (iii) `I_dyn` covers only n = 30 nodes per fold, taken as the first 30 lexicographically sorted IDs. §8.3 itself concedes this may correlate with generator roles, and ρ over 30 points has a per-fold 95% CI of roughly ±0.3;
- (iv) `I_dyn` has a test–retest floor as low as 0.74, which is never used to disattenuate or bound the reported agreement;
- (v) learned engines are never evaluated on `I_dyn` or `I_comp`, so "GNNs match the count" holds only on `I*`.

Required changes:

- use a random stratified sample, or the full population on the smaller folds, and report per-fold n and CIs;
- report partial correlations of each ranker with `I_dyn` controlling for `I*` (or for Analytic `I*`), to show signal beyond the first-order term;
- evaluate the learned engines on `I_dyn`/`I_comp`;
- remove "without architectural circularity" and "genuine … rather than … artifacts".

**M4. `I_comp` contradicts the recommended default, and this is reported selectively.**
On `I_comp`, `Topo-QoS` (0.702) beats `InDeg` (0.650), and `InDeg` wins only 3/12 folds (artifact: `wins_vs_topoqos = 3/12`, p = 0.110). The abstract lists `I_comp` ρ = 0.650 as *support* for dependency counting. §3.2 states that "unweighted topological structures drive performance across both static and dynamic simulations". Table 11 recommends `InDeg` "across both static and dynamic failure modes".

The honest reading is oracle-dependent: counts win on reachability and queue-flow oracles, and QoS-weighted betweenness wins on the multi-criteria oracle. The abstract, highlights and Table 11 must reflect this. Also define `I_comp` precisely in the main text (weights, dimensions, tiers). One sentence is not enough for a quantity used to rank the recommended methods.

**M5. Confirmatory versus exploratory: the headline comes from post-primary amendments.**
The only pre-specified primary contrast (HGT-QoS vs. Topo-QoS) is null. The abstract, title, §1.2 and §9 lead with `InDeg`, `Reach`, GAT-P-QoS and Analytic `I*`. These are Amendments 7, 9 and 10 plus a post hoc ceiling, registered after the primary sweep had produced results. The Holm p = 0.002 for `InDeg` is computed within the Amendment 7 family, not within the 13-contrast omnibus.

§1.3's "planned exploratory investigations … revealed the true driver" is HARKing-adjacent language. Please:

- state in the abstract that the registered primary contrast was null and that the dependency-count results are exploratory;
- give the amendment registration dates relative to data access in the main text, not only in §S24;
- drop causal wording ("true driver", "unlocks").

**M6. The RQ4 cost claims, and the "sustainability" framing central to this special issue, are not supported by measurement.**
"Sub-millisecond" appears in the abstract, §1.1, §8.1, §9 and highlight 5, but no timing of `InDeg` or `Reach` is reported anywhere in the manuscript or supplement. Table 10 times only the HGT pipeline.

It is also unclear whether the `DEPENDS_ON` derivation, which `InDeg` requires, is part of the "Analyze" stage (239 s at 2,000 nodes). If it is, `InDeg` inherits that cost and "sub-millisecond" is wrong by five orders of magnitude.

Contribution 5 says "milliseconds", while the abstract and highlight 5 say "sub-millisecond". "Computational sustainability" is asserted from wall-clock time alone.

Please:

- time derivation + `InDeg`/`Reach` and `I*` on the same inputs and hardware, with scaling up to at least 10⁴ components;
- separate the derivation cost from the feature-extraction cost;
- either measure energy (e.g., CodeCarbon or RAPL) or drop "sustainability" in favour of "latency".

**M7. The RQ2 design is confounded, and the conclusions about typing are over-generalized.**
(a) The "QoS-off" arm zeroes `w_in`, which §7.2 says is a weighted in-degree, "thereby removing the dependent count itself". The QoS main effect in Table 8 therefore mixes QoS content with the single most predictive structural signal. Yet the Figure 5(C) caption still reads "QoS inputs raise both models by about 0.07". Re-run the 2×2 with the in-degree channel kept in both arms, or relabel the factor.

(b) The claim that "parameter-heavy relational typing introduces severe optimization instability" (abstract, §1.2, §8.2, §9) rests on one untuned HGT-P-QoS configuration (width 100, loss weights 0.5/0.3/0.1 described as "standard defaults", which they are not). The paper cites Errica et al. [87] on fair GNN comparison but performs no per-arm hyperparameter search. Either add a modest, equal-budget search per arm (learning rate, width, dropout, layers) and report seed variance, or restrict the claim to "in the single configuration evaluated" everywhere, including the abstract and conclusion.

(c) The 18-D base feature block already contains in-degree. That a GAT on the same graph reaches, but does not exceed, a feature it is given is close to expected. Hybrid-GAT-P, anchored on `InDeg`, adds nothing (0.758 vs. 0.764). Please add a learner ablation without the in-degree and `w_in` features, to show whether message passing recovers the signal on its own.

**M8. The explanation layer (contribution 4) is not evaluated.**
§8.4 concedes the layer is "an uncalibrated design proposal … not validated". Its own ranking correlation (0.200–0.319, §5.2 and §S1) is below every baseline. It carries around 20 hand-set or AHP weights and a shrinkage λ = 0.70, and nothing tests them. §8.1 nevertheless recommends combining `InDeg` with the Tukey fence of Q(v).

Either evaluate the layer, for example with one of:

- agreement of FT/A/M attribution with the dimension-level components of `I_comp`;
- a remediation counterfactual (apply the named remediation in the model and re-simulate);
- a small architect study;

or move it out of the contribution list into a clearly marked "proposed" subsection or future work. An unvalidated component should not be a numbered contribution in a Q1 paper.

**M9. The zero-shot RQ3 evidence is weak, and it is reported with mixed harnesses.**
The five models (22–41 applications each) were authored by one author who also designed the simulator. Two of them are RPC systems re-expressed as pub-sub meshes with invented brokers. A bootstrap CI over five systems ([0.879, 0.991]) is not meaningful.

`Reach`'s active-stratum ρ jumps from 0.286 (synthetic) to 0.871 (system models). That suggests the hand-authored models differ structurally (e.g., chain-like) from the generator, not that `Reach` generalizes. The RQ3 summary says learned engines transfer "far above closed-form centrality", while Table 12 reports Topo = 0.891 on Online Boutique.

The Table 9 footnote shows `Reach` and `InDeg` were scored by a different harness from the other rows, and `Topo-QoS` differs between the two (0.582 vs. 0.526). Mixing harnesses in one comparison table is not acceptable; re-score all rows in one harness. "Zero-shot" and "held-out" are also meaningless for training-free rankers, and calling InDeg's 12/12 "held-out architectures" overstates it: nothing was held out.

Please:

- report per-system values in the main table;
- add structural descriptors (depth, fan-in distribution, share of zero-impact nodes) comparing the synthetic and system models;
- have at least one model independently authored by a second modeller (the single-modeller threat is noted but not mitigated);
- rename the protocol for training-free methods to "per-scenario evaluation".

**M10. Critical-set identification contradicts "highly accurate".**
`InDeg`'s Overlap@K is 0.504 and `Reach`'s is 0.344. §8.1 itself concedes "approximately 50% false positives and false negatives" at the top-20% gate, which is the operational use case in the paper's CI/CD framing. The prescribed safety margin ("inspect the top 30–35% to encompass 80–90% of true critical hubs") is not backed by any reported analysis. I could find no such curve in the manuscript, supplement or experiment pages.

Please report recall@k curves (or cost–recall curves) for each ranker, derive the safety margin from them, and state how ties are broken in both Spearman and Overlap@K. Integer counts produce many ties, and §8.1 attributes the churn to "tie-breaking".

**M11. Reproducibility drift is larger than several reported effects.**
§8.3 reports metric drift of up to 0.172 (±0.041 on HGT-QoS) across revision cycles and devices. That exceeds the primary contrast (0.069), the QoS effect (0.073) and the Hybrid-HGT gain (0.103). Four supporting artifacts also lack provenance (Data Availability).

Report whether each learned-engine cell in Tables 6, 8 and 9 reproduces from the tagged commit on a stated device. Give seed-level spread for every learned row, not just HGT-P-QoS.

**M12. Overclaiming and verbosity.**
The prose frequently outruns the evidence:

- "Crucially" ×8, "essential" ×4, "genuine" ×4, "rigorous(ly)" ×5;
- "highly accurate", "outstanding predictive accuracy", "fully legible", "exceptional accuracy", "decisive advantages";
- "Occam's razor", "providing an essential empirical benchmark".

With ρ_{>0} = 0.516 on active components and Overlap@K ≈ 0.5, "moderate-to-strong rank agreement with a reachability simulator" is the accurate description.

The headline numbers (0.764, +0.211, 12/12, 0.938, 0.748, 5.6×) are repeated in the abstract, §1.2, §1.4, the §7.1 summary, §8.1 and §9. Content that is not needed for the main claim could shrink or move to the supplement:

- the introduction's thesis paragraph and contributions list overlap almost completely;
- §8.1 (two-tier taxonomy) and §9 restate the same claims;
- the QoS weighting apparatus (§3.2, 16-D edge vector, AHP) is shown not to help (§3.2, §7.1);
- Rules 2–4 and 6 are unexercised;
- 13+ predictors are compared, when the recommended set is two counts and one GAT.

## 4. Minor Comments

1. **Abstract length (Guide: ≤ 250 words).** The abstract is about 266 words even with each math expression counted as one word, so it exceeds the limit. It also uses non-standard abbreviations (`InDeg`, `I*`, `I_dyn`, `I_comp`, ρ_{>0}, PR-AUC), which the Guide asks authors to avoid or define. It should also mention that the primary contrast was null (see M5).
2. **Highlights.** Lengths comply (77–81 characters, 5 bullets). Highlight 4 ("confirms robust predictive validity") and highlight 5 ("Sub-millisecond … matches complex GNNs") restate unsupported claims (M3, M6).
3. **Dangling cross-references:**
   - "Supplementary Table S36" (§7.3): the supplement ends at S33.
   - "`HGT-QoS-U`, §7.2" and "`GBM-Feat` … (§7.2)" in §8.2 and Table 11: neither appears in §7.2, and GBM-Feat is never introduced in the body before Table 11.
   - Please audit all §S references.
4. **Equation numbering** skips (4): equations are tagged 1, 2, 3, 5. The Topo/Topo-QoS formula in §6.2 is untagged.
5. **Registered comparator defect.** The articulation-point term of Topo/Topo-QoS "reads zero for every node". Keeping a buggy comparator "as registered" is defensible for the confirmatory test. It should be labelled as a defect in Table 6, not only in prose, and the corrected version (§S22) given at least one row.
6. **Averaging ρ across folds.** State whether means are arithmetic or Fisher-z, and whether the bootstrap resamples folds or nodes. With 12 folds of very different size (26–300 apps), consider size-weighted summaries as a sensitivity check.
7. **Markdown generator artifacts** in `manuscript.md`: `<span id="sec:rq1-loso" …>[sec:rq1-loso]`, `[tab:9b]`, and unnumbered "#### Summary" headings. The Guide requires numbered sections and subsections. Check that the PDF uses numbered headings or plain italic lead-ins.
8. **"Formal" derivation.** Table 3 is a rule table, not a formal definition. Either give the rules as set-builder definitions with a short lemma proving `InDeg` = two-hop subscriber count (including the transitive `USES` case), or drop "formal". A CI test is not a proof.
9. **Terminology.** Standardize "sub-millisecond" versus "milliseconds". Drop "genuine" before `I_comp`. Avoid "zero-shot" and "held-out" for training-free methods. "Hybrid" means a learned correction on a Topo-QoS prior in one place and an `InDeg` prior in another (Hybrid-GAT-P); give the latter a distinct name.
10. **Table 5** is seven columns wide, with a free-text "Empirical Role" column. **Table 11** has multi-sentence cells. Both will typeset poorly in single-column elsarticle; move the prose to the text.
11. **Table 7.** Add CIs and fold-win counts for every cell, not just those quoted in the text. Note that ρ and ρ_{>0} coincide on `I_comp` by construction, and drop the redundant columns.
12. **§4.3.** "`probe.labeled_node_ids`" and file paths such as `tests/test_independence_guarantee.py` and `results/qos_attribution_controls.json` are implementation details. Move them to the replication pages.
13. **§3.2** says `CR = 0.016`, while §2.3 points to back-filled matrices (§S4). State in the body whether the shipped topic-QoS matrix was elicited independently or back-solved.
14. **References:**
    - [3] "Data distribution service (dds)": fix capitalization (DDS).
    - [65]–[68] "(square)" should read "(SQuaRE)".
    - [56] Santos et al., TSE 47(10) is dated 2019 but the issue is 2021 (online 2019).
    - [82] (ICPE 2025) lacks pages or DOI.
    - [89] is a Springer book, not a journal volume.
    - Several conference entries lack DOIs; JSS asks for complete data.
15. **Length justification** says 24 pages, but the compiled PDF is 25 pages. The substance is fine (< 36).
16. **Graphical abstract** is encouraged by the Guide and currently absent. Figure 2's running example would adapt well.
17. **Data statement.** Four artifacts "predate provenance stamping". Either regenerate them or state which reported numbers depend on them.
18. **§1.1** "demonstrate that simple, training-free structural counts accurately rank cascade vulnerability in sub-milliseconds" belongs in results, not in the motivation.
19. **Figure 1 caption** refers to a "triage" route into the explanation layer that the body never defines.

## 5. Recommendation

**Major Revision.**

The study is careful in ways the field rarely is: preregistration, amendment logging, a reported null primary, matched controls and mechanical reconciliation. Its central empirical message, that a simple count matches GNNs, is a legitimate and useful negative result for JSS.

Acceptance in its current form is not possible, for three reasons:

- **Construct validity.** The headline predictor is the first-order term of the oracle it is scored against (M3). The oracle can itself be run from the same inputs at lower cost (M1). The claimed benefit of the derivation is not separated from the benefit of choosing afferent coupling as the metric (M2).
- **Headline claims not backed by measurement.** "Sub-millisecond" and "sustainable" have no timing or energy measurement (M6). "Without circularity" rests on a 30-node lexicographic subsample (M3). `InDeg` is presented as robust across failure modes although it loses 9/12 folds on `I_comp` (M4). Typing is called optimization-unstable on the basis of one untuned configuration (M7).
- **Scope and presentation.** The headline results are exploratory amendments presented as the core finding (M5). An unevaluated explanation layer is listed as a contribution (M8). The abstract exceeds the Guide's word limit.

Most of these can be fixed by reframing plus a bounded set of additional analyses, so this is not a rejection:

- raw-graph fan-in and reverse-reach baselines;
- a stratified, larger `I_dyn` sample with partial correlations;
- learned engines scored on `I_dyn`/`I_comp`;
- measured counting latency;
- recall@k curves;
- an equal-budget hyperparameter search, or narrowed typing claims.

If the revision cannot show value beyond reproducing `I*` (M1), the authors should reposition the paper as a negative-results benchmark study, where its rigour would be a genuine asset.
