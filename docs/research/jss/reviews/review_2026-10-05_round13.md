# Referee Report — Round 13 (post-condensation manuscript)

**Manuscript:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?*
**Venue:** Journal of Systems and Software — Special Issue on AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems
**Reviewed version:** [manuscript.md](../manuscript.md) at `9f304cf9` (24 pp. `elsarticle` preprint, 3p; supplement 46 pp.)
**Date:** 2026-10-05

---

## Summary

The paper asks when graph learning adds value over analytical rankings for predicting cascading-failure impact in publish–subscribe architectures, before deployment, from deployment manifests alone. It derives an explicit `DEPENDS_ON` dependency graph from a typed pub-sub multigraph. It then compares a training-free betweenness baseline, dependency-count "references", hybrid and pure GNNs (HGT, GAT, GIN), and gradient-boosted models. Labels come from three in-house simulators under leave-one-scenario-out (LOSO) over twelve synthetic architectures, plus zero-shot evaluation on five hand-authored models of open-source systems. The main contribution is a carefully controlled negative result: the registered primary contrast is null. A GAT on the derived graph (ρ = 0.748) cannot be told apart from counting direct dependents (0.764). For the expensive queue-flow simulator, a post hoc rate-weighted first-order formula (0.830) matches or beats learned approximations. The authors also propose a "reference criterion" for marking rankings that restate a simulator's propagation rule.

## Overall Impression & Assessment

**Originality.** Moderate. Deriving dependencies from pub-sub manifests extends the authors' RASSE 2025 paper [32]. In practice the new derivation is Rule 5 (App→Library), since Rules 2, 3, 4 and 6 are defined but never exercised. The idea behind the reference criterion is sound but not new: compare a learner against the label-generating rule. It is close to existing ideas on shortcut learning and simulation validation, which the paper cites [52–54]. The most original part is the set of controls that pull apart representation, direction, degree information, aggregator and oracle-aligned features. That control design is unusually thorough for this literature.

**Significance.** Potentially high as a cautionary benchmark. Papers that apply GNNs to architecture criticality seldom test against the simplest rule-aligned baseline, and this one shows why they should. The significance is limited by the design itself, though. All three oracles are well approximated by low-order functions of the dependency graph, and the authors say so: "no regime where learning could exceed one was exercised" (Abstract; §7.4). The title asks *when* learning improves ranking, but the study supplies only one side of that boundary. It shows where learning is not needed and never shows where it is.

**Methodological soundness.** Transparency is exemplary: a pre-specified plan, logged amendments, Holm families, a Nadeau–Bengio correction, seed censuses, mechanical reconciliation of 1,815 figures, and candid disclosure of a defect in the registered comparator. Rigor is uneven, however:

- the learned arms are under-resourced and in one key comparison under-informed;
- the dependency graph that drives the representation gain is itself oracle-aligned, and no control addresses this;
- all ground truth is simulated by the authors' own simulators, and no real failures are used;
- most headline claims are exploratory or "registered" only after the primary null was known.

**Suitability for a Q1 journal and this special issue.** The topic is in scope (reliability, performance surrogates, sustainability), and JSS explicitly welcomes negative results. In its current form, however, the paper reads as an extremely careful audit of one benchmark rather than generalizable knowledge about when AI techniques help. That gap can be closed, either by adding a regime where learning should help or by narrowing the title and claims to what was tested.

---

## Major Comments

### M1. The central research question cannot be answered by this design: there is no positive control.

The title and §2.4 frame the study as identifying *when* learning adds value. Yet §7.4 concedes that "the benchmark cannot exhibit a regime in which learning exceeds an aligned analytical ranking", and §7.5 (items 4–5) lists exactly that regime as future work. Without a condition where learning is expected to succeed, the reader cannot tell apart three explanations:

- (a) learning adds nothing for this problem class;
- (b) these oracles are too simple;
- (c) these learners, at this training budget (11 small graphs, one fixed configuration), are too weak.

The paper's own evidence fits (b) and (c) at least as well as (a).

*Request.* Add at least one oracle with a non-first-order mechanism that the authors can control: backpressure onto publishers, retry amplification, heterogeneous service capacity, or queue coupling across shared brokers. Show that the analytical references degrade there and report what learners do. If learners still fail, the negative result becomes far stronger. If they succeed, the paper delivers the "when" its title promises. If this is infeasible, retitle (for example, "…: A Benchmark Where Dependency Analysis Suffices") and remove "when does" language from the abstract, §1.4 and §8.

### M2. The representation gain is confounded with oracle alignment of the graph itself.

§3.5 and F11 remove oracle-aligned *features*, but the oracle-aligned *structure* stays. By Remark 1, every edge into an Application in the `DEPENDS_ON` projection is exactly a Rule-1 subscriber→publisher edge. That is the support of I\*'s first propagation wave, which the paper itself classifies as a reference (§4.4). A GNN on this graph therefore receives the reference's support as its adjacency. F11's "GIN-P-QoS-const reaches 0.719 from structure alone" shows exactly this: a sum aggregator over the projected adjacency recovers the in-degree.

The claim that "making dependencies explicit enables graph learning" (§1.3; §7.1) is then hard to separate from "handing the learner the reference's support lets it approximate the reference". By the logic of the reference criterion, the derived graph is itself an order-1 artifact of I\*.

The cross-oracle evidence supports this reading:

- On I_dyn, GAT-P-QoS (0.615) falls below InDeg (0.664) and below the raw-graph rankers' partial correlations (Table 6).
- On I_comp it collapses to 0.274.

The dependency-graph advantage does not survive a change of oracle.

*Request.* (i) State explicitly that the projection's edge set is oracle-aligned for I\*, and soften the representation claim to "for oracles whose first wave is the projection's in-neighborhood". (ii) As a control, train on a projection whose edges are *not* the I\* support, such as a Rule-5-only or library/host-mediated projection, or a degree-preserving rewiring of the Rule-1 edges. This separates "explicit dependencies" from "the oracle's adjacency". (iii) Report the F8/F11 contrasts on I_dyn as well as I\*.

### M3. The learned-versus-analytical comparison on I_dyn is not like-for-like.

The headline queue-flow claim compares Eq. 7 (0.830) with learned approximations (Abstract; Table 6; F12). Eq. 7 reads declared publication rates. The GNN trained on I_dyn "received neither rates nor payloads" (§6.1, end of "What declared rates add…"). The only learner that *did* receive rates, GBM-P-QoS→dyn, reaches 0.799 alone and 0.830 when Eq. 7 is added as a column (+0.000). That is the expected outcome when a tree ensemble with 11 training graphs gets a near-sufficient statistic.

Two further issues:

- Eq. 7 was formulated post hoc by the simulator's authors after the I_dyn results were seen (§5.2), so it uses privileged knowledge of the simulator internals that no learner had.
- Table 6 mixes aggregation conventions. GNN rows are 5-seed prediction ensembles, while the I_dyn-trained rows (★) are per-seed means. GAT-P-QoS trained on I\* (0.615, ensemble) therefore appears *better* on I_dyn than GAT-P-QoS→dyn trained on I_dyn (0.598, per-seed). This is at least partly an artifact.

*Request.* (i) Retrain GAT-P-QoS→dyn with the same rate and payload node and edge inputs the GBM receives, and report it under the same aggregation as every other row. (ii) Report every Table 6 row both per-seed and as an ensemble, or use one convention throughout. (iii) Report per-fold *headroom*, meaning the noise ceiling √r minus Eq. 7's ρ, for each fold. Where Eq. 7 is already near the ceiling, "learning adds nothing" means nothing. The ceilings (0.89–0.996) against 0.830 suggest headroom on some folds, and that is where the learners' failure is informative.

### M4. Negative claims about learning need well-tuned learners; here they rest on one fixed configuration.

The registered nested hyperparameter selection was replaced by a single configuration fixed "before the first sweep" (§4.2; §5.3), and its provenance is not stated. When the registered rule was run later, it moved HGT-QoS by +0.055 and GAT-P-QoS by −0.054 (§6.2). Other sources of variance are just as large:

- Node order alone adds about 0.044 per fold (F13/17b).
- HGT-P-QoS has a within-fold seed SD of 0.254 and anti-correlated seeds (§7.2).
- Some reported values drifted by up to 0.172 across revision cycles and compute devices (§7.4).

These variances are as large as, or larger than, the differences the conclusions rest on. GAT-P-QoS versus InDeg is −0.016. The "no learned model exceeds…" claim needs learners tuned with a budget comparable to the effort spent on the analytical side. The paper cites exactly these pitfalls itself [31, 80, 81].

*Request.* (i) State how the fixed configuration was chosen and whether any of the 12 scenarios informed it. (ii) Run a modest, nested, per-fold search for at least GAT-P-QoS, GIN-P-QoS and HGT-P-QoS, and report best-of-protocol alongside the fixed configuration. (iii) Either fix HGT-P-QoS (learning-rate warm-up, normalization, seed screening on validation loss) or drop it from the main text, since §7.2 already concedes it is uninformative about typing.

### M5. The early-stopping and loss design introduce avoidable bias.

(i) Early stopping uses a 20% *node-level* split of the largest training scenario (§4.2). The validation nodes share a graph, and therefore message-passing neighborhoods, with training nodes. This is a transductive selection inside an inductive protocol, and it favors whichever scenario happens to be largest in the fold. A leave-one-training-scenario-out validation would be consistent with LOSO.

(ii) ListMLE breaks ties among the 31% zero-impact Applications in *generator creation order*. If the generator creates components in a role-correlated order, this leaks label information. The 17b result does not settle the question: the published order (0.748) beats all three permutations (0.712, 0.740, 0.734), which is the pattern a favorable order would produce. A tie-aware loss (Plackett–Luce with ties, or simply dropping intra-tie pairs) is cheap and appears only under "further extensions" (§7.5). It should be done now, because the main GAT-P-vs-InDeg comparison (0.016) sits inside the node-order spread.

### M6. The reference criterion is post hoc, loosely defined, and applied asymmetrically.

The criterion was "defined post hoc, after simple dependency counts performed strongly" (§4.4), yet it is listed as Contribution 2 and determines which rankers may carry contrasts. Its definition ("what O computes when truncated after k waves, *with stated simplifications*") leaves room for interpretation:

- `Reach` is classed as the "unweighted support of the *untruncated* cascade", which is not a truncation at all.
- InDeg is "the support" of the first wave rather than its computation.
- For I_comp, Topo-QoS and raw degree are "term-aligned" and "read with the same caution". Yet Topo-QoS keeps a contrast and is *recommended* in Table 10, while InDeg on I\* is barred from contrasts.

Readers need a decision procedure that a third party could apply to a new simulator and reach the same classification.

*Request.* (i) Give a formal, checkable definition: which simplifications are admissible, and whether "support" and "untruncated" count. (ii) Apply it identically across the three oracles. (iii) Demote it from a headline contribution to a reporting practice unless it is validated on at least one external simulator-labeled benchmark from the literature.

### M7. The status of the evidence is weaker than the presentation suggests.

Both confirmatory contrasts are null. Everything in the abstract, the highlights and Table 10 is "registered secondary" or exploratory. The "registered secondary" arms were committed *after the primary null was known* (§5.3), across 17 self-hosted amendments. The plan is stored in the authors' own repository with commit timestamps, not lodged with a third-party registry. Calling these results "registered" lends them confirmatory weight they do not have. Contrast families were also chosen by the authors: Holm is applied "within its family", and the omnibus correction covers 13 "decision-bearing" contrasts.

*Request.* (i) Relabel post-null arms as "protocol-logged post hoc" (or similar) throughout. (ii) Add a single main-text table listing every claim in the abstract and highlights with its status, its registration date relative to when the data were seen, its family, and its corrected p. (iii) Apply the exploratory label to every exploratory number in the abstract, not only one. (iv) Commit, as the paper's strongest forward-looking element, to the confirmation corpus of §7.5 item 2. Ideally, generate it and report it in this revision, since the generator and harness already exist.

### M8. External validity: every target is simulated, and every "real" system is a single-author stylization.

No ranker is validated against an observed failure. The 12 LOSO folds come from one generator family. The five "open-source" models were hand-authored by the first author from documentation, with brokers, QoS, code metrics and hosts "partly assumed", and two of them are RPC systems recast as pub-sub. RQ3 is therefore a transfer test across one modeler's style, as the authors acknowledge. On these models `Reach` reaches 0.938 and ρ>0 = 0.871, so I\* there is essentially transitive reach. The stylized models appear close to trees or chains, which makes RQ3 an easy and unrepresentative target.

*Request.* At minimum: (i) extract at least one model automatically from real artifacts. The paper itself names HAROS/ROSDiscover for ROS 2 launch files and docker-compose/Kubernetes manifests for Online Boutique and Train-Ticket. (ii) Have a second modeler independently author at least two of the five systems and report agreement. (iii) Ideally, run even a small fault-injection campaign on Train-Ticket [33] (which ships with replayable faults) to see whether any of I\*, I_dyn or Eq. 7 orders observed impact. One small real-failure data point would greatly raise the paper's value for this special issue.

### M9. The cost and sustainability analysis (RQ4) is too coarse for a sustainability special issue, and partly mis-specified.

(i) Energy is estimated as wall-clock time × the package's 28 W *base* power (§6.4; §7.4), but the I_dyn figure multiplies **CPU-hours** (12.7) by package power. If the sweep ran on several cores in parallel, which is likely on a 14-core i7-1370P, this over-counts by up to the degree of parallelism. The 355.6 Wh figure, which supports the "expensive oracle" argument, may then be inflated by close to an order of magnitude. State the wall-clock time and core count, or use per-core power.

(ii) Several comparisons are true by construction: a count is cheaper than a simulation, and a 2.5 ms count is cheaper than a 1.46 s feature extraction. They add little. The informative quantity is the per-architecture cost of I_dyn at realistic CI cadence, roughly 1 CPU-hour per architecture and parallelizable, against the decision value of its ranking.

(iii) Most of the feature cost (88–91%) is articulation/CDI, which is a choice of feature set. Report learned rankers on a cheap feature set so the cost comparison is not a strawman.

### M10. The weight of the study sits on the oracle that does not need approximating.

§1.2 and §7.3 say I\* is cheap and should simply be run ("running the oracle is therefore cheaper than approximating it with a learned ranker"). Yet the confirmatory contrast, the 2×2, F1–F11, F13, the transfer study and Figure 4 are all on I\*. The genuine surrogate-modelling problem, I_dyn at 12.7 CPU-hours, gets two learners: one deprived of rates (M3), and one exploratory contrast. For a paper about when learning adds value, the balance should be reversed. The I_dyn analysis should carry the core experimental design: the full control set, a tuned learner and the transfer study. I\* can then serve as the calibration case the introduction says it is.

### M11. Literature gaps relevant to the special issue.

The hybrid "analytical prior + learned correction" design, and the finding that the learned correction adds little once the analytical model is good, is the central topic of *gray-box performance modeling*. That literature is not cited, for example Didona et al., "Enhancing performance prediction robustness by combining analytical modeling and machine learning" (ICPE 2015), and the follow-up work on hybrid analytical/ML models. Eq. 7 is a first-order flow approximation, and the paper should relate it to classical analytical approximations of queueing networks: mean-value analysis (Reiser & Lavenberg, JACM 1980) and layered queueing networks (e.g., Franks et al., IEEE TSE 2009). Palladio [35] is cited but not engaged. Situating the result against these lines would sharpen its novelty and fit with a "performance and reliability" special issue.

### M12. Readability and verbosity.

The paper is compact in pages but very dense:

- About 13 model labels (GAT-P-QoS-min, GIN-P-QoS-const, GAT-QoS-R, Hybrid-GAT-AP, …).
- Arm codes F1–F13 that are never defined in the main text but are cited in §6.1 before Table 7 appears.
- Amendment numbers.
- Paragraphs carrying 8–12 numbers each, for example §6.1 "What declared rates add…" and §6.2 "Message passing, node order…".

§1.3 "Findings in Brief" largely restates the abstract and §6, with the same numbers. Key evidence (the 2×2 in Table S25, and the full control tables S69–S71) is only in a 46-page supplement. That supplement also carries material unrelated to the current claims: the explanation-layer sensitivity, a "diagnostic remediation card", anti-pattern detection, and HGT attention analysis. These are residues of earlier framings.

*Request.* Drop or merge §1.3 into the abstract and introduction. Define arm codes, or replace them with descriptive names. Move the 2×2 cell table into the main text. Prune the supplement to material that supports the current claims. Limit each results paragraph to the numbers needed for its claim and move the rest to tables.

---

## Minor Comments

1. **Abstract length.** 247–257 words depending on how inline math is counted, against the guide's 250-word limit. Trim to stay under the limit under any counting rule. The abstract also contains about ten numerical results; JSS abstracts should state purpose, principal results and conclusions concisely.
2. **Title.** Beyond M1, "Software-as-a-Graph:" names the framework rather than the finding. Consider leading with the finding.
3. **Contribution 1** (§1.5) presents "typed rules", but only Rules 1 and 5 are used, and Rule 1 comes from [32]. Move Rules 2, 3, 4 and 6 (Table 2, marked †) to the supplement, or state plainly that the novel, evaluated part is Rule 5.
4. **[32] and the AP defect.** Topo-QoS "continues the score family of [32]" and has a zero-articulation defect (§5.2). State whether [32]'s published results used the same implementation. If they did, the conference paper's results are also affected and should be noted.
5. **QoS machinery.** §3.2 (AHP, CR = 0.016) and the 16-D QoS edge vector take up space, yet QoS "carries no measurable signal" on either oracle. For I\* this is by design. For I_dyn, state whether the simulator actually changes behavior with RELIABLE/BEST_EFFORT or durability settings. If not, "QoS policies are inert" is a property of the simulator, not of the world. The QoS-disabled I_dyn run in §7.5 is cheap and should be done now.
6. **HGT edge input.** Table 4 gives HGT and HGT-QoS the same parameter count (434,620), although one reads a 1-hot and the other a 16-D edge vector. PyG's `HGTConv` does not consume `edge_attr`. Explain how the 16-D vector enters HGT. If it does not, the HGT "QoS channel" arm is mislabeled.
7. **"Won" as effect size** (§5.3). A win count is not an effect size. Δρ already serves; add a paired standardized effect (for example matched-pairs rank-biserial) if one is wanted.
8. **Bootstrap intervals.** Percentile bootstrap intervals over 12 folds (and over 5 systems) under-cover. Use BCa or a t-based interval on the paired differences, or state the limitation.
9. **Nadeau–Bengio under LOSO.** The correction factor assumes random subsampling with fixed n_test/n_train. Under LOSO with fold sizes of 26–300 Applications, state which ratio was used.
10. **Spearman–Brown on Spearman ρ** (§4.3) is an approximation, so note it. The bound √r assumes a noise-free ranker, which holds here, but say so.
11. **Size weighting** (§6.1). Weighting folds by |V_app| lifts Topo-QoS to the level of the raw-graph learners (0.596 vs. 0.595–0.606). This deserves more than one sentence, because it means the raw-graph learned-vs-baseline gap comes from small folds.
12. **Table 5 p-value column.** The header is "p (p_Holm)", but the GAT-P-QoS and HGT-P-QoS entries are single values with ‡. State whether 0.0068 is nominal or Holm-adjusted, and within which family.
13. **Arithmetic.** In §6.1, "0.007 above it as a five-seed ensemble (0.772)" reads as 0.772 − 0.764 = 0.008. This is presumably rounding of unrounded values; state that differences are computed before rounding, or align the numbers.
14. **Hybrid logit.** σ(z + α·logit p) with p rank-normalized to [0, 1] is undefined at p ∈ {0, 1}. State the clipping.
15. **Remark 1** says the learners' in-degree feature differs from InDeg for 940 of 1,321 Applications. This matters for the circularity argument (M2) and belongs in §3.5, not inside a remark.
16. **Figures.** The Figure 4 caption carries results (TOST p = 0.17, +0.030) that belong in the text. The Figure 1 caption refers to an "explanation layer" that is outside the study; remove it from the figure. Per the guide, check that line art meets 1000 dpi and that bitmap figures meet the 300 dpi minimum. Figure 5's grey dashed versus black dash-dotted curves need to be distinguishable in grayscale.
17. **Table 10** recommends "Run I\* directly" as the "ranker" for I\*. This is a reasonable practical point, but it shows that RQ1 on I\* has no practical stake (M10).
18. **Two-tier triage protocol** (§7.3) is unevaluated. Keep it to two sentences or move it to future work.
19. **Mixed spelling.** "favourable" (§6.2) against US spelling elsewhere ("modeling", "favor").
20. **References.** [81] is an arXiv preprint and should be marked as such or replaced by a peer-reviewed version if one exists. [2] (NetDB 2011) lacks pages and a URL. Several entries ([8], [9], [13], [14], [36], [37], [39], [42], [43], [51], [55]–[57], [64], [65], [68]–[70], [76], [82], [87], [90]) lack DOIs; the guide asks for DOIs where available.
21. **Data availability.** The repository link is pinned to `f3352f0c`, which contains the Amendment 17 experiment page. Make sure the Zenodo deposit matches that commit and the manuscript version, since the manuscript has since moved to `9f304cf9`.
22. **Generative-AI declaration.** Compliant in placement and title. Consider noting that AI-assisted analysis scripts were checked by the mechanical reconciler, since the guide stresses verifying AI-generated output.
23. **Highlights.** Compliant (5 bullets, ≤ 85 characters each). Highlight 5's "learners started from the formula add nothing" overstates F12. GBM+Eq.7 gives Δ = +0.000 [−0.013, +0.015], which is "no detectable gain", not "nothing".

---

## Recommendation: **Major Revision**

**Justification.** This is an unusually honest and well-instrumented study. Its controls (direction, degree, aggregator, oracle-aligned features, node order), its candor about the null primary contrast and the comparator defect, and its replication package exceed what most submissions in this area provide. A carefully established negative result about GNNs for architecture criticality would be valuable to JSS readers.

The paper is not acceptable in its current form, for five reasons:

1. It promises to characterize *when* learning helps, but never constructs a condition in which it could (M1).
2. Its central representation claim is confounded with the oracle alignment of the derived graph itself (M2).
3. Its key analytical-versus-learned comparison on the one expensive oracle withholds the decisive inputs from the GNN and mixes aggregation conventions (M3), and the learners are untuned at a variance level comparable to the claimed differences (M4, M5).
4. Its reference criterion is post hoc and applied asymmetrically (M6), and its "registered" evidence was committed after the primary null (M7).
5. No target is a real failure or a real extracted topology (M8).

None of these is fatal. M1–M5 can be addressed with the existing generator, simulators and harness, and M6–M8 with reframing plus one modest extraction or fault-injection exercise. Rejection would therefore be disproportionate. Minor revision would not suffice, because several headline sentences in the abstract and highlights may change once M2–M4 are addressed.
