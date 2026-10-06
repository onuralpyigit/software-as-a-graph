# Referee Report — Round 13 (2026-10-05)

**Journal:** Journal of Systems and Software — Special Issue VSI: AI4MSS (AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems)
**Manuscript:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?*
**Version reviewed:** `docs/research/jss/manuscript.md` (24-page condensed body, PR #99 + 8ec64ee0), with the highlights, cover letter and length statement in `latex/`
**Recommendation:** **Major Revision**

---

## 1. Summary

The paper derives an explicit `DEPENDS_ON` dependency graph from publish–subscribe deployment manifests and asks whether graph learning ranks Applications by simulated cascade impact better than analytical rankings computed from the same graph. It compares GNNs (GAT, HGT, GIN), hybrids that correct a closed-form prior, gradient-boosted models and closed-form "references" against three author-built simulators (reachability $I^*$, SimPy queue-flow $I_{\text{dyn}}$, composite $I_{\text{comp}}$). The evaluation uses LOSO over twelve generator-produced architectures plus zero-shot transfer to five hand-authored system models. The main contribution is a negative, carefully controlled result. Learned rankers improve on the derived graph (GAT-P-QoS $\rho = 0.748$), but they never exceed a direct-dependent count ($0.764$) or a rate-weighted first-order formula ($0.830$ on $I_{\text{dyn}}$). A post hoc "reference criterion" separates rankings that restate a simulator's rule from rankings that predict its outcome.

## 2. Overall Impression and Assessment

**Strengths.** The paper is unusually candid. Both confirmatory contrasts are reported as null, every headline result is tagged registered-secondary or exploratory, the comparator defect is disclosed, and the post hoc origin of the reference criterion is admitted. The control battery is better than most GNN-for-SE papers:
- an edge-direction control (F8);
- removal of every oracle-aligned feature (F11);
- a featureless GIN (F11);
- learners started from the analytical formula (F12);
- a node-order permutation gate (F13);
- the Nadeau–Bengio correction for overlapping folds.

The replication package is strong: a CI-checked corpus manifest, Zenodo archiving, and 1,815 figures mechanically reconciled against artifacts. JSS explicitly welcomes negative results, and the methodological warning (compare a learned surrogate against the simplest truncation of the simulator it approximates) is useful and fits the special issue's call for evaluation methodologies.

**Originality.** Moderate. Three of the "new" components are known results in new clothing:
- **Dependency rules.** Only two of the six rules are exercised, and Rule 1 comes from the authors' RASSE 2025 paper.
- **`InDeg`.** It is shown (Remark 1) to be classical afferent coupling.
- **GNN-mechanism findings.** Attention cannot count, sum aggregation can, and a featureless GIN recovers degree. These largely re-confirm known GNN expressivity theory.

The novel parts are the reference criterion and the systematic evidence that learned surrogates add nothing over aligned truncations on these oracles.

**Significance.** It is limited by the design. All three oracles are the authors' own, and the authors concede they are low-order functions of the dependency graph (§7.4). So the null is close to entailed by the benchmark, and the title question ("*when* does graph learning improve…") is never answered positively. No topology comes from a real manifest and no label comes from a real failure, so practical significance for "modern software systems" is not yet established.

**Methodological soundness.** The within-benchmark statistics are careful. But there are three structural problems:
- the learned models are data-starved and untuned, which biases toward the null;
- the reference criterion is post hoc and applied asymmetrically;
- several conclusions in the abstract, introduction and discussion are stronger than the equivalence evidence supports.

**Suitability for a Q1 venue.** Suitable *in principle* as a rigorous negative-result/benchmark-methodology paper. It is not yet suitable in its current form because of (i) the falsifiability gap, (ii) the absence of any real-system anchor and (iii) overclaiming relative to the evidence. These are addressable.

---

## 3. Major Comments

### M1. The study design cannot produce a positive answer to its own title question

All three simulators propagate failures by mechanisms whose first-order terms the authors can write down in closed form:
- $I^*$ is BFS reachability.
- $I_{\text{dyn}}$'s first-order effect "is again subscriber loss" (§4.3) and agrees with $I^*$ at $\rho = 0.711$.
- $I_{\text{comp}}$ is term-aligned with degree and betweenness (§4.4).

§7.4 states it outright: "the benchmark cannot exhibit a regime in which learning exceeds an aligned analytical ranking." The central finding is therefore a property of the oracles, which the authors also designed, rather than a finding about learning. §7.2 then lists the mechanisms where learning *might* help (backpressure onto publishers, retry storms, heterogeneous service capacity, queue congestion), but none is simulated.

**Requested:**
- **(a)** Add at least one oracle with a genuinely non-first-order mechanism. The SimPy $I_{\text{dyn}}$ harness already models queues, so finite broker capacity with backpressure to publishers, or retry amplification with heterogeneous service rates, is an incremental extension. Then show whether the order-$k$ references lose their advantage there.
- **(b)** For $I_{\text{dyn}}$, report each ranker's per-fold *noise-ceiling-normalized* agreement. Eq. 7 reaches $0.830$ against a ceiling of $\sqrt{r} \approx 0.89$–$0.996$ (§4.3). If Eq. 7 is near the ceiling on most folds, F12's null is uninformative: no method could have exceeded it. Readers need to know how much reliable variance was available for learning to capture.
- **(c)** If (a) is infeasible, retitle and reframe the paper as a benchmark-methodology/negative-result study. For example: "*…Graph Learning Does Not Exceed Aligned Analytical Baselines on First-Order Cascade Simulators*". Remove the implication that "when" has been characterized.

### M2. No real-system anchor: topologies, oracles and system models are all author-produced

- **Corpus.** Twelve topologies come from one generator. The five "open-source system models" were hand-authored by the first author from documentation, with no second modeler (§5.1, §7.4). Two of them are RPC systems re-expressed as pub-sub.
- **Oracles.** All three are author-built and none is validated against observed failures.
- **Simplicity of the system models.** `Reach` attains $\rho = 0.938$ and $\rho_{>0} = 0.871$ on them (Table 8). That suggests these models are structurally simple enough that transitive reach is nearly the oracle.

For a special issue on the reliability of *modern software systems*, at least one of the following is needed:
- **(a)** One system model extracted *automatically* from real artifacts. Examples: Autoware launch files via HAROS/ROSDiscover (both already cited), docker-compose/Kubernetes manifests plus AsyncAPI specs of an event-driven application, or the Train-Ticket deployment.
- **(b)** A small fault-injection study on one deployed system (e.g., kill-one-service experiments on Train-Ticket or a ROS 2 demo). It should check whether $I^*$ or $I_{\text{dyn}}$ rank components consistently with observed delivery loss. Even $n \approx 20$ components would anchor the oracle construct.
- **(c)** Independent re-authoring of at least two of the five system models by a second modeler, with reported agreement (e.g., Jaccard on edges, $\rho$ on resulting $I^*$).

Without one of these, every conclusion is conditional on three layers of author design: the generator, the simulator and the models.

### M3. Learned models are data-starved and untuned, biasing toward the null

- **Data size.** Each LOSO fold trains a ~430k-parameter GNN (Table 4) on roughly 1,100 labeled Applications from eleven graphs. The generator is free, and $I^*$ labels cost seconds per architecture (Table 9), yet training is limited to twelve scenarios.
- **Hyperparameters.** They were fixed a priori. The registered nested selection, run later, moves `HGT-QoS` by $+0.055$ and `GAT-P-QoS` by $-0.054$ (§6.2). That is the same magnitude as several reported effects ($+0.041$ direction, $+0.031$ Eq. 7 vs. GBM, $-0.016$ GAT-P vs. `InDeg`).
- **Stability.** `HGT-P-QoS` is unstable (within-fold seed SD $0.254$) and is described as "one untuned configuration" (§7.2).

**Requested:**
- **(a)** A learning curve. Train on 11, ~50 and ~200 generated scenarios, held out by domain so the test folds stay fixed, and plot $\rho$ for `GAT-P-QoS`, a GIN and GBM against `InDeg`. If learners plateau below the reference at every training size, the null becomes much more convincing. If they cross it, that is precisely the "when" the title asks about.
- **(b)** A matched, documented tuning budget per model family (the same number of configurations for GAT, GIN, HGT and GBM), selected by nested CV.
- **(c)** Smaller-capacity models (e.g., a 2-layer GIN with roughly $10^4$ parameters). They are more appropriate for this data scale.

As it stands, "learning adds no value" cannot be separated from "these configurations in this data regime add no value".

### M4. Several claims exceed the statistical evidence

The paper correctly states that "matches" means "not significantly different" and that equivalence at $\pm 0.05$ fails (TOST $p = 0.17$; §6.1). Yet the front and back matter draw equivalence-style conclusions:
- **Abstract.** "explicit dependency representation renders complex learning superfluous"; "Representation matters more than model complexity."
- **§1.3.** "establishing that in first-order regimes learning adds no predictive advantage."
- **§7.2.** "rendering model complexity superfluous."
- **Abstract.** "All three simulators admit such approximations." This contradicts §4.4 ("For $I_{\text{comp}}$ no ranking is a truncation"); `Topo-QoS` and degree are only term-aligned with $I_{\text{comp}}$.

Note that on $I_{\text{dyn}}$ the evidence *does* approach equivalence: GBM+Eq.7 vs. Eq. 7 gives $\Delta = +0.000$ $[-0.013, +0.015]$. On $I^*$ it does not. The language should differ accordingly.

§6.2 also proposes a sound rule: learned-ranker differences smaller than the ~0.044 per-fold node-order spread "should not be read as differences between models". Apply this rule consistently. GAT-P-QoS vs. `InDeg` ($-0.016$), the direction effect ($+0.041$) and Eq. 7 vs. GBM ($+0.031$) all fall within or near that spread, yet the last one appears in the abstract ("matching or exceeding") and in Highlight 4/5.

Please also stop reporting "Won" (folds won) as an effect size. It is a sign count. Add a paired effect size such as the matched-pairs rank-biserial correlation.

### M5. The reference criterion is post hoc, under-specified, and applied asymmetrically

The criterion was defined "after simple dependency counts performed strongly" (§4.4). That is acknowledged, but three problems remain:
- **(a) Asymmetry.** References "carry no contrast" and "comparisons with [them] are not claims of superiority". Yet the paper's main messages *are* such comparisons: "no learned model exceeds the analytical rankings" (§6.1), "the rate-weighted reference exceeds [GBM] by $+0.031$" (§6.1), Table 10 recommending references because they are "at least as accurate", and Highlights 3–5. The authors must choose. If references are comparators, they need pre-specification and multiplicity control. If they are not, the headline cannot rest on them.
- **(b) Discretionary scope.** A reference is "what $O$ computes when truncated after $k$ waves, *with stated simplifications*". Nothing bounds the simplifications. Eq. 7 drops queueing entirely. Why is it a reference while GBM-with-Eq.7-as-input is a predictor? Is a GNN with Eq. 7 as a prior (F12) a reference? Give a formal definition, e.g., a truncation operator $T_k$ on the simulator's propagation semantics with an enumerated, closed set of admissible simplifications. Then show that Eqs. 6–7 and `InDeg`/`Reach` satisfy it and that the learned rankers do not.
- **(c) Confirmation corpus.** §7.5 item 2 lists a confirmation corpus as future work. Because the generator regenerates byte-identically from configs and $I^*$ labeling is cheap, it should be done *in this revision*. Freeze the analysis, generate a fresh set of scenarios with new seeds and parameter settings, and re-test the headline registered-secondary and exploratory claims. This single step would turn the paper's exploratory headline into confirmatory evidence, and it is the most cost-effective improvement available.

### M6. The registered comparator is defective and weak, and Table 10 still recommends it

- **The defect.** `Topo-QoS` has an implementation defect: its articulation term is identically zero (§5.2).
- **It is still the reported baseline.** The registered (defective) value is the one reported throughout. The Table 5 footnote calls this a "defect-controlled configuration where the articulation point index evaluates to zero". That phrasing is euphemistic; please state plainly that it is a bug.
- **Both confirmatory contrasts depend on it.** Both are against this baseline, and the one positive learned result ("hybrids beat the baseline") turns out to "belong to the baseline's weakness" (§6.1).
- **Table 10 still recommends it for $I_{\text{comp}}$.** This happens even though (i) it is defective and (ii) raw total degree is higher ($0.719$ vs. $0.702$, §6.1).

**Requested:** report the corrected baseline as the primary row, with the registered value footnoted. Recommend total degree (or the corrected score) in Table 10. Reconsider whether "learning beats only a weak baseline" deserves headline space (Highlight 5, §1.3), since a comparator chosen before the strong one was known says little about learning.

### M7. The cost and energy analysis — the special issue's sustainability angle — is thin and partly misattributed

- **(a) Training cost is missing.** The energy comparison (§6.4, §7.4) covers feature extraction (0.54 Wh), the $I^*$ sweep (0.086 Wh) and $I_{\text{dyn}}$ labeling (355.6 Wh). It omits the dominant term for learned approaches: training 12 folds × 5 seeds × up to 300 epochs × every arm. Please report training wall-clock and energy.
- **(b) Feature cost belongs to the feature set.** The authors themselves state that articulation/CDI account for 88–91% of feature extraction, "so this cost belongs to the chosen feature set, not to learned ranking as such" (§6.4). And a featureless GIN reaches $0.719$ (F11). §7.4 nonetheless concludes that "Deploying heavy graph learning for pre-deployment gating thus incurs unjustified environmental overhead." That sentence contradicts §6.4 and should be removed or qualified.
- **(c) The estimator is not a lower bound.** Wall-clock × 28 W base power is called "an order-of-magnitude lower bound". It is not a bound in either direction: a single-threaded load on a P-series mobile part can draw well below base power, and turbo can draw well above it. Use powercap/RAPL (often readable with a udev rule rather than root), CodeCarbon, or an external meter. Otherwise, drop "lower bound".
- **(d) Report break-even, not absolute differences.** Absolute differences of fractions of a watt-hour are practically negligible. The sustainability-relevant quantity is the break-even: how many evaluations of an expensive oracle a learned surrogate must replace to amortize labeling plus training. Report that for $I_{\text{dyn}}$, against Eq. 7's near-zero cost.

### M8. Inconsistent reporting protocols impair cross-table comparison

- **Three conventions in two tables.** Table 5 reports per-seed means. Table 6 reports the mean of five seeds' *predictions* for GNNs. The $\to$dyn rows of Table 6 revert to per-seed means. GAT-P-QoS therefore appears as $0.748$ and $0.772$ against the same oracle.
- **The learned $I_{\text{dyn}}$ GNN is not a credible contender.** In Table 6, the $I^*$-trained GAT-P-QoS ensemble scores $0.615$ on $I_{\text{dyn}}$, *above* the GAT trained on $I_{\text{dyn}}$ itself ($0.598$). That indicates the $I_{\text{dyn}}$-trained GNN did not fit its target, partly because it received neither rates nor payloads (§6.1). It is therefore not a credible learned contender for $I_{\text{dyn}}$.

**Requested:** use one protocol in all tables (per-seed mean, with the ensemble as a supplementary row). Add a GNN that receives the declared rate and payload as node or edge features before concluding anything about learned GNN approximations of $I_{\text{dyn}}$.

### M9. Readability: the evaluation apparatus overwhelms the science

The body carries several layers of apparatus:
- a 17-amendment registration history;
- control codes F1–F13;
- roughly twenty ranker names built from a suffix grammar (`-P`, `-QoS`, `-R`, `-deg`, `-min`, `-const`, `-AP`, `-perm`, `$\to$dyn`, `+Eq7`);
- a status tag on most claims;
- at least 17 pointers into a supplement of 43 sections and 71 tables.

"Findings in Brief" (§1.3) restates the abstract with more numbers. §7.3 and §7.4 restate §6.

Some passages also read as rhetoric rather than analysis, and contradict the body:
- §7.3 says closed-form counts provide "exact, deterministic, and instant guarantees essential for automated deployment gating". Three paragraphs later, the same rankings are "unsuitable as automated pass/fail filters". "Guarantees" is also unjustified.
- §7.3 attributes "within-fold standard deviations up to 0.254" to "deep GNN models" in general. That figure is the untuned `HGT-P-QoS`; the GATs are 0.024–0.030 (§7.4).

**Requested:**
- **(a)** Collapse the naming scheme (e.g., a two-letter model code plus a graph superscript).
- **(b)** Move the amendment and status bookkeeping into one appendix table, keeping one status sentence per RQ in the body.
- **(c)** Make the four RQs answerable from the body alone.
- **(d)** Remove the repetition and the rhetorical passages identified above.

### M10. The scope of the dependency-derivation contribution is overstated

Contribution 1 presents six typed rules, but only Rules 1 and 5 are exercised (Table 2):
- Rule 1 comes from [32].
- Rule 5 edges end at Libraries, so Application-level `InDeg` is unchanged by the derivation (Remark 1).
- The derivation's measured value is $+0.058$ for `Reach` (§3.3).

Rules 2–4 and 6 are "formalized for completeness" but never tested, yet they appear in Figure 2 as if operative.

**Requested:** either evaluate Broker/Host ranking with Rules 2–4 and 6, or move them to the supplement and narrow Contribution 1 to what is evaluated. Likewise, position the message-passing findings (§6.2, §7.2) explicitly as confirmations of known expressivity results [76–78], not as new insights.

### M11. Literature gaps and citation mismatches

**Missing literature.** The following directly bear on the paper's thesis and are absent:
- **Hybrid analytical + ML performance modelling.** The hybrid-prior design (Figure 3) and the "learned surrogate vs. analytical model" question are the core of this literature, e.g., Didona et al., *Enhancing performance prediction robustness by combining analytical modeling and machine learning* (ICPE 2015), and related gray-box modelling work. This is central for a special issue on AI for performance.
- **"Simple vs. deep" in SE.** Fu & Menzies, *Easy over hard* (ESEC/FSE 2017); Majumder et al., *500+ times faster than deep learning* (MSR 2018). Both directly anticipate the cost/accuracy argument.
- **Architecture-based reliability.** Cheung's user-oriented reliability model (IEEE TSE 1980), as the origin of architecture-based reliability.
- **Learned performance surrogates.** E.g., DeepPerf (Ha & Zhang, ICSE 2019), and causal/ML failure analysis in microservices such as Sage (Gan et al., ASPLOS 2021). These should be contrasted with the pre-deployment setting.

**Citation mismatches.**
- **[10] (Albert et al.)** is cited for "a slow subscriber fills a broker queue and starves its publishers" (§1.1). That paper studies the error/attack tolerance of scale-free networks and says nothing about broker queues.
- **[8, 9]** (Motter–Lai; Buldyrev et al.) are cited for propagation "through brokers, shared topics, colocated hosts and shared libraries". Neither concerns middleware.
- **[11, 13, 14]** (Avizienis et al.; Perry & Wolf; Garcia et al.) are cited for pre-deployment cost and for SPOF/QoS-mismatch failures. They do not support those specific claims.

---

## 4. Minor Comments

1. **Abstract (244 words).** It is within the 250-word limit, but it is dense with point estimates and status tags. Its final sentence ("All three simulators admit such approximations…") contradicts §4.4 (see M4).
2. **Highlights.** They comply with JSS (5 bullets, ≤ 85 characters each). Highlight 3 ("No learned ranker outperforms counting direct dependents") frames `InDeg` as a competitor, which §4.4 says it is not. Highlight 5 ("add nothing") is stronger than F12's evidence for the GBM residual arm ($-0.006$, $p = 0.68$).
3. **Title.** See M1(c). The question in the title is not answered by the design.
4. **§1.4, first sentence.** "Addressing when graph learning is needed, establishing when learning is unnecessary is as informative…" is a dangling construction. Please rewrite.
5. **Spelling.** British and US spellings are mixed: "favourable" (§6.2) and "surrogate-modelling" (§1.2) against "modeling" elsewhere. Standardize to one.
6. **Table 5 footnote §.** See M6: replace "defect-controlled configuration" with a plain statement of the bug. The § and ¶ footnote markers in Tables 4–5 refer to the same defect; unify them.
7. **Table 7 vs. text.** The table says `GIN-P-QoS`; §6.2 says "a GINE network". Unify the names and state how edge features enter GINE.
8. **Table 6 underline.** On $I_{\text{comp}}$, `Topo-QoS` is underlined as the highest predictor, but the text reports raw total degree at $0.719$. Add the degree row or qualify the underline.
9. **Feature counts.** §3.5 has "18 normalized metrics" shared by all types, while Table 4 gives GBM "18 columns: 9 dependency counts, 9 QoS/rate". These are different 18-element sets. Name them distinctly.
10. **Eqs. 6–7.** They are defined in §5.2 (Experimental Setup) but are methods; move them to §4.4 next to the criterion. Eq. 7's `\tag` placement also differs from Eqs. 1–6.
11. **§3.1.** "A Library takes the largest weight among its topics and consuming Applications, amplified by its fan-out." Give the formula, or reference the supplement explicitly.
12. **§4.3, $I^*$.** "Scaled by a declared, uncalibrated QoS severity ladder" and "five seeds that decide ties at the propagation threshold" are too terse to reproduce. Add pseudocode (an appendix is fine) covering the threshold, the ladder values and what is randomized.
13. **$I_{\text{comp}}$ weights.** The weights (0.35/0.25/0.25/0.15) are declared, not elicited. Please report the stability of the $I_{\text{comp}}$ rankings under a weight perturbation (e.g., Dirichlet sampling around the declared vector).
14. **§4.2, ties.** ListMLE breaks the ~31% tied zero labels in creation order, and the node-order spread (~0.044) is as large as several effects. Use a tie-aware listwise loss now (e.g., ListNet or LambdaLoss with tie groups) rather than deferring it to future work (§7.5).
15. **§5.3, statistics.** The Nadeau–Bengio correction was derived for repeated random train/test splits. Justify its use for LOSO, including the $n_{\text{test}}/n_{\text{train}}$ ratio used, given unequal fold sizes (26–300 Applications).
16. **Overlap@K.** State the tie handling for Overlap@K, as you do for Figure 5. With 31–51% zero labels, it matters.
17. **Partial $\rho$ (Table 6).** Specify the computation (residualization of ranks, per fold, then averaged) and the bootstrap unit.
18. **Figure 2.** It shows Rule 2 (Application→Broker) edges. Note in the caption that the evaluated projection uses Rules 1 and 5 only.
19. **§6.3.** "which `Reach` does perfectly" overstates it. `Reach`'s $\rho_{>0}$ is $0.871$, and the inert/active separation should be quantified (e.g., AUROC for $I^* > 0$).
20. **§6.4.** "running the oracle is therefore cheaper than approximating it with a learned ranker" holds trivially when the oracle costs under a second. State it once and drop it from Table 10's evidence column.
21. **Markdown figure captions.** The Markdown rendering of the Figure 4–5 captions loses mathematics ("win" for $w_{\text{in}}$, "1, 321", "+ 0.030", "Idyn"). The LaTeX sources are correct, but please check the submitted PDF.
22. **Replication link.** The repository link pins commit `f3352f0c` (Amendment 17 results), which predates the condensed manuscript. Pin the submission tag instead. Cite the software [94] with a Zenodo DOI rather than a GitHub URL, per the Guide's software-citation guidance.
23. **Keywords.** "Graph neural networks" and "graph learning" overlap. Consider replacing one with "surrogate modeling" or "benchmark methodology", which better index the contribution.
24. **Supplement size.** The supplement (43 sections, 71 tables) exceeds the body. JSS reviewers are not obliged to read it, so every claim in the abstract and highlights must be checkable from the body (see M9).
25. **Graphical abstract.** A graphical abstract exists in `latex/figures/` and is encouraged by the Guide. Make sure it is uploaded as a separate file, and that its numbers match the post-revision tables.

---

## 5. Recommendation: **Major Revision**

**Why not Reject.** The paper is methodologically honest, unusually well-controlled within its benchmark and fully reproducible. Its negative result, together with the warning to benchmark learned surrogates against the simplest truncation of the simulator they approximate, is a genuine and timely contribution for the AI4MSS special issue. JSS explicitly welcomes negative results.

**Why not Minor Revision.** In its present form, the conclusions are largely entailed by a benchmark the authors designed end-to-end: the generator, all three oracles and all five system models. No real-system anchor exists. The learned models are data-starved and untuned, which biases toward the null. The headline claims rest on a criterion defined post hoc and on comparisons the criterion itself says are not contrasts. Several statements ("renders learning superfluous", "establishing", "all three simulators admit such approximations", "unjustified environmental overhead") exceed the evidence.

**Minimum required for acceptance:**
1. Re-test the headline claims on a confirmation corpus generated after the analysis is frozen (M5c).
2. Add a training-size learning curve and a matched tuning budget (M3).
3. Add at least one real-system anchor: an automatically extracted topology, a small fault-injection validation, or second-modeler replication (M2).
4. Either add a non-first-order oracle or retitle/reframe as a negative-result benchmark study (M1).
5. Calibrate claims to the equivalence evidence throughout the abstract, highlights, §1.3, §7 and §8 (M4).
6. Correct the comparator and the Table 10 recommendation (M6).
7. Repair the energy analysis: training cost, a valid estimator and break-even (M7).

Items M8–M11 and the minor comments are expected but straightforward.
