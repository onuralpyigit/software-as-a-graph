# Referee Report — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems"

**Manuscript:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?*
**Version reviewed:** `manuscript.md` / `latex/manuscript.pdf` (27 pp.) and `supplementary.md` (48 pp.) on `main` at 7b6f475a (2026-10-06)
**Recommendation:** **Major Revision**

---

## 1. Summary

The paper proposes Software-as-a-Graph (SaG). SaG turns a publish–subscribe architecture into a typed multigraph and derives `DEPENDS_ON` edges from it with publish–subscribe rules. It then asks whether graph learning ranks Applications by simulated cascading-failure impact better than analytical rankings computed from the same dependencies. Using leave-one-scenario-out evaluation over twelve generator-produced architectures, three failure simulators (reachability $I^*$, queue-flow $I_{\text{dyn}}$, composite $I_{\text{comp}}$), and zero-shot transfer to five hand-authored system models, the authors find three things:

- A GAT on the derived dependency graph ($\rho = 0.748$) beats raw-multigraph learners.
- No learned model significantly beats a direct-dependent count ($0.764$).
- A closed-form, rate-weighted first-order formula ($0.830$) beats a learned approximation of the expensive queue-flow simulator.

The claimed contributions are the dependency derivation, a "reference criterion" that separates rankings that restate a simulator's rule from rankings that predict it, a controlled comparison of analytical, hybrid and learned rankers, and a reproducible benchmark with a cost analysis.

## 2. Overall Impression & Assessment

**Originality.** Moderate. Three pieces are useful and not common in the software-architecture literature:

- the explicit framing of simulator-labelled benchmarks as potentially circular;
- the order-$k$ reference criterion;
- the controls that separate representation, edge direction, degree features and aggregator.

The dependency projection is incremental over the authors' own conference paper [31]. The only newly evaluated rule is Rule 5 (Library), worth $+0.058$ on `Reach`. Four of the six rules are formalized but never exercised. The finding that simple structural counts match learned models confirms a well-known pattern (Zimmermann & Nagappan; Premraj & Herzig; Fu & Menzies; Kitsak et al. for short-range spreading). It does not overturn it.

**Significance.** This is an honest, carefully controlled negative result, and JSS explicitly welcomes such studies. Its significance is limited by a structural problem that the authors acknowledge: all three oracles are first-order by construction. So the study cannot reach the regime its title asks about. In effect, the benchmark measures how well learners recover the labelling rule. That a low-order truncation of the labelling rule wins is close to guaranteed by the design. There is a second issue. If publish–subscribe afferent coupling "restates" $I^*$, then the main lesson concerns the oracle (it is mostly a coupling metric), not the relative merit of learning. The paper does not draw this conclusion clearly.

**Methodological soundness.** Execution quality is high:

- paired fold-level tests with Holm and Nadeau–Bengio corrections;
- TOST used correctly to refuse equivalence claims;
- partial correlations;
- label-reliability ceilings;
- node-order permutation controls;
- disclosure of an implementation defect in the registered comparator;
- mechanical reconciliation of 1,817 reported figures.

The confirmatory evidence base is nevertheless thin. Both co-primary contrasts are null. Every headline finding is registered-secondary or exploratory, and the secondary arms were registered after the primary null was known. There were seventeen amendments, and the registered hyperparameter selection was replaced by a fixed configuration. Configuration effects ($\pm 0.055$), node-order effects ($\approx 0.044$ per fold) and cross-device drift (up to $0.172$) are the same size as the key learned-model contrasts.

**Suitability for a Q1 journal and the special issue.** The topic fits JSS (architecture, reliability, AI for SE), and the reliability strand fits the special issue. The sustainability strand is weak. Energy is a nameplate estimate (28 W × wall-clock), and the paper itself concludes that energy differences between rankers are "practically negligible". In its current form the manuscript is not yet at Q1 standard, for three reasons:

1. The title question cannot be answered by the design.
2. Several inexpensive but decisive controls are missing. By the authors' own account, the generator makes them nearly free.
3. The text is so heavily qualified that the contribution is hard to extract.

All three are fixable. The study's transparency is exemplary and should be kept.

---

## 3. Major Comments

### M1. The design cannot answer the title question; either extend the design or retitle.

The title and RQ1 ask *when* graph learning improves cascade-impact ranking. §1.4, §7.2, §7.4 and §7.5(3) concede that all three simulators are first-order by construction. In particular, $I_{\text{dyn}}$ "by construction propagates no failure beyond one hop", has no brokers, never blocks publishers, and lets a starved consumer keep publishing. The paper therefore "cannot characterize the regimes in which learning would exceed" analytical truncations. Section 7.5(3) says these regimes are "the only ones in which this study's title question could receive a positive answer". A Q1 paper should not carry a title whose question its design rules out.

I ask for one of the following:

- **(a) Preferred.** Add at least one non-first-order oracle. A natural candidate is a variant of the existing SimPy simulator with publisher backpressure on full queues, finite broker capacity, or retry amplification, run on the twelve folds. Then report whether (i) Eq. 7 or an order-2 truncation still tracks it, and (ii) learners close any of the remaining gap. The infrastructure exists; §7.5 already lists this as priority 3.
- **(b)** Retitle and reframe the paper as a benchmark-methodology and negative-result paper. For example: "Explicit dependency derivation, not graph learning, explains cascade-impact ranking on first-order simulators". Then remove "when" from the RQs.

### M2. The circularity argument cuts against the oracles, and the reference/predictor split is post hoc and partly arbitrary.

The reference criterion (§4.4) is the paper's most original idea, but it raises three problems.

1. **It was defined post hoc,** after the dependency counts performed strongly (§4.4, §7.4). The admissible simplifications S1–S5 are generous. S4 allows uniform weights, and S5 allows replacing a weighted loss with its support. These appear tailored so that exactly `InDeg`, `Reach`, Eq. 6 and Eq. 7 qualify. The paper should say why S1–S5 and no others. It should also show that the criterion classifies some ranking *against* the authors' interest, for example a ranking that scores well but is classified as a predictor. Otherwise it reads as a rule designed to exclude the winners from contrasts.
2. **Afferent coupling (Martin's $C_a$) is an established architecture metric** with a 20-year literature (§2.2 cites it as such). The paper concludes that $C_a$ "restates" $I^*$. That is mainly evidence that $I^*$ adds little beyond a metric architects already compute, so $I^*$'s validity as "ground truth for cascade impact" is in question. The Discussion should state this plainly. The current framing implies that the analytical side "wins".
3. **The split confuses the practical message.** References are excluded from inferential contrasts and from "best predictor" underlining. Yet Table 10 recommends them, and the registered comparator (`Topo-QoS`) is a defective betweenness score that the authors themselves call the wrong choice "in hindsight". The decision-relevant comparison for a practitioner is learned model vs. afferent coupling.

I ask the authors to:

- run the confirmation corpus of §7.5(1) now. By the authors' own account, regenerating topologies and labelling with $I^*$ takes seconds;
- pre-specify `InDeg` (afferent coupling) as the comparator there;
- re-test the GAT-P-QoS vs. `InDeg` and Eq. 7 vs. GBM→dyn comparisons confirmatorily.

This one step would turn the headline findings from exploratory into confirmatory and would apply the reference criterion prospectively, as §4.4 itself admits is needed.

### M3. "Representation versus model" is confounded with aggregator expressivity and hop count.

The central RQ2 claim is that the dependency graph beats the raw multigraph by $+0.072$ beyond edge direction, and by $+0.231$ with oracle-aligned features removed. The derived Rule-1 edge collapses a typed two-hop pattern (publisher → topic ← subscriber) into a single hop. The direction control `GAT-QoS-R` uses softmax attention. As the paper itself notes (§2.3, §6.2), attention averages and cannot count. Counting subscribers on the raw graph therefore requires an attention model to count a two-hop typed neighbourhood. That is a known expressivity limitation, not a property of "dependency semantics".

No raw-graph, bidirectional, sum-aggregation control was run. I found no `GIN-QoS-R` or R-GCN arm in either document. Please add:

- **(a)** a GIN/GINE on the raw multigraph with reverse edges and at least two layers, with full features and with the `-min`/`-const` features;
- **(b)** a direction-typed or relation-typed bidirectional model (R-GCN, or HGT with reverse relations as separate types), which §7.4 lists as untested.

If (a) closes most of the $+0.231$ gap, then the projection precomputes a composition that a sum aggregator can learn. That would be a different and more nuanced message than "representation mattered more than model complexity". The result should be reported either way.

### M4. The learned approximation of $I_{\text{dyn}}$ is handicapped, and the surrogate-modelling framing is overstated.

1. **The GNN trained on $I_{\text{dyn}}$ received no rates.** `GAT-P-QoS→dyn` received neither declared rates nor payloads (§6.1). The paper itself shows that rates are the only declared information carrying signal ($+0.097$). The GNN's 0.598 is therefore not a fair test of learned surrogate modelling, as the authors acknowledge ("a GNN given the declared rates was not run"). It is the obvious arm for a paper about whether learning adds value. Please run it, with rates as node features and as edge features.
2. **The "expensive" cost is an implementation artifact.** $I_{\text{dyn}}$ is described as "expensive", which makes approximating it "a genuine surrogate-modeling problem" (§1.2). Yet by construction it is a one-hop delivered-rate computation embedded in a discrete-event engine. The 12.7 CPU-hours (≈ 7 s per fault run) come from implementing a first-order quantity with SimPy, not from any intrinsically hard dynamics. Under these conditions, a closed-form first-order formula approximating it is the expected outcome. The abstract and highlights ("approximates a 12.7 CPU-hour simulator") should not present this as a surprising surrogate-modelling result.
3. **The break-even argument (§6.4) is close to circular.** It follows directly from the fact that a training-free formula exists for this simulator.

### M5. The registered comparator is a straw man, and the abstract and highlights lean on comparisons against it.

`Topo-QoS` is QoS-weighted betweenness alone. Its articulation term is zero because of a defect. It is the weakest analytical ranker, and the paper says it supersedes the prior paper's recommendation of it. Even so:

- the abstract reports "Learned models transfer zero-shot better than the training-free baseline (ρ ≈ 0.81 vs. 0.53)";
- §1.3 leads with "Learning adds measurable value relative to the training-free baseline";
- Highlight 5 states "hybrids beat the baseline".

The hybrid result is uninformative by the authors' own analysis: hybrids nest the comparator, do not beat their base learners, and reproduce `InDeg` when given it as a prior. Please remove comparisons against `Topo-QoS` from the abstract and highlights and lead with comparisons against the meaningful baseline (afferent coupling). The hybrid material can shrink to a paragraph plus a supplementary table.

### M6. The inferential base is fragile; resolve the noise sources rather than only disclosing them.

The paper discloses the following noise sources candidly:

- the primary null;
- 17 amendments;
- secondary arms registered after the primary null was known;
- the registered nested selection replaced by a fixed configuration;
- nested selection moving `GAT-P-QoS` by $-0.054$ and `HGT-QoS` by $+0.055$, in opposite directions;
- node-order spread of $0.044$ per fold, caused by ListMLE breaking ties in creation order with 31% of labels tied at zero;
- cross-device drift of a learned cell by up to $0.172$ (§7.4).

The representation effect ($+0.072$) is only about 1.6× the node-order spread and is about the size of the configuration swing. Disclosure is necessary but not sufficient. Please:

- **(a)** report the dependency-graph contrasts (F8, F11, GAT-P-QoS vs. `InDeg`) under the registered nested protocol as well as the fixed configuration;
- **(b)** replace ListMLE's arbitrary tie order with a tie-aware listwise loss, for example ListNet on graded relevance or a Plackett–Luce model with tie groups. This removes a documented artifact instead of characterizing it;
- **(c)** explain the source of the 0.172 cross-device drift (nondeterministic kernels? data loading? hash order?) and reproduce the headline learned cells on a second device. A drift that large is a reproducibility red flag for a paper whose contribution is a benchmark;
- **(d)** consider a mixed-effects analysis (seed nested in fold) instead of Wilcoxon on seed means, so that seed variance (up to 0.254 for `HGT-P-QoS`) enters the uncertainty.

### M7. The small-data regime is not characterised, although the paper says this would be free.

The learners are ~430k-parameter GNNs trained on eleven architectures (~1,000–1,300 labelled Applications) from one generator family. §7.4 notes that this "biases a learned-versus-analytical comparison toward the null". §7.5(4) notes that "the generator makes these [learning curves] free". Training-set size is a primary determinant of *when* learning adds value. Please add:

- a learning curve over the number of training architectures (e.g., 2, 5, 11, 25, 50 generated scenarios);
- at least one smaller-capacity model;
- a matched tuning budget across model families, or a justification for not having one.

Without these, "learning did not exceed the references" cannot be separated from "learning was starved".

### M8. External validity: the "deployment manifests" claim is not supported, and the zero-shot evidence is weak.

1. **Manifest claim.** The abstract and §1.2 say SaG "derives explicit dependency graphs from deployment manifests". §7.4 states that "the released tooling has no importer for launch files or for compose or Kubernetes manifests" and that "no topology in this study comes from a real deployment manifest". The claim must be qualified throughout. Better, demonstrate automatic extraction on at least one real system. Autoware launch files via HAROS/ROSDiscover, or Online Boutique's Kubernetes manifests, are both feasible.
2. **Zero-shot corpus.** The five systems were hand-authored by the first author. Two are RPC systems re-expressed as publish–subscribe meshes with no reply topics, and more than 50% of Applications are inert. With $n = 5$, percentile bootstrap intervals are not meaningful (the paper calls them "descriptive only"), and full-population $\rho$ mostly reflects separating inert from active components. The authors' own inter-modeler protocol (`reproduce/model_agreement.py`) has not been executed by anyone. A second modeler re-authoring even two of the five systems would substantially strengthen RQ3. As it stands, RQ3 supports only the claim stated in §6.3.
3. **Uncertainty on $I_{\text{comp}}$ and $I_{\text{dyn}}$.** Both are labelled only on the twelve synthetic folds. Contribution 5 ("seventeen architectures and three simulators") should say so.

### M9. Unevaluated machinery inflates the framework section.

Large parts of §3 do not bear on the results:

- Rules 2, 3, 4 and 6 (not evaluated);
- the explanation layer in Figure 1 ("outside the empirical evaluation");
- the AHP-derived QoS weights (CR = 0.016);
- the power-mean ($p = 3$) Application weight;
- the Library weight with its $0.15 \log_2$ fan-out amplification;
- the 16-D edge vector's seven QoS parameters.

The paper reports that declared QoS policies carry *no* measurable signal on either the reachability or the queue-flow oracle, and that the reported counts are unweighted. Please trim §3 to what is evaluated: Rules 1 and 5, `InDeg`/`Reach`, and the node features used. Move the rest to the supplement, and remove the explanation layer from Figure 1. This would also free space to address M1, M3 and M7 within the page budget.

### M10. Writing: the manuscript is over-hedged, dense, and reads as a response to past reviewers.

The scientific honesty is admirable, but the presentation obscures the findings. In the main text, "reference" occurs ~69 times, "exploratory" 22 times, "first-order" 36 times, and "not significantly different" 9 times. The same caveats (status of results, first-order oracles, single modeler, fixed configuration) are restated in the Abstract, §1.3, §1.4, §4.4, §5.3, §6.x, §7.2, §7.4 and §8. Arm names stack suffixes (`GIN-P-QoS-const`, `GAT-QoS-R-min`, `Hybrid-GAT-AP`). Control families are labelled F1, F2, F4, F8–F13, with F3 and F5–F7 only in the supplement. Revision-history artefacts appear in the text, for example "Across revision cycles and compute devices", "seventeen numbered amendments", and "the counts were reclassified as references after all results were final".

Specific requests:

- Order the introduction as problem → RQs → findings. "Findings in Brief" currently precedes the RQs.
- State the status of every claim once, in one table (claim, status, test, section), and then report results plainly.
- Rewrite the abstract's second paragraph. "+0.072 above the same model with reverse edges on the raw multigraph, where the registered co-primary contrasts were null" mixes two unrelated facts in one clause.
- Rename arms to a readable scheme, and renumber the control families contiguously within the main text.
- Prune the 48-page supplement (S1–S43, tables up to S71). Much of it documents the revision history rather than the science.

---

## 4. Minor Comments

1. **Abstract length.** The Guide sets a 250-word limit. With mathematical expressions counted as words, the abstract is at or above it (~250–265 depending on the counting convention). Tighten it; M5 and M10 already suggest content to drop. Avoid "(exploratory)" parentheticals in the abstract; state status once.
2. **Highlights.** All five are within 85 characters (75–82). However, Highlight 1 describes the method, not a result, and Highlight 5 joins two unrelated claims, one of which is the weakest finding. Consider replacing them with "Afferent coupling matches the best graph learner on a reachability simulator" and "Simulator-labelled benchmarks need an order-k reference check".
3. **Keywords.** "Reliability" and "dependability" overlap. Consider "afferent coupling" or "surrogate model", which are more discoverable for this contribution.
4. **Citation style.** The manuscript uses numeric citations. The Guide accepts any consistent style at submission but specifies author–year in-text citation for the journal. `elsarticle-harv` would avoid a later conversion.
5. **Remark 1 / §3.5.** "The in-degree feature the learners receive also follows `USES` links; the two differ for 940 of the 1,321 Applications." `USES` points App → Library, so it is unclear how it changes an Application's in-degree. Define the feature precisely, and on which graph it is computed. The figure also means the feature differs from `InDeg` for 71% of Applications, which sits uneasily with §3.5's statement that the in-degree feature is "close to `InDeg`".
6. **Table 1.** Entity types and edge types share one table with a header row in the middle. Split it into two tables or two clearly separated panels.
7. **Table 4, parameter matching.** GAT (437,496) and GAT-QoS (429,992) differ by 1.7%. State the matching tolerance used for the 2×2 explicitly; §4.1 states "within 1%" only for HGT-P.
8. **§4.2, early stopping.** Early stopping uses a 20% node-level split of the largest *training* scenario, so validation nodes share a graph with training nodes. Justify this choice over a held-out scenario. The nested protocol, which does use a held-out scenario, moves results by ±0.055.
9. **§4.3.** "The Validate-stage failure simulator" is code-base jargon. Describe $I_{\text{comp}}$ by function, and justify or cite the declared weights (0.35/0.25/0.25/0.15) or test their sensitivity. §7.5 lists this as not done.
10. **§4.3 / §6.1, reliability ceiling.** The Spearman–Brown correction applied to Spearman correlations is approximate; say so. "The formula's mean of 0.830 thus leaves roughly 0.06–0.17" compares a mean with per-fold bounds. Report per-fold headroom, $\sqrt{r_f} - \rho_f$, instead.
11. **§6.1.** "0.007 above it as a five-seed ensemble (0.772)": $0.772 - 0.764 = 0.008$ at the displayed precision. Either use one consistent rounding or add a note.
12. **"Nominal Holm p"** (§6.1, Table 7 F13) is a contradiction in terms. Report the nominal $p$ and say which family it would enter.
13. **Overlap@K in Table 5** breaks ties by identifier order, and about 31% of labels are tied at zero. Figure 5 already resolves ties in expectation. Drop Overlap@K from Table 5 or replace it with the tie-aware recall.
14. **§1.3 and §7.3** state that recovering 80% of the critical set requires 40–45% of Applications. That holds for $I^*$. On $I_{\text{dyn}}$, Eq. 7 recovers 0.81 at 20% and 0.96 at 30%. Qualify the general statement, which affects the "unsuitable as gates" conclusion.
15. **Table 10, first row** recommends "Run $I^*$ directly". As practitioner guidance this is circular unless $I^*$ is shown to be valid. Phrase it as "if $I^*$ is the accepted impact definition…".
16. **Figure 1** includes the unevaluated explanation layer (see M9). **Figure 3(a)** describes the hybrid as $\sigma(z + \alpha\,\mathrm{logit}\,p)$ but does not state how $\alpha$ is initialised or constrained.
17. **§5.3** says the plan is "pre-specified in the replication repository with commit timestamps, not lodged with a third-party registry". Name the commit hash of the original plan in the text, so readers can verify which elements predate the primary result.
18. **Equation formatting.** Eq. 7 places its `\tag` after the display, unlike Eqs. 1–6, and Eq. 1 is tagged inline in running text. Make the display style consistent, per the Guide's "display equations separately… numbering them consecutively".
19. **Related work gaps.**
    - The ML data-leakage literature is directly relevant to the circularity argument (e.g., Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 2023).
    - The simulation-metamodelling literature (e.g., Kleijnen, *Design and Analysis of Simulation Experiments*) is relevant to the "learned approximation of an expensive simulator" framing (Contribution 4).
    - Microservice architecture-recovery tools beyond ROS (e.g., MicroART) are relevant to M8.
20. **Data availability.** The GitHub link is pinned to 56d9bff8, while the current `main` contains later commits (2e90073d, 7b6f475a). Confirm that the pinned commit contains the final scripts and the reconciliation target. The 1,817-figure mechanical reconciliation is excellent practice and could be mentioned in the main text, not only in Declarations.
21. **Generative AI declaration.** The declaration is present and follows the required section title. Since AI tools were used to develop analysis scripts, a sentence in §5.1 noting how those scripts were verified (e.g., by the regression tests and reconciler) would make the declaration's "verified" claim concrete.

---

## 5. Recommendation

**Major Revision.**

**Justification.** The manuscript is an unusually transparent, well-instrumented study. Its central negative message is plausible and useful to the JSS readership: on simulator-labelled benchmarks, explicit dependency derivation plus simple counts matches graph learning, and learned results must be read against the simulator's own low-order truncation. It is not ready for acceptance, for three reasons:

1. **The design cannot answer the question in the title** (M1). Every oracle is first-order by construction, so a positive answer is ruled out in advance.
2. **Several decisive controls are missing but cheap**, by the authors' own account:
   - a raw-graph sum-aggregation direction control (M3);
   - a rate-aware learned surrogate (M4);
   - a learning curve (M7);
   - a prospective confirmation corpus that would make the headline claims confirmatory (M2, M6).
3. **The written contribution is buried** under repeated qualification, revision-history artefacts and unevaluated framework machinery (M9, M10).

Rejection is not warranted: the core data, tooling and controls are sound, and the requested additions reuse existing infrastructure. Minor Revision is not appropriate either, because M1–M3 can change what the paper claims. If the authors add a non-first-order oracle (or retitle and reframe), run the confirmation corpus with afferent coupling as the comparator, add the sum-aggregation and rate-aware controls, and condense the text, the paper would make a solid JSS contribution.
