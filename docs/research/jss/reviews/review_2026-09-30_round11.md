# Referee Report — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems", round 11 (second review)

**Manuscript:** *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?* (revised)
**Material reviewed:** `manuscript.md` (34 pp. PDF, 11 tables, 5 figures, 87 references), `latex/highlights.tex`, `reviews/response_round10.md`, and the JSS Guide for Authors. The assessment is of the revised manuscript as a stand-alone article.
**Date:** 2026-09-30 · **Recommendation:** Major Revision (narrowed; see the end)

---

## Summary

SaG derives a typed `DEPENDS_ON` dependency graph from publish–subscribe deployment manifests. It then compares analytical, hybrid and learned rankers of Application cascade impact against three simulators, under leave-one-scenario-out evaluation on twelve synthetic architectures and zero-shot on five hand-authored system models.

The revision reframes the paper as a carefully bounded negative result. The registered primary contrast is null. No learned model exceeds an analytical ranking aligned with the simulator: afferent coupling and a first-order expansion recover the reachability simulator, and a rate-weighted expansion (0.830) beats a learned approximation of the queue-flow simulator (0.799). Learning helps only relative to a weak training-free baseline.

The main contributions are now a benchmark with an explicit restatement-versus-prediction criterion, and evidence that simulator-labelled evaluations of graph learning should be run against simulator-aligned analytical baselines.

## Overall Impression & Assessment

The revision is substantial and honest. Every factual problem I raised in round 10 was either corrected or explicitly disclosed:

- the cost labels;
- the HGT-P-QoS parameter count;
- the Table 6 underline;
- the §7.2 selection-rule contradiction;
- the articulation-term explanation.

The authors also found and fixed errors I had not raised: the $w_V$ description, six wrong bibliography entries, and an unbacked p-value. They have removed "matches" throughout and now state the primary null in the abstract. They also state plainly that the benchmark cannot exhibit a regime in which learning wins. Figure 5B is now drawn on the full $I_{\text{dyn}}$ population, which I welcome. The paper is now accurate about what it shows, and that is a real improvement.

The difficulty is that it now shows less. The revision took a strictly text-only route, so the requested controls were turned into disclosures:

- the direction control and the corrected-prior hybrids (M4, M5);
- a confirmation corpus (M7);
- a real-system anchor (M2).

Contribution 1 ("the derived graph is the principal source of predictive signal") and the positive half of contribution 3 ("hybrids significantly outperform the baseline") now carry caveats that the paper itself says it cannot resolve.

Two of the missing experiments are cheap. They reuse existing harnesses and take about one CPU-hour each. They would turn a caveat into a finding in either direction. I therefore still cannot recommend acceptance. The remaining work is narrow and well defined, and the paper is close.

**Originality.** Modest but genuine. The reference criterion and its consequences for simulator-labelled benchmarks are the paper's most transferable idea.
**Significance.** Moderate for a negative result of this kind, and limited by the simulated-only ground truth.
**Soundness.** Transparent. Two claims remain under-supported (M1, M2 below).
**Fit.** Good fit for the special issue as a sober evaluation of AI techniques. The sustainability evidence remains estimate-only.

---

## Assessment of the Response to Round 10

| Round-10 comment | Status | Comment |
|---|---|---|
| M1 scope | **Addressed in text** | The scope statements are clear: "no regime found… a property of these oracles". The title is kept; I accept this given the explicit answer in §7.5 and §8. |
| M2 ground truth | Disclosed, not done | Acceptable only together with the clearer scoping now in the abstract. |
| M3 reference criterion | **Adequately rebutted** | The post hoc origin is disclosed and the "term-aligned" category added. I accept the rebuttal; see minor comment 3 for one residual inconsistency. |
| M4 baseline / hybrids | Partly addressed | The text is now honest, but see major comment M2. |
| M5 direction confound | Disclosed, not resolved | See major comment M1. |
| M6 "matches" | Addressed | One residue remains (minor comment 1). |
| M7 post hoc / confirmation | Disclosed, not done | See major comment M3. |
| M8 drift | Addressed | The interpretability rule is added. |
| M9 external validity | Addressed | Per-system values and the descriptive-CI wording are welcome. |
| M10 cost / energy | Addressed | One new inconsistency appears (minor comment 5). |
| M11 length / jargon | Largely addressed | A few deferrals and one plan code remain (minor comments 2 and 8). |
| M12 literature | Addressed | See minor comment 9 on positioning. |

---

## Major Comments

### M1. The representation claim still rests on a comparison the paper now concedes is confounded

The paper's lead claim, "explicit dependency representations contributed more to ranking performance than model complexity", appears in:
- the abstract;
- highlight 2;
- contribution 1;
- §3.4;
- §6.2's summary;
- §7.1;
- §8.

Its learned-ranker evidence is `GAT-P-QoS` (0.748) against `GAT-QoS` (0.635). §7.5 now correctly states that the raw-graph GAT receives no messages at Applications, and that no reverse-edge control was run. So the +0.113 "cannot be attributed to the derived dependency semantics alone". The same concession is not carried through the paper:

- **Abstract.** It still ends "Representation mattered more than model complexity" without the caveat.
- **§3.4.** It still says the raw multigraph "allows the effect of the representation to be separated from the effect of the learning architecture". The paper now says this is not true.
- **§1.2 and §6.2 summary.** §1.2 says the design "isolates the contribution of explicit dependency representation from learning algorithm complexity". §6.2's summary says the representation gain "exceeds every change of model architecture tested".

**Requested (preferred): run the control.** The GAT harness needs only a reverse-edge option: append flipped edges and duplicate edge features. At about 5–10 min wall-clock per arm on CPU, a matched-capacity `GAT-QoS` with reverse edges on $G_{\text{structural}}$ answers the question. Both outcomes are informative:

- **If the bidirectional raw-graph GAT approaches 0.748,** the derivation's value for learning is mostly direction, and contribution 1 should say so.
- **If it stays near 0.635,** the claim stands, and it becomes the paper's cleanest positive finding.

**Minimum otherwise.** Make the abstract, highlight 2, §1.2, §3.4 and the §6.2 summary consistent with §7.5. Replace "representation mattered more than model complexity" with a statement the design supports. An example: "analytical rankings on the dependency graph were not exceeded by any learned model".

### M2. The hybrid result is still reported as a positive finding for learning, and the corrected-prior retrain is cheap

The text now states the caveats: a weak and defective baseline that is nested in the hybrids, and no difference from the base learners. It still headlines the result:

- the abstract ("Hybrids correcting a closed-form prior outperform that baseline");
- contribution 3;
- the §6.1 paragraph heading ("Hybrid rankers secure statistically significant gains beyond baseline priors");
- §7.2 ("learning helped in two ways").

Two things undercut this headline:
- The hybrids were trained on the defective prior, and the paper says so in §5.2.
- The paper's own evidence is that the hybrids are indistinguishable from their base learners. So the "gain" is a property of the comparator, not of the correction.

**Requested.** Retrain both hybrids with the corrected (AP-restored) prior, and ideally with an `InDeg` prior. The registry already supports a prior kind per variant, and `indeg_prior` exists. The cost is about 1–2 CPU-hours.

If this is declined:
- remove the hybrid gain from the abstract;
- retitle the §6.1 paragraph descriptively, e.g. "Hybrid rankers versus the training-free baseline and their base learners";
- rephrase §7.2's "learning helped in two ways" so the first "way" is not presented as evidence for learning. The paper's own sentence, "modest evidence for learning", is the right register.

### M3. The headline recommendation remains a post hoc, exploratory contrast

Three prominent statements rest on an analysis that was added last and is labelled exploratory:
- Table 11 recommends the rate-weighted expansion for queue-flow impact.
- The abstract reports that it exceeds the learned approximation.
- §7.4 builds its Tier-1 protocol on it.

The +0.031 is nominally significant (10/12 folds). But it is post hoc, and the formula was written with knowledge of what $I_{\text{dyn}}$ reads. The response declines a confirmation corpus at about 13 CPU-hours.

I do not insist on a full corpus. A lighter confirmation would suffice: for example, new generator seeds for the six cheapest domains, which is well under 2 CPU-hours of $I_{\text{dyn}}$ labels by the paper's own per-architecture costs, Table 10 and §6.4. Scoring only the two training-free rankings (Eq. 6 and Eq. 7) and the already-trained GBM would convert the paper's one practical recommendation from exploratory to confirmed. If the authors decline, Table 11's queue-flow row and §7.4's Tier 1 should say "exploratory" as prominently as the abstract does.

### M4. The paper's definition of what rankers read is internally inconsistent

The input–label separation is central to the paper's validity argument. As written, it is contradicted by the paper's own setup:

- §1.2 says "Rankers read only the analysis graph". §3.4 says "All ranker inputs are computed on $G_{\text{analysis}}$", and §7.5 says rankers "consume $G_{\text{analysis}}$".
- Table 4 lists the raw-multigraph learners and hybrids as reading "$G_{\text{structural}}$ + features". §4.1 says `HGT-QoS` message-passes over the raw relations.
- §3.1 defines $G_{\text{analysis}}$ as "the logical `DEPENDS_ON` projection". §3.4 defines it as the structural graph that "adds the `DEPENDS_ON` edges and code metrics".

Nothing here suggests label leakage: the raw-graph learners read structure, not simulator output. But a reader cannot tell which graph each ranker reads, or what "input–label separation" guarantees. **Requested:** give one definition of $G_{\text{analysis}}$. State explicitly that the raw-graph learners read the edges of $G_{\text{structural}}$ plus node features computed on $G_{\text{analysis}}$, and that the separation guaranteed is "no simulator output is an input", not "different graphs".

---

## Minor Comments

1. **§6.1 "Two patterns emerge"** still says analytical rankings "match or exceed every learned model". Replace with the non-significance wording used elsewhere.
2. **Leftovers from the revision:**
   - §6.1 ends "(worst-case $\rho$ $-0.121$, Holm $p = 0.001$; )", an empty trailing clause.
   - §6.2 still names "Arm N".
   - §7.5 calls closed-form centrality "proximate" to $I_{\text{comp}}$, while §4.4 now uses "term-aligned".
3. **§3.2 vs §4.3 on QoS.** §3.2 says the $I_{\text{dyn}}$ QoS null "is informative" because the simulator reads QoS. §4.3 says the simulator's sensitivity to QoS "is not quantified" because no QoS-off run exists. If the simulator is itself insensitive to QoS policies, the null is uninformative for the same reason as on $I^*$. Soften §3.2 to "may be informative", or run the QoS-off labels on a subset of folds.
4. **Figure 5B.** Thank you for redrawing it on the full population. The panel still omits the rate-weighted expansion (Eq. 7), which is the reference for $I_{\text{dyn}}$ and the ranker Table 11 recommends. Its grey curves are $I^*$'s references, and the caption says so. Add Eq. 7 to panel B. This is cheap: the recall harness already scores arbitrary rankers.
5. **Energy bound (§6.4 vs §7.5).**
   - §6.4 calls TDP × wall-time "an upper bound for single-threaded work".
   - §7.5 notes that turbo power reaches up to 64 W, above the 28 W base.

   Base power × time is therefore not an upper bound. Drop "upper bound" or explain why turbo is excluded.
6. **Abstract, last analytical sentence.** "No learned model exceeded analytical rankings aligned with the simulator" is true against the best aligned ranking (Eq. 6: 0.808). It is not true against every aligned ranking: the `GAT-P-QoS` seed ensemble (0.772) numerically exceeds `InDeg` (0.764). Write "the best analytical ranking aligned with each simulator".
7. **§2.4 statistics sentence.** Arcuri & Briand [78] recommend standardized effect sizes (e.g., Vargha–Delaney $\hat{A}_{12}$). The paper reports raw $\Delta\rho$. Either add $\hat{A}_{12}$ for the registered contrasts, or cite [78] only for the non-parametric testing it supports.
8. **Remaining repository deferrals where the supplement holds the material:**
   - notation (§3.1) → supplement notation section;
   - per-scenario composition (Table 3 caption) → S8 / S14;
   - PR-AUC / $F_1@\tau$ / nDCG (§5.3) → S15.

   Also, §5.2 says both that the other baselines are "reported in Supplementary §S40" and that "We archive the unweighted Topo baseline in the replication repository"; reconcile the two.
9. **§2.1 positioning.** The claim that classic error-propagation and model-based approaches "require operational profiles or fault matrices unavailable at commit time" is too broad for AADL EMV2 and HiP-HOPS, which are design-time notations. The distinction is that they need component failure-mode annotations, which deployment manifests do not carry. Say that.
10. **Seed failures (§7.5).** "Single HGT seeds differ from others in the same fold by more than 1.0" means some seeds produce negative $\rho$. Are these convergence failures? If so, report how many seeds failed. Averaging them into per-seed means depresses the HGT rows in a way that says more about training stability than about typing.
11. **Figure 5 caption rendering.** In the Markdown, "1, 321" should read "1,321" (use `{,}` in math mode consistently in captions).
12. **Highlight 2** ("gain on derived dependency graphs") inherits M1. It should be adjusted with whatever wording M1 settles on.

---

## Recommendation

**Major Revision (narrowed).**

The manuscript is now accurate, transparent and well scoped, and the negative result is valuable to the special issue. What prevents acceptance is that two of its headline claims rest on comparisons the paper itself declares inconclusive:
- representation over model complexity (M1);
- learning's value via the hybrids (M2).

In both cases the resolving experiment is cheap and uses existing tooling.

I would expect to recommend acceptance, subject to minor edits, if the authors do the following:
1. run the reverse-edge raw-graph control (M1) and retrain the hybrids with the corrected prior (M2), and state the results whichever way they go;
2. make the stated scope consistent throughout (M1 minimum, M4);
3. either lightly confirm the Eq. 7 recommendation (M3) or label it exploratory wherever it is recommended.

If the authors prefer not to run M1 and M2, the alternative is to remove both claims from the abstract, highlights and contributions. What would remain is the benchmark, the reference criterion and the queue-flow approximation result. That paper could still be publishable, but as a narrower contribution, and it would need a correspondingly narrower title.
