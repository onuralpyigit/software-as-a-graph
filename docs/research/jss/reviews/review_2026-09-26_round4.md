# Referee Report (Round 4) — JSS Special Issue "AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems"

**Manuscript (third revision):** *Software-as-a-Graph: Dependency-Graph Analysis and Learning for Pre-Deployment Simulated Cascade-Impact Ranking in Publish–Subscribe Systems* (Yigit, Buzluca)

**Material reviewed:** everything in commit `4d6b7cc`:
- [`manuscript.md`](../manuscript.md), the LaTeX sources and the compiled [`latex/manuscript.pdf`](../latex/manuscript.pdf) (23 pp.);
- [`latex/highlights.tex`](../latex/highlights.tex), [`latex/vitae.tex`](../latex/vitae.tex), [`latex/LENGTH_JUSTIFICATION.md`](../latex/LENGTH_JUSTIFICATION.md) and [`latex/refs.bib`](../latex/refs.bib);
- the new harness [`reproduce/independent_oracle_evaluation.py`](../../../../reproduce/independent_oracle_evaluation.py) and the code it calls ([`reproduce/convergent_validity.py`](../../../../reproduce/convergent_validity.py), [`reproduce/icomp_sensitivity.py`](../../../../reproduce/icomp_sensitivity.py), [`reproduce/training_free_suite.py`](../../../../reproduce/training_free_suite.py)).

The supplement ([`latex/supplementary.tex`](../latex/supplementary.tex)) and [`PREREGISTRATION.md`](../PREREGISTRATION.md) are unchanged in this revision.

**Independent checks run by the reviewer:**
- **I\* scoring.** I re-ran the released harness's I\* scoring of `InDeg` and the analytic first-order approximation over the twelve folds (`PYTHONHASHSEED=0`).
- **I_dyn candidate set.** I probed which components the I_dyn labeller selects on three folds.

**Checked against:** *Guide for Authors — Journal of Systems and Software* (retrieved 13 Sep 2026).
**Review model:** single-anonymised.
**Date:** 2026-09-26

---

## 1. Summary

The paper derives a `DEPENDS_ON` graph from typed publish–subscribe architecture models and benchmarks training-free counts, closed-form centrality, graph neural networks and hybrids at ranking Applications by simulated cascade impact. The evaluation uses leave-one-scenario-out cross-validation over twelve synthetic architectures and zero-shot transfer to five hand-authored system models.

The paper now frames itself as an empirical negative result for graph learning. Its central findings are:
- `InDeg`, which it now correctly identifies as publish–subscribe afferent coupling, ranks the reachability oracle I\* at ρ = 0.764 and beats closed-form centrality on 12 of 12 folds;
- attention networks on the dependency graph match `InDeg` but do not exceed it;
- the registered primary contrast (HGT vs centrality) failed.

The main new contribution of this revision is an evaluation of the rankers against two further oracles, a discrete-event queue-flow simulator (I_dyn) and a "composite" oracle (I_comp), intended to show that the count's accuracy is not an artifact of how I\* is built.

## 2. Overall Impression and Assessment

**Much of this revision responds well to the previous report.** The following changes are real improvements and bring the paper close to an honest, publishable empirical study:
- **Title and claims.** The title now says "Simulated Cascade-Impact Ranking".
- **Primary outcome.** The failed primary registered contrast is in the abstract and §1.3.
- **The `InDeg` identity.** It is stated in the main text (§3.3, §6.2).
- **QoS material.** The QoS equations and the AHP derivation have left the main text.
- **Active stratum and ceiling.** Table 6 reports ρ₍>0₎ and an analytic first-order ceiling for I\*. Both are exactly what was needed.
- **Construct overlap.** §4.4 now describes it frankly.
- **The MLP finding.** §8.2 plainly states that the raw-multigraph GNNs acted as per-node MLPs.
- **Top-K caveat.** The engine recommendation is qualified by Overlap@K.
- **Related work.** It now covers the prior work I listed and argues special-issue fit through the simple-baselines literature.
- **Paperwork.** The vitae are complete.

**However, the revision's central new evidence (Table 7) does not currently support the conclusions drawn from it**, and several new numbers cannot be traced to any released code or artifact:
- **The "I_comp" column is not a failure oracle.** It is SaG's own explanation-layer score Q(v).
- **The GAT-P-QoS row of Table 7 has no code behind it.**
- **The I_dyn column is scored on an undisclosed subsample.** It covers the first 30 Application identifiers per fold in string order.
- **The QoS-column control has no reported number, script or artifact.**
- **Table 6's analytic row does not match** what the authors' own harness produces.
- **Several caveats were removed from the text** although they still apply (drift, anti-conservative tests, single-modeller system models).
- **New claims were added that no evidence supports** ("orders of magnitude", "pre-registered", QoS "shown in our independent dynamic evaluations").

**Assessment against the usual criteria:**
- **Originality:** modest but now correctly positioned.
- **Significance:** would be established if the I_dyn evidence holds up.
- **Methodological soundness:** currently compromised by the provenance problems in M1.
- **Q1 suitability:** the paper is within reach, but a journal that reconciles "every reported figure against released artifacts" cannot accept figures that have no artifact.

### Status of the round-3 major comments

| Round-3 item | Status |
|:---|:---|
| M1 — score rankers on I_dyn / I_comp; active stratum; analytic ceiling; reword claims | **Partly done.** The active stratum and the ceiling were added and the title reworded. The I_dyn/I_comp evaluation was run, but the I_comp column is Q(v), I_dyn is subsampled, and the GAT row has no provenance (M1, M2 below). |
| M2 — why predict a cheaper simulator | **Partly.** §7.4 and §8.1 now concede the point. The stated advantage rests on the unvalidated explanation layer and on "robustness across paradigms", which Table 7 does not show (M3). |
| M3 — strawman "raw connections" comparator | **Done.** The identity is stated, and the +0.565 has left the abstract and highlights. |
| M4 — QoS columns confounded with weighted `InDeg` | **Acknowledged, but the control is unreported** (M1(d)). |
| M5 — typing never tested where message passing works | **Worsened.** The claim was scoped more broadly instead of narrowed (M3). |
| M6 — thesis contradicted by analyzer effects | **Done.** §1.2 was restated. |
| M7 — primary outcome; registered vs exploratory | **Mostly done**, but "pre-registered" is new and unsupported (M3). |
| M8 — scope, recommendation rule, special-issue fit | **Fit argued.** The recommendation rule is now contradicted by Table 7 (M2). |
| M9 — literature | **Done.** One new reference has a wrong author list (minor 4). |

---

## 3. Major Comments

### M1. Several new results cannot be traced to released code or artifacts, and one is mislabelled. (Must be resolved before further review.)

**(a) The "I_comp" column measures agreement with SaG's own predictor, Q(v), not with a failure simulator.**
- `independent_oracle_evaluation.py` reads I_comp as the `q_score` field of `results/icomp_scenario_cache_jss12.json` (`get_all_icomp_labels`).
- The cache builder (`icomp_sensitivity.py`, `extract_scenario_data`) fills `q_score` with `c.scores.overall` from `PredictionService().predict_quality(...)`. That is the RM composite Q(v) of §5.
- The simulator's impact components are stored separately in the same record as `reachability`, `fragmentation`, `throughput` and `flow_disruption`, and the harness never reads them.

Three things are consistent with this and not with a simulated target:
- ρ and ρ₍>0₎ are identical in every script-produced I_comp cell, because Q(v) has no zeros. The one exception is the GAT-P-QoS row (0.386 vs 0.345), which the script does not produce; see (b).
- `Topo-QoS` "wins" (0.669).
- §7.1's explanation that I_comp "incorporates QoS-weighted betweenness directly into its multi-criteria formula" describes Q(v): its M(v) term puts weight 0.35 on betweenness. It does not describe the four simulated impact components.

The I_comp column, and every sentence that relies on it (§7.1, §8.3, §9, Contribution 3), must be withdrawn or recomputed from the simulated components with the shipped weights (0.35/0.25/0.25/0.15).

**(b) The GAT-P-QoS row of Table 7 is not produced by the released script.** The script's GAT-P-QoS loader is a stub:

```python
if loso_dep_path.exists():
    ...
    pass
```

Its ranker list is `["InDeg", "Reach", "Topo-QoS", "Analytic-I*"]`. No other committed script scores a learned engine against I_dyn or Q(v). Please release the code and artifact that produced 0.588 / 0.421 (I_dyn) and 0.386 / 0.345 (I_comp), or remove the row. This matters because "`InDeg` … matching the neural engine `GAT-P-QoS`" on I_dyn is the only learned-vs-count evidence on a non-reachability target.

**(c) No result artifacts are committed.** None of the following exist in the repository:
- `results/independent_oracle_evaluation.json`;
- the I_dyn cache `results/idyn_scenario_cache_jss12.json`;
- the Q(v) cache `results/icomp_scenario_cache_jss12.json`.

The supplement has no section, per-fold table or provenance for Table 7, the analytic ceiling or the new Table 6 and Table 9 ρ₍>0₎ cells. The Data Availability statement still says "511 reported figures" are reconciled. By the paper's own standard ("every reported value is reconciled against released artifacts", §8.3), these figures do not yet qualify.

**(d) The QoS-column control is asserted, not reported.** §7.2 says a control replaced (w_in, w_out) with unweighted degree and permuted QoS profiles, and that "unweighted in-degree preserves over 90% of the gain". It reports:
- no Δρ, no confidence interval, no fold wins and no test;
- no result at all for the permutation arm.

No script or artifact for either arm is in the commit. This is a retraining experiment (12 folds × 5 seeds per arm). Please report it as a table with its artifact and state whether it was registered, or remove the sentence. As written, the paper's resolution of the round-3 QoS confound rests on an unsupported number.

**(e) The analytic ceiling differs between Tables 6 and 7, and Table 6's version is not what the harness produces.**

| Source | ρ | ρ₍>0₎ |
|:---|:---:|:---:|
| Table 6 | 0.814 | 0.548 (Overlap@K 0.582, Δ +0.261, 12/12, p = 0.0005) |
| Table 7 | 0.808 | 0.632 |
| My re-run of `_compute_analytic_first_order` in the released harness (`PYTHONHASHSEED=0`, twelve folds) | 0.808 | 0.631 |

The same re-run reproduces `InDeg` exactly (0.764 / 0.516). Table 6's analytic row (ρ, ρ₍>0₎, Overlap@K, CI, Δ and test) therefore needs to be regenerated from an artifact. "Ceiling" should also be qualified: on my run the approximation is *below* `InDeg` on four folds (AV, Financial Trading, Healthcare, Industrial SCADA). It bounds the mean, not every fold.

**(f) Table 10's first row changed without an explanation that fits the data.** The 249-node forward pass is now 13.5 ms (previously 26.5 ms, "carrying warm-up"), but its p10–p90 range (13.0–34.4 ms) is unchanged. A median lying 0.5 ms above its own p10, with an unchanged spread, is not what a re-measurement produces. Please re-measure, or report the artifact the new median comes from.

**(g) Table 9 reintroduces an undisclosed cross-harness comparison.** The previous version marked `Reach` and `InDeg` with a dagger: they were scored by the Amendment 7 harness, in which `Topo-QoS` scores 0.582, not 0.526. The dagger is gone, but the numbers are unchanged, and `supp_amendment7.tex` still documents the 0.582 vs 0.526 discrepancy. The main table now silently places scores from two pipelines side by side. This was round-2 M3. Re-score in one pipeline, or restore the disclosure.

I raise (a)–(g) not to question the authors' good faith, which the history of this manuscript amply demonstrates, but because the revision appears to have been assembled quickly (the harness commit follows the round-3 report by under an hour). The paper's credibility rests on its reconciliation discipline. The editor should require the artifacts before another round of review.

### M2. What the I_dyn evaluation shows, and what it does not.

The I_dyn column is the right experiment and potentially the paper's strongest new evidence. As implemented, it is weaker than the text claims.

1. **An undisclosed subsample.**
   - The harness calls `_message_flow_labels(..., max_candidates=30)`, which keeps the first 30 entries of `labeled_node_ids`.
   - Those are sorted as strings. On the three folds I probed they are Applications `A0, A1, A10, A100, A101, …`.
   - I_dyn therefore scores 30 of 300 Applications on Enterprise and 30 of 200 on IoT Smart City, selected by identifier string order rather than at random.
   - The generator's identifier order may correlate with component role.
   - Table 7's caption says "Application population". Please report n per fold, justify the selection or replace it with a seeded random sample, and show that the conclusions hold on the full population for at least the folds where it is affordable.
2. **No inference.** Table 7 reports means only: no confidence intervals, fold wins or paired tests, although the harness computes CIs and imports `wilcoxon`. I_dyn runs at one seed, and its own test–retest is 0.74–0.97 (Supplementary S3.1). The paper removed that floor from §4.3; it must be restated beside Table 7.
3. **"Without architectural circularity" overstates independence.** I_dyn simulates message flow over the same publisher→topic→subscriber structure. A failed publisher lowers the delivered rate of exactly its subscribers. I_dyn is independent of I\*'s breadth-first propagation, threshold and damping rules, not of the subscriber relation that `InDeg` counts. What Table 7 shows is informative but narrower:
   - `InDeg` (0.610) predicts I_dyn about as well as I\*'s own first-order term (0.636);
   - `InDeg` clearly beats `Topo-QoS` (0.393) under a queueing model.

   Please reword the abstract ("without architectural circularity"), §7.1 ("confirms that … not an artifact"), §8.3 and §9 accordingly.
4. **The recommendations are now contradicted by the paper's own table.**
   - `Reach` drops to 0.505 on I_dyn, while Table 11 and §8.1 still recommend `Reach` for "unfamiliar" systems on the strength of the I\*-only 0.938.
   - §8.3 states that "the dependency counts and attention networks maintain robust rank correlations across these independently formulated oracles". That is not true of `Reach`, whatever becomes of the I_comp column.

   State that `Reach`'s advantage is specific to I\*, or evaluate it on I_dyn for the system models.
5. **Registration status.** The I_dyn/I_comp evaluation has no amendment in `PREREGISTRATION.md`. Label it post hoc and exploratory in the caption and text.

### M3. New claims that the evidence does not support, and caveats removed that still apply.

**Claims added in this revision:**
- **"Orders of magnitude" (abstract).** "Neural feature extraction takes orders of magnitude longer than direct simulation" is false by the paper's own §7.4: 2.0–17.7×, median 5.6×.
- **"Pre-registered" (abstract, highlight 5).** The previous version correctly said *registered rather than pre-registered*, because the plan is a file in the authors' repository without a third-party timestamp, and several amendments were written after related results were known. §6.3 dropped that sentence. Restore it, and remove "pre-registered".
- **QoS and "our independent dynamic evaluations" (§3.2).** "Declared contracts provide actionable signals primarily when explicit queueing dynamics or SLA penalties are modeled, as shown in our independent dynamic evaluations" is unsupported. No QoS-aware vs QoS-unaware comparison exists on I_dyn, and the one QoS-weighted ranker in Table 7 (`Topo-QoS`) is the *worst* on I_dyn.
- **Typing (§7.1, §8.2).** "Relational parameterization impairs convergence on sparse dependency graphs" and "relational parameterization introduces optimization instability without delivering predictive gains" generalize from a single untuned configuration: `HGT-P-QoS` at width 100, one learning rate, reported as registered without tuning. Round 3 asked for this conclusion to be scoped; it was broadened. Either run a small, registered tuning sweep for `HGT-P-QoS` (learning rate × width, inner-CV), or restate: "in the one untuned configuration we ran, HGT did not train stably; typing remains untested where message passing reaches the scored nodes."
- **"Robustness across independent failure paradigms" (§8.1)** is offered as a practical advantage over simulation. See M2(4) and M1(a).
- **The practical case for static analysis (§7.4 and §8.1)** now rests on "explainable ISO/IEC 25010 remediation profiles". §8.4 calls the same layer "an uncalibrated design proposal … not validated". The paper cannot use an unvalidated component as its answer to the utility question. The other argument ("design-time triage before simulation parameters are calibrated") is reasonable for the counts, which have no parameters, but it should be stated as an argument, not as a finding.

**Caveats that were removed but still apply.** Please restore them, concisely:
- **The drift ledger.** Learned cells move by up to 0.172 across code revisions and devices, and HGT-QoS by 0.041. Several effects reported here are of the order of 0.07, so this is essential context for every learned-engine figure.
- **Anti-conservative tests.** The Wilcoxon tests are anti-conservative because folds share ten of eleven training scenarios.
- **Single-modeller system models.** The five system models were each written by one author, no second modeller has re-derived them, and the agreement tool exists.
- **Evaluation scope.** Rules 2–4 and 6 are not validated by the Application-population evaluation (still in §3.3, but gone from §8.4), and no published learned-criticality model (FINDER, DrBC) was reproduced.
- **The I_dyn/I\* agreement and floors in §4.3.** I_dyn agrees with I\* at ρ = 0.627, and I_dyn has its own test–retest floor. Readers need both to interpret Table 7.
- **Zero-shot inert/active separation (§7.3).** The counts separate inert from active components by construction, which makes the system models an easier target for them.

### M4. Items from earlier rounds that remain open.

- **Figures 1 and 3** still depict "three ranking engines" and a "QoS-weighted `DEPENDS_ON` projection", with no dependency-count path. The counts are the paper's recommended instrument.
- **Data Availability.** It still links the tag `jss-submission-v4`, which is not in the repository and, per Amendment 9, names an older manuscript version. It still states "511 reported figures". Confirm that the cited Zenodo *version* DOI contains the Amendment 7–10 artifacts and the new Table 7 artifacts (M1(c)).
- **The supplement is unchanged.** It has no section for the new analyses, and parts of it contradict the main text (the 0.582 / 0.526 discrepancy; M1(g)).
- **Emphasis on hybrids.** They remain Contribution 2 and a summary headline, although Table 11 no longer recommends them. That is acceptable as a registered result, but keep it to one sentence.

---

## 4. Minor Comments

**Text and cross-references**
1. **`manuscript.md` is out of sync with the LaTeX source.**
   - It contains duplicated top-level headings (§4 and §8 each appear twice).
   - It contains raw BibTeX keys in the text ("[meiklejohn2021service]", "[53, rud2006ais]"; [53] *is* Rud et al.).
   - Its reference list still has 93 entries and none of the new ones.
   - `sections/*.md` is stale throughout.

   Regenerate the Markdown from the LaTeX source, or remove it from the submission package.
2. **Broken cross-references after renumbering.**
   - §6.2 cites "§4.1.1", which no longer exists.
   - §6.2 says "four centralities still carry QoS (§3.4)". §3.4 is now "Dual Graph Views", and the sentence explaining that four centralities are QoS-weighted was deleted from §3.5. Restore it, because it explains why the "QoS-off" arms are not QoS-free.
   - §5 cites "§3.4" for node properties; that is now §3.5.
3. **§4.1 is titled "… and Attention Networks" but never describes the GAT arms.** Add one sentence on their layers, heads and widths (296/288 on the projection).

**References**

4. **New references to check:**
   - **Meiklejohn et al. (SoCC '21).** The bibliography lists "Meiklejohn, Sharma, Miller". To my knowledge the paper is by C. S. Meiklejohn, A. Estrada, Y. Song, H. Miller and R. Padhye. Please verify against DOI 10.1145/3472883.3486997.
   - **MicroRank.** Please verify the last two authors against DOI 10.1145/3442381.3449905.

   The Guide requires every reference to correspond to a real source with correct data.

**Tables**

5. **Table 6.**
   - Mark the analytic row as a post hoc reference, outside any registered family. Its p-value currently reads as part of the Amendment 7 family.
   - `Hybrid-GAT-P` has the highest ρ₍>0₎ (0.537), above `InDeg` (0.516). Mention it in the text, or note that it is within noise.
6. **Table 7 content.**
   - Add n per fold, fold count, seed, candidate selection, confidence intervals and win counts (M2).
   - Drop or recompute the I_comp columns (M1(a)).
   - The caption calls I_comp "the multi-criteria failure simulator"; as scored, it is not (M1(a)).
7. **Table 9 lost its per-system rows.** Table 11 and §8.2 still cite per-system and RPC-split values (e.g. ρ₍>0₎ = 0.813–0.976 on the RPC-derived models) that can no longer be traced in the main text. Restore a compact per-system view, or point to the supplementary table.

**Guide compliance**

8. **`LENGTH_JUSTIFICATION.md` is stale.** It says 26 pages (the PDF has 23), "96+ entries" and 11 tables (there are 12). It also states that the supplement contains "independent oracle evaluations", which it does not. Correct it, or drop it, since 23 pages needs no justification.
9. **Highlights.** Each is ≤ 85 characters ✓. Two changes are needed:
   - `highlights.tex` uses LaTeX math (`$\rho = 0.76$`), but Elsevier's highlights are a plain-text field. Use "ρ = 0.76".
   - Highlight 5 ("Pre-registered study shows limits of complex GNNs over structural counts in CI") has the registration issue from M3, and "in CI" is unclear.
10. **Abstract.** 250 words or fewer ✓. Beyond the corrections in M2 and M3, give the I_dyn figure and its scope, e.g. "ρ = 0.610 on a 30-Application sample per fold". "Evaluated against an independent … simulator … maintains predictive rank correlation" is otherwise uncheckable.
11. **Vitae.** Completed, 84 and 87 words ✓.
12. **Declaration of generative AI.** Present and correctly placed ✓. Since the new harness and the tables built from it were produced during a very short revision, say whether AI assistance extended to generating or checking reported numbers.

---

## 5. Recommendation

**Major Revision.**

**Justification.** The authors took most of the round-3 requests seriously. The paper is now honestly framed, correctly positioned against afferent coupling and the simple-baselines literature, and much tighter. The I_dyn experiment it adds is the right one, and on the part I could check, the released harness reproduces the I\* figures for `InDeg`.

The paper cannot be accepted in its current form, because the evidence introduced to answer the central circularity question is not yet sound:
- one of its two "independent oracles" is the framework's own explanation score (M1(a));
- the learned-engine row on those oracles has no code behind it (M1(b));
- the I_dyn column is an undisclosed, identifier-ordered 30-component subsample with no inference (M2);
- the QoS confound is "resolved" by an unreported experiment (M1(d));
- several numbers disagree with the released harness or with each other (M1(e)–(g));
- new unsupported claims were added and valid caveats removed (M3).

**Conditions for acceptance.** These are largely editorial and computational, not conceptual:
- **(i) Artifacts.** Commit the artifacts and scripts for every new number, extend the reconciler to cover them, and add a supplementary section with per-fold values.
- **(ii) I_comp.** Recompute it from the simulated impact components, or remove it.
- **(iii) I_dyn.** Re-run it on a disclosed, randomly seeded sample, or on the full population, with confidence intervals and paired tests. State its seed floor.
- **(iv) The QoS-column control.** Report it in full, or drop the claim.
- **(v) Text.** Correct the claims listed in M2 and M3 and restore the removed caveats.

If (i)–(iii) confirm that `InDeg` leads closed-form centrality and matches the attention networks on I_dyn, the paper will be a clean, well-evidenced negative result for graph learning in architecture-level dependability analysis, well suited to JSS and defensible for this special issue.
