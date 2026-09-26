# Response to the round-7 referee report

The report is [review_2026-09-26_round7.md](review_2026-09-26_round7.md), and the revision is on branch `jss-revision-round7`.

All new analyses are **Amendment 12** in [PREREGISTRATION.md](../PREREGISTRATION.md). It was registered before its arms ran, and its deviations are logged beneath it. The new analysis script is [reproduce/referee_round7.py](../../../../reproduce/referee_round7.py).

We thank the referee. Several comments changed the paper's conclusions, not just its wording:
- **M2:** the derivation claim was withdrawn.
- **M4:** the recommendation now depends on the oracle.
- **M10:** the safety margin was replaced by a measured one that is much less favourable.

A defect found while answering M2 also corrected a published supplementary number (Amendment 10's Pubs-raw arm).

## Major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | Why predict `I*` when it is computable from the same inputs? | Stated plainly in §1.2 and §8.1: all ground truth is simulated, and `I*` reads the same manifest. We make no claim about observed outages. The remaining practical case is measured, not asserted: counting costs 1–29 ms per corpus architecture against 0.08–4.5 s for `I*`, and `I*`'s labelling time grows roughly quadratically with size (R6). The paper is repositioned as a benchmark with a largely negative result for graph learning (title, abstract, §1.4, §2.4, §9). External validation is listed as open. | Title, abstract, §1.2, §8.1, §8.4, §9 |
| M2 | "Unlocks" is unsupported; add raw-graph baselines | Proposition 1 proves `InDeg` = typed two-hop count, verified max \|Δ\| = 0 on 17 graphs. Raw-graph rankers were added (R1): degree 0.199, reverse PageRank 0.089, PageRank undefined (constant), topics published 0.731, Reach-R1 0.674. The paper now says the +0.211 is a property of choosing afferent coupling, not of the projection; the projection is credited only with Rule 5's +0.058 for Reach. "Unlocks" and "formal derivation" are removed. | §3.3, §6.2, Table 5, Table 6, §7.1 |
| M3 | Circularity acknowledged but not resolved; `I_dyn` is a 30-node lexical sample | "Without architectural circularity" and "genuine … rather than artifacts" are removed. §4.4 now states the construct overlap and bounds it two ways. (1) Partial correlations (R2): `InDeg` keeps 0.259 [0.143, 0.367] with `I_dyn` beyond `I*`, and 0.163 beyond the first-order expansion; Reach keeps nothing. (2) Learned engines are scored on both other oracles (R3); none beats `InDeg` on `I_dyn`. **Not done:** the full-population `I_dyn` labels. Amendment 11's labelling was stopped at the authors' request after an out-of-memory failure, so every `I_dyn` number still rests on the lexical n = 30 sample, and the paper says so wherever `I_dyn` appears. Disattenuation is replaced by citing the published test–retest range (0.74–0.97) as a bound. | §4.3, §4.4, Table 7, §7.1, §8.3 |
| M4 | `I_comp` contradicts the default and was reported selectively | Topo-QoS (0.702) and raw degree (0.719) beat `InDeg` (0.650, 3/12 wins) on `I_comp`. This is now in the abstract, §3.2, §7.1 and §8.1. Table 11 is now organised by failure notion (oracle). `I_comp` is defined exactly (weights 0.35/0.25/0.25/0.15 and the fragmentation split). | Abstract, §3.2, §4.3, §7.1, Table 11 |
| M5 | Headline results are exploratory amendments | The abstract, §1.2 and §1.3 open with the null registered primary. §6.3 has a new paragraph, "what is confirmatory and what is not", giving amendment dates relative to the primary result. "True driver" and "unlocks" are removed. | Abstract, §1.2–1.3, §6.3, §8.3 |
| M6 | "Sub-millisecond" and "sustainability" are unmeasured | "Sub-millisecond" is removed everywhere; `InDeg` alone is ~0.1 ms but the projection is not. Counting-path latency is now measured per corpus fold and by size up to 10,000 components (R6, new cost table in §7.4). "Sustainability" is replaced by "wall-clock cost", and we state that no energy was measured. Separately, the `inference-latency` make target is fixed to write the Table 10 artifact it did not reproduce. | §7.4, §8.1, highlights, `reproduce/Makefile` |
| M7a | QoS factor confounded with weighted in-degree | Table 8 relabels the factor as "QoS" inputs, and the caption says it includes `w_in`. The text states the confound and that the rerun with `w_in` kept was not performed. Figure 5(C) title and caption are corrected. | Table 8, §7.2, Fig. 5 |
| M7b | Typing claim rests on one untuned configuration | Claims are restricted to "the single untuned configuration" in the abstract, §1.4, §7.2, §8.2 and §9. "Standard defaults" is corrected to "fixed before the first sweep, not tuned". The seed ensemble lifts HGT-P-QoS to 0.618, which we report. **Not done:** a hyperparameter search. | §4.2, §7.2, §8.2, §8.4 |
| M7c | Learners are given in-degree as a feature | Stated in §3.5, §7.1 and §8.2. GAT-P+InDeg adds nothing to the count. **Not done:** a learner without the in-degree features. | §3.5, §7.1, §8.2 |
| M8 | Explanation layer unevaluated | §5 is retitled "A Proposed Explanation Layer (Not Evaluated)" and shortened to one subsection. Its formulas moved to the new Supplementary section S-rm-formulas. It is removed from the contributions and from Table 11. | §1.4, §5, Supp. |
| M9 | Zero-shot evidence weak; mixed harnesses | Table 9 is now scored on one harness; `InDeg` and `Reach` are unchanged on the learned engines' labels. New structural descriptors show the system models have more inert Applications (0.51 vs 0.31) and more concentrated fan-in (Gini 0.65 vs 0.50), which explains Reach's jump. The CIs are marked indicative. "Zero-shot" and "held-out" are no longer applied to training-free rankers (§6.3 "per-scenario evaluation"). The single-modeller threat is stated as unmitigated. **Not done:** a second modeller. | §6.3, Table 9, §7.3, §8.3 |
| M10 | Overlap@K ≈ 0.5 contradicts "highly accurate"; safety margin unsupported | Tie-aware recall@k curves were added (R4, Figure 6). The claimed "top 30–35% catches 80–90%" was **wrong**: at 30–35% `InDeg` catches 62–68%, and 80% needs the top 45%. §8.1 now derives the margin from the curve and concludes the count is an ordering for review, not a sharp gate. Tie handling is documented in §6.3. | §6.3, §7.1, Fig. 6, §8.1 |
| M11 | Drift exceeds effects | Seed spread is now reported for every learned engine: the mean within-fold SD is 0.099 for HGT-QoS and 0.254 for HGT-P-QoS, and single HGT seeds differ by more than 1.0. The earlier "seed spread 0.208" was a constant typed into the reconciler; it is now read from an artifact. §8.3 states that learned cells come from named CPU sweeps whose per-seed logs reproduce the published means, and that the conclusions rest on the training-free rows. | §8.3, Supp. S-referee |
| M12 | Overclaiming and verbosity | Every "Crucially", "outstanding", "exceptional", "decisive", "genuine", "rigorous(ly)", "fully legible", "essential benchmark" and "Occam's razor" is removed. The abstract, §1 and §9 are rewritten; §1.2's thesis paragraph and the two-tier taxonomy are replaced; the QoS apparatus is trimmed; Rules 2–4 and 6 are marked "defined, not exercised". | Throughout |

## Minor comments

| # | What changed |
|---|---|
| 1 | The abstract is 246 words (raw count) and uses no undefined abbreviations. The reconciler now enforces ≤ 250 words. |
| 2 | Highlights rewritten, 71–80 characters each. The reconciler enforces ≤ 85. |
| 3 | `HGT-QoS-U` and `GBM-Feat` are now introduced in §7.2. The "Table S36" reference is correct: it is Supplementary *Table* S36 (the per-system table), not a section, so the report's comment was mistaken on that point. |
| 4 | The Topo equation is labelled. In the PDF it was already numbered; the gap existed only in the Markdown rendering. |
| 5 | The comparator defect is marked in Table 6 (¶) with the corrected values (0.329 and 0.533) in the caption and in §6.2. |
| 6 | Averaging and bootstrap unit stated in §6.3. Fisher-z preserves every row's order; size weighting preserves the top five but lifts Topo-QoS level with the raw-graph GNNs, and this is reported. |
| 7 | "Summary" lead-ins are paragraph headings in the LaTeX; the Markdown is regenerated. |
| 8 | "Formal" removed; Proposition 1 with proof added. |
| 9 | "Sub-millisecond" removed. "Zero-shot" and "held-out" are no longer applied to training-free rankers. "Hybrid-GAT-P" is renamed **GAT-P+InDeg** (registry label and legacy map), so "Hybrid" always means a Topo-QoS prior. |
| 10 | Table 5 narrowed to four columns; Table 11 rebuilt with short cells. |
| 11 | Table 7 rebuilt: the redundant `I_comp` ρ>0 columns are gone; CIs and a partial-ρ column added; learned rows added. |
| 12 | Implementation paths removed from §4, §6 and §8 prose. |
| 13 | §3.2 states that the topic-QoS matrix was stated independently (CR 0.016, non-degenerate) and that three other matrices encode a declared vector. |
| 14 | DDS and SQuaRE capitalization fixed. Beliakov is now a book with a DOI; Khodabandeh now has pages and a DOI. **Reference [56] (Santos et al., TSE 47(10) 2019) could not be found in Crossref** and was replaced by the verifiable HAROS paper (IRC 2019, DOI 10.1109/IRC.2019.00018); the replacement was confirmed by the authors on 2026-09-26. |
| 15 | Length justification updated to the compiled page count. |
| 16 | Graphical abstract added (`latex/figures/graphical_abstract.pdf`, 13 × 5 cm, 1535 × 590 px ≥ 1328 × 531 required), every number read from an artifact (`reproduce/render_graphical_abstract.py`). |
| 17 | Data statement now says which reported numbers depend on the four unstamped artifacts (only §4.3's 0.965/0.977). |
| 18 | Results claims removed from §1.1. |
| 19 | Figure 1 caption now says what the dashed route is: the proposed layer is applied after ranking and receives no predictor output. |

## Corrections found during the revision

- **Amendment 10, Pubs-raw.** Scored on publishers only because of a population defect. Corrected from 0.431 to 0.731, so `InDeg`'s margin over it falls from +0.334 to +0.033 (decision E1 still applies). The artifact is regenerated from a clean tree; the supplementary table, amendment log and experiment page are updated.
- **Reconciler.** Two hard-coded "truths" (seed spread 0.208; GAT-P-QoS 0.748 in §8.2) are replaced by artifact reads.
- **GenAI declaration.** It now also covers drafting of revision text; the updated declaration was approved by the authors on 2026-09-26.

## Author-initiated change after this response (Amendment 13)

The authors decided not to use `InDeg` and `Reach` as predictors, because they restate the primary oracle's propagation rule (Proposition 1). They are now reported as *references* next to the first-order expansion of `I*`, and carry no contrast against `Topo-QoS`. They no longer appear in the title, abstract, highlights, conclusion, predictor taxonomy or guidance table, and `GAT-P+InDeg` has moved to the supplement. No number changed. The M3 answer above still holds: the partial correlation is kept as the bound on how much of the reference survives outside `I*`. The new title is "Software-as-a-Graph: Benchmarking Centrality and Graph Learning for Pre-Deployment Cascade-Impact Ranking in Publish–Subscribe Systems". See `PREREGISTRATION.md` Amendment 13 and `experiments/amendment13-reference-demotion.md`.

## Not done, and stated as limitations (§8.4)

- validation against observed outages;
- a hyperparameter search;
- the 2×2 rerun with `w_in` kept;
- a learner without in-degree features;
- full-population `I_dyn` (Amendment 11, stopped);
- a second modeller;
- energy measurement.
