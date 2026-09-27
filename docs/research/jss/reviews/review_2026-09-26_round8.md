# Referee Report (R1) — JSS Special Issue VSI:AI4MSS, round 8

**Manuscript:** *Software-as-a-Graph: Benchmarking Centrality and Graph Learning for Pre-Deployment Cascade-Impact Ranking in Publish–Subscribe Systems*
**Material reviewed:** `manuscript.md` (generated from `latex/`, 27 pp. PDF), `highlights.tex`, JSS Guide for Authors; `PREREGISTRATION.md` and `experiments/rq4-cost.md` for timing claims.
**Date:** 2026-09-26 (round 8) · **Recommendation:** Major Revision · Response: [response_round8.md](response_round8.md)

---

## Summary

The paper models publish–subscribe architectures as typed multigraphs, derives an explicit `DEPENDS_ON` dependency graph, and benchmarks closed-form centrality, GNNs (GAT, HGT) and hybrid learners against three simulators under LOSO over twelve synthetic architectures and zero-shot on five hand-authored models of open-source systems. Its central claim is a largely negative result: the registered primary contrast (HGT-QoS vs. Topo-QoS) is null, only centrality-corrected hybrids beat centrality, and no learner exceeds dependency counts that the authors reclassify as "references" because they restate the primary oracle's first propagation wave.

## Overall assessment

- **Originality — moderate.** Pre-deployment cascade ranking for pub-sub is under-served; the typed dependency derivation is a sensible engineering contribution, but its novelty over [25] is unclear. The benchmark framing is honest, and JSS welcomes negative results.
- **Significance — limited as it stands.** The primary oracle is a cheap deterministic function of the same manifest the predictors read (0.08–4.5 s) and its first wave is, by Proposition 1, the in-degree. The benchmark therefore largely measures how well models approximate a known computable function, which the paper concedes.
- **Methodological soundness — mixed.** Transparency is exemplary; but the confirmatory/exploratory boundary is drawn inconsistently, the decisive ablations are "not run", no arm is tuned, seed/device drift exceeds several reported effects, the circularity argument is applied asymmetrically across oracles, and the cost comparison contradicts the paper's own scaling tables.

## Major comments

- **M1. The primary task restates its own ground truth.** Either reframe as surrogate modelling evaluated where the oracle is expensive, or anchor part of the evaluation on ground truth the predictors do not restate (EdgeX, Home Assistant, Train-Ticket are runnable; Train-Ticket has a published fault benchmark).
- **M2. Circularity is applied asymmetrically.** I_comp's fragmentation/flow terms respond to removal of high-betweenness/high-degree nodes, so Topo-QoS (0.702) and Degree-raw (0.719) "winning" there is the same kind of restatement. State one operational criterion for a reference and apply it to all three oracles. Say plainly that afferent coupling matches every learner under a reachability notion of impact.
- **M3. "Never exceeds the reference" rests on a non-significant difference** (Holm p = 1.000) and on the aggregation statistic: the five-seed ensemble (0.772) is above InDeg (0.764). Use an equivalence test (TOST with a declared margin) or "statistically indistinguishable".
- **M4. Confirmatory status of the hybrids, and the nested comparator.** Amendments 5–6 postdate the primary null, the status the paper uses to call Amendments 7/9/10 exploratory. The hybrid nests Topo-QoS; test hybrid vs its own base learner and vs unweighted betweenness (0.591) / constant weights (0.595). The prior hurts zero-shot (0.662 vs 0.805) and on I_comp. Report the hybrid contrasts under |V_app| weighting.
- **M5. "Registered" overstates what was done.** Self-hosted plan, 13 amendments in 20 days, one post-results reclassification. Provide external timestamps or say "version-controlled analysis plan"; move the amendment chronology out of the main text.
- **M6. RQ2 cannot answer its own question; decisive ablations are cheap.** The QoS factor zeroes w_in; every learner reads in-degree. Run the 2×2 with w_in held, and learners without in-degree/w_in. Softmax attention cannot count neighbours (Xu et al. 2019; Corso et al. 2020): add a sum-aggregation arm without degree features.
- **M7. No tuning; seed and device instability larger than the effects.** Equal-budget nested selection (Errica et al., cited by the paper); hierarchical fold×seed bootstrap; no negative typing claim without minimal tuning.
- **M8. The cost claim contradicts the paper's own tables.** On the generated graphs of Tables 10–11 the analysis stage (1.74/8.32/44.5/239 s) is cheaper than one I* pass (4.0/15.1/58.5/347 s). Reconcile; per-feature breakdown; qualify the headline; drop the sustainability framing or measure energy.
- **M9. Zero-inflation drives headline numbers.** Report ρ>0 in the abstract; "learned engines ≈ 0.81" holds for two of five.
- **M10. External validity.** Single generator; five single-author models; I_dyn on a lexicographic n = 30 sample. Complete Amendment 11 or use a stratified random sample; second-modeller replication.
- **M11. Scope, novelty, verbosity.** State overlap with [25]; Proposition 1 is a definitional identity; move the unevaluated §5 and Figure 4 to the supplement; consolidate the circularity caveat (≥ 10 repetitions); simplify main tables.

## Minor comments

1. Abstract at the 250-word limit; add ρ>0.
2. Table 9 lacks block sub-headers (Topo/Topo-QoS read as references).
3. Every row label is bold in Tables 6/7/9, so "bold = best" is ambiguous.
4. Table 7 bolds Topo-QoS partial ρ as "best" though it is the only predictor computed; compute learned partial ρ.
5. "0.017 below" vs displayed 0.764 − 0.748 = 0.016.
6. Table 6 ‡ note vs which contrast p = 0.0068 belongs to; Holm p = 1.000 vs InDeg contradicts "no contrast against references".
7. What do the five "tie-breaking seeds" of a BFS oracle randomise?
8. â₂ objective undefined; ListMLE with ~31% tied labels; inner validation split scenario- or node-level?
9. Are I_comp's AHP weights among the back-filled matrices of S4?
10. Proposition 1 vs Rule 1 "incl. transitive USES": do library-mediated subscribers and Libraries count?
11. Rules 2–4 and 6 "not exercised": mark or move.
12. Nadeau–Bengio corrected test or permutation test alongside Wilcoxon.
13. Tercile grouping on the comparator's own score induces regression to the mean.
14. Missing literature: GNN counting/expressivity; GNN centrality approximation; Train-Ticket benchmark; Hellendoorn & Devanbu 2017.
15. Typography ("explicitly; Because", "documentation ;", mixed British/American spelling).
16. Regenerate `qos_label_ablation.json` with provenance (it backs main-text ρ = 0.965/0.977).
17. Duplicated URL in [87].
18. Define p(v) in Figure 3's caption.
19. Compliance checks pass (highlights 70–83 chars, 7 keywords, 27 pp, GenAI declaration title, graphical abstract).

## Recommendation

**Major Revision.** Transparent and reproducible; a well-argued negative result fits JSS. Not acceptable as is because of M1, M2/M3, M4, M6/M7 and M8; each is fixable within a revision cycle, mostly with CPU sweeps the tooling already supports.
