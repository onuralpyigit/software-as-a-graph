# Referee Report (R2) — JSS Special Issue VSI:AI4MSS, round 8

**Date:** 2026-09-26 (round 8) · **Recommendation:** Major Revision · Response: [response_round8.md](response_round8.md)

> **Editorial note (authors).** This report was evaluated against the round-7 manuscript before
> the revision. Points already satisfied by that text, or resting on stale wording, are answered
> as such in the response letter. In particular: the quotation "parameter-heavy relational typing
> introduces severe optimization instability" (Major 3) is from a pre-round-7 draft and is not in
> the reviewed manuscript; Major 1 compares the generated-graph analysis times (1.74–239 s) with
> the corpus oracle times (0.08–4.5 s); and Major 2's statement that the QoS-off arms remove the
> raw in-degree column is incorrect (only `w_in` is zeroed).

---

## Summary

The manuscript investigates pre-deployment cascading-failure impact prediction in distributed publish–subscribe architectures, where asynchronous broker- and topic-mediated decoupling obscures structural fault propagation paths. The authors present Software-as-a-Graph (SaG), a static framework that derives an explicit `DEPENDS_ON` dependency projection from deployment manifests and benchmarks closed-form centrality metrics, graph neural networks (GAT, HGT), and hybrid models against three simulation oracles across twelve synthetic scenarios (under leave-one-scenario-out cross-validation) and five hand-modeled open-source systems. The primary contribution is a rigorous empirical benchmark—largely conveying a negative result for complex graph learning—demonstrating that GNNs fail to outperform simple structural reference counts (which essentially restate the reachability simulator's initial propagation wave) while imposing a feature-extraction latency that exceeds the cost of running the reachability simulation directly.

## Overall Impression & Assessment

1. **Originality.** The dependency derivation rules are an elegant contribution to architecture modeling; `InDeg` is proven (Proposition 1) equivalent to pub-sub afferent coupling; the ML architectures are standard. The originality lies in the critical benchmarking role.
2. **Significance.** Rigorous negative results have community value (Fu & Menzies 2017; Premraj & Herzig 2011; Ferrari Dacrema et al. 2019). Tempered by the question: if `I*` takes 0.08–4.5 s from the same manifests, the operational incentive for proxies is narrow.
3. **Methodological soundness.** Exemplary transparency (13 amendments, procedural isolation, null reported, 1,347 reconciled figures). Vulnerabilities: (1) construct circularity, with in-degree features given to the GNNs; (2) the RQ2 2×2 zeroes `w_in` in the QoS-off arm; (3) untuned neural baselines; (4) `I_dyn` on 26–30 lexicographically sorted nodes per fold; (5) the unvalidated §5 explanation layer.
4. **Special Issue fit.** Fit with "Reliability" and "Performance" is good; with "AI Techniques" and "Sustainability" strained; sustainability rests on wall-clock time only, with no energy, RAPL or CO₂e measurement.

## Major Comments

1. **The operational paradox.** Why train a 430k-parameter GNN (Overlap@20% 0.454) with a 5.6× feature-extraction penalty to approximate a deterministic simulation executable in seconds? Articulate this in the introduction and discussion; frame the paper as a diagnostic investigation into the limits of AI for static reliability analysis.
2. **Confounded factorial design (RQ2) and incomplete feature ablation.** The QoS-off arm zeroes `w_in`; no learner without `in_degree`/`w_in` was evaluated. *Required:* run the ablation, or downgrade RQ2's claims.
3. **Overgeneralized claims from untuned baselines.** Conclusions rest on one untuned configuration (width 64/288, lr 3×10⁻⁴, loss weights 1.0/0.5/0.3/0.1, 3 layers); Errica et al. [87]. *Required:* restrict typing claims to the single configuration; add an under-tuning threat to §8.3.
4. **Sampling bias in `I_dyn`.** First 26–30 Applications per fold in lexicographic order (Amendment 11 aborted after an out-of-memory failure); the partial correlation 0.259 rests on it. *Required:* flag the convenience sample in §7.1 and Table 7; moderate the construct-validation claim.
5. **Unvalidated explanation layer (§5).** A design proposal; three AHP matrices encode declared vectors; moving toward elicited judgments lowers ρ to 0.200. *Required:* condense into §8.4 and move Figure 4 to the supplement.
6. **External validity of RQ3.** Single-modeller bias; two RPC systems re-expressed as pub-sub; 51% inert Applications inflate full-population ρ; ρ>0 ≤ 0.342. *Required:* call them hand-crafted architectural abstractions; emphasise weak active-stratum transfer.
7. **Sustainability.** Wall-clock time is not sustainability; no energy, power or carbon measurement. Quantify energy (RAPL, CodeCarbon) or frame a Green-AI critique.

## Minor Comments

1. Abstract is 252 words by raw count; trim to ≤ 250.
2. Highlights comply (76, 82, 83, 73, 70 characters); keep as a separate editable file.
3. Label the defective Topo rows in Table 6 (e.g. "Topo-QoS (Betweenness only)").
4. Mention Degree-raw's 0.719 on `I_comp` in the abstract.
5. Harmonise node- vs edge-weight notation across §3.1, Table 1 and Table 3.
6. Strengthen Green-AI-in-SE grounding in §2.1.
7. Re-stamp the four provenance-less artifacts on Zenodo.

## Recommendation

**Major Revision** — unresolved RQ2 confound; generalisations from one untuned configuration; `I_dyn` sampling bias; speculative §5; no empirical sustainability metrics.
