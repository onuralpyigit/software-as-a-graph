# Response to advisor review notes v5 and v6 (2026-09-29)

Branch `jss-advisor-v6-revision`. Sources: `advisor_review_notes_v5.docx`, `notlarv6.docx`, `Eq_X.docx` and the accompanying email.

## New central claim

The question is now *"When does graph learning provide value beyond explicit dependency analysis?"*. The paper's centre is deriving dependency semantics that make quantitative impact analysis possible, and approximating simulators. "Surrogate" has been replaced by "(learned) approximation" throughout the main text, the highlights and the supplement. Engine IDs such as `GBM-P-QoS→dyn` are unchanged.

## What was adopted verbatim

| Item | Where |
|:--|:--|
| Title: *Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish–Subscribe Systems?* | `manuscript.tex`, `title_page.tex` |
| Abstract (243 words, under the 250 limit) | `sections/abstract.tex` |
| Intro 25–29, 31–48, Findings in brief (49–77), §1.3 lead-in and RQ1–RQ4, the l.95 wording, Contributions (99–132) | `sections/sec1_introduction.tex` (exceptions below) |
| Conclusion as prose | `sections/sec9_conclusion.tex` |
| Discussion restructured into 7.1 *Dependency Representation versus Model Complexity*, 7.2 *When Learning Adds Value, and Relative to What*, 7.3 *Restatement versus Prediction*, 7.4 *Practical Consequences* (old 7.1), 7.5 *Threats* (old 7.3), 7.6 *Limitations and Future Work* (old 7.4), with every paragraph replacement you listed | `sections/sec8_discussion.tex` |
| Energy wording: "upper bound" became "estimate" in §6.4 and §7.5; the turbo-boost sentence was added; "19× speedup and massive energy savings" was deleted | `sec7_results.tex`, `sec8_discussion.tex` |

Cross-references use LaTeX labels, so the renumbering you listed for lines 47, 76, 151, 301, 398, 547 and 607 happens automatically. We checked it in the compiled PDF.

## Eq_X: Option A

- Eq. (6), the first-order expansion of I*, moved from §6.1 into the §5.2 reference paragraph.
- The rate-weighted expansion was added right after it as **Eq. (7)** (`eq:rate-expansion`), followed by your sentence.
- §6.1 now points to Eq. (6) instead of defining it.
- §4.4 got your replacement sentence.
- Table 6 has a new reference row, "Rate-weighted (Eq. (7))", with its CI and partial correlations, and your caption text.
- The inline definition in §7.2 is gone; §7.2 now cites Eq. (7).
- Numbering holds: Eq. (5) is Topo-QoS, Eq. (6) is Î*₁, Eq. (7) is the rate expansion. No later equation was renumbered.

## Changes to your text, and why

1. **The queue-flow signal comes from declared rates and payloads, not QoS contracts.** Your texts say "declared QoS contracts carry measurable signal only on this simulator" (Findings), "QoS information, which adds signal only on the queue-flow simulator" (Contribution 3), and "the only simulator on which declared QoS contracts carry measurable signal" (§7.4). The registered Family C "QoS feature block" also contains each Application's declared publication rate and rate × payload. We split it (Amendment 15, exploratory), using the same learner, seeds and I_dyn labels:

   | Features on I_dyn (LOSO) | ρ |
   |:--|:--:|
   | S (structural only) | 0.704 |
   | S + rate, payload | **0.801** (+0.097, 9/12, Holm p = 0.021) |
   | S + QoS-derived columns only (w(t)-weighted scores and policy shares) | 0.700 (−0.005, p = 0.68) |
   | S + all Q (published) | 0.799 |

   The three sentences now say that declared publication rates and payload sizes carry the signal and QoS policies add none. On I*, QoS inputs were already n.s. (+0.030 with w_in held). So QoS policy information shows no measurable effect on either simulator.

2. **Citations.** [40, 41] (Zimmermann 2008, Premraj 2011) and [65, 66] (Hellendoorn 2017, Errica 2020) are correct for the defect-prediction and simple-baseline sentence. In "graph learning has recently been applied to dependency structures and software architectures [54, 55, 62]", refs [54, 55] are FINDER and DrBC, which are general network key-player methods. That clause now reads "applied to network criticality [54, 55] and to microservice architectures [62]". "[28, 40] … reliability and criticality analysis" became "reliability and defect analysis", because [40] is defect prediction.

3. **Two headings in §7.4** would both have read "Choosing a ranker". The second one is now "Choosing a ranker by failure notion".

## Evidence behind the numbers you added

The value ρ = 0.830 had no artifact before this revision. It now has one:
- **Script:** `reproduce/idyn_rate_expansion.py`, run with `make -f reproduce/Makefile rq-rate-expansion`.
- **Artifact:** `data/benchmarks/idyn_rate_expansion.json`, built from a clean commit.
- **Record:** Amendment 15 in `PREREGISTRATION.md` (post hoc, exploratory).
- **Gates:** it reproduces Amendment 11's Analytic-I* exactly and the published `gbm_dep_qos_dyn` value of 0.799.

Results:
- **Eq. (7) on I_dyn:** ρ = 0.830 [0.778, 0.872] on LOSO and 0.893 zero-shot. It takes at most 1.2 ms per architecture, which backs "in milliseconds".
- **Eq. (7) vs the learned approximation:** +0.031 [+0.013, +0.049], 10/12 folds, nominal Holm p = 0.009. So "matching or exceeding" is conservative.
- **Beyond I\*:** Eq. (7) keeps partial ρ = 0.578 with I_dyn after I* is removed. The rate term is exactly what I* lacks.
- **Other variants:** rate × payload reaches 0.748, and publication rate alone 0.786.
- **Saturation-aware variant:** not built. I_dyn runs at utilisation 0.65, and the plain rate form already exceeds the learned model.

`reproduce/reconcile_manuscript.py` has a new check, `check_rate_expansion`, covering the Table 6 row, every cell of the three supplement tables, and the quoted values. Mutation-tested: three planted errors were all caught. The full reconciler passes at 1,596 figures.

The other claims in your text were checked against the artifacts and hold:
- hybrids +0.103 and +0.130 over the baseline (11/12);
- hybrids are the most consistent learned models across the three simulators (worst case 0.585 and 0.573, against about 0.48 for the next);
- zero-shot 0.81 vs 0.526, with references at 0.863–0.938;
- ρ>0 ≤ 0.342;
- −0.136 without degree features;
- raw degree 0.719 and Topo-QoS 0.702 on I_comp.

## Edits outside the sections you reviewed (waiting for your reading)

These are consistency fixes only, so that §§2–6 no longer contradict the new thesis:

- **§2.4 last paragraph:** "positive for learned surrogates when the oracle is costly" became "on the two oracles for which a closed-form first-order expansion exists, that expansion … matches or exceeds every learned model".
- **§3.2:** QoS weights help the learned approximation through rates and payloads, not QoS policies.
- **§4.3:** "Independent queue-flow oracle" became "Queue-flow oracle". We added one clause saying "simulator" and "oracle" are used interchangeably, because your new texts say "simulator" and the rest of the paper says "oracle".
- **§5.2:** Eq. (6) and Eq. (7) (see above), and the amendment count changed from fourteen to fifteen. The analysis-plan paragraph lists Eq. (7) and the attribution as exploratory.
- **§6.1:**
  - the Summary box drops "highly accurate cascade-impact ranking" and "capturing declared QoS signals (+0.095)";
  - the queue-flow paragraph reports the attribution and the Eq. (7) contrast;
  - the Table 6 block is renamed "Learned approximations of I_dyn".
- **§6.4:** the summary box and last paragraph now say the rate reference, not a learned model, is what saves the ~355 Wh.
- **§7.4, Table 11:** the caption now recommends a reference where one exists, since it is cheaper and at least as accurate, and not as a skill claim. The I_comp row adds raw total degree 0.719. The unfamiliar-topology row adds "dependency counts rank higher (0.863–0.938)".
- **Highlights:** all five rewritten to the new thesis, each ≤ 85 characters.
- **Supplement:**
  - new Section S39 (Amendment 15, three tables);
  - an A15 row in the amendment log;
  - the A11 row notes that Family C's gain is carried by rate and payload.

## Open

- Fixed during this revision: five supplement citations were undefined, because their `refs.bib` entries had been pruned in `b6dac343`. `kato2018autoware`, `edgexfoundry2024`, `homeassistant2024` and `google2024onlineboutique` are restored verbatim from history. The Train-Ticket row now cites the existing `zhou2018trainticket`, which is the same TSE 2021 paper as the old `zhou2021fault`.
- The other sections are waiting for your reading.

## Pre-submission fixes (2026-09-29, after the v6 merge)

These came from a full read of the merged v6 manuscript. They are on branch `jss-presubmission-fixes`.

1. **§6.2 contradicted Amendment 15.** The "QoS factor" paragraph ended "on I_dyn, which uses the contracts, it does (Family C)". It now says the Family C gain comes from declared rates and payload sizes, and QoS-policy content adds nothing.
2. **w(t) is not pure QoS policy.** `w(t) = 0.75·QoS + 0.15·log size + 0.10·log rate`. So three of the seven attribution columns (the w(t)-weighted `Topo-QoS`, `Reach-QoS` and QoS-weighted in-degree) carry declared rate and size, log-compressed at a quarter of the weight.
   - §3.2 and §6.1 now describe the block as "QoS-derived columns (w(t)-weighted scores and policy shares)", and so do the supplement and this note.
   - The Findings, Contribution 3 and §7.4 now say "QoS policies and QoS-weighted scores add none".
   - No number changes, and the conclusion stands.
3. **The I_dyn "test–retest 0.74–0.97 upper bound" in §4.3 was false.** On the full-population labels, one seed agrees with another at 0.43–0.96 per fold, and Eq. (7) exceeds that on 8/12 folds.
   - The labels are five-seed means, with an estimated reliability of 0.79–0.99 (Spearman–Brown). No Table 6 ranker exceeds its fold's bound.
   - §4.3 now says this. The values are computed in `idyn_rate_expansion.json` (`label_reliability`) and checked by the reconciler.
4. **"Supplementary §S.36" in §5.1 pointed to Amendments 9–10.** No supplement section documents the RPC→pub-sub conversion. It now points to the per-system composition (S14) and the RQ3 experiment page.
5. **§6 headings now match the new RQs:** "RQ1: Ranking Accuracy", "RQ2: Sources of Predictive Performance", "RQ4: Cost".
6. **Triage protocol.** Both formulas run in milliseconds, so the direct-dependent count and Eq. (7) are now Tier 1 (commit). Tier 2 (staging) runs the simulators on the Tier-1 shortlist, which keeps your "reserve full simulation for those components".
7. **The alert-fatigue paragraph now leads with the recommended training-free ranking:** `InDeg` recovers 0.49 at 20% and needs the top 45% for 80% recall. The GAT figures are kept for comparison.
8. **Contribution 4** now marks the 0.830 result "exploratory, added after the registered analyses". The abstract is unchanged.

Also:
- The AI-use declaration now names Claude Opus 5 and Opus 5.5 and Gemini Flash 3.8.
- The build statistics in `latex/README.md` are refreshed: 30 pages, 75 references.

Still open before submission:
- A new Zenodo version that includes the Amendment 15 artifact, then a check of the DOI in Data Availability and `refs.bib`.
- Your reading of §§2–6.
