# Response to the round-10 referee report

This answers [review_2026-09-30_round10.md](review_2026-09-30_round10.md). The revision is on branch `jss-revision-round10`, based on `99848f0b`.

**No new experiments.** Every change is to text, tables, references, or reconciler checks. Each requested experiment is either answered below or recorded as a limitation and a future-work item (§7.5, §7.6). The title and the advisor's thesis are unchanged. The scope is now stated more sharply.

We thank the referee. The points that changed what the paper claims:

- **Scope (M1).** The abstract, §1.2, §7.5 and §8 now state three things:
  - The primary registered contrast was null.
  - The headline findings are registered secondary or exploratory.
  - All three oracles are well approximated by low-order functions of the dependency graph.

  The benchmark therefore cannot exhibit a regime in which learning exceeds an aligned analytical ranking. The paper says that no such regime was found, not that none exists.
- **"Matches" is gone (M6).** Where the evidence is non-significance, the text reads "not statistically distinguishable; equivalence at ±0.05 not established". Where the data show more, as for Eq. 7 against the learned queue-flow approximation (+0.031, 10/12 folds), the text says "exceeds (exploratory)".
- **Hybrids (M4).** Every hybrid claim now sits next to three facts: the baseline is the weakest analytical ranker; it carries the articulation defect and is nested in the hybrids; and neither hybrid differs from its own base learner.
- **Directionality (M5).** Contribution 1, §7.1, §7.5 and §8 now say the +0.113 dependency-graph gain was not separated from edge direction. A raw-graph learner with reverse edges is listed as future work.
- **Cost labels (M10).** §6.4 had mislabelled two corpus totals (details under Corrections below). The prose is now correct, and the reconciler checks all of these figures.

## Major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | The design cannot answer "when does learning help" | The scope sentences above were added. The title is kept (advisor's framing). A future-work item names non-first-order oracles: backpressure onto publishers, retry amplification, heterogeneous capacity. | Abstract, §1.2, §7.5, §7.6, §8 |
| M2 | No empirical ground truth | **Not done.** Real fault injection is out of scope for a text revision. The abstract now says the findings "concern agreement with failure simulators, not prediction of physical failures". §7.6 item 1 names Train-Ticket's fault benchmark and ROS 2, Kafka and MQTT deployments. | Abstract, §1.2, §7.5, §7.6 |
| M3 | Reference criterion is post hoc and asymmetric | Partly accepted; see the rebuttal below. §4.4 now says the criterion was introduced post hoc and names InDeg as classical afferent coupling. Topo-QoS and raw degree on $I_{\text{comp}}$ are labelled *term-aligned* rather than references. Circularity is quantified from the reference rows: the first-order expansion recovers 0.808 of $I^*$ and the rate-weighted expansion 0.830 of $I_{\text{dyn}}$. InDeg keeps its reference status. | §4.4, §7.5 |
| M4 | Weak, defective baseline; hybrid wins | §5.2 now presents the executed score (QoS-weighted betweenness), states the defect once, and notes that the hybrids' prior inherits it. Contribution 3, §7.2 and §8 put the base-learner nulls (+0.035, p = 0.73; +0.048, p = 0.30) next to the hybrid gains. Highlight 5 now reads "Learning beats only a weak baseline, not dependency-based references." §1.4 states that [32]'s closed-form score family is the weakest analytical ranker here, and that these results supersede its recommendation. **Not done:** retraining with the corrected or InDeg prior (§7.6). | §1.4, §5.2, §7.2, §8, highlights |
| M5 | Representation confounded with edge direction | §4.1 spells out why forward message passing never reaches Applications on $G_{\text{structural}}$. Contribution 1 is tempered: the derivation mostly organises signal that a typed two-hop query can also recover (Remark 1). A construct-validity threat and a future-work control were added. **Not done:** the bidirectional raw-graph GAT. | §1.4, §4.1, §7.1, §7.5, §7.6, §8 |
| M6 | "Matches" contradicts the equivalence results | Reworded throughout: abstract, highlight 3, §1, §2.4, §6.1, §7.1–7.5, §8, Table 11, Figure 4 caption. §7.5 adds that 12 folds give a ±0.076 interval, so the study is underpowered for equivalence at ±0.05. | as listed |
| M7 | Headline claims are post hoc; confirmation set | §5.3, §6, §7.5 and §8 now call the two confirmatory contrasts "co-primary, both null", and state that the headline findings are registered secondary or exploratory. §5.3 points to the supplement's amendment log for family membership. **Not done:** a confirmation corpus. It would need about 13 CPU-h of $I_{\text{dyn}}$ labels plus sweeps; it is listed in §7.6. | §1.3, §5.3, §6, §7.5, §7.6, §8 |
| M8 | Drift exceeds the reported effects | §7.5 now says learned-model differences smaller than the observed drift (up to 0.172) should not be interpreted, and names the affected contrasts. HGT-P-QoS width is stated as 100 (430,680 parameters) in §4.1 and Table 4. §7.2 says HGT-P-QoS ran only in its fixed configuration. | §4.1, §7.2, §7.5, Table 4 |
| M9 | External validity; five single-author models | §6.3 now gives per-system GAT-P-QoS values (0.774–0.841) and says their narrow spread comes from separating inert components. Table 9's caption now calls the five-system intervals descriptive (126 resamples, not inferential). It also notes that no test is applied and that the zero-shot comparator carries the defect. **Not done:** a second modeler or mechanical extraction (§7.6). | §6.3, Table 9 |
| M10 | Cost and energy claims | §6.4 corpus totals relabelled (see Corrections below). $I_{\text{dyn}}$ cost is now reported per architecture: 0.01–5.88 CPU-h, median 0.41, with the slowest seed about 76 min. The unsupported "mere minutes" claim is removed. The feature cost is attributed to the chosen feature set. The energy-validity paragraph is shortened. **Not done:** RAPL measurement (§7.6). | §6.4, §7.5, §7.6 |
| M11 | Length, repetition, jargon | "Findings in brief" was replaced by a short organisation paragraph. Plan codes are removed from the main text (rule F1a/F2b/F6b, arm N, Family A–C, equivalence arm F3, control F5, rule R5). F1, F2 and F4 stay only as row labels defined in Table 8's caption. Deferrals to the "replication repository" now point to the supplement where it holds the material: `supp:hgt`, `supp:baselines`, `supp:taxonomy`, `supp:features`. The file path was removed from §7.2. | §1.2, §4.1, §5.2, §6, §7.2 |
| M12 | Literature gaps | §2.1 adds error-propagation analysis (Abdelmoez et al. 2004; Popic et al. 2005; Cortellessa & Grassi 2007; Hiller et al. 2004), HiP-HOPS (1999) and the AADL Error-Model Annex (2014). It also adds Eadro (ICSE 2023). §2.4 adds Chen et al. 2020, Lv et al. 2021, Shchur et al. 2018, Huang et al. 2021 and Arcuri & Briand 2011. Every entry was checked against the publisher record (see Corrections below). | §2.1, §2.4, refs.bib |

### Rebuttal to M3

We keep InDeg and Reach as references for $I^*$. This follows the Amendment 13 ruling.

The paper does not claim that afferent coupling is unimportant to practitioners. §7.4 recommends it for routine use. Its role in the evaluation is different: InDeg is exactly the set that $I^*$'s first wave reaches (Remark 1). A contrast of a learned model against it therefore measures how closely the learner restates the oracle, not predictive skill.

The asymmetry the referee points to is real. A learner trained on $I^*$ labels is also tied to $I^*$. The paper states this in §4.4: learners use in-degree features and are trained on the oracle they are scored on. §6.2 measures it: −0.136 without the degree features.

The paired GAT-P-QoS vs. InDeg comparison is still reported in §6.1, both as a difference and as a TOST. It is simply not counted as a registered contrast.

## Minor comments

| # | What changed |
|---|---|
| 1 | Keywords reduced to seven. |
| 2 | Highlights 3 and 5 reworded; every highlight is ≤ 85 characters. |
| 3 | The abstract states the primary null and the simulator-only scope. It is ≤ 250 words (the reconciler checks this). |
| 4 | Figure files are not renamed. A production mapping from printed figure numbers to file names is in `LENGTH_JUSTIFICATION.md`. |
| 5 | Funding now uses Elsevier's standard sentence. |
| 6 | The generative-AI declaration is unchanged. |
| 7 | The experiments URL points at tag `v1.0-jss`, which must be created at submission. |
| 8 | `LENGTH_JUSTIFICATION.md`: now 34 pages, 87 references, S1–S40, with a corrected figure mapping. |
| 9 | §7.2 now says the selection rule was applied to HGT-QoS **and** GAT-P-QoS. |
| 10 | §7.4 and Table 11 attribute Topo-QoS's $I_{\text{comp}}$ lead to its betweenness term, and note the articulation term is zero. |
| 11 | Table 6: the underline is on GAT-P-QoS (0.772), with a caption note on ensemble vs. per-seed values. |
| 12 | Table 6 mixes statistics; the caption explains the ordering. |
| 13 | "Co-primary" is used throughout. |
| 14 | §3.3 now says InDeg, Reach, Topo-QoS and the `-P` graphs use direct subscriptions only, and that only the node-property in-degree feature follows `USES` links. |
| 15 | Fixed. |
| 16–17 | §4.2 discloses the ListMLE tie ordering and the transductive split. |
| 18 | §4.3 describes how $I_{\text{dyn}}$ reads QoS and says no QoS-off run exists. §3.2 says the $I^*$ QoS null is largely fixed by design, while the $I_{\text{dyn}}$ null is informative. The overgeneralising parenthetical in contribution 3 is removed. |
| 19 | $I_{\text{comp}}$ grammar fixed. §3.2 and §4.3 now say plainly that the weights were chosen, not elicited. |
| 20 | §3.1 is corrected (see Corrections below). |
| 21 | "depart-mode flag" is now "transport departure mode". |
| 22 | PR-AUC and the other identification metrics stay in Supplementary `tab:identification`; Table 9 already reports PR-AUC. |
| 23 | The fold-weighting caveat was added to §7.5's conclusion validity. |
| 24 | **Done (follow-up).** Figure 5B is redrawn on the exhaustive $I_{\text{dyn}}$ labels (all 1,321 Applications) via `reproduce/referee_round7.py recall --idyn-full` → `data/benchmarks/referee_round10_recall_idyn_full.json`. The §6.1 prose now reads: GAT-P-QoS 0.74 at 40% and 0.82 at 50%, InDeg 0.80 at 40% (previously 0.65 / 0.80 / 0.83 on the n = 30 sample). The reconciler checks these figures against the new artifact. The $I^*$ panel is unchanged. |
| 25–26 | Typography fixed; "SAG" → "SaG". |
| 27 | Table 1 already separates entities and edges with a rule and a header row; unchanged. |
| 28 | Eq. 5 is presented as executed (see M4). |
| 29 | "cheaper" |
| 30 | The file path was removed from §7.2. |
| 31 | "Architecture–Code Gap" removed. |
| 32 | §1.2 now describes energy as TDP-based estimates; the §1.1 motivation sentence is unchanged. |
| 33 | The BibTeX escape was fixed in the new entries. |

## Corrections found during the revision

- **§6.4 corpus totals were mislabelled** (these errors predate round 10):
  - "11.1 s, 0.086 Wh" was described as the dependency count. It is the five-seed $I^*$ labelling sweep (`results/energy_estimate.json` → `oracle_s`).
  - "106.9 s, 0.83 Wh" was described as feature extraction. It is the full detection gate (`gate_s`).
  - Over the twelve folds, feature extraction actually takes 70.0 s and the count 0.037 s, about 1 J (`data/benchmarks/referee_round8_cost.json`).
  - The reconciler now checks all six figures.
- **"Mere minutes on multi-core CI runners" (added in this round's first pass) was false.** A seed-parallel run is bounded by its slowest seed, which takes about 76 min on Financial Trading.
- **§3.1 $w_V$ was wrong in both the submitted text ($1 - \text{CQP}$) and the first-pass edit ($w_V = 1.0$).**
  - In the implementation (`saag/infrastructure/memory_repo.py`), an Application's $w_V$ is a power mean of its topics' QoS weights.
  - A Library takes the largest weight among its topics and its consumers, amplified by fan-out.
  - CQP and its four inputs are only node features of the learned rankers.
- **Table 4:** HGT-P-QoS has 430,680 parameters, not 434,620.
- **Figure 4 caption:** a "p = 0.62" for GAT-P-QoS vs. InDeg had no backing artifact. The per-fold Wilcoxon from `referee_round8_tost.json` gives 0.47 per seed. The caption now quotes only the archived TOST p = 0.17.
- **Bibliography.** Six of the twelve entries added in the first pass were wrong. Each was corrected against its publisher record:

  | Entry | Error | Corrected to |
  |---|---|---|
  | Eadro | Wrong authors and title | Lee et al., ICSE 2023, pp. 1750–1762 |
  | Popic et al. | Wrong authors and pages | Popic, Desovski, Abdelmoez & Cukic, ISSRE 2005, pp. 53–62 |
  | EPIC | Wrong title, issue and pages | "Profiling the propagation and effect of data errors in software", IEEE TC 53(5):512–530 |
  | HiP-HOPS | Wrong venue | SAFECOMP 1999, LNCS 1698, pp. 139–152 |
  | AADL EMV2 | Wrong venue | SEAA 2014, pp. 361–368 |
  | Abdelmoez et al. | Five of eight authors, wrong pages | All eight authors, pp. 384–393 |

  The Lv et al. author list was also completed. The §2.1 prose was reworded to match what each paper does.
- **Per-system zero-shot sentence (§6.3):** "within active components … across all systems (ρ>0 ≤ 0.342)" was not supported per system. It now reads "mean ρ>0 = 0.342". "Cloud Microservices" was renamed Online Boutique.

## Not done, stated as limitations

| Requested | Why not | Where recorded |
|---|---|---|
| Fault injection on a real deployment | Text-only revision | §7.5, §7.6 item 1 |
| Second modeler / mechanical extraction | Text-only revision | §7.5, §7.6 item 3 |
| Post-freeze confirmation corpus | About 13 CPU-h of $I_{\text{dyn}}$ labels plus sweeps | §7.6 item 5 |
| Bidirectional raw-graph GAT | New experiment | §7.5, §7.6 item 5 |
| Hybrids with corrected / InDeg prior | New experiment | §5.2, §7.6 item 5 |
| RAPL-measured energy | New measurement (RAPL needs root on the test machine) | §7.5, §7.6 |
| QoS-off $I_{\text{dyn}}$ | About 12.7 CPU-h | §4.3, §7.6 |

## Checks

| Check | Result |
|---|---|
| `make` (manuscript + supplement) | 34 + 44 pages; 0 undefined references or citations; 87 references |
| Abstract / highlights | ≤ 250 words; ≤ 85 characters each (69–80) |
| `reproduce/reconcile_manuscript.py` | 1,624 figures match (8 new cost checks; the Eq. 7 prose check was re-pointed at the new wording, with both figures still checked) |
| `reproduce/render_manuscript_md.py --check` | up to date |
| `scripts/check_doc_links.py docs/` | OK |
| `pytest -m "not integration"` | 1,422 passed |
