# Response to the round-8 referee reports

The reports are [review_2026-09-26_round8.md](review_2026-09-26_round8.md) (R1) and
[review_2026-09-26_round8_r2.md](review_2026-09-26_round8_r2.md) (R2). The revision is on branch
`jss-revision-round8`.

New runs are **Amendment 14** in [PREREGISTRATION.md](../PREREGISTRATION.md). It was committed
(`714e70fc`) before any of its arms ran, together with a **deviation record** for the plan's
selection rule and one **status-tier rule** that replaces the earlier mixed use of "confirmatory".
The analysis script is [reproduce/referee_round8.py](../../../../reproduce/referee_round8.py); the
experiment page is [amendment14-round8.md](../experiments/amendment14-round8.md).

We thank both referees. Several comments changed conclusions, not just wording:
- **Tiers (R1 M4, M5).** Only the original plan is confirmatory now. The matched 2×2 and the hybrids
  become *registered secondary*: Amendment 2 had declared itself exploratory, and Amendments 5–6
  were written after the primary null.
- **Hybrids (R1 M4).** They beat the registered comparator but not their own base learners. Only
  Hybrid-GAT beats unweighted betweenness after correction.
- **"Never exceeds" is withdrawn (R1 M3).** The learner and `InDeg` are statistically
  indistinguishable but not equivalent at ±0.05.
- **Degree dependence (R1 M6, R2 M2).** The dependency-graph GAT's approach to the reference rests
  on its degree features (−0.136 without them).
- **QoS (R1 M6).** The "QoS" effect of the 2×2 was mostly the weighted in-degree (+0.073 → +0.030).
- **Full-population `I_dyn` (R1 M10, R2 M4).** Amendment 11 had in fact completed, and it changes
  the `I_dyn` story. A surrogate trained on queue-flow labels beats every closed-form approximation.
  Declared QoS contracts carry signal there, and not on `I*`.
- **Registered selection rule (R1 M7).** It was never applied. That deviation is now disclosed and
  the rule has been run (§7.2).
- **Cost (R1 M8).** The referee was right that the published ratio mixed stages, sessions and graphs. Re-timed like for like on the same graphs, the direction holds: learned feature extraction costs 4.5–72.5× (median 16.9×) one `I*` pass on the corpus, and 5.7–18.7× on generated graphs. The count is one to two orders of magnitude cheaper than a pass, not two to three.

## R1 — major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | The primary task restates its own ground truth; frame as surrogate modelling or add independent ground truth | §1.2 and §8.1 now distinguish two regimes. Where the oracle is cheap (`I*`, seconds), run it; the direct-dependent count is a cheap stand-in for it. Where it is expensive (`I_dyn`, 12.7 CPU-h for the corpus), a learned surrogate earns its place: GBM-Dep-QoS→dyn reaches 0.799 on held-out architectures, above the first-order expansion (+0.094, Holm 0.019; Amendment 11 B). The GNN surrogate does not (0.598; Amendment 14 F5b). Validation on a running deployment is still not done; §2.1 and §8.4 name Train-Ticket's fault benchmark and the MQTT systems as first targets. | §1.2, §7.1, §8.1, §8.4, Table 12 |
| M2 | Circularity applied asymmetrically; no operational criterion | §4.4 states one criterion: a closed-form truncation of an oracle's own computation is a reference for it. It is applied to all three oracles. On `I_comp`, Topo-QoS and raw degree are called *proximate* to its fragmentation and flow terms, and Table 12 and §7.1 read their lead with the same caution. The circularity caveat now lives in §4.4 and §8.3, with pointers elsewhere. §8.1 states the practical headline plainly: under reachability, afferent coupling ranks as well as every GNN. | §4.4, §7.1, §8.1, §8.3, Tables 7, 12 |
| M3 | "Never exceeds" rests on a non-significant difference and on the statistic | Withdrawn everywhere. TOST at the registered ±0.05 fails: p = 0.17 per seed and 0.13 for the ensemble; the 90% CI reaches ±0.076. The ensemble (0.772) is reported as 0.007 *above* `InDeg`. Text: "statistically indistinguishable, not equivalent". The margin was fixed after a preview, which the amendment discloses. | Abstract, highlights, §1, §7.1, §9, Supp. Table `tab:r8-tost` |
| M4 | Confirmatory status of the hybrids; nested comparator | One tier rule (§6.3). Confirmatory = the plan only. The hybrids are registered secondary, with their dates and the fact that the primary null was known stated. Against their own base learners (registered by Amendments 5/6, previously unreported): +0.035 (p 0.73) and +0.048 (p 0.30). Against unweighted betweenness and constant weights (F7): Hybrid-GAT +0.092/+0.088 (Holm 0.049), Hybrid-HGT +0.065/+0.061 (Holm 0.085). Both hybrids survive size weighting (sign-flip p 0.038/0.005) and Nadeau–Bengio (p 0.019/0.014). The nesting, the zero-shot cost (0.695/0.662 vs 0.760/0.805) and the `I_comp` shortfall are stated in §7.1 and §7.3. The Topo-QoS defect is flagged in Table 5. | §6.3, §7.1, §7.3, Table 5, Supp. `tab:r8-hybrid` |
| M5 | "Registered" overstates; self-hosted; 13 amendments | "Registered analysis plan" is now "version-controlled analysis plan". §6.3 and the supplement's amendment log give the commit hash of each amendment and say the timestamps are the repository's own, with no independent registry. Amendment numbers are removed from most of the running text. | §6.3, Supp. amendment log |
| M6 | RQ2 confound; degree-free learner; GIN arm | All run (Amendment 14): the 2×2 with `w_in` held (QoS effect +0.030, Holm 0.33); degree-free GATs (−0.136/−0.161, Holm 0.005); GINE at matched budget, which keeps 0.721/0.711 without degree features, although the aggregator contrast misses Holm (0.157). §2.4 now cites GNN counting theory (Xu et al. 2019; Corso et al. 2020; Chen et al. 2020). | §2.4, §7.2 (Table `tab:a14`), §8.2 |
| M7 | No tuning; seed/device instability | We found that the plan's own selection rule had never been applied. This is recorded as a deviation (§4.2, §6.3), and the rule was run as arm N. Nested selection moves HGT-QoS 0.622 → 0.677 and GAT-P-QoS 0.748 → 0.693, neither significant; nested HGT-QoS vs Topo-QoS is +0.123 (Holm 0.19). So F6b holds: the primary stays null. The search harness also early-stops on a held-out scenario rather than the published node split. Its reproduction gate failed on the three folds where it chose the published hyperparameters (up to 0.10 apart). This is disclosed, and no difference is attributed to selection alone; per the stopping rule the arm was not re-run. A hierarchical fold→seed bootstrap is added (Supp. `tab:r8-hier`); it widens intervals only slightly. The comparators re-run in the Amendment 14 invocation reproduce exactly. | §4.2, §6.3, §7.2, §8.3, Supp. S-round8 |
| M8 | Cost claim contradicts the paper's own tables | New Table `tab:cost-ll` times every region on the same graphs in one session: the count, one `I*` pass over Applications, the published five-seed three-type sweep, the app-layer feature extraction and the system-layer gate. Features cost 4.5–72.5× one pass on the corpus and 5.7–18.7× on generated graphs up to 5,000 components; the AP/CDI phase is 88–91% of that. Tables 10–11 are relabeled: they are from different sessions, and Table 11's oracle column is the three-type sweep. The Analyze:Forward framing is dropped, and the green-SE framing cut. `I_dyn` labeling cost (12.7 CPU-h) is reported as the case for a surrogate. No energy was measured. | §7.4, §8.1, Table 11, highlights |
| M9 | Zero-inflation; abstract reports only full-population ρ | The abstract reports ρ>0. "Learned engines ≈ 0.81" is now "the two best learned engines". The RQ3 summary states that the hybrids transfer worse. | Abstract, §7.3 |
| M10 | External validity; `I_dyn` sample | Full-population, five-seed `I_dyn` (Amendment 11) replaces the n = 30 sample everywhere; n = 30 is kept as the sensitivity check (Supp. `tab:r8-n30`). `I*`~`I_dyn` rises to 0.711. Partial correlations are computed for every ranker, learned ones included. The single-modeller threat is unchanged and stated as unmitigated. | §4.3, §7.1, Table 7, §8.3 |
| M11 | Overlap with [25]; Proposition 1; §5; verbosity | §1.4 states exactly what [25] contained: the pub-sub graph, Rule 1, a 0.7/0.3 betweenness–articulation score and a reachability-loss check. Proposition 1 is now Remark 1. §5 and Figure 4 move to the supplement (S-explanation). The circularity caveat is consolidated in §4.4/§8.3. | §1.4, §3.3, §4.4, Supp. |

## R1 — minor comments

| # | What changed |
|---|---|
| 1 | Abstract re-drafted to ≤ 250 words, with ρ>0. |
| 2 | Table 9 has block sub-headers. |
| 3 | Tables 6, 7, 9: best values are underlined; row labels keep the uniform bold style. |
| 4 | Table 7 gives partial ρ for every ranker, learned included, given `I*` and given the first-order expansion. |
| 5 | "0.017" corrected to 0.016 (per-seed); the ensemble difference is +0.007. |
| 6 | The contrast of `GAT-P-QoS` with `InDeg` is now reported as an equivalence test, not a Holm p. |
| 7 | §4.3 says what the seeds randomise (processing order within a propagation wave, which decides ties at the threshold). |
| 8 | §4.2 names the auxiliary heads (reliability, supervised by `I*_R`; maintainability, masked), describes ListMLE tie handling and the inner validation *scenario*, and corrects early-stopping patience from 30 to the 60 actually used. |
| 9 | `I_comp`'s weights are one of the three declared-vector AHP matrices; stated in §4.3. |
| 10 | Remark 1 states that the evaluated projection applies Rule 1 to direct subscriptions (Applications or Libraries); the repository derivation behind the node features also follows USES chains. Table 3 no longer says "incl. transitive USES". |
| 11 | Rules 2–4, 6 are marked † (not exercised) in Table 3. |
| 12 | Nadeau–Bengio and \|V_app\|-weighted sign-flip tests added for every registered contrast. |
| 13 | Table 13 caption notes the regression-to-the-mean artifact of grouping on the comparator's score. |
| 14 | Added: Xu et al. 2019, Chen et al. 2020, Corso et al. 2020, Maurya et al. 2021, Zhou et al. (Train-Ticket), Hellendoorn & Devanbu 2017. |
| 15 | Typos fixed; American spelling throughout. |
| 16 | Not regenerated; the Data Availability statement still says which main-text numbers rest on the unstamped artifact. |
| 17 | Duplicated URL in [87] removed. |
| 18 | p(v) defined in Figure 3's caption. |
| 19 | Compliance rechecked after the revision (see "Checks"). |

## R2 — comments

R2 was evaluated against the reviewed manuscript. Where a request was already met by that text, we
say so and point to it; the stale quotation in R2 Major 3 is from a pre-round-7 draft.

| # | Comment | Response | Where |
|---|---|---|---|
| Major 1 | Operational paradox | Already stated in §1.2/§8.1; now reframed as two regimes (see R1 M1). R2's comparison of 1.74–239 s (generated graphs) with 0.08–4.5 s (corpus) mixes two measurements; the like-for-like timing now settles the ratio (see R1 M8). | §1.2, §7.4, §8.1 |
| Major 2 | RQ2 confound; degree-free ablation | Run (see R1 M6). Note: the QoS-off arms zeroed only `w_in`, not the raw in-degree column. | §7.2 |
| Major 3 | Untuned GNNs; soften | Typing claims were already restricted to one configuration (§8.2); the registered selection rule has now been run (see R1 M7). | §7.2, §8.2 |
| Major 4 | `I_dyn` sampling bias | Full population now (see R1 M10). | §4.3, §7.1 |
| Major 5 | §5 unvalidated | Moved to the supplement (see R1 M11). | Supp. S-explanation |
| Major 6 | RQ3 external validity | The "hand-authored models" wording was already present (§6.1); ρ>0 is now in the abstract. | §6.1, §7.3 |
| Major 7 | Sustainability | We make no sustainability claim, and the green-SE framing is cut to one sentence. We did not add energy measurement and list it as a limitation. The special issue does not require every paper to address every pillar. | §8.1, §8.4 |
| Minor 1–2 | Abstract length; highlights | Checked after revision. | — |
| Minor 3 | Label the defective Topo rows | Table 5 now says "betweenness only (AP term zero)"; the ¶ note stays in Tables 6–7. | §6.2 |
| Minor 4 | Put Degree-raw 0.719 in the abstract | Not done: raw degree is proximate to `I_comp`'s terms (§4.4), so featuring it would repeat the circularity R1 M2 warns about. | — |
| Minor 5 | Notation | w(e) = w_E(e) on DEPENDS_ON; Table 3 uses w_E. | §3 |
| Minor 6 | Green-AI literature | Not expanded; the framing was cut instead (see Major 7). | — |
| Minor 7 | Provenance stamping | See R1 minor 16. | — |

## Corrections found during the revision

1. **Amendment 11 had completed.** Its full-population labels and three contrast families were
   committed but the manuscript said "not completed". Its registered decision rules now govern the
   text: A″, B, C, Z, R.
2. **The plan's selection rule was never applied**, and Supplementary Table A1 said it governed every
   learned result. Recorded as a deviation; run as arm N.
3. **"0.200 at the shipped setting" was wrong**: the shipped λ = 0.70 gives 0.234; 0.200 is raw AHP.
4. **Early-stopping patience** was 60 in every run, not 30.
5. **The 2.0–17.7× cost ratio** compared the full detection gate with the five-seed sweep over three
   node types, not with "one run" of `I*`; Table 11's "one I* pass" column was the same sweep.
6. **The amendment log** counted twelve amendments; there were thirteen, now fourteen.
7. **Harness path fallback.** From a worktree, the training-free `topo_qos` row silently used a
   different substrate (relative cache path). Recorded in Amendment 14's G0; no contrast used it.

## Not done, stated as limitations

Validation against observed outages; a second modeller; automatic graph extraction; energy
measurement; hyperparameter search beyond the stage-1 grid and beyond two engines; synchronous call
edges; a sum-aggregation GNN surrogate for `I_dyn` (a natural next amendment).

## Checks

- `cd docs/research/jss/latex && make`: 0 errors, 0 undefined references; manuscript 29 pages, supplement 40.
- Abstract 245 words (math counted as one word); highlights 76–82 characters, five bullets.
- `python reproduce/reconcile_manuscript.py`: 1,486 figures match their artifacts, 0 skipped. New checks: `check_independent_oracles` (round-8 Table 7), `check_round8` (Table `tab:a14`, F3/F6/F7 quotes, withdrawn phrases), `check_cost_ll`.
- `python reproduce/render_manuscript_md.py --check`: up to date. `python scripts/check_doc_links.py docs/`: all links resolve.
- `pytest -m "not integration"`: 1,422 passed.

