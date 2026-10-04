# Response to the round-12 referee report

This answers [review_2026-10-04_round12.md](review_2026-10-04_round12.md). The revision is on branch
`jss-revision-round12`, based on `387f75e7` (`main`).

We ran the cheap experiments the referee asked for as **Amendment 17** in
[PREREGISTRATION.md](../PREREGISTRATION.md), committed (`b8710f75`) before anything was trained, with
contrasts and decision rules F11–F13. F13 triggered its flag, so we registered **Amendment 17b** (two
more node-order permutations) before running it.

| Item | Location |
|:---|:---|
| Sweep targets | `make -f reproduce/Makefile rq-amendment17 rq-amendment17b` |
| Analysis script | [reproduce/referee_round12.py](../../../../reproduce/referee_round12.py) |
| Artifacts | `data/benchmarks/referee_round12_{f12,amendment17,descriptive,perm}.json` |
| Experiment page | [amendment17-round12.md](../experiments/amendment17-round12.md) |
| New table | main Table `tab:a17`; the matched 2×2 moved to Supplementary `tab:contrasts_matched` |

We thank the referee. The four experiments came out as follows:

- **Oracle-aligned features (M3): the representation claim gets stronger.** We zeroed every feature
  that computes part of `I*`: in-degree, `w_in`, reverse PageRank, the articulation score, MPCI, FOC
  and CDI.
  - The raw-graph learners collapse without them: `GAT-QoS` falls to 0.369 and the reverse-edge
    control to 0.378.
  - The dependency-graph GAT keeps 0.610. It beats the equally stripped reverse-edge control by
    **+0.231 [+0.128, +0.340], 10/12, Holm p = 0.0068** (rule F11a).
  - With no node features at all, a sum-aggregation GNN on the dependency graph reaches 0.719, which
    is 0.045 below `InDeg`. It learns most of the count from structure alone.
  - Without these features, sum aggregation beats attention (+0.115, Holm p = 0.027). This is the
    first significant aggregator contrast in the study.
- **Learning on top of Eq. 7 (M4/M1): no gain** (rule F12b).
  - Stacking Eq. 7 as a column ties it (0.830, +0.000, Holm p = 0.97).
  - Residual learning gives 0.824 (−0.006).
  - The `I_dyn` GAT given Eq. 7 as a prior gains +0.214 over itself but stays below the formula
    (0.812, −0.018, Holm p = 0.0029).
- **Node order (M5): a variance source, not a favourable order.**
  - The registered permutation lowered `GAT-P-QoS` to 0.712 (−0.035, p = 0.012), which triggers rule
    F13a.
  - Two more permutations, registered as Amendment 17b, give 0.740 and 0.734. The three average
    0.729, 0.019 below the published value, inside the mean per-fold spread of 0.044 (rule P2).
  - The paper now reports node order as a source of variance of about 0.04 and says that
    learned-ranker differences of that size are not interpretable.

## Major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | The design cannot answer "when" | **Partly run, partly text.** The title is kept (author decision). The abstract, §1.2, §7.2 and §8 now state the boundary: all three simulators admit low-order approximations, so no regime where learning could exceed one was exercised. §1.2 says the cheap oracle `I*` serves to calibrate the method, and the practical case is surrogate modelling of `I_dyn`. The direct test, learners started from Eq. 7 (F12), is run and null. A non-first-order oracle is not built (see "Not done"). | Abstract, §1.2, §6.1, §7.2, §7.6, §8 |
| M2 | Reference criterion: post hoc, not operational, asymmetric, novelty overstated | §4.4 restates the criterion as an order-*k* truncation of the oracle's computation. It gives the derivation for each reference: Eq. 6, order 1; `InDeg`, the order-1 support; `Reach`, the untruncated support; Eq. 7, rate-weighted order 1. §4.4 states that learners fitted to the oracle are at least as circular. For symmetric accounting, partial ρ(·, `I*` given Eq. 6) is now computed for the learned rows (artifact; e.g. GAT-P-QoS 0.387 vs `InDeg` 0.148). §2.1 now cites Kitsak et al. 2010, Geirhos et al. 2020 and Sargent 2013, and contribution 2 is reworded as an evaluation guideline. | §1.4, §2.1, §2.5, §4.4 |
| M3 | Oracle-aligned features survive the degree ablation | **Run** (F11, plus the const arm). Results as above. §3.5 now lists which features compute part of `I*`, and corrects the articulation feature: it is one continuous directed score stored in two columns, not two binary flags. | §3.5, §6.2, Table `tab:a17` |
| M4 | Headlines are post hoc; confirmation cheap; Eq. 7 stacking | **Partly run.** Eq. 7 stacking, residual and prior arms run (F12, null). "Registered" is now defined once as pre-specified in a version-controlled plan with commit timestamps, not a third-party registry (§1.3, §5.3). The confirmation corpus is not run (see "Not done"); it is §7.6 item 2. | §1.3, §5.3, §6.1, §7.6 |
| M5 | Drift rule contradicts the headline; harness discrepancy; HGT instability; tie order; early stopping | **The drift rule is restated**: drift concerns cells from different sweeps or devices. Every learned-vs-learned contrast is paired within one sweep whose re-run comparators reproduce exactly (one within 1.2×10⁻⁴), and paired contrasts were not repeated on a second device. **Harness**: it trains on one scenario fewer and stops on a held-out scenario, so it compares two protocols (§4.2, §6.2). **HGT**: typing claims on the dependency graph are withdrawn; HGT-P's instability is called an optimisation failure of one untuned configuration (§7.2). **Tie order**: run (F13, 17b; above). **Early stopping**: §4.2 now states that the next-largest scenario serves when the largest is held out. | §4.2, §6.2, §7.2, §7.5 |
| M6 | RQ2 typing/QoS drawn on an inert substrate; one-hot in "untyped" GAT | The matched 2×2 moved to the supplement with a scope note. §6.2 and §7.2 say it compares per-node models and carries no information about relational typing. §4.1 discloses the one-hot relation encoding in the GAT-QoS edge vector: it is inert on the raw graph and constant on the projection. | §4.1, §6.2, §7.2, Supp. `supp:matched-2x2` |
| M7 | No real artifact, no observed failure | **Text.** §5.1 and §7.5 state plainly that no topology comes from a real manifest. ROSDiscover is cited beside HAROS. Real extraction and fault injection are now §7.6 item 1. Neither is run (see "Not done"). | §2.2, §5.1, §7.5, §7.6 |
| M8 | Defective, weakest comparator | Table 5 gains an indented corrected-baseline row (0.533 [0.404, 0.650], from `topo_ap_sensitivity.json`, checked by the reconciler). The hybrid paragraph is removed from the abstract; §1.3 and §7.2 state the hybrid result as no gain over the base learner. §5.2 explains why `InDeg` was not the comparator: the plan predates the count analysis, and the comparator continues the RASSE score family. | Abstract, §1.2, §5.2, §6.1 |
| M9 | Overlong, repetitive, process vocabulary, unevaluated components | Body 21,071 → 18,687 words (−11.3%), 35 → 32 pages. Repeated headline numbers are cut from summaries and §7–§8, and "Amendment N" is removed from the body text. Several items moved out of the main text or were cut: explanation-layer prose (to the supplement); the AHP framing of `I_comp`'s weights (now "declared, not elicited"); infrastructure-rule weight prose; the §3.5/§5 intros. Captions are tightened. *Findings in Brief* is kept (advisor structure) as its own subsection. | Whole paper |

## Minor comments

| # | What changed |
|---|---|
| 1 | Abstract 247 words. "Is matched by" → "is not distinguishable from". "On that graph" → "on the raw multigraph". |
| 2 | Highlights rewritten (≤ 83 characters): "No learned ranker outperforms counting direct dependents"; "Spearman" replaces "rho"; "publish-subscribe" replaces "pub-sub"; highlight 5 states the weak-baseline result and the Eq. 7 null. |
| 3 | Keywords kept as the advisor's choice (not done). |
| 4 | *Findings in Brief* is its own subsection (§1.3). |
| 5 | Contribution 1 is limited to Rules 1 and 5; the infrastructure rules are "defined but not evaluated". |
| 6 | §3.5 lists all 18 base metrics, with the articulation score correctly described. |
| 7 | §3.3 reports the disagreement: the in-degree feature exceeds the `InDeg` count for 940 of 1,321 Applications (never below), with per-fold rank agreement 0.55–1.00. |
| 8 | §4.1 states per model which graph and directions are used. |
| 9 | §4.2 defines $I^*_R$ and drops the masked maintainability head. |
| 10 | §4.3 calls the severity ladder "declared, uncalibrated". |
| 11 | AHP framing of `I_comp` removed in §2.3 and §4.3; §3.2 no longer discusses the other matrices. |
| 12 | §1.3 and §5.3 now say the dependency-graph learners' contrasts were declared exploratory when the arms were registered (Amendment 9), which matches Table 5's ‡. |
| 13 | −0.008 is against `GAT-P-QoS[I*-App]`, the Application-label control in the same sweep; named in §6.1 and §7.2. |
| 14 | Table 6's caption names the statistic for each block. |
| 15 | Table 12's GNN-approximation cell says it received no rates or payloads. |
| 16 | RAPL could not be read: the counters are root-only on the measurement machine, as §6.4 and §7.6 now say. Tier 1's "0.086 mWh, 0.31 J" is replaced by "milliseconds, well under a joule". |
| 17 | Overlap@K kept, with the tie note; Figure 5 remains the tie-aware view. |
| 18 | "Won" is named as the paired effect size; Â12 is not computed (it is unpaired). The 13 omnibus contrasts are listed in §5.3. |
| 19 | Figure files renamed to match their numbers (`Figure_4`, `Figure_5`, `Figure_S3`), with render scripts, Makefile and production note updated. |
| 20 | DOIs added to nine references (each checked against Crossref or doi.org). arXiv DOI for Shchur et al., marked as a preprint. SimPy cited as version 4.1.1 (2023). A separate software reference is added. |
| 21 | The repository link points to an immutable commit (`f3352f0c`) instead of the `jss-submission` tag. |
| 22 | Supplement titles "Referee Analyses" / "Round-8 Analyses" renamed. The HGT formulation is its own section (S2), and the informal heading is renamed. |
| 23 | "Variance-Stabilised" → "Variance-Stabilized". |
| 24 | §6.1 now has the |V_app|-weighting sentence. |
| 25 | Related work adds metastable failures (Bronson et al. 2021; Huang et al. 2022), Luo et al. 2021 and GENI (Park et al. 2019); all verified. |

## Deviations recorded in Amendment 17's results log

- **Sweep relaunch.** The first launch was stopped before any arm completed and relaunched after the
  arm code was committed, because artifacts record a dirty tree. Nothing from it was kept.
- **Â12** was listed as descriptive but not computed; the contrasts are fold-paired.
- **F12 tabular arms** run in `reproduce/referee_round12.py`, so the Amendment 15 artifact is not rewritten.
- **Amendment 17b** was registered after F13 was seen. It is labelled as such, and its decision rule is
  descriptive.

## Not done, stated as limitations

| Requested | Why | Where |
|---|---|---|
| M1(b) non-first-order oracle | Outside the agreed scope (cheap arms only); it needs a new simulator | §7.6 item 5 |
| M4 confirmation corpus | Needs a corpus-directory parameter through the label and LOSO harnesses, Neo4j cache builds and new `I_dyn` labels | §7.6 item 2 |
| M7 real-artifact extraction, live fault injection, second modeler | No importer for launch or compose files exists; fault injection needs a deployment | §7.5, §7.6 items 1 and 4 |
| HGT-P-QoS tuning | Typing claims withdrawn instead | §7.2 |
| Corrected baseline in Table 10 | Recomputing it on the system models does not reproduce Table 10's label files (0.582 vs. 0.526 as run), so a corrected row would not be comparable | Table 10 caption |
| Gate column out of Table 11 | Kept: the RQ4 energy prose quotes it, and it is labelled "for reference" | Table 11 |
| Keywords | Kept as the advisor's choice | — |
| `supplementary.md` completeness | `scripts/generate_merged_papers.py` (pandoc) drops nine supplement sections (S28–S36) when it converts the whole document; the cause was not isolated. The submitted supplement is the LaTeX PDF, which is complete | — |

## Checks

| Check | Result |
|---|---|
| `make` | 32 pages (supplement 45); 0 undefined references or citations |
| Abstract / highlights | 247 words; ≤ 83 characters each |
| `reproduce/reconcile_manuscript.py` | 1,814 figures match (new `check_amendment17`: Table `tab:a17`, the F11/F12/F13/17b prose, the corrected-baseline row, and the §3.3 feature-vs-reference count; the matched-2×2 check now reads the supplement) |
| `render_manuscript_md.py --check`, `check_doc_links.py docs/` | up to date; all links resolve |
| `pytest -m "not integration"` | 1,434 passed |
