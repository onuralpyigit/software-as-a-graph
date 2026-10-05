# Response to the round-13 referee report

This answers [review_2026-10-05_round13b.md](review_2026-10-05_round13b.md) (M1–M11, 25 minor comments). It also cross-references the separately committed round-13 report, [review_2026-10-05_round13.md](review_2026-10-05_round13.md) (M1–M12), which overlaps with it.

The revision is on branch `jss-revision-round13`. That branch is `4530b230` (the payload-aware simulator option and Amendment 18), merged with `main` at `8ec64ee0`.

**Scope: a text-only revision.** By the author's decision, no new experiment was run in this round, including the registered Amendment 18 relabel. Every number in the paper is unchanged and still reconciles against its artifact. Requested experiments are answered in two ways:
- **Calibration.** Claims are narrowed to what the existing evidence supports.
- **Limitations.** The requested runs are listed as prioritized future work, in the table at the end.
- **Title.** Kept, as the author decided.

We thank the referee. Two corrections came out of checking the paper against the code during this revision. Neither depends on the referee's requests, and both are now in the text.

- **The queue-flow oracle reads no payload.** A code survey for Amendment 18 found that `I_dyn` gives every message the same size, so declared payload never affects its labels. The paper said `I_dyn` used "declared rates, payload sizes" and that "rates and payload sizes" carried signal. Both statements are corrected to *declared rates*.
  - **Where.** §4.3, §6.1, §7.2, §7.3, Table 10, and Supplement A15 and the amendment log.
  - **The attribution.** The +0.097 gain is rate signal.
  - **Consistent evidence.** The payload-weighted closed form (`r_t B_t`) scores below Eq. 7 on all twelve folds (0.748 vs. 0.830).
  - **Amendment 18 is logged as registered, not run.** See [PREREGISTRATION.md](../PREREGISTRATION.md).
- **The two RPC system models are not request/reply decompositions.** §5.1 said Online Boutique and Train-Ticket were "decomposed into request and reply topics mediated by broker channels", keeping "the bidirectional dataflow". The committed models re-express them as event-driven meshes:
  - **Online Boutique:** 20 event topics on four assumed brokers, none of which the reference deployment uses.
  - **Train-Ticket:** 30 topics, of which three carry requests or commands and none carries replies. The Eureka service-discovery server is modeled as a broker.
  - **Corrections.** §5.1 now says this. Supplement §S15, which the body cited for the decomposition but which never described it, now has a paragraph that does.
  - **Code.** The `saag/adapters/__init__.py` docstring, which claimed the adapter "transcribes … launch graphs, Docker Compose / Kubernetes manifests", now says it hand-encodes five models and parses no manifests.

Two further descriptions were made accurate after reading the code:
- **I\*'s seeds.** They do not "decide ties at the propagation threshold". They drive a stochastic failure draw in waves after the first. §4.3 now states the threshold, the damping and the QoS ladder, and Supplement §S11 has I\* pseudocode.
- **The `Topo-QoS` defect.** It is a missing metrics key (`ap_c_score`), not an "index mismatch".

## Major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | The design cannot positively answer "when does learning help"; noise ceiling; reframe | **Text.** The paper now states that all three oracles are dominated by first-order propagation *by construction*: `I_dyn` propagates no failure beyond one hop, has no brokers, backpressure or retries, and its silenced consumers keep publishing. So the study tests whether learning exceeds aligned truncations of such oracles, and cannot characterize where it would. "Superfluous" and "renders complex learning superfluous" are gone. A new paragraph, "How much is left to learn", gives the headroom above Eq. 7 from the reported reliabilities (√r 0.89–0.996 vs. 0.830 leaves 0.06–0.17). Non-first-order oracles are future-work item 3, "the only regimes in which this study's title question could receive a positive answer". The title is kept (author's decision). | Abstract; §1.3; §1.4; §4.3; §6.1; §7.2; §7.4; §7.5; §8 |
| M2 | No real-system anchor | **Not done; stated.** §7.4 now states that no importer exists for launch, compose or Kubernetes files, that the models are hand-encoded, that no fault injection was run, and that the ready re-modeling protocol ([`reproduce/model_agreement.py`](../../../../reproduce/model_agreement.py)) has not been executed by a second modeler. Real topologies, fault injection and independent modelers are merged into future-work item 2. The RPC-model description was corrected (above). | §5.1; §7.4; §7.5; Supp. §S15 |
| M3 | Learners data-starved and untuned | **Not done; stated.** Every learned conclusion is now scoped to "eleven architectures, about 1,000–1,300 labeled Applications per fold, roughly 430,000 parameters, one fixed configuration". The nested-selection swing (±0.055) is set against the size of the reported effects, and §7.4 notes that this regime biases a learned-versus-analytical comparison toward the null. Learning curve, matched tuning budget, smaller models, tie-aware loss and a rate-fed GNN for `I_dyn` are future-work item 4. | §1.3; §6.2; §7.2; §7.4; §7.5; §8 |
| M4 | Claims exceed the equivalence evidence | **Text.** Removed or replaced: "renders complex learning superfluous", "Representation matters more than model complexity", "All three simulators admit such approximations" (`I_comp` is only term-aligned), "establishing that…", "rendering model complexity superfluous", and "matching or exceeding" (abstract and highlights). The text now separates near-equivalence on `I_dyn` (F12 interval [−0.013, +0.015] inside ±0.05) from `I*` (TOST fails). The 0.044 node-order rule is applied explicitly to −0.016 (GAT-P vs. `InDeg`), +0.041 (direction) and +0.031 (Eq. 7 vs. GBM); the last is now read as "does not exceed", not superiority. "Won" is defined as a sign count rather than an effect size, and the column is renamed "Folds won". "Matches" becomes "not distinguishable". | Abstract; highlights 3 and 5; §1.3; §5.3; §6.1; §6.2; §7.2; §7.4 |
| M5 | Reference criterion post hoc, under-specified, asymmetric; confirmation corpus | **Text, partly.** §4.4 now defines the criterion with a truncation operator `T_k`, a closed list of admissible simplifications (S1 expectation over seeds; S2 omitted loss rescaling; S3 omitted add-on mechanisms; S4 uniform weights; S5 support), and the rule that no parameter may be fitted to the oracle. It states which simplifications make Eq. 6, `InDeg`, `Reach` and Eq. 7 references, and that `Topo-QoS`, the hybrids and every learner are predictors. Asymmetry: comparisons with references are now described as descriptive anchors, with the single exploratory Eq. 7 vs. GBM contrast flagged, and "no learned model exceeded the references" is labeled descriptive. **Confirmation corpus: not run;** it is future-work item 1, with the reason it is cheap. | §4.4; §6.1; §7.5 |
| M6 | Defective comparator; Table 10 recommends it | **Text.** The defect is stated plainly (a missing `ap_c_score` key falls back to zero), and the Table 5 footnote and markers are unified (¶). §5.2 explains why registered contrasts stay against the registered value (changing a registered comparator would be an undeclared deviation). The corrected value stays beside it, and F9 compares hybrids with the corrected prior. **Table 10 `I_comp` row now recommends raw total degree (0.719)**, with `Topo-QoS` noted as defective and term-aligned. §1.3 now presents the hybrid result as reflecting the baseline's weakness, not as a learning finding. Making the corrected baseline the primary row was declined (above). | §5.2; Table 4; Table 5; Table 10; §1.3 |
| M7 | Energy analysis thin and misattributed | **Text.** Training cost (7.7 CPU-h for four arms, ≈0.22 kWh; Supp. §S32) is added to §6.4. A **break-even paragraph** is added: an `I_dyn` surrogate pays ≈30 Wh of labels per architecture plus training, and at equal accuracy repays that only after about a dozen architectures; Eq. 7 has no break-even cost and was at least as accurate. Feature cost is attributed to the feature set (a featureless GIN reaches 0.719). "Order-of-magnitude lower bound" becomes "nameplate estimate, not a bound in either direction". "Unjustified environmental overhead" is deleted. RAPL measurement: not done (root-only on this machine). The cost page's "upper bound" wording was aligned. | §6.4; §7.4; [rq4-cost.md](../experiments/rq4-cost.md) |
| M8 | Mixed protocols; `I_dyn` GNN not credible | **Text.** The Table 6 caption now makes the two protocols explicit (ensemble vs. per-seed means, with the 0.772/0.748 example). §6.1 notes that the `I*`-trained ensemble (0.615) beats the `I_dyn`-trained GAT (0.598) on `I_dyn`, so the latter "did not fit its target and is not a credible learned approximation". It is removed from Table 10's evidence. A rate-fed GNN is future work. Recomputing every row under one protocol was not done (no runs). | Table 6; §6.1; Table 10; §7.5 |
| M9 | Readability | **Text, partly.** §1.3 is rewritten with fewer numbers and a "What this does and does not show" paragraph. The rhetorical §7.3 paragraph is replaced by two factual sentences: the "guarantees essential for gating" claim is gone, and the 0.254 seed SD is attributed to HGT-P-QoS only, against 0.024–0.030 for the GATs. Table 4's caption gains a legend for the control suffixes. Ranker names are kept, because the supplement, the experiment pages and the reconciler key on them. The per-claim status tags were already consolidated in §5.3. | §1.3; §7.3; Table 4 |
| M10 | Contribution 1 overstated | **Text.** Contribution 1 is narrowed to the evaluated Rules 1 and 5, with "four infrastructure rules … formalized but not evaluated". The Fig. 2 caption marks the Rule 2 edges as not part of the evaluated projection. §6.2 and §7.2 now present the attention-vs-sum and featureless-GIN results as confirming known expressivity results [Xu et al.; Corso et al.; Chen et al.]. Evaluating Rules 2–4 and 6 is future-work item 5. | §1.5; Fig. 2; §6.2; §7.2 |
| M11 | Literature gaps; citation mismatches | **Text.** Six references are added, each verified against its DOI: Didona et al. (ICPE 2015), Fu & Menzies (ESEC/FSE 2017), Majumder et al. (MSR 2018), Cheung (TSE 1980), Ha & Zhang (ICSE 2019) and Gan et al. (ASPLOS 2021). They are cited in §2.1 (Cheung, Sage), §2.3 (DeepPerf; Didona as the analytical-plus-residual design that SaG's hybrids and F12 learners follow) and §2.4 (simple-vs-deep). Mismatches fixed: `albert2000error` is dropped from the broker-starvation clause; [8, 9] are rescoped to "much as overload cascades through complex and interdependent networks"; [11, 12] to design-time fault prevention and architecture evaluation; and [13, 14] to emergent architectural properties and module-invisible smells. | §1.1; §2.1; §2.3; §2.4; `refs.bib` |

## Minor comments

| # | What changed |
|---|---|
| 1 | Abstract recalibrated (M4); 239 words by `detex`. |
| 2 | Highlight 3: "The best learner does not exceed a direct-dependent count (0.748 vs 0.764)." Highlight 5: "Learners started from the formula do not improve on it; hybrids beat a weak baseline." Both ≤ 85 characters. |
| 3 | Title kept (author's decision); see M1. |
| 4 | §1.4 opening sentence rewritten. |
| 5 | "favourable" → "favorable"; "surrogate-modelling" → "surrogate-modeling". |
| 6 | Table 5 footnote rewritten; § and ¶ markers unified to ¶. |
| 7 | "GIN (GINE layers, which add the edge vector to each message)" in §6.2. |
| 8 | Table 6 caption notes that raw total degree (0.719, not tabulated) exceeds the underlined `I_comp` value. |
| 9 | GBM's columns are named the "tabular S+Q design" in Table 4, distinct from the 18 shared node features of §3.5. |
| 10 | **Declined:** moving Eqs. 6–7 into §4 would renumber equations cited across the body, supplement, highlights and experiment pages. §4.4 now refers to them by number. |
| 11 | Library weight formula given in §3.1 (`saag/infrastructure/memory_repo.py`); Application power mean `p = 3` stated. |
| 12 | `I*` described precisely in §4.3 (threshold 0.2, ladder ×1.2/×1.15/×1.05, damping 0.15 per wave, floor 0.25), with pseudocode in a new unnumbered subsection of Supp. §S11 (`supp:istar`), so no S-number shifts. |
| 13 | **Not run:** `I_comp` weight sensitivity is listed under further extensions (§7.5). |
| 14 | **Not run:** tie-aware listwise loss is future-work item 4. |
| 15 | §5.3 states the Nadeau–Bengio ratio (1/11, scenarios as the unit) and that Applications as the unit changes no conclusion (Hybrid-HGT p = 0.019 vs. 0.020; `results/referee_round8_hybrid.json`). |
| 16 | §5.3 states the Overlap@K tie rule (deterministic but arbitrary: NumPy's default sort over identifier-ordered components); Figure 5 resolves ties in expectation. |
| 17 | §5.3 states the partial ρ computation (per-fold Pearson of rank residuals after linear regression on the controlled rank; fold mean with the same bootstrap). |
| 18 | Fig. 2 caption: Rule 2 edges are drawn for completeness, not evaluated. |
| 19 | "which `Reach` does perfectly" → "and `Reach` separates it well". AUROC for `I* > 0` was not computed (no runs). |
| 20 | Table 10 `I*` row: "cheaper than any learned ranker's features" is removed. |
| 21 | Markdown regenerated from the LaTeX; the compiled PDF captions render correctly. |
| 22 | **Kept the immutable-commit link** (round 12 moved from a tag to a commit deliberately, because tags can move). It is re-pinned to this revision's commit in a follow-up commit. The software citation keeps its GitHub URL; the Zenodo DOI already covers the replication package [93]. |
| 23 | **Declined:** keywords kept (author's decision). |
| 24 | The body is self-contained for every claim in the abstract and highlights; the supplement is supporting detail. |
| 25 | Graphical abstract unchanged; no figure value changed in this revision. |

## Cross-reference to the committed round-13 report (M1–M12)

| Committed report | Answered by |
|---|---|
| M1 no positive control | M1 (first-order by construction; non-first-order oracle as future-work item 3) |
| M2 representation confounded with oracle alignment | M5 (formal criterion); the F11 controls already in the paper |
| M3 `I_dyn` comparison not like-for-like | M8 and the payload correction |
| M4 tuning | M3 |
| M5 early stopping and loss bias | M3; minor 14 |
| M6 reference criterion | M5 |
| M7 status of evidence | M4; §5.3 |
| M8 external validity | M2; the RPC-model correction |
| M9 cost and sustainability | M7 |
| M10 weight on the oracle that needs no approximating | M1; M7 (break-even) |
| M11 literature | M11 |
| M12 readability | M9 |

## Not done, stated as limitations

| Requested | Why | Where |
|---|---|---|
| M5(c) confirmation corpus | Text-only revision (author's decision) | §7.5 item 1 |
| M2 real-artifact extraction, fault injection, second modeler | No importer exists; fault injection needs a deployment; a second modeler needs another person | §7.4; §7.5 item 2 |
| M1(a) non-first-order oracle | Text-only revision; needs simulator work | §7.5 item 3 |
| M3 learning curve, tuning budget, smaller models; tie-aware loss; rate-fed `I_dyn` GNN | Text-only revision | §7.5 item 4 |
| M10 Rules 2–4 and 6 | Needs Broker/Host ranking experiments | §7.5 item 5 |
| M7 RAPL measurement | RAPL counters are root-only on the measurement machine | §7.4 |
| M8 single-protocol Table 6 | Needs re-scoring every learned row | Table 6 caption |
| M6 corrected baseline as primary row | Registered contrasts stay against the registered comparator; the corrected value is shown beside it (as in round 12) | §5.2 |
| Amendment 18 (`I_dyn-size`) | Registered before this round, not run (author's decision) | [PREREGISTRATION.md](../PREREGISTRATION.md); §7.5 |
| `supplementary.md` completeness | `scripts/generate_merged_papers.py` still drops sections when converting the whole supplement (round 12); the submitted supplement is the LaTeX PDF | — |

## Checks

| Check | Result |
|---|---|
| `make` | 27 pages (was 25; supplement 48); 0 undefined references or citations |
| Abstract / highlights | 239 words (`detex`); ≤ 85 characters each |
| `reproduce/reconcile_manuscript.py` | 1,815 figures match. Five prose checks were kept intact by wording the revised sentences so that their numbers are still verified |
| `render_manuscript_md.py --check`, `check_doc_links.py docs/` | up to date; every link resolves |
| `pytest -m "not integration"` | 1,440 passed, 6 deselected |
