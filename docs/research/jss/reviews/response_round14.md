# Response to the round-14 referee report

This responds to [review_2026-10-06_round14.md](review_2026-10-06_round14.md): major comments M1–M10 and 21 minor comments. The revision is on branch `jss-revision-round14`.

## Scope

The revision combines four new, cheap experiment families (Amendment 19) with targeted text changes.

- **Registration.** Amendment 19 was registered, with its code, at `48c196fb`, before any run.
- **Not run (author's decision):**
  - a non-first-order oracle;
  - a confirmation corpus;
  - R-GCN;
  - a manifest importer.
- **Title.** The advisor's title is kept, and the scope statement is sharpened instead.
- **Structure.** The Abstract, Introduction and Conclusion keep the advisor-v9 structure.

We thank the referee. Five results came out of the new runs (Amendment 19; [experiment page](../experiments/amendment19-round14.md)).

- **Aggregator (M3, F14a).** The representation claim survives a sum-aggregation control.
  - On the raw multigraph with every edge reversed, a GINE network reaches 0.668, level with its attention counterpart.
  - The dependency graph still wins under matched sum aggregation: +0.239 without the oracle-aligned features (12/12 folds, Holm p = 0.0015), +0.064 with them, and +0.289 with no node features. In the last case the raw-graph GNN learns 0.430, against 0.719 on the dependency graph.
  - Sum aggregation on the raw graph closes 46% of the attention-model gap.
- **Rate-fed queue-flow GNNs (M4, F15b).**
  - Given declared rates as node and edge inputs, the GNNs improve on the rate-blind one. The best, a GIN, reaches 0.665 (+0.067).
  - They stay far below the gradient-boosted approximation (0.799) and Eq. 7 (0.830; −0.165, Holm p = 0.0015).
  - The rule fired on its "significantly above the rate-blind GNN" condition, not on "within 0.02 of GBM". The text reports the remaining 0.134 gap.
- **Tie-aware listwise loss (M6b, F16 and S2).**
  - GAT-P-QoS is unchanged under the new loss (0.747 vs 0.748).
  - The direction-controlled gain holds (+0.069, Holm p = 0.019).
  - Node-order spread stays at 0.047, against 0.044 under ListMLE. Node-order variance therefore does not come from tie order.
- **Learning curve (M7, LC-c).**
  - GAT-P-QoS rises from 0.605 (K = 1) through 0.670 (K = 4) to 0.748 (K = 11), narrowing its gap to afferent coupling from 0.160 to 0.017 without closing it.
  - The step from K = 8 to 11 is small (+0.015 [+0.003, +0.028]), so by the registered rule the curve neither saturates nor keeps rising.
  - The raw-graph GAT is flat from K = 4.
  - A width-64 GAT loses 0.110, so the published width is not over-parameterized.
- **Robustness (M6a, M6d, m10; descriptive).**
  - Under the registered nested protocol, GAT-P-QoS sits 0.071 below afferent coupling (p = 0.11).
  - A mixed-effects model with seeds nested in folds reproduces every registered contrast.
  - The per-fold `I_dyn` headroom above Eq. 7 is 0.047–0.291. The earlier "0.06–0.17" was wrong and is corrected.

### Corrections to a parallel text pass

A separate session made a text-only pass at this report before this revision. It is preserved as commit `a1120b49`, and its responses `response_round14` (since replaced by this file) and `response_round15.md` remain in git history. Checking it against the artifacts and the code found errors that the figure reconciler does not catch, because they were in prose or in an unreconciled table. All are corrected here.

- **§5 claim-status table.**
  - The primary contrast was given as "+0.008, p = 0.622"; it is +0.069, p = 0.266.
  - The hybrid gain was given as "+0.072"; it is +0.103/+0.130.
  - The typing effect was given as "+0.003"; it is −0.014.
  - "QoS edge features +0.072" was the direction control, not the QoS effect.

  The table was rebuilt from reconciled values (`tab:claims`).
- **Plan commit.** It was given as `56d9bff8` (the v9 merge). The plan was committed at `44713326` on 6 September 2026, before the revised harness produced any result.
- **MicroART reference.** It was fabricated: "Walker, Jin, Kazman, ICSA-C 2020". Crossref gives Granchelli et al., ICSA Workshops 2017, DOI 10.1109/ICSAW.2017.9. The entry is replaced.
- **The 0.172 drift.** It was attributed to "non-deterministic scatter/gather … and BLAS variations". The repository documents three causes, and §7.4 now gives them:
  - a since-fixed PyG device-placement defect;
  - stale checkpoint resumption;
  - non-deterministic CUDA reductions.
- **Remark 1.** The in-degree feature was described as counting "all incoming edges … including incoming library links". It is the in-degree on the full Neo4j `DEPENDS_ON` graph, where Rule 1 also follows `USES` chains of up to three hops.
- **§4.4.** The text claimed that centralities failing the criterion "confirm … the criterion is structurally grounded rather than post hoc". The criterion is post hoc and is now stated as such, with score-independent examples instead.
- **§4.3.** It asserted a headroom "≥ 0.05" that had not been computed, and named LightGBM. Both are removed.
- **§6.2.** It attributed the 0.044 node-order spread to ListMLE tie-breaking. F16/S2 refutes this.
- **§6.2.** It stated the mechanism ("direct 1-hop sum aggregation or localized attention directly counts") before any test. Attention cannot count; the sentence is replaced by the F14 result.
- **§7.1.** A paragraph on "Byzantine failures" and GNN "expressive capacity" was speculative. It is replaced by the oracle-validity point of M2.

## Major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | The design cannot answer the title question | **Text; title kept (author's decision).** The abstract now says that, because the simulators are first-order by construction, "the study shows when learning is unnecessary, not when it helps". §1.4 states that the title question is answered for that regime only. RQ2 adds the aggregator and the number of training architectures. Future-work item 3 names non-first-order oracles as the only regime in which the title question could get a positive answer. **Not run:** a non-first-order oracle. | Abstract; §1.4; RQ2; §7.5 |
| M2 | The circularity cuts against the oracles; S1–S5 post hoc | **Text.** §4.4 justifies S1–S5: each removes exactly one mechanism stated in §4.3, and none changes which components a wave reaches or the direction of propagation. It also lists what is not admitted, and gives cases where the classification does not follow the scores: Eq. 7 is a *predictor* on I* (0.756); `Reach` is a reference scoring below the best predictor; GAT-QoS+InDeg (0.763) and GBM→dyn (0.799) are predictors scoring at reference level. A new §7.1 paragraph states that afferent coupling recovering I* is first a statement about I*'s value as ground truth. §1.3 and §8 lead with the comparison against afferent coupling. The nested-protocol result (−0.071) is reported. **Not run:** confirmation corpus (§7.5 item 1). | §4.4; §1.3; §6.2; §7.1; §8 |
| M3 | Representation confounded with the aggregator | **Experiment, F14a.** GIN-QoS-R, -min and -const were added. Under matched sum aggregation the derived graph still wins (+0.239, 12/12, Holm 0.0015). The claim is now stated "beyond edge direction and aggregator". **Not run:** R-GCN. §7.4 notes that HGT-QoS is relation-typed and bidirectional but attention-based. | §1.3; §6.2; Table 7; Supp. Table S-a19; §7.1; §7.2; §7.4; highlight 2 |
| M4 | The I_dyn GNN was handicapped; surrogate framing overstated | **Experiment, F15b, and text.** Three rate-fed arms were added (node rate; node plus edge rate share; GIN). The best reaches 0.665, below GBM 0.799 and Eq. 7 0.830. "Genuine surrogate-modeling problem" is gone; §1.2 says a closed form is expected for a one-hop simulator. Contribution 4 and the highlights were reworded. The cost section keeps the break-even as accounting. | §1.2; §1.5; §6.1; Table 4; Table 7; Table 10; highlight 3 |
| M5 | Straw-man comparator in the abstract and highlights | **Text.** The abstract no longer compares against `Topo-QoS`. The highlights no longer mention hybrids. §1.3 and §8 lead with afferent coupling. The hybrid paragraph is unchanged in substance, and the corrected-prior hybrid rows moved from Table 7 to the supplement. | Abstract; highlights; §1.3; §8; Table 7 |
| M6 | Fragile inferential base | **(a) Descriptive:** nested protocol vs afferent coupling −0.071 (p = 0.11). F8/F11 under the nested protocol were not run. **(b) Experiment, F16 and S2:** the tie-aware loss changes nothing, and node-order spread is not from ties. **(c) Text:** documented drift causes; all comparators reproduced bit-exactly (G0) in every amendment sweep. A second device was not run. **(d) Descriptive:** mixed-effects estimates confirm F8b, F11a, F14a and F16b, and GAT-P-QoS vs InDeg (−0.017, p = 0.62). | §6.2; §7.4; Supp. `supp:controls` |
| M7 | Small-data regime uncharacterized | **Experiment, LC-c.** Learning curve for three learners × K ∈ {1, 2, 4, 8, 11} × 3 draws. Dependency-graph learners approach afferent coupling with K (gap 0.160 → 0.017) without passing it, and the raw-graph GAT is flat. A small-capacity arm loses 0.110. §4.2 states that no family had its own tuning budget. §7.4 replaces "biases toward the null" with what the curve shows. **Not run:** K > 11 (needs new scenarios, caches and labels). | §6.2 and Figure 6; §1.3; §4.2; §7.4; §7.5; Supp. Table S-a19lc |
| M8 | The "deployment manifests" claim | **Text.** "Deployment manifests" and "declarative specifications" become "declared publish–subscribe architecture models" in the abstract, §1.1, §1.5, §3, Figure 1, §7.3 and §8, with "no manifest importer is evaluated" (§3). Contribution 5 now separates twelve synthetic architectures labeled by three simulators from five system models labeled by I*. MicroART is cited. **Not run:** importer and second modeler. | Abstract; §1; §3; §7.3; §8; §2.2 |
| M9 | Unevaluated machinery | **Text.** The power-mean and Library-weight formulas, and Rules 2–4 and 6, moved to a new supplement section (`supp:weights`). The AHP detail is cut to one clause. The explanation layer was removed from Figure 1 (source and caption) and appears only in the supplement. | §3.1–3.3; Figure 1; Supp. `supp:weights` |
| M10 | Over-hedged, dense, revision-history artifacts | **Text, targeted.** A claim-status table (`tab:claims`) replaces the status paragraph. "(exploratory)" parentheticals were removed from the abstract and the Table 10 row. Repeated first-order and fixed-configuration caveats were cut from §1.3, §6.2 and §7.1. "Seventeen numbered amendments" and "Across revision cycles…" are gone. Table 7 panels are named by what they test, and the caption maps F-numbers to the supplement. **Declined (author's decision):** reordering the introduction, renaming arms, heavy supplement pruning. | §5.3; §1.3; Table 7; §7; §8 |

## Minor comments

| # | What changed |
|---|---|
| m1 | Abstract: 238 words by the reconciler's count (math as one word), 246 by raw split. |
| m2 | New highlights (70–77 characters): afferent coupling; derived graphs beyond direction and aggregator; rate formula vs rate-fed GNNs; learning curve; reference check. |
| m3 | Keyword "dependability" replaced by "afferent coupling". |
| m4 | **Declined.** Numeric citations are kept; the Guide accepts any consistent style at submission, and the journal style is applied at proof. |
| m5 | Remark 1 defines the in-degree feature exactly. §3.5 "close to" → "rank-correlated with". The reconciled "940 of the 1,321" and ρ range are kept. |
| m6 | Table 1 split into panels A and B. |
| m7 | §5.2: "within 1.1% of HGT's 434,620 parameters (429,992–437,496)". |
| m8 | §4.2 justifies the node-level early-stopping split and points to the nested protocol as the sensitivity check. |
| m9 | §4.3: "Validate-stage" removed; weights are "declared and uncalibrated, and the sensitivity of ranker order to them was not tested". |
| m10 | §4.3 notes that Spearman–Brown is approximate for rank correlations. §6.1 gives per-fold headroom: 0.047–0.291, mean 0.132. |
| m11 | §5.3: differences are computed from unrounded values. |
| m12 | "nominal Holm p" → "Holm p = 0.009 within its exploratory family". The node-order gate is "tested alone (unadjusted p)". The reconciler regex was updated. |
| m13 | Overlap@K is kept. The Table 5 caption states its tie-breaking and points to the tie-aware recall in Figure 5. |
| m14 | §1.3 and §7.3 qualify the 80%-recall statement by oracle (I*: 40–45%; I_dyn, Eq. 7: 81% at 20%) and restrict "unsuitable as gates" to reachability impact. |
| m15 | Table 10 row 1: "If I* is accepted as the impact definition, run I* directly". |
| m16 | Figure 1 per M9. Figure 3 caption: α is initialized to 1 and unconstrained; p is clipped to [0.01, 0.99] (`baselines.py`, `core.py`). |
| m17 | §5.3 names the plan commit `44713326` (6 Sep 2026). |
| m18 | Equations are numbered consistently in the LaTeX source (`\label` inside each display). The Markdown rendering's tag placement is a renderer artifact and is unchanged. |
| m19 | Added Kapoor & Narayanan 2023 (Patterns), Kleijnen 2015 and Granchelli et al. 2017 (MicroART), each checked against Crossref. |
| m20 | 56d9bff8 → 7b6f475a only re-pins the link. §5.1 states that a script checks every reported figure; the count (2,016) is in the Data Availability statement. The link is re-pinned after merge. |
| m21 | §5.1 states that the AI-assisted analysis scripts are covered by the regression tests and the reconciler. |

## Deviations

- **Provenance.** During the main sweep another session re-rendered `manuscript.md` in the main checkout, which stamped the artifact `dirty`. The sweep was re-stamped from a clean detached worktree at `48c196fb`:
  - `--resume` reused every fingerprinted fit, with identical per-seed values;
  - the zero-shot runs were repeated.

  The learning curve and the analysis ran in that worktree. Every Amendment 19 artifact is clean.
- **F15b fired on its second condition only** (see above). The text reports the remaining gap.

## Not done, stated as limitations

| Requested | Why | Where |
|---|---|---|
| Non-first-order oracle (M1) | Author's decision: new simulator variant and ~13 CPU-h relabel | §7.5 item 3 |
| Confirmation corpus (M2) | Author's decision: needs new scenarios, Neo4j caches and labels, kept outside the corpus digest | §7.5 item 1 |
| R-GCN / relation-typed summing model (M3b) | Author's decision | §7.4 |
| F8/F11 under the nested protocol (M6a) | The nested protocol was reported for GAT-P-QoS vs afferent coupling only | §6.2 |
| Second compute device (M6c) | No second device in this round | §7.4 |
| K > 11 (M7) | Needs newly generated scenarios and labels | §7.4; §7.5 item 4 |
| Manifest importer, second modeler (M8) | Not available in this round | §7.4; §7.5 item 2 |
| Introduction reorder, arm renaming (M10), author–year citations (m4) | Author's decision (targeted edits) | — |

## Checks

| Check | Result |
|---|---|
| `make -C docs/research/jss/latex` / `make supplement` | Manuscript 30 pages (limit 36), supplement 50 pages, 0 undefined references |
| Abstract / highlights | 238 words (reconciler count) / 70–77 characters |
| `python reproduce/reconcile_manuscript.py` | 2,016 / 2,016 figures match, 0 stale, 0 dirty |
| `render_manuscript_md.py --check`, `check_doc_links.py docs/` | Up to date; every link resolves |
| `pytest -m "not integration"` | 1,456 passed |
| Corpus digest | `3afa81f0…c5acb`, unchanged |
