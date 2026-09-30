# Response to advisor notes v7 (notlarv7.docx)

**Branch:** `jss-advisor-v7-revision` (from `main` @ `2cbfdf50`)
**Scope:** Section 6 (Results) text edits; four global consistency items; one integrity fix.
**Status:** Manuscript 31 pp and supplement 44 pp (both unchanged). No undefined references. The reconciler reports `1603 figures match` (1601 before, plus 2 new checks). `pytest -m "not integration"` passes: 1422 passed.

Line numbers in the "Note" column are those of the reviewed PDF. Every change is in `latex/sections/sec7_results.tex` unless another file is named. The advisor's replacement text was applied verbatim, with LaTeX cross-references (`Eq.~\eqref`, `Section~\ref`, `Table~\ref`) substituted for hard-coded numbers. The only departures are marked **Deviation**.

## Section 6

| Note | Change |
|---|---|
| before 527 | Inserted the three-dimensions paragraph (representation, analytical/hybrid/learned, restate vs. predict). |
| 533–536 | The RQ1 summary now opens "For the reachability and queue-flow oracles… The strongest learned model… (GAT-P-QoS, ρ = 0.748) approaches but does not exceed the direct-dependent reference (0.764…)". |
| 539–540 | The summary now reads "…reaches 0.830, matching or exceeding the learned approximation (+0.031, 10 of 12 folds; exploratory)". This follows the wording decision below. |
| 542–543 | Removed "0.553; corrected 0.533" from the parenthesis. |
| after 548 | Inserted the "Two patterns emerge…" paragraph. |
| 564–566 | Now reads "Trained on the dependency graph rather than the raw multigraph, the strongest learned model, GAT-P-QoS, achieves ρ = 0.748, approaching but not exceeding the direct-dependent reference." |
| 591–593 | Now reads "…retain most of their value (0.212–0.217), indicating… beyond the unweighted first-order term, although less than the rate-weighted reference (0.573)." |
| 593–611 | Replaced with the reordered paragraph (unweighted → rate-weighted → GBM → GNN). All numbers are unchanged, and it uses "at least as effectively as the learned model". |
| 629–630 | Now reads "In the exploratory n = 30 sample of I_dyn (right panel of Figure 6)". |
| 631–635 | Softened as proposed: "would likely cause alert fatigue", "This suggests", "architectural reviews", "Section 7.4 proposes a two-tiered triage protocol". |
| 637–638 | The RQ2 summary now opens "Attention-based learners improve… (GAT-P-QoS 0.748 vs. GAT-QoS 0.635), whereas the typed transformer does not… (HGT-P-QoS 0.514 vs. HGT-QoS 0.622)". |
| after 643 | Appended the ablation-magnitude sentences (−0.136/−0.270 degree; −0.013 typing; −0.016 aggregator; +0.113 representation). |
| 649 | "the earlier QoS effect" → "the QoS effect in Table 7". |
| 690 | "the author wrote all five" → "the first author wrote all five". |
| 716–718 | Now reads "processor's base power (28 W TDP)" and "355.6 Wh". Dropped "In contrast,". Added "Cost should be read together with ranking quality…". The body text's "host CPU's base package power (28.0 W…)" was aligned to "processor's base power (28 W…)". |
| 719–722 | Replaced with "Table 10 times each stage on the same graphs in one session…". **Deviation:** "because no ranker requires them" → "…requires them *at ranking time*". The five-seed sweep produces the learned engines' training labels, so the unqualified sentence would be false. The Table 10 caption's "the stage behind the earlier ratio of the detection gate" became "(the full detection gate), listed for reference". |
| 739–742 | De-duplicated the energy figures: 355.6 Wh (≈ 1.28 MJ), 0.83 Wh (2.99 kJ), 0.086 Wh (310 J). |
| 742–744 | Now reads "…running full dynamic simulations, or extracting the features that learned engines need, on every commit is the high-overhead, energy-intensive path; neural inference itself is negligible." Lines 744–747 are unchanged. |

## Outside Section 6

1. **"+0.08 to +0.11".** The range comes from the three graph-attention pairs (GAT-P/GAT, GAT-P-QoS/GAT-QoS, Hybrid-GAT-P/Hybrid-GAT: +0.075 to +0.113). No sum-aggregation network was run on the raw multigraph, so the suggested "(…for the attention and sum-aggregation networks)" would not be accurate. HGT-P-QoS falls by −0.107. The range is therefore replaced everywhere by the attention-based statement (GAT-P-QoS 0.748 vs. GAT-QoS 0.635):
   - Findings in brief (l.55). This also notes that the typed transformer does not improve.
   - Contribution 1 (l.107).
   - §7.1 (l.753).
   - Conclusion (l.985). "Graph learning benefited substantially" becomes "Attention-based learners benefited".
   - **Also the highlights**, which carried the same range. That highlight now reads "Graph-attention learners gain on derived dependency graphs (rho 0.748 vs 0.635)." (80 characters.)

   The supplement's "GAT arms gain +0.075 to +0.113" is already scoped correctly and was kept.
2. **"Exceeding" vs. "matching or exceeding".** Adopted "matching or exceeding" (or "at least as well/effectively as") for every qualitative claim. The abstract, Findings in brief, Contribution 4, §7.1, §7.2, §7.4, §7.5 and the Conclusion already used it. A literal numeric contrast keeps its literal wording ("which exceeds it by +0.031 [+0.013, +0.049] on 10 of 12 folds"), as in the advisor's own paragraph. The supplement's "already exceeds the learned approximation" became "already matches or exceeds".
3. **"Earlier version" (l.380, §4.3).** The sentence now reads "A sensitivity check restricted to the first 30 Applications per fold in lexicographic order is reported in Supplementary Section S38 (Table S62)." **Deviation:** the text cites the supplement rather than the replication repository, because that is where the check is. Other revision-history wording was also removed:
   - In the main text: §6.1, §6.2, §6.4 and the Table 10 caption.
   - In the supplement: "A previous version of this supplement…", "Earlier drafts measured…", "The earlier version of this figure is withdrawn…", "…which is what earlier revisions did", "earlier lexicographic sample", and "The earlier ratio of Table S…".

   Legitimate uses remain ("earlier naming scheme", "an earlier session", "an earlier zero-shot configuration").
4. **American spelling.** The following were corrected:
   - modelling → modeling (§5.1, vitae)
   - labelled → labeled (§7.3)
   - single-modeller → single-modeler (§7.5 heading)
   - utilisation → utilization
   - relabelled → relabeled
   - labelling → labeling
   - artefact → artifact (×3)
   - behavioural → behavioral
   - analyses (verb) → analyzes

   The two table generators that emit these strings (`reproduce/render_amendment7_tables.py`, `reproduce/render_referee_tables.py`) were updated too, so re-rendering cannot bring the British forms back. Citation titles in `refs.bib` were left as published.

## Integrity fix (not in the notes)

§5.1, the RQ3 paragraph (§6.3), §7.5 and §7.6 still reported an "exploratory calibration spot-check" by the second author (entity J ≥ 0.88, edge J ≈ 0.65–0.72). No such check was performed, and no `model_agreement` artifact exists; the withdrawal from the round-9 revision never reached `main`. All four passages now state that the first author created every model, that no second modeler re-derived any of them, and that no inter-modeler agreement is reported. The §7.5 wording is the one used before the claim entered in `52d88efc`. This also answers the l.690 question ("which author?") consistently across the paper.

## Tooling

- `reproduce/reconcile_manuscript.py`: the `sec:rq1 Eq. 7` quote check now anchors on the reordered sentences. A new check (`sec:rq1 summary Eq. 7`) verifies the +0.031 and 10/12 now quoted in the RQ1 summary.
- The following were regenerated: `manuscript.md`, `sections/*.md`, `supplementary.md`, `manuscript_flat.tex`, `supplementary_flat.tex`, both PDFs, and `submission_package.zip`. The flat and Markdown copies were already behind `2cbfdf50`/`338f7141`, so their diffs also carry those commits' edits.
