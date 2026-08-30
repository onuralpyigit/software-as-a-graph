# Thesis material scaffold

This folder exists because the [JSS special-issue draft](../jss/draft.md) was condensed from
~30,100 words to fit the journal's ≤36-single-column-page guidance, refocused around the paper's
graph-learning-and-dependability claim. The material that came out is not wrong or wasted — most of
it is exactly the kind of design and validation detail a thesis chapter has room for that a journal
paper does not. This folder is where it lives so it doesn't have to be re-derived or re-split out of
git history later.

**`jss_draft_full.md` is frozen.** It is a near-byte-identical copy of `docs/research/jss/draft.md`
as it stood at commit `f0cba41822820a79ebdab123d54a76072b8f1689`, immediately before the condensation
pass — one word drifted since (`HGL` → `HGT` on line 141, from the repo-wide HeteroGAT→HGT rename in
`904f64b`; harmless, but the file is no longer literally byte-identical to that commit). Write thesis
chapters *from* it — don't edit it in place. If you need to compare against the current (condensed) JSS
draft, diff the two files directly. **It is also RMAV-era throughout** (the retired four-dimension
Reliability/Maintainability/Availability/Vulnerability model — 42 occurrences, e.g. §4.2 "RMAV
Formulas") and its results tables are superseded by later reruns; see
[`outline.md`](outline.md#source-integrity) for the source-precedence rules that govern when this file
may and may not be used directly.

**`material/*.md`** are the individual sections lifted out of the full draft, verbatim, each with its
own provenance header. They exist so a specific piece of content (say, the RM formulas) doesn't
require re-reading all 30k words of `jss_draft_full.md` to find.

## Map: JSS section → thesis material → what happened to it in the journal paper

| JSS §(§) | Thesis file | Condensed in JSS to | Why it's here rather than in the paper |
|---|---|---|---|
| §3.2 Tables 3–4, §3.4, §3.5 Table 6 | [`material/model_details.md`](material/model_details.md) | Two sentences + a pointer to the replication package | Supporting detail (QoS-weight tables, the `cm_*` SCA metric list, the four-layer table) that the condensed §3 states the conclusion of but doesn't need to re-derive |
| §4.1–§4.6 (RM dimensions, formulas, composite score, worked example) | [`material/rm_attribution.md`](material/rm_attribution.md) | §4 retitled "Interpretable Attribution as a Baseline", ~1,200 words, one merged table | §8.3 shows the stated dimension weighting does not improve ranking accuracy over equal weights — the paper keeps that result and the dimension *definitions*, but not the full formula derivation, the three-weighting-paths account, or the worked example |
| §4.7 (Relationship Criticality) | [`material/relationship_criticality.md`](material/relationship_criticality.md) | Removed entirely | Defined but not validated — §8.2 states outright that the edge-removal measurement does not validate it, since the two are computed over populations that barely intersect. A thesis chapter can present this as a design contribution without the journal paper's validation bar |
| §5.4 (library blast, negative result), §5.5 (stratified correlation) | [`material/oracles_and_labels.md`](material/oracles_and_labels.md) | One paragraph each | Both were already restated nearly in full in §8.2 — the condensation removes duplication, not the finding. §5.1 (the three-oracle definitions) and §7.5 (their measured agreement) stay in the journal paper in full; this file includes them for self-containedness |
| §6.1–§6.7 (remediation operators, acceptance criterion, CI/CD gate, yield analysis) | [`material/remediation_and_gating.md`](material/remediation_and_gating.md) | Two subsections, ~700 words; §6.7's yield table becomes two sentences folded into §8.4 | The mechanism (Generate→Verify, the four operators, the gate's exit codes) stays in the paper; the design-rationale prose around each (why per-edit rather than aggregate verification, the independence invariants spelled out separately from §5.3) is thesis-chapter depth |
| §9.2 (six instrument defects) | [`material/threats_and_instrument_defects.md`](material/threats_and_instrument_defects.md) | One paragraph naming only the two defects that touched reported figures | The full six-defect account is a methodology finding in its own right — silent instrumentation bugs that produced normal-looking wrong numbers — worth a thesis discussion on validation discipline, but more than a journal Discussion section can carry alongside everything else in §9.2 |
| *(none — new material)* | [`material/why_not_simulate.md`](material/why_not_simulate.md) | Not in the journal paper | **Written for the thesis, not lifted from the condensation.** Answers the "if simulation defines criticality, why train a predictor?" question that `../jss/outline.md` records as an unresolved reviewer risk. It is the upstream text that `docs/criticality.md` §7.2 and the JSS introduction condense from |

## Suggested chapter mapping

**Superseded by [`outline.md`](outline.md).** That file is now the authoritative eleven-chapter plan
(the "natural first cut" below was five bullets and explicitly non-binding); it also carries the
thesis's central claim, the publication map with co-authorship boundaries, and the source-integrity
rules this folder needs. Keep the table below for its provenance detail — which JSS subsection each
material file came from and why it was cut — but plan chapters from `outline.md`, not from here.

None of this is binding — the actual thesis structure depends on what else surrounds the SaG work —
but a natural first cut:

- **Ch. "The SaG model"** ← `model_details.md` + the (unabridged) §3 of `jss_draft_full.md`.
- **Ch. "Interpretable attribution"** ← `rm_attribution.md` + `relationship_criticality.md`. This
  is the natural home for the full RM derivation and the edge-criticality construction that the
  journal paper can only state the conclusion of.
- **Ch. "Failure-impact prediction"** ← the unabridged §5 of `jss_draft_full.md` (the HGT and the
  learning results are the journal paper's core and stay there too) + `oracles_and_labels.md` for
  the full oracle-disagreement analysis + `why_not_simulate.md`, which motivates the stage as a
  whole and depends on that disagreement analysis, so the two belong in the same chapter.
  `why_not_simulate.md` §4 also forward-references the attribution chapter and the remediation
  chapter; if the thesis opens with a framework-overview chapter, it reads equally well there.
- **Ch. "Remediation and continuous verification"** ← `remediation_and_gating.md`.
- **Ch. "Methodology and validation discipline"** (or an appendix) ← `threats_and_instrument_defects.md`.
  This is good material for a thesis's methodology chapter specifically because it documents *how*
  each defect was found, not just that it existed.

## What's not here

Everything that survived the condensation in full or near-full — §1 (motivation, RQs), §2 (related
work), the three-oracle definitions of §5.1, the HGT architecture of §5.2, and all of §8's results
tables — is in [`docs/research/jss/draft.md`](../jss/draft.md) itself, condensed but not gutted, and
`jss_draft_full.md` for the pre-condensation wording if a thesis chapter wants the longer phrasing.
