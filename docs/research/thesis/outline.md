# Graph-Based Modeling and Analysis of Distributed Publish–Subscribe Systems

> **Structural outline for the doctoral monograph.** This file is the thesis's counterpart to
> [`../jss/outline.md`](../jss/outline.md) and [`../ause/outline.md`](../ause/outline.md): a
> chapter-by-chapter map of what each chapter argues, what backs it, which source it draws from, and
> what is still open. No chapter text exists yet — this is the plan, not the manuscript.
>
> **Source precedence is not negotiable** (see [§Source integrity](#source-integrity)). The frozen
> [`jss_draft_full.md`](jss_draft_full.md) is **RMAV-era** and must not be used for Chapter 4 or for
> any results table.

* **Degree / institution:** PhD, Istanbul Technical University, Department of Computer Engineering
* **Advisor:** Prof. Feza Buzluca
* **Format:** monograph (not article-compilation), English, ITU LaTeX template — *template not yet obtained*
* **Publications covered:** JSS (submitted), AuSE (submitted), RASSE 2025 (published), UYMS 2026 ×2 (co-authored)

---

## The claim

The thesis argues one thing that none of its constituent papers argues, because each paper collapses
the duality in a different direction:

> A single typed graph substrate supports **two parametrically independent pathways** — a
> deterministic, standards-grounded *diagnostic attribution* (Pathway A) and a learned *predictive
> ranking* (Pathway B) — which are not competitors but answer different questions, are composed at
> inference by a **triage bridge**, and are closed by counterfactually verified prescription.

**Why the papers cannot make this claim.**

| Venue | What it does to the duality |
|---|---|
| JSS | Demotes Pathway A: *"one primary contribution with a supporting layer, rather than two co-equal pathways"*; RM retitled *The Explanation Layer* (§5) |
| AuSE | Pathway A is the substrate under audit; the contribution is prescription |
| RASSE 2025 | Predates the split — one analytical criticality score, $\rho = 0.94$ |

**Why the claim is load-bearing rather than cosmetic.** It converts the project's most awkward
results into evidence. Pathway A's LOSO $\rho = 0.195$ and its anti-correlation at every AHP
shrinkage setting are not a weak baseline — they are the measurement establishing that a diagnostic
instrument is not a ranker. The separation is *architectural* (no shared parameters, per
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md#L49)) and *empirically testable*: the `λ_RM` coupling
term in [`saag/prediction/models/core.py`](../../../saag/prediction/models/core.py#L432) defaults to
0.0, and the $\lambda_{\text{RM}} = 0.1$ arm is the direct evidence the pathways are separable.
**No publication currently reports that ablation as separability evidence.**

---

## Chapter map

| Ch | Title | Primary sources | Status |
|:--:|---|---|---|
| — | Front matter (incl. Turkish `ÖZET`, jury page) | new | blocked on ITU template |
| 1 | Introduction: the Architecture–Code Gap and the dual-pathway thesis | JSS `sec1_introduction.tex`; `material/why_not_simulate.md` | to write |
| 2 | Background and Related Work | frozen §2 (8 subsections) ∪ AuSE §2 | to write, last |
| 3 | The Software-as-a-Graph Model | JSS `sec3_sag_model.tex`; `material/model_details.md`; frozen §3.6 | assemble |
| 4 | **Pathway A — Diagnostic Attribution** | `material/rm_attribution.md`; `material/relationship_criticality.md`; AuSE §4 | assemble + restore |
| 5 | **Pathway B — Predictive Failure-Impact Forecasting** | JSS `sec4_*.tex`; `material/oracles_and_labels.md`; `material/why_not_simulate.md` | assemble |
| 6 | **Pathway Integration: the Triage Bridge** | `saag/analysis/triage.py`; `ARCHITECTURE.md`; `λ_RM` ablation | **new — write first** |
| 7 | Prescriptive Remediation and CI/CD Gating | `material/remediation_and_gating.md`; AuSE §5–§7, §9.2–§9.5 | assemble |
| 8 | Experimental Design | JSS `sec6_experimental_setup.tex`; AuSE §8 | assemble |
| 9 | Results | JSS `sec7_results.tex` (RQ1–RQ5); AuSE §9 | assemble |
| 10 | Methodology and Validation Discipline | `material/threats_and_instrument_defects.md`; RASSE reconciliation | to write |
| 11 | Discussion, Threats, Conclusion, Future Work | JSS `sec8_discussion.tex`; AuSE §10–§11 | to write, last |
| A–D | Published papers; replication package; metric registry; anti-pattern catalog | existing | assemble |

---

## Chapters that differ from the sum of the papers

Chapters 3, 8 and 9 are largely assembly. **Chapters 4, 6 and 10 are where the monograph is not a
stapled set of papers**, and they are what a committee will read as the doctoral contribution.

### Chapter 4 — Pathway A restored to first class

The JSS paper compresses RM to ~1,200 words and one merged table, and **cuts relationship
criticality (frozen §4.7) entirely** because it is *defined but not validated* — the edge-removal
measurement runs over a population that barely intersects the one $D_2$ scores. A thesis can present
a design contribution without a journal's validation bar, provided it says so in those words rather
than letting the reader infer validation. This chapter carries:

- the full RM derivation, the composite $Q(v)$, adaptive box-plot classification, and the worked
  attribution example — from [`material/rm_attribution.md`](material/rm_attribution.md), already
  migrated to RM;
- relationship criticality $D_2$ — from
  [`material/relationship_criticality.md`](material/relationship_criticality.md), formulas verified
  against `_score_and_classify_edges`;
- the nineteen-pattern anti-pattern catalog — from AuSE §4, which is its authoritative home.

**Open:** `material/rm_attribution.md` Table 10 (worked attribution) is marked
`TODO(needs re-measurement)` and has never been re-run under the RM model. Regenerate; do not carry
the stale table forward.

### Chapter 6 — the Triage bridge (new)

The integration mechanism is implemented ([`saag/analysis/triage.py`](../../../saag/analysis/triage.py)),
tested ([`tests/test_triage.py`](../../../tests/test_triage.py)), wired into the pipeline and the
CLI (`--triage-k`), and documented in `ARCHITECTURE.md`, `docs/cli-pipeline-guide.md`,
`docs/criticality.md` and `docs/visualization.md` — but **appears in no manuscript beyond a clause**.
It is the thesis's own contribution and should be written first, because it depends on code rather
than on either paper's text and therefore blocks nothing.

What the chapter must establish:

1. **The join is by component id, never by reading a diagnosis off a ranking.** A
   `GNNAnalysisResult`'s `components` shim leaves `fault_tolerance`/`availability` at 0.0 and
   `profile` at `None`; the diagnosis always comes from the RM substrate.
2. **Cold-start degradation is graceful and declared** — `TriageResult.ranking_source ∈ {"gnn","rm"}`;
   with no checkpoint, Pathway A ranks *and* diagnoses.
3. **Separability is measured, not asserted** — the $\lambda_{\text{RM}}$ ablation.
4. **Simulation is never a triage input** — the independence guarantee, enforced by three separate
   architectural tests.

### Chapter 10 — methodology as contribution

Not filler. The six instrument defects in
[`material/threats_and_instrument_defects.md`](material/threats_and_instrument_defects.md) are silent
bugs that produced normal-looking wrong numbers, and the advisor-review round **reversed two RQ
conclusions** by fixing measurement rather than method (pooled-vs-stratified population; the
degenerate RM substrate). Together with the RASSE reconciliation below, this is a documented account
of how this class of experiment fails — which very few theses can show.

---

## Reconciling RASSE 2025

RASSE reports $\rho = 0.94$. Current measurement puts Pathway A at $\rho = 0.195$ (LOSO) and the
catalog at $\rho = 0.485$ — *below degree centrality at 0.519*. A committee member will find this.
The reconciliation belongs in Chapter 10 as a stated result, not in a threats subsection:

- **Different population.** RASSE pooled across node types. Pooled $\rho = 0.028$ falls *outside* the
  per-type range $[0.14, 0.50]$ — a Simpson's-paradox effect. Pooled and stratified correlations are
  not comparable quantities.
- **Different oracle.** RASSE validated against reachability loss; current work reports against
  $I^*$, $I_{\text{comp}}$ and $I_{\text{dyn}}$, which agree only at top-$K$ Jaccard 0.31–0.42.
- **Different composite.** The 4-D RMAV score RASSE used no longer exists.

Framed as measurement maturation, this strengthens the thesis. Smoothed over, it is a defence
liability.

---

## Publication map (Chapter 1)

ITU requires per-publication contribution disclosure. Two of the five works are co-authored with
other students and must have their boundaries stated explicitly.

| Work | Status | Feeds | Contribution note needed |
|---|---|---|---|
| RASSE 2025 (`10.1109/RASSE64831.2025.11315354`) | published | Ch. 4, Ch. 10 | superseded by Ch. 4's model; reconciled in Ch. 10 |
| JSS (VSI:AI4MSS) | submitted | Ch. 3, 5, 8, 9 | — |
| AuSE (CI/CD-DevOps-TD SI) | submitted | Ch. 4, 7, 9 | companion-submission disclosure |
| UYMS 2026 — structural interaction patterns | published | Ch. 2, 3 | **co-authored (Çalışkan, Yiğit, Buzluca)** |
| UYMS 2026 — visualization | published | Ch. 3 (appendix) | **co-authored (Erşen, Çalışkan, Yiğit, Buzluca)** |
| Middleware 2026 research | rejected, dead | — | not claimed |
| Middleware 2026 industrial | abandoned, unsubmitted | Ch. 7 (optional) | check `data/clearance.md` before reuse |

---

## Source integrity

**[`jss_draft_full.md`](jss_draft_full.md) is RMAV-era and mostly unusable as a direct source.** It
carries 42 `RMAV` occurrences; §4.1 is *"Four Orthogonal Dimensions"* and §4.2 is *"RMAV Formulas"* —
the retired four-dimension model throughout. Its Tables 18–23 are superseded three times over: the
RM migration, the advisor-review reruns, and the 2026-08-29 weight revision.

Precedence, always:

1. **Formulas and model text** → `../jss/latex/sections/*.tex` (authoritative, RM-era, post-review),
   plus the two already-migrated files `material/rm_attribution.md` and
   `material/relationship_criticality.md`.
2. **Prescription and catalog** → `../ause/draft.md` (authoritative for AuSE).
3. **Every number** → the artifact under `results/` that generated it, via `reproduce/`. Never
   transcribe a figure from any draft; a hand-transcribed results figure has published superseded
   values in this project before.
4. **`jss_draft_full.md` is for pre-condensation *prose phrasing* only** — never for Chapter 4, never
   for a results table.

**Correction to [`README.md`](README.md):** it asserts `jss_draft_full.md` is byte-identical to commit
`f0cba41822820a79ebdab123d54a76072b8f1689`. It is not — the HGT rename (`904f64b`) changed one word
on line 141 (`HGL` → `HGT`). Benign, but the README's stated invariant is now false and should be
relaxed rather than left standing.

---

## Sequencing

JSS is due **30 September 2026** and AuSE is near-term. The thesis must not compete with either.

1. **Now, while both submissions are live.** Only work that draws on code rather than manuscript
   text: obtain the ITU template, build the skeleton and front matter, write the Chapter 1
   publication map, and draft **Chapter 6**. Regenerate the two missing artifacts (below).
2. **After JSS ships.** Chapters 3, 5, 8, 9 assemble from the frozen-at-submission LaTeX.
3. **After AuSE ships.** Chapter 4's catalog half, Chapter 7, and Chapter 9's prescription results.
4. **Last.** Chapters 2, 10, 11 — these must see all results before they can be written honestly.

## Outstanding

1. **ITU thesis template not obtained.** No thesis class, Makefile, jury page or `ÖZET` scaffold
   exists in the repo; the two journal builds vendor `elsarticle` and plain `article` respectively.
   This blocks the front matter and the LaTeX skeleton, nothing else.
2. **Table 10 (worked attribution) needs re-measurement** under the RM model.
3. **Figure 6 (HGT attention subgraph) has no rendered artifact** — referenced in two files;
   `reproduce/render_attention_subgraph.py` produces it.
4. **`README.md`'s byte-identity claim is false** — see Source integrity above.
5. **Chapter 2 must merge two related-work sections** written for different venues and audiences
   (frozen §2's eight subsections; AuSE §2's anti-pattern, refactoring-recommendation and SBSE
   strands). This is a rewrite, not a concatenation.
