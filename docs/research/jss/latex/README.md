# JSS submission — LaTeX sources

Elsevier `elsarticle` sources for the *Journal of Systems and Software* submission (Special Issue
VSI:AI4MSS). **This folder is authoritative**: it is what gets zipped for Editorial Manager, it is
where manuscript revisions land, and its results tables and figures are generated from committed
artifacts by scripts under `reproduce/` rather than written by hand.

[`../draft.md`](../draft.md) is a Markdown rendering of the same manuscript. It is now **generated**,
not maintained by hand:

```bash
make                                    # refresh manuscript.aux / manuscript.bbl first
python ../../../../reproduce/render_draft_md.py
```

Section, table, figure and citation numbers in `draft.md` are read out of the compiled `.aux` and
`.bbl`, so the Markdown carries exactly the numbering the PDF does. Run the generator after every
revision round; editing `draft.md` by hand reintroduces the drift that previously left it two
revision rounds behind (with superseded results tables, a withdrawn fallback-gate recommendation, and
a retracted cost claim still in it).

`draft.md` keeps two things these sources do not, and the generator preserves both: an ASCII
schematic of the pipeline and an ASCII diagram of the HGT layer stack (`Figure M1`), plus a header
note recording the review model. Its figure numbering matches the LaTeX; a note in the document
records the `Figure_N` file correspondence.

## Layout

```
latex/
├── manuscript.tex       — main file: preamble, frontmatter, abstract, \input of every section
├── sections/            — one .tex per manuscript section (sec1..sec9) + declarations.tex
│                         NOTE: sec4_* is the PREDICTIVE pathway (HGT), sec5_* the EXPLANATION layer (RM)
├── supplementary.tex    — SEPARATE document, Sections S1–S8 (see below); builds standalone
├── refs.bib             — 91 references, shared by the manuscript and the supplement
├── title_page.tex       — SEPARATE, non-anonymous title page for Editorial Manager
├── highlights.tex       — SEPARATE file, 5 bullets ≤85 chars (Elsevier requires "highlights" in the name)
├── LENGTH_JUSTIFICATION.md — text for the "Comments to the Editor" field
├── figures/             — Figure_1..3 + Figure_S1..S2 (.pdf + .png @300dpi); figures/src/ has the two
│                          graphviz .dot sources
├── vendor/              — elsarticle.cls + the .sty/.bst files this machine's TeX Live didn't ship
└── Makefile
```

## Build

```bash
make            # pdflatex -> bibtex -> pdflatex x2 -> manuscript.pdf
make figures    # regenerate all figures (delegates to reproduce/Makefile jss-figures)
make flat       # manuscript_flat.tex — single file, if a submission portal rejects \input
make zip        # submission_package.zip — everything Editorial Manager needs
make clean      # remove build artifacts, keep the PDF
make distclean  # remove the PDF too
```

The supplement is built separately:

```bash
pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

No system-wide LaTeX package installation is required — `vendor/` is self-contained and the Makefile
points `TEXINPUTS`/`BSTINPUTS` at it. Verified against a minimal `texlive-latex-base` install
(Debian, 2026), `pdflatex`, `bibtex`. `latexmk` and `xelatex` are not required or used.

**The `times` class option is not used.** URW Times is installed here and `[preprint,3p,times]`
builds cleanly at the same page count, so this is not a hard constraint — it is left off only to
avoid an unrequested change of appearance. The document uses `lmodern` instead, which microtype's
font-expansion feature needs anyway.

## Class options and page count

`manuscript.tex` uses **`[preprint,3p]`** — a one-column 10pt journal layout, single-spaced. This is
the layout JSS's "<36 pages single-column" guidance reads naturally against.

| Class options | Pages | Note |
|---|---:|---|
| **`[preprint,3p]`** | **39** | **current setting** |
| `[preprint,review,3p]` | — | 1.5-spaced reviewing copy; add `review` back if the editor asks for one |
| `[preprint]` | — | Elsevier's generic preprint layout (larger type/margins) |

Of the 39 pages, the reference list is the last 5. Reaching the 36 the Guide encourages would require
dropping an evaluation condition; `LENGTH_JUSTIFICATION.md` argues the case for the current length
and lists what has already been moved to the supplement.

**Re-measure, do not restate.** This file previously carried three different page counts at once (43,
43 and 36) against an actual 39. Take every count here from the build: `pdfinfo manuscript.pdf`,
`grep -c 'begin{table' sections/*.tex`, `grep -c '^\\bibitem' manuscript.bbl`.

## Supplementary material

`supplementary.tex` (8 pages, Sections S1--S8) carries the material moved out of the body during condensation:

| § | Content |
|---|---|
| S1 | Parameter sensitivity of the explanation layer (OFAT + Morris), Figure S1 (AHP shrinkage) |
| S2 | Zero-inflation sensitivity of the oracle-agreement figures |
| S3 | Domain-specific weighting and threshold sensitivity |
| S4 | AHP pairwise-comparison matrices and their consistency diagnostics |
| S5 | Generative parameters of the synthetic corpus |
| S6 | Anti-pattern detection benchmark and node-type stratification |
| S7 | Real-world evaluation of the explanation layer (RM / Q(v) against I_comp) |
| S8 | HGT relational attention-weight analysis, Figure S2 |

The two documents do not share an `.aux`, so cross-references from the supplement into the body are
written as literal text ("Section 7.1 of the main manuscript"), never as `\ref`. Keep it that way —
`\ref` into the other document renders as `??`.

## Figures

Three figures in the manuscript, each `\includegraphics`'d from a live section and cross-referenced
with `\ref`, plus two in the supplement:

| Fig. | File | Content | Section | Generator |
|:---:|---|---|---|---|
| 1 | `Figure_1.pdf` | end-to-end SaG pipeline | §1.3 | `figures/src/figure1_pipeline.dot` |
| 2 | `Figure_2.pdf` | running example: structural graph + `DEPENDS_ON` | §3.3 | `figures/src/figure2_running_example.dot` |
| 3 | `Figure_3.pdf` | results at a glance (LOSO ρ, F1@K, oracle agreement) | §7.1 | `reproduce/render_results_figure.py` |
| S1 | `Figure_S1.pdf` | AHP shrinkage sensitivity | Supp. S1 | `reproduce/render_shrinkage_figure.py` |
| S2 | `Figure_S2.pdf` | HGT attention-weight case study | Supp. S8 | `reproduce/extract_attention.py` + `render_attention_subgraph.py` |

File numbering and printed numbering now agree, as the JSS Guide for Authors requires ("number images
according to the order they appear within your article"): the manuscript's artwork is `Figure_1..3`
and the supplement's is kept in a separate `Figure_S*` series. Both generators default to the current
(`_v4`/`_v3`) artifacts, so `make figures` reproduces what is shipped; they previously defaulted to
superseded ones, which is how the figures went stale before.

The Graphviz figures must keep their **natural canvas width near the text block (~468pt)**. They are
included at `width=\linewidth`, so a canvas twice that width is scaled to ~0.5 and every font inside
is halved with it. After editing a `.dot`, re-measure with `pdfinfo figures/Figure_N.pdf` rather than
judging by eye.

## Verifying a revision

```bash
python ../../../../reproduce/reconcile_manuscript.py --verbose
```

Reconciles every reported table figure — currently **237** — against the artifact that produced it,
and flags any that is missing, stale against the corpus, or was produced from a dirty working tree.
It covers `supplementary.tex` as well as the body: the supplement restates body figures as literal
text (it cannot `\ref` across documents), and that is how S6/S7 once kept a superseded pooled ρ after
§7.3.6 was corrected. Run it after any
edit to a table in `sec6`/`sec7`. It does **not** check numbers that appear only in prose; that class
of defect has bitten this manuscript twice, so grep for headline values across both formats after a
revision:

```bash
grep -rnE '0\.680|0\.160|0\.695|0\.581|0\.568|0\.114|0\.054|0\.127|2,461|2,812' sections/ ../draft.md
```

Current state of the build: **39 pages**, 9 sections, 12 tables, 3 figures, 91 references (all cited),
**zero LaTeX errors, zero undefined references, zero undefined citations, zero overfull boxes**. The
supplement builds to 8 pages (S1--S8, 6 tables, 2 figures), also with zero undefined references.

## What's still a placeholder

- **Review model** — settled: JSS uses **single-anonymised** review (Elsevier Guide for
  Authors, confirmed September 2026), so `manuscript.tex` correctly carries the author block
  and `title_page.tex` is uploaded to Editorial Manager as a separate file. Do not anonymise
  the body.
- **Generative-AI declaration** — its own `\section*{}` at the end of `sections/declarations.tex`,
  immediately before the reference list, with the heading the Guide prescribes. Tool name filled in
  (Anthropic's Claude, for language and LaTeX typesetting only). Resolved.
- **Vitae** — `vitae.tex` exists but both biographies are placeholders. The Guide requires a
  ≤100-word biography per author, in an editable format.
- No **Acknowledgements** section is included; add one before submission if needed. It belongs in its
  own section directly before the reference list (and before the generative-AI declaration).
- **Graphical abstract** — encouraged by the Guide, not required; not produced here. If added:
  531 × 1328 px (h × w) or proportionally more, TIFF/EPS/PDF/MS Office, separate file.
- **Length** — 39 pages, over the "less than 36 pages single-column" the Guide encourages, so
  `LENGTH_JUSTIFICATION.md` must go into the "Comments to the Editor" field at submission.
