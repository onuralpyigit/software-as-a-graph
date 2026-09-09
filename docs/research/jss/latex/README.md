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
├── sections/            — one .tex per manuscript section (sec1..sec8) + declarations.tex
│                         NOTE: sec4_* is the PREDICTIVE pathway (HGT), sec5_* the EXPLANATION layer (RM)
│                         §9 (Conclusion) is a top-level \section at the end of sec8_discussion.tex
├── supplementary.tex    — SEPARATE document, Sections S1–S7 (see below); builds standalone
├── refs.bib             — 91 references, shared by the manuscript and the supplement
├── title_page.tex       — SEPARATE, non-anonymous title page for Editorial Manager
├── highlights.tex       — SEPARATE file, 5 bullets ≤85 chars (Elsevier requires "highlights" in the name)
├── LENGTH_JUSTIFICATION.md — text for the "Comments to the Editor" field
├── figures/             — Figure_1..Figure_5 (.pdf + .png @300dpi); figures/src/ has the two
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
| **`[preprint,3p]`** | **43** | **current setting** |
| `[preprint,review,3p]` | — | 1.5-spaced reviewing copy; add `review` back if the editor asks for one |
| `[preprint]` | — | Elsevier's generic preprint layout (larger type/margins) |

Of the 43 pages, the main text through the Conclusion is 37 and the reference list is 5. Reaching 36
for the whole PDF would require dropping an evaluation condition; `LENGTH_JUSTIFICATION.md` argues
the case for the current length and lists what has already been moved to the supplement.

## Supplementary material

`supplementary.tex` (7 pages) carries the material moved out of the body during condensation:

| § | Content |
|---|---|
| S1 | Parameter sensitivity of the explanation layer (OFAT + Morris), Figure S1 (AHP shrinkage) |
| S2 | Zero-inflation sensitivity of the oracle-agreement figures |
| S3 | Domain-specific weighting and threshold sensitivity |
| S4 | AHP pairwise-comparison matrices and their consistency diagnostics |
| S5 | Generative parameters of the synthetic corpus |
| S6 | Anti-pattern detection benchmark and node-type stratification |
| S7 | Real-world evaluation of the explanation layer (RM / Q(v) against I_comp) |

The two documents do not share an `.aux`, so cross-references from the supplement into the body are
written as literal text ("Section 7.1 of the main manuscript"), never as `\ref`. Keep it that way —
`\ref` into the other document renders as `??`.

## Figures

Four figures, each `\includegraphics`'d from a live section and cross-referenced with `\ref`, plus
one in the supplement:

| Fig. | File | Content | Section | Generator |
|:---:|---|---|---|---|
| 1 | `Figure_1.pdf` | end-to-end SaG pipeline | §1.3 | `figures/src/figure1_pipeline.dot` |
| 2 | `Figure_2.pdf` | running example: structural graph + `DEPENDS_ON` | §3.3 | `figures/src/figure2_running_example.dot` |
| 3 | `Figure_5.pdf` | results at a glance (LOSO ρ, F1@K, oracle agreement) | §7.1 | `reproduce/render_results_figure.py` |
| 4 | `Figure_3.pdf` | HGT attention-weight case study | §7.3 | `reproduce/extract_attention.py` + `render_attention_subgraph.py` |
| S1 | `Figure_4.pdf` | AHP shrinkage sensitivity | Supp. S1 | `reproduce/render_shrinkage_figure.py` |

Note that the *file* numbering and the *printed* numbering differ, because Figure_4 and Figure_5 were
orphaned during one revision and reinstated in different places. Both generators now default to the
current (`_v4`/`_v3`) artifacts, so `make figures` reproduces what is shipped; they previously
defaulted to superseded ones, which is how the figures went stale before.

The Graphviz figures must keep their **natural canvas width near the text block (~468pt)**. They are
included at `width=\linewidth`, so a canvas twice that width is scaled to ~0.5 and every font inside
is halved with it. After editing a `.dot`, re-measure with `pdfinfo figures/Figure_N.pdf` rather than
judging by eye.

## Verifying a revision

```bash
python ../../../../reproduce/reconcile_manuscript.py --verbose
```

Reconciles every reported table figure — currently **177** — against the committed artifact that
produced it, and flags any artifact older than the corpus it claims to describe. Run it after any
edit to a table in `sec6`/`sec7`. It does **not** check numbers that appear only in prose; that class
of defect has bitten this manuscript twice, so grep for headline values across both formats after a
revision:

```bash
grep -rnE '0\.680|0\.160|0\.695|0\.581|0\.568|0\.114|0\.054|0\.127|2,461|2,812' sections/ ../draft.md
```

Current state of the build: 43 pages, **zero LaTeX errors, zero undefined references, zero undefined
citations, zero overfull boxes**, 12 tables, 4 figures, 91 references (all cited). The supplement
builds to 7 pages, also with zero undefined references.

## What's still a placeholder

- **Review model** — settled: JSS uses **single-anonymised** review (Elsevier Guide for
  Authors, confirmed September 2026), so `manuscript.tex` correctly carries the author block
  and `title_page.tex` is uploaded to Editorial Manager as a separate file. Do not anonymise
  the body.
- **Generative-AI declaration** — present in `sections/declarations.tex`, to be confirmed by the
  authors.
- No **Acknowledgements** section is included; add one before submission if needed.
- **Graphical abstract** — encouraged by the Guide, not required; not produced here.
