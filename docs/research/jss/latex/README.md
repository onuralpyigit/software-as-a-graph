# JSS submission — LaTeX sources

Elsevier `elsarticle` sources for the *Journal of Systems and Software* submission (Special Issue
VSI:AI4MSS). **This folder is authoritative**: it is what gets zipped for Editorial Manager, it is
where manuscript revisions land, and its results tables and figures are generated from committed
artifacts by scripts under `reproduce/` rather than written by hand.

[`../draft.md`](../draft.md) is a maintained Markdown rendering of the same manuscript, kept in sync
after each revision round. It is useful for review and diffing, but where the two disagree, these
sources win. (The relationship used to run the other way — `draft.md` was the source and this folder
its conversion — which is why the two drifted: two rounds of reviewer revisions were applied here
first. If you edit `draft.md` alone, that drift returns.)

`draft.md` is not a mechanical mirror. It keeps two things these sources do not: **Table 0** (a
comparison of dependability-analysis paradigms) and an inline ASCII diagram of the HGT layer stack,
both dropped here during condensation to fit JSS's ≤36-single-column-page guidance. Its figure
numbering therefore differs from the numbering below; `draft.md` carries a note recording the
mapping. The pre-condensation text (~30,100 words, 23 tables, 6 figures) is preserved verbatim at
[`../../thesis/jss_draft_full.md`](../../thesis/jss_draft_full.md), and the material cut from it is
organised by topic under [`../../thesis/material/`](../../thesis/material/).

## Layout

```
latex/
├── manuscript.tex       — main file: preamble, frontmatter, abstract, \input of every section
├── sections/            — one .tex per draft.md section (mirrors its numbering), + declarations.tex
├── refs.bib             — 53 references + [Anon-A], transcribed from draft.md's numbered list
├── title_page.tex        — SEPARATE, non-anonymous title page (placeholders — see below)
├── highlights.tex        — SEPARATE file, 5 bullets ≤85 chars (Elsevier requires "highlights" in the name)
├── figures/              — Figure_1.pdf .. Figure_5.pdf (+ .png @300dpi); figures/src/ has the two
│                           graphviz .dot sources
├── vendor/               — elsarticle.cls + the handful of .sty/.bst files this machine's TeX Live
│                           didn't ship (see "Toolchain" below) — self-contained, no sudo needed
└── Makefile
```

## Build

```bash
make            # pdflatex -> bibtex -> pdflatex x2 -> manuscript.pdf
make figures    # regenerate all 5 figures (delegates to reproduce/Makefile jss-figures)
make flat       # manuscript_flat.tex — single file, if a submission portal rejects \input
make zip        # submission_package.zip — everything Editorial Manager needs
make clean      # remove build artifacts, keep the PDF
make distclean  # remove the PDF too
```

No system-wide LaTeX package installation is required — `vendor/` is self-contained and the Makefile
points `TEXINPUTS`/`BSTINPUTS` at it. Verified against a minimal `texlive-latex-base` install (Debian,
2026), `pdflatex`, `bibtex`. `latexmk` and `xelatex` are not required or used.

**The `times` class option is intentionally not used** (see the comment atop `manuscript.tex`): it
pulls in `mathptmx`, which needs URW Times Type 1 font metrics from `texlive-fonts-recommended` — not
present in the no-sudo toolchain this was built against. The document uses `lmodern` instead (needed
anyway for microtype's font-expansion feature, which requires scalable outlines). Swap `times` back in
if building on a machine with `texlive-fonts-recommended` installed.

## Class options and page count

`manuscript.tex` uses **`[preprint,3p]`**. elsarticle processes options in *declaration* order
(`preprint` → `review` → `3p`), so what matters is the net state: `3p` sets a one-column 10pt journal
layout, and omitting `review` leaves `\@blstr{1}` (single spacing). Measured on the current text:

| Class options | Pages | Note |
|---|---:|---|
| **`[preprint,3p]`** | **31** | **current setting** — single-spaced, one column; the layout JSS's "<36 pages single-column" guidance reads naturally against |
| `[preprint,review,3p]` | 43 | 1.5-spaced reviewing copy; add `review` back if the editor asks for one |
| `[preprint]` | 43 | Elsevier's generic preprint layout (larger type/margins) |

Only `[preprint,3p]` comes in under 36 pages. If an editor insists on the plain `preprint` layout,
the manuscript is 43 pages and would need either a length justification in the cover letter or a
further round of cuts — §8.5 and §9.2 are the next candidates.

## Figures

Five figures, each `\includegraphics`'d from a live section and cross-referenced with `\ref`:

| Fig. | File | Content | Source section | Generator |
|:---:|---|---|---|---|
| 1 | `Figure_1.pdf` | end-to-end SaG pipeline (two pathways) | §1.3 | `figures/src/figure1_pipeline.dot` |
| 2 | `Figure_2.pdf` | running example: structural graph + `DEPENDS_ON` | §3.3 | `figures/src/figure2_running_example.dot` |
| 3 | `Figure_3.pdf` | HGT attention-weight case study | §7.3 | `reproduce/extract_attention.py` + `render_attention_subgraph.py` |
| 4 | `Figure_4.pdf` | AHP shrinkage sensitivity | §7.3 | `reproduce/render_shrinkage_figure.py` |
| 5 | `Figure_5.pdf` | results at a glance (LOSO ρ, F1@K, oracle agreement) | §7.1 | `reproduce/render_results_figure.py` |

**Two production notes.** Figures 2–4 were once orphaned: the files existed and the text referred to
them by number, but no section actually included them — the `\includegraphics` lived in
`sections/sec8_results.tex`, which `manuscript.tex` stopped inputting when `sec7_results.tex`
superseded it. They are back, and the hard-coded "Figure~3"/"Figure~4" strings are now `\ref`s so the
numbering cannot silently drift again.

The Graphviz figures must also keep their **natural canvas width near the text block (~468pt)**. They
are included at `width=\linewidth`, so a canvas twice that width is scaled to ~0.5 and every font
inside is halved with it; Figure 1 once printed its 10.5pt labels at 5.4pt for exactly this reason.
Enlarging the float cannot fix it. After editing a `.dot`, re-measure with
`pdfinfo figures/Figure_N.pdf` rather than judging by eye.

**Two generators are retired** but deliberately kept, since their analyses still live in
`docs/research/thesis/material/` and either figure is one command from returning:
`render_pooled_vs_pertype_figure.py` (pooled vs. per-node-type ρ — §5.5 is now a single paragraph)
and `render_threshold_figure.py` (propagation-threshold sweep — §7.3 carries the same numbers).
Neither is part of `make figures` any more; both carry a RETIRED banner. Note that
`render_pooled_vs_pertype_figure.py` still defaults to writing `Figure_4`, which would now clobber the
shrinkage figure — redirect its output before running it.

## What's still a placeholder

- **`title_page.tex`** — author names, affiliations, and the corresponding-author email are all-caps
  placeholders. Fill in before submission; the manuscript body (`manuscript.tex` and everything under
  `sections/`) stays double-anonymised and must **not** gain author identity.
- **Declarations** (`sections/declarations.tex`) — CRediT roles and funding are stubbed per
  double-anonymised review, exactly as in `draft.md`'s own Declarations section. The data-availability
  and competing-interest text is filled in verbatim from the source.
- **Generative-AI declaration** — stub, per journal policy, to be completed by the authors.
- No **Acknowledgements** section is included — `draft.md` has none; add one before submission if
  needed.
- **Graphical abstract** — encouraged by the Guide, not required; not produced here.

## Regenerating from draft.md

There is no committed script for the Markdown→LaTeX conversion itself (it was a one-off, throwaway
pre-pass + `pandoc` + hand review, not meant to be re-run mechanically). If `draft.md` changes
materially, the affected `sections/*.tex` file(s) need re-conversion and re-review by hand; the
figure-generation scripts under `reproduce/` (`render_shrinkage_figure.py` and the two
`figures/src/*.dot` sources) are the reusable, re-runnable part.

Conventions to preserve when re-converting a section: `\hypertarget{slug}{% \section{...}\label{slug}}`
followed by `\label{sec:N}`; cross-references as `Section~\ref{sec:N.M}` / `Table~\ref{tab:N}` /
`Figure~\ref{fig:N}`; tables as `longtable` with the pandoc column-width spec and
`\caption{...\label{tab:N}}`; citations by bibtex key, never by draft.md's `[N]` number. Two gotchas
worth knowing: `draft.md`'s literal ①/② characters have no T1 glyph and must become
`\textcircled{\scriptsize 1}`; and pandoc's `\raggedright` columns do not hyphenate, so a long
unbreakable token (`RedundancyInsertion`, `MessageFlowSimulator`) needs its column widened rather
than left to overflow.

## Verification performed

Against the condensed `../draft.md` (~16,800 words, 13 tables, 4 figures):

- Clean build (`make distclean && make`) with `[preprint,3p]`: **31 pages**, **zero LaTeX errors,
  zero undefined references, zero undefined citations**, across a full
  `pdflatex → bibtex → pdflatex → pdflatex` cycle, with no outstanding "rerun to get
  cross-references" warning.
- All **13 tables** and **4 figures** are cited in the text, defined exactly once, and numbered with
  no gaps (verified both in the sources and in `pdftotext` output of the compiled PDF).
- All **54 bibliography entries** (53 + `[Anon-A]`) are cited and render; no uncited entries.
- No dangling references to the sections cut during condensation (§3.4, §4.6, §4.7, §6.5, §6.7) or to
  the retired tables/figures.
- Headline tables (8, 9, 10, 13) and the sensitivity tables (11, 12) diff clean against `draft.md` —
  the conversion perturbs no reported figure.
- Abstract: 253 words (≤250 target; 3 over). Highlights: 5 bullets, longest 83 chars (≤85 required).
- 4 remaining overfull `\hbox` warnings, worst 8.8 pt (~0.12 in) — cosmetic, inside table cells.
- `make zip` produces a complete `submission_package.zip` (all sections, `refs.bib`, `manuscript.bbl`,
  4 figure PDFs + 4 PNGs, `vendor/`, `title_page.tex`, `highlights.tex`).
- No author-identifying strings outside `title_page.tex`.
