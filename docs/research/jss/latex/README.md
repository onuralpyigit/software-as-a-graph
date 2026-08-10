# JSS submission — LaTeX sources

Elsevier `elsarticle` LaTeX conversion of [`../draft.md`](../draft.md), the authoritative manuscript
text for the *Journal of Systems and Software* submission (Special Issue VSI:AI4MSS). This folder —
not the Markdown — is what gets zipped for Editorial Manager.

## Layout

```
latex/
├── manuscript.tex       — main file: preamble, frontmatter, abstract, \input of every section
├── sections/            — one .tex per draft.md section (mirrors its numbering), + declarations.tex
├── refs.bib             — 53 references + [Anon-A], transcribed from draft.md's numbered list
├── title_page.tex        — SEPARATE, non-anonymous title page (placeholders — see below)
├── highlights.tex        — SEPARATE file, 5 bullets ≤85 chars (Elsevier requires "highlights" in the name)
├── figures/              — Figure_1.pdf .. Figure_6.pdf (+ .png @300dpi); figures/src/ has the two
│                           graphviz .dot sources
├── vendor/               — elsarticle.cls + the handful of .sty/.bst files this machine's TeX Live
│                           didn't ship (see "Toolchain" below) — self-contained, no sudo needed
└── Makefile
```

## Build

```bash
make            # pdflatex -> bibtex -> pdflatex x2 -> manuscript.pdf
make figures    # regenerate all 6 figures (delegates to reproduce/Makefile jss-figures)
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

## Figure numbering — read before touching any figure

**draft.md's own figure labels are not in physical reading order**, and this is not a bug introduced
during conversion — it is a pre-existing property of the draft. Its "Figure 6" (the attention-subgraph
case study, §5.2) physically appears *before* its "Figure 3" (pooled-vs-per-type correlation, §5.5).
LaTeX numbers floats by reading order regardless of what the source calls them, so the *printed*
number in the compiled PDF differs from draft.md's own caption text for four of the six figures:

| Printed as | File | Content | draft.md's own label | Source section |
|:---:|---|---|:---:|---|
| Fig. 1 | `Figure_1.pdf` | end-to-end SaG pipeline | Figure 1 | §1.4 |
| Fig. 2 | `Figure_2.pdf` | running example: structural graph + `DEPENDS_ON` | Figure 2 | §3.6 |
| Fig. 3 | `Figure_3.pdf` | HGT attention-weight case study | **Figure 6** | §5.2 |
| Fig. 4 | `Figure_4.pdf` | pooled vs. per-node-type Spearman ρ | **Figure 3** | §5.5 |
| Fig. 5 | `Figure_5.pdf` | AHP shrinkage sensitivity | **Figure 4** | §8.3 |
| Fig. 6 | `Figure_6.pdf` | propagation-threshold sensitivity | **Figure 5** | §8.3 |

Every in-text cross-reference (`Table~\ref{tab:N}`, `Fig.~\ref{fig:N}`, `Section~\ref{sec:N}`) uses
LaTeX's symbolic `\label`/`\ref`, keyed to draft.md's own numbers (e.g. `\label{fig:6}` on the
attention-subgraph float) — so every reference in the compiled text prints the *correct* number
automatically and consistently, regardless of this mismatch. The only place the mismatch is visible is
the **filename**, which is deliberately kept at the *printed* number (matching the Guide for Authors'
"number images according to the order they appear" rule) rather than draft.md's authored number. Do
not rename figure files to match draft.md's captions — that would break the printed/filename match the
Guide asks for.

The two generator scripts affected (`reproduce/render_pooled_vs_pertype_figure.py`,
`reproduce/render_shrinkage_figure.py`, `reproduce/render_threshold_figure.py`) each carry this same
note; regenerating them writes to the correct filename automatically.

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
figure-generation scripts under `reproduce/` (`render_pooled_vs_pertype_figure.py`,
`render_shrinkage_figure.py`, `render_threshold_figure.py`, and the two `figures/src/*.dot` sources)
are the reusable, re-runnable part.

## Verification performed

- Clean build (`make distclean && make`): 75 pages, zero undefined references, zero undefined
  citations, across a full `pdflatex → bibtex → pdflatex → pdflatex` cycle.
- All 23 tables and all 6 figures are cited in the text and numbered with no gaps.
- Abstract: 246 words (≤250 required). Highlights: 5 bullets, longest 83 chars (≤85 required).
- Headline numbers (Tables 18, 21, 22, 23) spot-checked character-for-character against `draft.md`.
- No author-identifying strings outside `title_page.tex`.
