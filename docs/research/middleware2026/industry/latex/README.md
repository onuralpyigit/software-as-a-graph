# Middleware 2026 Industrial Track — LaTeX build

Converted from [`../draft.md`](../draft.md). Single-file manuscript (`manuscript.tex`), ACM
`sigconf` format, 9pt, single-blind (author names/affiliation included, not anonymized) — see the
CFP requirements this satisfies in `../draft.md`'s history and `../outline.md`.

## Build

```bash
make            # pdflatex -> bibtex -> pdflatex x2 -> manuscript.pdf
make pages      # print compiled page count
make clean      # remove build artifacts, keep the PDF
```

## Dependencies

`vendor/acmart.cls` (+ `ACM-Reference-Format.bst`, `acm*.bbx/.cbx`, `acmdatamodel.dbx`) is vendored
already — built from the official CTAN `acmart.dtx`/`.ins` source via `tex acmart.ins`, since no
package manager here can pull `acmart.cls` itself without `sudo`.

**`acmart.cls` in turn requires ~20 further LaTeX packages** (`booktabs`, `xcolor`, `microtype`,
`unicode-math`, `libertine`/`zi4` fonts, `environ`, `manyfoot`, `xstring`, …) plus `tikz`/`pgf` for
the two figures and `enumitem` for compact lists. These are now installed on this machine via:

```bash
sudo apt install texlive-publishers texlive-latex-extra texlive-fonts-extra \
    texlive-latex-recommended texlive-fonts-recommended texlive-pictures
```

(If setting this up fresh elsewhere: hand-vendoring 20+ interdependent CTAN packages, some with
binary font files, is impractical next to the one `apt` command above, which resolves the whole
dependency tree correctly in one shot.)

**Verified compiling as of this revision.** `make` builds cleanly; body content (§1 through §7,
including "Future work") ends on page 6, and only the reference list itself spills onto page 7 —
confirmed via `pdftotext` in natural column order, not just total page count. Since references are
excluded from the CFP's 6-page limit, this satisfies it. Re-run `make pages` after any further
edits to confirm this still holds; the margin is thin (References starts partway down page 6's
second column), so a modest amount of added content could push it back over.

A handful of cosmetic `Overfull \hbox` warnings remain (long `\texttt{}` tokens and one figure
caption not breaking ideally in the narrow single-column width, up to ~46pt over) — `\sloppy` is
already enabled globally to mitigate this; they don't affect the page count and are not currently
visually severe, but a final pass before submission should eyeball the PDF for any that look bad.

## Open items before submission

- **`\author{}`/`\affiliation{}` are placeholders** (`TODO(author)` etc. in `manuscript.tex`).
  Single-blind review requires real names — fill these in. `refs.bib`'s `companion2026` entry uses
  the identical placeholder for the same author(s); fill both together.
- **`tao2019` in `refs.bib`** (backs the "Digital twins" paragraph's Industry-4.0 distinction) was
  recalled from general knowledge, not looked up live this session — verify volume/issue/page
  numbers against the published IEEE Transactions on Industrial Informatics version before
  submission.
- **References: 9 → ~18.** Tracked since the Markdown draft (see `../outline.md`, "Remaining work"
  item 3). The current 9 resolve every citation the prose actually makes; expanding further is
  additional related-work coverage, not a conversion blocker.
- **Figure 1 (architecture overview) is new** — the Markdown draft only had a "Figure 2" (CI/CD
  sequence) with no "Figure 1," a leftover from an earlier revision that dropped an architecture
  diagram without renumbering. This TikZ recreation is original to the LaTeX conversion, not a
  1:1 port of prior content — it renders correctly (verified in the compiled PDF), but is worth a
  final look for polish.
- **CCS Concepts (`\ccsdesc`) use real ACM 2012 category paths but omit the `\begin{CCSXML}`
  metadata block** (exact numeric `concept_id` values) — see the `TODO(refs)` comment above them in
  `manuscript.tex`. Fill in the real IDs from https://dl.acm.org/ccs before submission; the visible
  "CCS Concepts" section already renders correctly without them.
