# docs/research/jss/latex/Makefile
# ==================================
# Build the JSS manuscript PDF from the vendored, no-sudo TeX toolchain in
# vendor/. See README.md for what's in this folder and what's still a
# placeholder pending de-anonymization.
#
# Usage:
#   make            — full build (pdflatex -> bibtex -> pdflatex x2)
#   make figures    — regenerate all 5 figures, 3 body + 2 supplementary
#                     (delegates to reproduce/Makefile jss-figures)
#   make flat       — single-file manuscript_flat.tex, if a submission portal
#                     rejects \input
#   make clean      — remove build artifacts (keeps the PDF)
#   make distclean  — remove the PDF too
#   make zip        — submission_package.zip: everything Editorial Manager needs

PDFLATEX   ?= pdflatex
BIBTEX     ?= bibtex
MAIN       := manuscript
TEXINPUTS  := ./vendor:
BSTINPUTS  := ./vendor:
export TEXINPUTS BSTINPUTS

.PHONY: all figures flat clean distclean zip supplement

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex refs.bib figures/Figure_1.pdf figures/Figure_2.pdf \
             figures/Figure_3.pdf
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "✓ $(MAIN).pdf built."

figures:
	$(MAKE) -C ../../../.. -f reproduce/Makefile jss-figures

flat: $(MAIN).pdf
	@echo "Flattening $(MAIN).tex and all \\input{} targets into $(MAIN)_flat.tex ..."
	@python3 -c "\
import re, pathlib; \
src = pathlib.Path('$(MAIN).tex').read_text(); \
out = re.sub(r'\\\\input\{([^}]+\.tex)\}', lambda m: pathlib.Path(m.group(1)).read_text(), src); \
pathlib.Path('$(MAIN)_flat.tex').write_text(out)"
	@echo "✓ $(MAIN)_flat.tex written."

supplement: supplementary.pdf

clean:
	rm -f supplementary.aux supplementary.bbl supplementary.blg supplementary.log supplementary.out supplementary.spl \
	      $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).spl \
	      $(MAIN).synctex.gz $(MAIN).fls $(MAIN).fdb_latexmk missfont.log

distclean: clean
	rm -f $(MAIN).pdf $(MAIN)_flat.tex

zip: $(MAIN).pdf supplementary.pdf
	rm -f submission_package.zip
	zip -r submission_package.zip \
	    $(MAIN).tex sections/ refs.bib vendor/ figures/*.pdf figures/*.png \
	    title_page.tex highlights.tex vitae.tex $(MAIN).bbl \
	    supplementary.tex supplementary.bbl supplementary.pdf \
	    LENGTH_JUSTIFICATION.md -x '*.dot'
	@echo "✓ submission_package.zip ready (manuscript + supplement + title page + highlights + vitae)."

supplementary.pdf: supplementary.tex refs.bib figures/Figure_S1.pdf figures/Figure_S2.pdf
	$(PDFLATEX) -interaction=nonstopmode supplementary
	$(BIBTEX) supplementary
	$(PDFLATEX) -interaction=nonstopmode supplementary
	$(PDFLATEX) -interaction=nonstopmode supplementary
