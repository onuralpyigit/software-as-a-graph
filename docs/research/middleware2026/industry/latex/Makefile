# docs/research/middleware2026/industry/latex/Makefile
# =========================================================
# Build the Middleware 2026 Industrial Track manuscript PDF.
#
# acmart.cls (vendored in vendor/, built from the acmart.dtx/.ins source via
# `tex acmart.ins`) needs about 20 further packages this machine did not have
# preinstalled (booktabs, xcolor, microtype, unicode-math, libertine/zi4
# fonts, tikz, etc.) -- install them once via the system package manager
# rather than hand-vendoring each one from CTAN:
#
#   sudo apt install texlive-publishers texlive-latex-extra texlive-fonts-extra \
#       texlive-latex-recommended texlive-fonts-recommended texlive-pictures
#
# Usage:
#   make            — full build (pdflatex -> bibtex -> pdflatex x2)
#   make pages      — print the compiled page count (excl. references, per
#                     the CFP's 6-page limit) and the body font size in use
#   make clean      — remove build artifacts (keeps the PDF)
#   make distclean  — remove the PDF too
#   make zip        — submission_package.zip: manuscript + refs + vendor

PDFLATEX  ?= pdflatex
BIBTEX    ?= bibtex
MAIN      := manuscript
TEXINPUTS := ./vendor:
BSTINPUTS := ./vendor:
export TEXINPUTS BSTINPUTS

.PHONY: all pages clean distclean zip

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	@echo "✓ $(MAIN).pdf built."

pages: $(MAIN).pdf
	@pdfinfo $(MAIN).pdf | grep -i pages
	@echo "(References section page(s) are excluded from the CFP's 6-page limit --"
	@echo " check the References heading's page number in the PDF against total pages.)"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).spl \
	      $(MAIN).synctex.gz $(MAIN).fls $(MAIN).fdb_latexmk missfont.log $(MAIN).xmpdata

distclean: clean
	rm -f $(MAIN).pdf

zip: $(MAIN).pdf
	rm -f submission_package.zip
	zip -r submission_package.zip \
	    $(MAIN).tex refs.bib vendor/ $(MAIN).bbl
	@echo "✓ submission_package.zip ready."
