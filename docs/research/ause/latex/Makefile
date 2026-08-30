all: manuscript.pdf

manuscript.pdf: manuscript.tex refs.bib $(wildcard sections/*.tex)
	pdflatex -interaction=nonstopmode manuscript.tex
	bibtex manuscript
	pdflatex -interaction=nonstopmode manuscript.tex
	pdflatex -interaction=nonstopmode manuscript.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.spl *.fls *.fdb_latexmk sections/*.aux
