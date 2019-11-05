SCRIPT_TARGETS = playcount.pdf top_artists.csv
DOWNLOAD_TARGETS = date_downloaded.tex top_artists.json

PDF_LATEX = pdflatex $(PDF_LATEX_ARGS)
PDF_LATEX_ARGS = -synctex=1

buildsystem.pdf : buildsystem.tex \
						buildsystem.aux buildsystem.bbl \
						date_downloaded.tex \
						$(SCRIPT_TARGETS)
	$(PDF_LATEX) buildsystem.tex
	$(PDF_LATEX) buildsystem.tex
buildsystem.aux : buildsystem.tex $(SCRIPT_TARGETS)
	$(PDF_LATEX) buildsystem.tex
buildsystem.bbl : buildsystem.tex literature.bib
	bibtex buildsystem
$(DOWNLOAD_TARGETS) : 
	./download_charts.sh
$(SCRIPT_TARGETS): top_artists.json
	./parse_charts.py $@

.PHONY : clean extraclean
clean : 
	rm -f *.pdf *.log *.blg *.bbl *.aux
extraclean : clean
	rm -f buildsystem.pdf $(DOWNLOAD_TARGETS) $(SCRIPT_TARGETS)
