SCRIPT_TARGETS = playcount.pdf top_artists.csv
DOWNLOAD_TARGETS = date_downloaded.tex top_artists.json

buildsystem.pdf : buildsystem.tex literature.bib \
						$(DOWNLOAD_TARGETS) \
						$(SCRIPT_TARGETS)
	latexmk -pdf buildsystem.tex
$(DOWNLOAD_TARGETS) : 
	./download_charts.sh
$(SCRIPT_TARGETS): $(DOWNLOAD_TARGETS)
	./parse_charts.py $@

.PHONY : clean extraclean
clean : 
	rm -f *.pdf *.log *.blg *.bbl *.aux
	latexmk -c
extraclean : clean
	rm -f buildsystem.pdf $(DOWNLOAD_TARGETS) $(SCRIPT_TARGETS)
	latexmk -C
