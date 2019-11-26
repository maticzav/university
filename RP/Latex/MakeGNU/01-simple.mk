buildsystem.pdf : buildsystem.tex \
						buildsystem.aux buildsystem.bbl \
						date_downloaded.tex \
						top_artists.csv playcount.pdf 
	pdflatex buildsystem.tex
	pdflatex buildsystem.tex
buildsystem.aux : buildsystem.tex playcount.pdf top_artists.csv
	pdflatex buildsystem.tex
buildsystem.bbl : buildsystem.tex literature.bib
	bibtex buildsystem
date_downloaded.tex top_artists.json : 
	./download_charts.sh
playcount.pdf : top_artists.json
	./parse_charts.py playcount.pdf
top_artists.csv : top_artists.json
	./parse_charts.py top_artists.csv

clean : 
	rm *.pdf *.log *.blg *.bbl *.aux
extraclean : clean
	rm buildsystem.pdf top_artists.json date_downloaded.tex playcount.pdf top_artists.csv
