buildsystem.pdf : buildsystem.tex \
						buildsystem.aux buildsystem.bbl \
						date_downloaded.tex \
						top_artists.csv playcount.pdf 
	pdflatex buildsystem.tex
	pdflatex buildsystem.tex
buildsystem.aux buildsystem.bbl : buildsystem.tex
	bibtex buildsystem
date_downloaded.tex top_artists.json : 
	./download_charts.sh
playcount.pdf : top_artists.json
	./parse_charts.py playcount.pdf
top_artists.csv : top_artists.json
	./parse_charts.py top_artists.csv
