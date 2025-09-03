all: instructions.pdf

%.pdf: %.md
	pandoc -f markdown -t latex $< -o $@ -V papersize:a4 -V fontsize=12pt  -V geometry:margin=1in
