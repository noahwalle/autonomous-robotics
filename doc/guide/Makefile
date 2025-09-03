all: guide.pdf

guide.pdf: guide.tex

%.pdf: %.tex
	pdflatex -shell-escape $*
	pdflatex -shell-escape $*

%.pgf: %.py
	cd $(dir $@); python3 $(notdir $<)

clean:
	-rm *.aux *.log
