all: report-template.pdf report-template.docx

%.docx: %.tex
	-pandoc -t docx -N -o $@ $<

%.pdf: %.tex
	pdflatex $*
	pdflatex $*

clean:
	-rm *.aux *.log
