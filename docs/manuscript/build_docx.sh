#!/bin/bash
# Build main.docx from main.tex via pandoc (PNG figures, numeric Nature-style citations).
# Requires pandoc (>=3). Run from docs/manuscript/.  PDF is built separately with `tectonic main.tex`.
set -e
cd "$(dirname "$0")"
[ -f nature.csl ] || curl -sL https://raw.githubusercontent.com/citation-style-language/styles/master/nature.csl -o nature.csl
# docx build copy: PNG figures (Word can't embed PDF) + \citealp -> \citep (pandoc compat)
sed -e 's#\(figs/fig[0-9]\)\.pdf#\1.png#g' -e 's/\\citealp/\\citep/g' main.tex > main_docx.tex
pandoc main_docx.tex --citeproc --bibliography=refs.bib --csl=nature.csl \
  -M title="NullState: a reference-anchored framework for detecting, diagnosing, and quantifying the off-target compartment of human neural organoids" \
  -M author="Luiz Takahashi" \
  -M subtitle="Voiland College of Engineering and Architecture, Washington State University, Pullman, WA, USA. Correspondence: l.takahashidosreis@wsu.edu" \
  -o main.docx
echo "wrote main.docx"
