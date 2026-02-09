#!/usr/bin/env bash
# Cross-platform build script (Unix-like systems)
FILE="rpaper.tex"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode "$FILE"
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error "$FILE"
  pdflatex -interaction=nonstopmode -halt-on-error "$FILE"
else
  echo "No LaTeX engine found. Install TeX Live or MacTeX and ensure pdflatex or latexmk is on PATH."
  exit 1
fi
