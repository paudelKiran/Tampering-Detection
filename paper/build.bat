@echo off
rem Build script for Windows (uses latexmk if available, falls back to pdflatex)
set FILE=rpaper.tex

where latexmk >nul 2>&1
if %errorlevel%==0 (
    latexmk -pdf -interaction=nonstopmode %FILE%
    goto :eof
)

where pdflatex >nul 2>&1
if %errorlevel%==0 (
    pdflatex -interaction=nonstopmode -halt-on-error %FILE%
    pdflatex -interaction=nonstopmode -halt-on-error %FILE%
    goto :eof
)

echo No LaTeX engine found. Install MiKTeX or TeX Live and ensure pdflatex or latexmk is on PATH.
exit /b 1
