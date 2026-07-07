#!/usr/bin/env bash
# Render the forward-model architecture figure (forward_model_V3.tex) to PNG+PDF
# for the docs site. The figure was authored with \documentclass{standalone};
# on TeX installs that lack the `standalone` class (e.g. TeX Live 2020 on
# Snellius) we wrap it in a plain `article` page (wrapper.tex) and rasterise via
# ghostscript. Requires: pdflatex, gs, python3. Re-run after editing the figure.
set -euo pipefail
cd "$(dirname "$0")/forward_model"

# Extract the figure body (verbatim, between \begin{document} and \end{document}).
python3 - <<'PY'
src = open("forward_model_V3.tex").read().splitlines()
def find(m):
    for i, l in enumerate(src):
        if l.strip() == m:
            return i
    raise SystemExit(f"marker {m!r} not found")
a = find(r"\begin{document}")
b = find(r"\end{document}")
open("body.tex", "w").write("\n".join(src[a+1:b]) + "\n")
PY

pdflatex -interaction=nonstopmode -halt-on-error wrapper.tex
cp wrapper.pdf forward_model_V3.pdf
gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r200 \
   -dGraphicsAlphaBits=4 -dTextAlphaBits=4 \
   -sOutputFile=forward_model_V3.png wrapper.pdf
echo "Wrote forward_model_V3.{pdf,png}"
