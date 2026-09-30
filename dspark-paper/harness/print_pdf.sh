#!/bin/zsh
# paper.html -> paper.pdf with headless Chrome, using the page's print rules (US Letter, light
# theme). Needs network for the web fonts, which the PDF embeds.
set -eu
cd "$(dirname "$0")/.."
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --no-pdf-header-footer --virtual-time-budget=15000 \
  --print-to-pdf="$PWD/paper.pdf" "file://$PWD/paper.html" 2>/dev/null
ls -l paper.pdf
