#!/usr/bin/env bash
# wordcount.sh — Compute MPhil submission word count.
#
# Oxford spec (Tom's Inspera note): "The word count applies to the entirety
# of the thesis including references, etc." Therefore this script counts
# main text, footnotes, figure notes (\floatfoot), captions, headers, and
# the bibliography. Math is counted as one word per equation (texcount default).
#
# How it works:
#   1. texcount -sum on the .tex source counts everything texcount sees:
#      text + headers + captions + math + float content + footnote/floatfoot
#      contents (since no macros are marked [ignore]).
#   2. The bibliography is BibTeX-rendered into a .bbl file and is therefore
#      not visible to texcount when it parses the .tex source. The script
#      detects a sibling .bbl file (same basename, parent or current folder)
#      and adds its word count to the total.
#   3. If a sibling .pdf exists, the script also reports a pdftotext-based
#      cross-check count. This is the most faithful "what is printed" number.
#
# Usage:
#   ./wordcount.sh                              # counts M.Phil_Submission_Version.tex
#   ./wordcount.sh path/to/other.tex            # counts a different file

set -euo pipefail

cd "$(dirname "$0")"

FILE="${1:-M.Phil Submission Version.tex}"

if [ ! -f "$FILE" ]; then
  echo "Error: file '$FILE' not found in $(pwd)." >&2
  exit 1
fi

if ! command -v texcount >/dev/null 2>&1; then
  echo "Error: texcount not found in PATH. Install via TeX Live or MacTeX." >&2
  exit 1
fi

# texcount resolves \input{} relative to the .tex file's folder. The current
# draft references appendix_tables and appendix_figures from the parent
# model/ folder, so we add -dir=.. as a search path. Once the user finalises
# a self-contained submission tex, the flag is harmless.
DIR_FLAG="-dir=.."

echo "============================================================"
echo "  Word count for: $FILE"
echo "  Counting EVERYTHING per Oxford spec: main text, footnotes,"
echo "  figure notes, captions, headers, math, and bibliography."
echo "============================================================"
echo

# 1) Source-side count via texcount.
#    -sum            sum all categories (text, headers, captions, math, float).
#    -inc            follow \input{}.
#    -incbib         include the compiled .bbl bibliography in the count.
#    -1              print the bare total on one line.
#    -merge          fold per-included-file subcounts into the parent total.
#    -q              quiet: just the number.
TEXCOUNT_FLAGS="$DIR_FLAG -inc -sum -merge -q"
COUNT_NOBIB=$(texcount $TEXCOUNT_FLAGS -1 "$FILE" 2>/dev/null \
              | head -n1 | tr -dc '0-9')
COUNT_WITHBIB=$(texcount $TEXCOUNT_FLAGS -incbib -1 "$FILE" 2>/dev/null \
                | head -n1 | tr -dc '0-9')
COUNT_NOBIB=${COUNT_NOBIB:-0}
COUNT_WITHBIB=${COUNT_WITHBIB:-0}
INCBIB_DELTA=$((COUNT_WITHBIB - COUNT_NOBIB))

# 2) Bibliography fallback: if -incbib did not add anything (delta == 0) and a
#    sibling .bbl exists, count its words via detex. This handles the case
#    where the .tex and .bbl live in different folders, which defeats
#    texcount's auto-detection of the .bbl.
BASENAME="${FILE%.tex}"
BBL_FILE=""
for candidate in "${BASENAME}.bbl" "../${BASENAME}.bbl" "../$(basename "$BASENAME").bbl"; do
  if [ -f "$candidate" ]; then
    BBL_FILE="$candidate"
    break
  fi
done

BBL_FALLBACK=0
if [ "$INCBIB_DELTA" -eq 0 ] && [ -n "$BBL_FILE" ]; then
  if command -v detex >/dev/null 2>&1; then
    BBL_FALLBACK=$(detex "$BBL_FILE" 2>/dev/null | wc -w | tr -d ' ')
  else
    BBL_FALLBACK=$(sed -E 's/\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^}]*\})*//g; s/[{}]//g' \
                   "$BBL_FILE" | wc -w | tr -d ' ')
  fi
  BBL_FALLBACK=${BBL_FALLBACK:-0}
fi

TOTAL=$((COUNT_WITHBIB + BBL_FALLBACK))

echo "Source words (text + footnotes + floatfoot + captions + headers + math) : $COUNT_NOBIB"
if [ "$INCBIB_DELTA" -gt 0 ]; then
  echo "Bibliography (via -incbib, auto-detected .bbl)                          : $INCBIB_DELTA"
elif [ "$BBL_FALLBACK" -gt 0 ]; then
  echo "Bibliography (detex fallback on sibling .bbl)                           : $BBL_FALLBACK   [from $BBL_FILE]"
else
  echo "Bibliography                                                            : 0   [no .bbl found]"
fi
echo "------------------------------------------------------------"
echo "TOTAL submission word count                                             : $TOTAL"
echo "20,000-word limit headroom                                              : $((20000 - TOTAL))"
echo

# 3) Optional cross-check via pdftotext if a sibling PDF exists.
PDF_FILE=""
for candidate in "${BASENAME}.pdf" "../${BASENAME}.pdf" "../$(basename "$BASENAME").pdf"; do
  if [ -f "$candidate" ]; then
    PDF_FILE="$candidate"
    break
  fi
done
if [ -n "$PDF_FILE" ] && command -v pdftotext >/dev/null 2>&1; then
  PDF_COUNT=$(pdftotext "$PDF_FILE" - 2>/dev/null | wc -w | tr -d ' ')
  echo "Cross-check (pdftotext on $PDF_FILE):"
  echo "  Words in rendered PDF              : $PDF_COUNT"
  echo
fi

# 4) Verbose breakdown for sanity checking which sections drive the count.
echo "------------------------------------------------------------"
echo "Per-section breakdown (texcount detail):"
echo "------------------------------------------------------------"
texcount $DIR_FLAG -inc -merge -sub=section "$FILE" || true
