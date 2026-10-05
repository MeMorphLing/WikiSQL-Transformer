#!/usr/bin/env bash
# Appendix A.2: download WikiSQL next to starter/ and link it for the starter scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d WikiSQL ]; then
  git clone --depth 1 https://github.com/salesforce/WikiSQL
fi
if [ ! -f WikiSQL/data/dev.jsonl ]; then
  (cd WikiSQL && tar xvjf data.tar.bz2)
fi
cd starter
[ -e WikiSQL ] || ln -s ../WikiSQL WikiSQL
echo "WikiSQL ready."
