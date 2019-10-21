#!/bin/bash

set -e
set -o pipefail

cd "$(dirname "$0")"

cd "$(dirname "$0")"
cp /dev/null chr/corpus-immersion.chr.tmp

find chr/ -iname '*.pdf' | while read pdf; do
    echo "$pdf"
    pdftotext -enc UTF-8 -eol unix "$pdf" - \
    | perl -C -lpe 's/[\N{U+00}-\N{U+09}\N{U+0B}-\N{U+1F}]//g' \
    | perl -C -lpe 's/[^\s\w\N{U+13A0}-\N{U+13FF}\p{Punctuation}]/ /g' \
    | perl -C -lpe 's/ +/ /g' \
    | perl -C -lpe 's/^[\d\s.,:]+//g' \
    | perl -C -lpe 's/^\(.\)//g' \
    | perl -C -lne 'print if /^((?![A-Za-z]).)*$/' \
    | perl -C -lne 'print if /[\N{U+13A0}-\N{U+13FF}]{2,}\s+[\N{U+13A0}-\N{U+13FF}]{2,}\s+[\N{U+13A0}-\N{U+13FF}]{2,}/' \
    | (grep -v -e '^[[:space:]]*$' || true) \
    | (grep -v '___' || true) >> chr/corpus-immersion.chr.tmp
done

LC_ALL="C" sort -u chr/corpus-immersion.chr.tmp | shuf > chr/corpus-immersion.chr
rm chr/corpus-immersion.chr.tmp

