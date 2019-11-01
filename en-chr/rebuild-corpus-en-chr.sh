#!/bin/bash -v

set -e
set -o pipefail

cd "$(dirname "$0")"
cwd="$(pwd)"

MARIAN=/git/marian

export PATH="$PATH:$MARIAN/build/"

if [ ! -e $MARIAN/build/marian ]
then
    echo "marian is not installed in $MARIAN/build, you need to compile the toolkit first"
    exit 1
fi

L1="en"
L2="chr"

WORKDIR="/data/work.$L1-$L2"
MODELDIR="/data/model.$L1-$L2"
TEMPDIR="/data/temp.$L1-$L2"
CORPUS_SRC="/data/corpus.src"
CORPUS_SRC_DEV="/data/corpus.src.dev"
CORPUS="$WORKDIR/corpus"
DEVCORPUS="$WORKDIR/dev-corpus"
TESTCORPUS="$WORKDIR/test-corpus"
MONODIR="/data/mono.src"

if [ ! -d "$CORPUS_SRC" ]; then echo "MISSING $CORPUS_SRC"; exit -1; fi

cd "$WORKDIR"

#the generated corpus is supplied as a tsv, split out for use in the corpus combiner
cut -f 1 "$CORPUS_SRC/generated-corpus.$L2-$L1.tsv" > "$CORPUS_SRC/generated-corpus.$L2"
cut -f 2 "$CORPUS_SRC/generated-corpus.$L2-$L1.tsv" > "$CORPUS_SRC/generated-corpus.$L1"

cp /dev/null "$CORPUS".$L1
cp /dev/null "$CORPUS".$L2

for x in "$CORPUS_SRC/"*.$L1; do
    y="${x/.$L1/.$L2}"
    if [ ! -f "$y" ]; then echo "MISSING $y"; exit -1; fi
    cat "$x" | sed '/^\s*$/d' >> "$CORPUS".$L1
    cat "$y" | sed '/^\s*$/d' >> "$CORPUS".$L2
done


paste "$CORPUS".$L1 "$CORPUS".$L2 \
    | cat -n \
    | LC_ALL=C.UTF-8 sort -k2 -k1n \
    | LC_ALL=C.UTF-8 uniq -f1 \
    | LC_ALL=C.UTF-8 sort -nk1,1 \
    | cut -f2- \
    > "$CORPUS".$L1-$L2.tsv

cut -f 1 "$CORPUS".$L1-$L2.tsv > "$CORPUS".$L1
cut -f 2 "$CORPUS".$L1-$L2.tsv > "$CORPUS".$L2

# create a dev set with data not in training corpus
cp /dev/null "$DEVCORPUS.$L1-$L2"
cp /dev/null "$DEVCORPUS.$L1"
cp /dev/null "$DEVCORPUS.$L2"

cut -f 2 "$CORPUS_SRC_DEV/dev-set.$L2-$L1.tsv" >> "$DEVCORPUS.$L1"
cut -f 1 "$CORPUS_SRC_DEV/dev-set.$L2-$L1.tsv" >> "$DEVCORPUS.$L2"

tail -n +600 "$CORPUS_SRC_DEV/corpus-exodus.$L1-$L2.tsv" > /tmp/temp3.txt

cut -f 1 /tmp/temp3.txt >> "$DEVCORPUS.$L1"
cut -f 2 /tmp/temp3.txt >> "$DEVCORPUS.$L2"

paste "$DEVCORPUS.$L1" "$DEVCORPUS.$L2" > "$DEVCORPUS.$L1-$L2"

# create a test set with data not in training corpus
cp /dev/null "$TESTCORPUS.$L1-$L2"
cp /dev/null "$TESTCORPUS.$L1"
cp /dev/null "$TESTCORPUS.$L2"

cut -f 2 "$CORPUS_SRC_DEV/dev-set.$L2-$L1.tsv" >> "$TESTCORPUS.$L1"
cut -f 1 "$CORPUS_SRC_DEV/dev-set.$L2-$L1.tsv" >> "$TESTCORPUS.$L2"

head -n 601 "$CORPUS_SRC_DEV/corpus-exodus.$L1-$L2.tsv" > /tmp/temp3.txt

cut -f 1 /tmp/temp3.txt >> "$TESTCORPUS.$L1"
cut -f 2 /tmp/temp3.txt >> "$TESTCORPUS.$L2"

paste "$TESTCORPUS.$L1" "$TESTCORPUS.$L2" > "$TESTCORPUS.$L1-$L2"

rm /tmp/temp.txt 2> /dev/null || true
rm /tmp/temp2.txt 2> /dev/null || true
rm /tmp/temp3.txt 2> /dev/null || true
