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
CORPUS="$WORKDIR/corpus"
DEVCORPUS="$WORKDIR/dev-corpus"
TESTCORPUS="$WORKDIR/test-corpus"
MONODIR="/data/mono.src"

if [ ! -d "$CORPUS_SRC" ]; then echo "MISSING $CORPUS_SRC"; exit -1; fi

cd "$WORKDIR"

cp /dev/null "$CORPUS".$L1
cp /dev/null "$CORPUS".$L2

for x in "$CORPUS_SRC/"*.$L1; do
    y="${x/.$L1/.$L2}"
    if [ ! -f "$y" ]; then echo "MISSING $y"; exit -1; fi
    cat "$x" | sed '/^\s*$/d' >> "$CORPUS".$L1
    cat "$y" | sed '/^\s*$/d' >> "$CORPUS".$L2
done

# create a random dev set
cp /dev/null "$DEVCORPUS.$L1-$L2"
cp /dev/null "$DEVCORPUS.$L1"
cp /dev/null "$DEVCORPUS.$L2"

paste "$CORPUS_SRC"/corpus-nt-bbe.$L1 "$CORPUS_SRC"/corpus-nt-bbe.$L2 | sort -u | shuf > "$WORKDIR/tmp1.$L1-$L2"
sed -i '/^\s*$/d' "$WORKDIR/tmp1.$L1-$L2"
head -n 50 "$WORKDIR/tmp1.$L1-$L2" > "$DEVCORPUS.$L1-$L2"
rm "$WORKDIR/tmp1.$L1-$L2"

cut -f 1 "$DEVCORPUS.$L1-$L2" > "$DEVCORPUS.$L1" 
cut -f 2 "$DEVCORPUS.$L1-$L2" > "$DEVCORPUS.$L2"

# create a random test set
cp /dev/null "$TESTCORPUS.$L1-$L2"
cp /dev/null "$TESTCORPUS.$L1"
cp /dev/null "$TESTCORPUS.$L2"

paste /data/corpus.src/corpus-genesis-bbe.$L1 /data/corpus.src/corpus-genesis-bbe.$L2 | sort -u | shuf > "$WORKDIR/tmp1.$L1-$L2"
sed -i '/^\s*$/d' "$WORKDIR/tmp1.$L1-$L2"
head -n 50 "$WORKDIR/tmp1.$L1-$L2" > "$TESTCORPUS.$L1-$L2"
rm "$WORKDIR/tmp1.$L1-$L2"

cut -f 1 "$TESTCORPUS.$L1-$L2" > "$TESTCORPUS.$L1" 
cut -f 2 "$TESTCORPUS.$L1-$L2" > "$TESTCORPUS.$L2"

