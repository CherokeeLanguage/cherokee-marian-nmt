#!/bin/bash -v

set -e
set -o pipefail

#keep this memory consuming monster from crashing my workstation with the swap of death
#ulimit -Sv 20480

cd "$(dirname "$0")"
cwd="$(pwd)"

MARIAN=/git/marian

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

# create work directory
if [ ! -d "$WORKDIR" ]; then echo "UNABLE TO RESUME - NO DATA!"; exit -1; fi
if [ ! -f "$DEVCORPUS.$L1" ]; then echo "UNABLE TO RESUME - NO DATA!"; exit -1; fi
if [ ! -f "$TESTCORPUS.$L1" ]; then echo "UNABLE TO RESUME - NO DATA!"; exit -1; fi
if [ ! -d "$MODELDIR" ]; then echo "UNABLE TO RESUME - NO MODEL!"; exit -1; fi

cd "$WORKDIR"

ALIGNEDCORPUS="$WORKDIR"/corpus.align.$L1-$L2

for e in $(seq 1 1 1000); do
    sed -i "/^version.*$/d" "$MODELDIR/model.npz.yml"
    # train nmt model
nice $MARIAN/build/marian \
    --after-epochs $e \
    --mini-batch 16 \
    --maxi-batch 64 \
    --cpu-threads 16 \
    --allow-unk \
    --no-restore-corpus \
    -w 1024 \
    --type s2s \
    --model "$MODELDIR"/model.npz \
    --dim-vocabs 2000 8000 \
    --train-sets "$CORPUS.$L1" "$CORPUS.$L2" \
    --vocabs "$MODELDIR"/vocab.$L1.spm "$MODELDIR"/vocab.$L2.spm \
    --sentencepiece-options "--hard_vocab_limit=false --character_coverage=1.0" \
    --layer-normalization \
    --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
    --early-stopping 5 --max-length 100 \
    --valid-freq 1000 --save-freq 1000 --disp-freq 1 \
    --cost-type ce-mean-words --valid-metrics ce-mean-words bleu-detok \
    --valid-sets "$DEVCORPUS.$L1" "$DEVCORPUS.$L2"  \
    --log "$TEMPDIR"/train.log --valid-log "$TEMPDIR"/validation.log --tempdir "$TEMPDIR" \
    --keep-best \
    --seed 1111 --exponential-smoothing \
    --normalize=0.6 --beam-size=6 --quiet-translation \
    --guided-alignment "$ALIGNEDCORPUS" \
    --valid-translation-output "$TEMPDIR/validation-translation-output.txt"

    # some simple tests
    echo "A man and a woman are walking." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/man-woman.$L2".output

    echo "The men and women are walking." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/men-women.$L2".output

    echo "The fox will be eating the chicken tomorrow." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/fox-chicken-will-be-eating.$L2".output

    echo "The fox will eat the chicken." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/fox-chicken-will-eat.$L2".output

    echo "The fox is eating the chicken." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/fox-chicken-eating.$L2".output

    echo "The fox ate the chicken." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/fox-chicken-ate.$L2".output

    echo "The fox did eat the chicken." \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/fox-chicken-did-eat.$L2".output      

done

 # translate dev set
cat "$DEVCORPUS.$L1" \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$DEVCORPUS.$L2".output

    # translate test set
cat "$TESTCORPUS.$L1" \
    | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$TESTCORPUS.$L2".output

