#!/bin/bash -v

set -e
set -o pipefail

#keep this memory consuming monster from crashing my workstation with the swap of death
ulimit -Sv $((32 * 1024 * 1024))

cd "$(dirname "$0")"
cwd="$(pwd)"

MARIAN=/git/marian

if [ ! -e $MARIAN/build/marian ]
then
    echo "marian is not installed in $MARIAN/build, you need to compile the toolkit first"
    exit 1
fi

DOALIGN=0

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
L1COUNT=32768
L2COUNT=32768

# create work directory
if [ ! -d "$WORKDIR" ]; then echo "UNABLE TO RESUME - NO DATA!"; exit -1; fi
if [ ! -f "$DEVCORPUS.$L1" ]; then echo "UNABLE TO RESUME - NO DATA!"; exit -1; fi
if [ ! -f "$TESTCORPUS.$L1" ]; then echo "UNABLE TO RESUME - NO DATA!"; exit -1; fi
if [ ! -d "$MODELDIR" ]; then echo "UNABLE TO RESUME - NO MODEL!"; exit -1; fi

cd "$WORKDIR"

ALIGNEDCORPUS="$WORKDIR"/corpus.align.$L1-$L2

# calculate max word counts
function getMaxWordCount {
    cat $CORPUS.$1 | /git/marian/build/spm_encode --model "$MODELDIR/vocab.$1.spm" | wc -w -l | while read words lines; do
        echo "$(($lines / $words))"
        return
    done
}

l1Maxwords=$(getMaxWordCount $L1)
l2Maxwords=$(getMaxWordCount $L2)
maxwords=$((($l1Maxwords+$l2Maxwords+2)*16))

echo "$L1: $l1Maxwords maxwords, $L2: $l2Maxwords maxwords"
echo "Mini batch maxwords: $maxwords"

ALIGNEDCORPUS="$WORKDIR"/corpus.align.$L1-$L2
if [ $DOALIGN != 0 ]; then    
	ALIGN_ARGS="--guided-alignment \"$ALIGNEDCORPUS\""
else
	ALIGN_ARGS=""
fi

#get previous epoch value
eStart="$(grep 'after-epochs:' "$MODELDIR/model.npz.yml" | cut -f 2 -d ' ')"
eStart=$(($eStart + 1))

echo "Starting at epoch $eStart"

#start at the previous "after-epoch value" for a quicker start up
#then resuming from an interrupted training

#for e in $(seq $eStart 5 10000); do
    sed -i "/^version.*$/d" "$MODELDIR/model.npz.yml"
    # train nmt model
    # --after-epochs $e \
nice $MARIAN/build/marian \
    --after-epochs 100000 \
    --mini-batch-fit \
    --devices 0 \
    --no-restore-corpus \
    -w 4096 \
    --type s2s \
    --model "$MODELDIR"/model.npz \
    --dim-vocabs $L1COUNT $L2COUNT \
    --train-sets "$CORPUS.$L1" "$CORPUS.$L2" \
    --vocabs "$MODELDIR"/vocab.$L1.spm "$MODELDIR"/vocab.$L2.spm \
    --sentencepiece-options "--character_coverage=1.0" \
    --layer-normalization --tied-embeddings-all \
    --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
    --early-stopping 5 --max-length 512 \
    --valid-freq 1000 --save-freq 5000 --disp-freq 5 \
    --cost-type ce-mean-words --valid-metrics ce-mean-words bleu-detok \
    --valid-sets "$DEVCORPUS.$L1" "$DEVCORPUS.$L2"  \
    --log "$TEMPDIR"/train.log --valid-log "$TEMPDIR"/validation.log --tempdir "$TEMPDIR" \
    --keep-best \
    --seed 1111 --exponential-smoothing \
    --normalize=0.6 --beam-size=6 --quiet-translation \
    $ALIGN_ARGS \
    --valid-translation-output "$TEMPDIR/validation-translation-output.txt" \
    --lr-report --overwrite

    # some simple tests
    (
        echo "Hello."
        echo "Hello?"
        echo "Hello!"
        echo "Hello again."
        echo "Hello it's me again."
        echo "A man and a woman are walking."
        echo "The men and women are walking."
        echo "The fox will be eating the chicken tomorrow."
        echo "The fox will eat the chicken."
        echo "The fox is eating the chicken."
        echo "The fox ate the chicken."  
        echo "The fox did eat the chicken."        
        echo "The fox ate the chicken and thought it tasted good."
        echo "The fox ate the chicken and thought it tasted very good."
        echo "The fox ate the chicken and thought it tasted fantastic."
    ) > /tmp/simple-test.en
    cat /tmp/simple-test.en | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml \
        --devices 0 -b 6 -n0.6 > /tmp/simple-test.chr
    paste /tmp/simple-test.en /tmp/simple-test.chr > "$WORKDIR/_test-simple.$L1-$L2".tsv
    cat "$TESTCORPUS.$L1" | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml \
        --devices 0 -b 6 -n0.6 > /tmp/test-corpus-output.chr
    paste "$TESTCORPUS.$L1" /tmp/test-corpus-output.chr "$TESTCORPUS.$L2" > "$WORKDIR/_test-corpus-output.$L1-$L2-$L2".tsv
#done

 
