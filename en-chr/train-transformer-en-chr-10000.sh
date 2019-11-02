#!/bin/bash -v

set -e
set -o pipefail

cd "$(dirname "$0")"
cwd="$(pwd)"

MARIAN=/git/marian

EPOCHS=10000

export PATH="$PATH:$MARIAN/build/"

if [ ! -e $MARIAN/build/marian ]
then
    echo "marian is not installed in $MARIAN/build, you need to compile the toolkit first"
    exit 1
fi

DOALIGN=1

L1="en"
L2="chr"

WORKDIR="/data/work.$L1-$L2"
MODELDIR="/data/model.$L1-$L2"
TEMPDIR="/data/temp.$L1-$L2"
CORPUS_SRC="/data/corpus.src"
CORPUS_SRC_DEV="/data/corpus.src.dev"
CORPUS="$WORKDIR/corpus"
STARTINGCORPUS="$WORKDIR/starting-corpus"
SYNTHETICCORPUS="$WORKDIR/synthetic-corpus"
DEVCORPUS="$WORKDIR/dev-corpus"
TESTCORPUS="$WORKDIR/test-corpus"
MONODIR="/data/mono.src"
L1COUNT=8192
L2COUNT=8192

GPUS=0

function reformatAsSentences {
    cat "$1" | dos2unix \
    | perl -0 -C -lpe 's/\n([^\n])/ $1/g' \
    | perl -0 -C -lpe 's/\n+\s*/\n/g' \
    | perl -C -lpe 's/\0/\n/g' \
    | perl -C -lpe 's/ +/ /g' \
    | perl -C -lpe 's/([.?!;:])\s+/$1\n/g' \
    | perl -C -lpe 's/_//g'
}

if [ ! -d "$CORPUS_SRC" ]; then echo "MISSING $CORPUS_SRC"; exit -1; fi

# create work directory
if [ -d "$WORKDIR" ]; then rm -rfv "$WORKDIR"; fi
mkdir "$WORKDIR"
# create the model directory
if [ -d "$MODELDIR" ]; then rm -rfv "$MODELDIR"; fi
mkdir "$MODELDIR"
# create the temp directory
if [ -d "$TEMPDIR" ]; then rm -rfv "$TEMPDIR"; fi
mkdir "$TEMPDIR"

cd "$WORKDIR"

cp /dev/null "$CORPUS".$L1
cp /dev/null "$CORPUS".$L2

for x in "$CORPUS_SRC/"*.$L1; do
    y="${x/.$L1/.$L2}"
    if [ ! -f "$y" ]; then echo "MISSING $y"; exit -1; fi
    cat "$x" | sed '/^\s*$/d' >> "$CORPUS".$L1
    cat "$y" | sed '/^\s*$/d' >> "$CORPUS".$L2
done

#(re)generate corpus
bash "$cwd/rebuild-corpus-$L1-$L2.sh"

#save a copy for use as base corpus for reset during switch ups between backwards and forwards training steps
paste "$CORPUS".$L1 "$CORPUS".$L2 > "$STARTINGCORPUS".$L1-$L2.tsv

# pre-train sentencepiece

cd "$MODELDIR"
rm "/data/model.spm/vocab.$L1.spm" 2> /dev/null || true
rm "/data/model.spm/vocab.$L2.spm" 2> /dev/null || true

cp /dev/null "$MONODIR/corpus-sentencepiece.$L2"
for x in "$MONODIR/$L2/"*".$L2"; do
    reformatAsSentences "$x" >> "$MONODIR/corpus-sentencepiece.$L2"
done

sed -i 's/---/ /g' "$MONODIR/corpus-sentencepiece.$L2"
sed -i '/^\s*$/d' "$MONODIR/corpus-sentencepiece.$L2"

$MARIAN/build/spm_train --input="$MONODIR/corpus-sentencepiece.$L2" \
    --max_sentence_length 32768 \
    --model_prefix="/data/model.spm/vocab.$L2.spm" \
    --vocab_size=$L2COUNT --character_coverage=1.0

function wgetIfNeeded {
    if [ -f "$1" ]; then return; fi
    wget -N -O /tmp/wget-temp.$$.txt "$2"
    mv -v "/tmp/wget-temp.$$.txt" "$1"
}

#Additional monolingual corpus for $L1 English
wgetIfNeeded "$MONODIR/$L1/Frankenstien.$L1" 'https://www.gutenberg.org/files/84/84-0.txt'
wgetIfNeeded "$MONODIR/$L1/Pride-and-Prejudice.$L1" 'https://www.gutenberg.org/files/1342/1342-0.txt'
wgetIfNeeded "$MONODIR/$L1/Moby-Dick.$L1" 'https://www.gutenberg.org/files/2701/2701-0.txt'
wgetIfNeeded "$MONODIR/$L1/Dr-Jekyll.$L1" 'https://www.gutenberg.org/files/43/43-0.txt'
wgetIfNeeded "$MONODIR/$L1/Sherlock-Holmes.$L1" 'https://www.gutenberg.org/files/1661/1661-0.txt'
wgetIfNeeded "$MONODIR/$L1/Dracula.$L1" 'https://www.gutenberg.org/ebooks/345.txt.utf-8'
wgetIfNeeded "$MONODIR/$L1/Grimms-Fairy-Tales.$L1" 'https://www.gutenberg.org/files/2591/2591-0.txt'
wgetIfNeeded "$MONODIR/$L1/Jungle-Book.$L1" 'https://www.gutenberg.org/ebooks/35997.txt.utf-8'
wgetIfNeeded "$MONODIR/$L1/Jungle-Book-2.$L1" 'https://www.gutenberg.org/ebooks/37364.txt.utf-8'

cp /dev/null "$MONODIR/corpus-sentencepiece.$L1"
for x in "$MONODIR/$L1/"*".$L1"; do
    reformatAsSentences "$x" >> "$MONODIR/corpus-sentencepiece.$L1"
done

sed -i '/^\s*$/d' "$MONODIR/corpus-sentencepiece.$L1"
sed -i 's/\r$//g' "$MONODIR/corpus-sentencepiece.$L1"

$MARIAN/build/spm_train --input="$MONODIR/corpus-sentencepiece.$L1" \
    --max_sentence_length 32768 \
    --model_prefix="/data/model.spm/vocab.$L1.spm" \
    --vocab_size=$L1COUNT --character_coverage=1.0

mv -v "/data/model.spm/vocab.$L1.spm.model" "/data/model.spm/vocab.$L1.spm"
mv -v "/data/model.spm/vocab.$L2.spm.model" "/data/model.spm/vocab.$L2.spm"

#copy in pretrained models for SPM
cp -v /data/model.spm/vocab.*.spm "$MODELDIR/"

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

# create alignment helper to improve word associations between languages. (In theory.)
# requires pretrained modles for SPM!
function doAlign {
    if [ $DOALIGN != 1 ]; then
        ALIGN_ARGS=""
        ALIGN_ARGS2=""
        return
    fi

   	cat $CORPUS.$L1 | /git/marian/build/spm_encode --model "$MODELDIR/vocab.$L1.spm" > $TEMPDIR/align.temp.$L1
	cat $CORPUS.$L2 | /git/marian/build/spm_encode --model "$MODELDIR/vocab.$L2.spm" > $TEMPDIR/align.temp.$L2
    
    ALIGNEDCORPUS="$WORKDIR"/corpus.align.$L1-$L2
    ALIGN_ARGS="--guided-alignment $ALIGNEDCORPUS"
    paste $TEMPDIR/align.temp.$L1 $TEMPDIR/align.temp.$L2 > $TEMPDIR/align.temp.$L1-$L2
    sed -i 's/\t/ ||| /g' $TEMPDIR/align.temp.$L1-$L2
	/git/fast_align/build/fast_align -vdo -i $TEMPDIR/align.temp.$L1-$L2 > $TEMPDIR/forward.align.$L1-$L2
	/git/fast_align/build/fast_align -vdor -i $TEMPDIR/align.temp.$L1-$L2 > $TEMPDIR/reverse.align.$L1-$L2
	/git/fast_align/build/atools -c grow-diag-final -i $TEMPDIR/forward.align.$L1-$L2 -j $TEMPDIR/reverse.align.$L1-$L2 > "$ALIGNEDCORPUS"
    
    ALIGNEDCORPUS2="$WORKDIR"/corpus.align.$L2-$L1
    ALIGN_ARGS2="--guided-alignment $ALIGNEDCORPUS2"
    paste $TEMPDIR/align.temp.$L2 $TEMPDIR/align.temp.$L1 > $TEMPDIR/align.temp.$L2-$L1
    sed -i 's/\t/ ||| /g' $TEMPDIR/align.temp.$L2-$L1
    /git/fast_align/build/fast_align -vdo -i $TEMPDIR/align.temp.$L2-$L1 > $TEMPDIR/forward.align.$L2-$L1
	/git/fast_align/build/fast_align -vdor -i $TEMPDIR/align.temp.$L2-$L1 > $TEMPDIR/reverse.align.$L2-$L1
	/git/fast_align/build/atools -c grow-diag-final -i $TEMPDIR/forward.align.$L2-$L1 -j $TEMPDIR/reverse.align.$L2-$L1 > "$ALIGNEDCORPUS2"
}

function translateEnToChr {
    cat "$1" | $MARIAN/build/marian-decoder -c "$MODELDIR"/model-$L1-$L2.npz.decoder.yml \
        -w 4096 --devices 0 -b 6 -n0.6
}

function translateChrToEn {
    cat "$1" | $MARIAN/build/marian-decoder -c "$MODELDIR"/model-$L2-$L1.npz.decoder.yml \
        -w 4096 --devices 0 -b 6 -n0.6
}


for loops in $(seq 1 1 5); do

echo "LOOP: $loops" >> "$TEMPDIR"/validation.log

#corpus reset
cut -f 1 "$STARTINGCORPUS".$L1-$L2.tsv > "$CORPUS".$L1
cut -f 2 "$STARTINGCORPUS".$L1-$L2.tsv > "$CORPUS".$L2

if [ -f "$SYNTHETICCORPUS".$L1-$L2.tsv ]; then
    cut -f 1 "$SYNTHETICCORPUS".$L1-$L2.tsv >> "$CORPUS".$L1
    cut -f 2 "$SYNTHETICCORPUS".$L1-$L2.tsv >> "$CORPUS".$L2
fi

#reset model if it exists
rm "$MODELDIR"/model-$L1-$L2.npz 2> /dev/null || true
rm "$MODELDIR"/model-$L1-$L2.npz.yml 2> /dev/null || true

echo "$L1-$L2" >> "$TEMPDIR"/validation.log

doAlign

# train nmt model - forwards
nice $MARIAN/build/marian \
    --model "$MODELDIR"/model-$L1-$L2.npz \
    --type transformer \
    --train-sets "$CORPUS.$L1" "$CORPUS.$L2" \
    --max-length 100 \
    --vocabs "$MODELDIR"/vocab.$L1.spm "$MODELDIR"/vocab.$L2.spm \
    --mini-batch-fit \
    -w 4096 \
    --maxi-batch 1000 \
    --early-stopping 10 \
    --cost-type ce-mean-words \
    --valid-freq 5000 --save-freq 5000 --disp-freq 10 \
    --valid-metrics ce-mean-words perplexity bleu-detok \
    --valid-sets "$DEVCORPUS.$L1" "$DEVCORPUS.$L2"  \
    --beam-size 6 \
    --normalize 0.6 \
    --log "$TEMPDIR"/train.log \
    --valid-log "$TEMPDIR"/validation.log \
    --enc-depth 6 \
    --dec-depth 6 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.1 \
    --label-smoothing 0.1 \
    --learn-rate 0.0003 \
    --lr-warmup 16000 \
    --lr-decay-inv-sqrt 16000 \
    --lr-report \
    --optimizer-params 0.9 0.98 1e-09 \
    --clip-norm 5 \
    --tied-embeddings-all \
    --devices $GPUS \
    --sync-sgd \
    --seed 1111 \
    --exponential-smoothing \
    --overwrite --keep-best

#generate synthetic corpus en->chr
BOOKLIST="Frankenstien Pride-and-Prejudice Moby-Dick Dr-Jekyll Sherlock-Holmes Dracula Grimms-Fairy-Tales Jungle-Book Jungle-Book-2"
#BOOKLIST="Jungle-Book"
cp /dev/null "$SYNTHETICCORPUS".$L1-$L2.tsv
cp /dev/null "$TEMPDIR/combined-books.$L1"
for book in $BOOKLIST; do
    reformatAsSentences "$MONODIR/$L1/${book}.$L1" >> "$TEMPDIR/combined-books.$L1"
done
translateEnToChr "$TEMPDIR/combined-books.$L1" > "$TEMPDIR/combined-books.$L2"
paste "$TEMPDIR/combined-books.$L1" "$TEMPDIR/combined-books.$L2" > "$SYNTHETICCORPUS".$L1-$L2.tsv

#reset corpus
cut -f 1 "$STARTINGCORPUS".$L1-$L2.tsv > "$CORPUS".$L1
cut -f 2 "$STARTINGCORPUS".$L1-$L2.tsv > "$CORPUS".$L2

if [ -f "$SYNTHETICCORPUS".$L1-$L2.tsv ]; then
    cut -f 1 "$SYNTHETICCORPUS".$L1-$L2.tsv >> "$CORPUS".$L1
    cut -f 2 "$SYNTHETICCORPUS".$L1-$L2.tsv >> "$CORPUS".$L2
fi

#reset model if it exists
rm "$MODELDIR"/model-$L2-$L1.npz 2> /dev/null || true
rm "$MODELDIR"/model-$L2-$L1.npz.yml 2> /dev/null || true

echo "$L2-$L1" >> "$TEMPDIR"/validation.log

doAlign

# train nmt model - backwards
nice $MARIAN/build/marian \
    --model "$MODELDIR"/model-$L2-$L1.npz \
    --type transformer \
    --train-sets "$CORPUS.$L2" "$CORPUS.$L1" \
    --max-length 100 \
    --vocabs "$MODELDIR"/vocab.$L2.spm "$MODELDIR"/vocab.$L1.spm \
    --mini-batch-fit \
    -w 4096 \
    --maxi-batch 1000 \
    --early-stopping 10 \
    --cost-type ce-mean-words \
    --valid-freq 5000 --save-freq 5000 --disp-freq 10 \
    --valid-metrics ce-mean-words perplexity bleu-detok \
    --valid-sets "$DEVCORPUS.$L2" "$DEVCORPUS.$L1"  \
    --beam-size 6 \
    --normalize 0.6 \
    --log "$TEMPDIR"/train.log \
    --valid-log "$TEMPDIR"/validation.log \
    --enc-depth 6 \
    --dec-depth 6 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.1 \
    --label-smoothing 0.1 \
    --learn-rate 0.0003 \
    --lr-warmup 16000 \
    --lr-decay-inv-sqrt 16000 \
    --lr-report \
    --optimizer-params 0.9 0.98 1e-09 \
    --clip-norm 5 \
    --tied-embeddings-all \
    --devices $GPUS \
    --sync-sgd \
    --seed 1111 \
    --exponential-smoothing \
    --overwrite --keep-best

#generate synthetic corpus chr->en
cut -f 2 "$SYNTHETICCORPUS".$L1-$L2.tsv > "$SYNTHETICCORPUS".$L2
cp /dev/null "$SYNTHETICCORPUS".$L1-$L2.tsv
translateChrToEn "$SYNTHETICCORPUS".$L2 > "$SYNTHETICCORPUS".$L1
paste "$SYNTHETICCORPUS".$L1 "$SYNTHETICCORPUS".$L2 >> "$SYNTHETICCORPUS".$L1-$L2.tsv

done
