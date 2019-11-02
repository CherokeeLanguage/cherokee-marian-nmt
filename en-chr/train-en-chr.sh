#!/bin/bash -v

set -e
set -o pipefail

cd "$(dirname "$0")"
cwd="$(pwd)"

EPOCHS=10000

. ./en-chr.vars
. ./en-chr.funcs

export PATH="$PATH:$MARIAN/build/"

if [ ! -e $MARIAN/build/marian ]
then
    echo "marian is not installed in $MARIAN/build, you need to compile the toolkit first"
    exit 1
fi

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

l1Maxwords=$(getMaxWordCount $L1)
l2Maxwords=$(getMaxWordCount $L2)
maxwords=$((($l1Maxwords+$l2Maxwords+2)*16))

echo "$L1: $l1Maxwords maxwords, $L2: $l2Maxwords maxwords"
echo "Mini batch maxwords: $maxwords"

for loops in $(seq 1 1 1); do

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

#split training up into 9 training sets and 1 training check set

shuf -o "$CORPUS.$L1-$L2.shuf.tsv" "$CORPUS.$L1-$L2".tsv

for s in $(seq 1 1 10); do

    echo "CHUNK: $s" >> "$TEMPDIR"/validation.log

    corpusSplit "$CORPUS.$L1-$L2.shuf.tsv"

    doAlign

    # reset stalled count before running next training cycle
    if [ -f "$MODELDIR/model-$L1-$L2.npz.progress.yml" ]; then
        sed -i 's/^stalled: .*$/stalled: 0/g' "$MODELDIR/model-$L1-$L2.npz.progress.yml"
    fi

    # train nmt model - forwards
    nice $MARIAN/build/marian \
        --after-epochs $EPOCHS \
        --mini-batch-fit \
        --devices 0 \
        --no-restore-corpus \
        -w 4096 \
        --type s2s \
        --model "$MODELDIR"/model-$L1-$L2.npz \
        --dim-vocabs $L1COUNT $L2COUNT \
        --train-sets "$CORPUS.$L1" "$CORPUS.$L2" \
        --vocabs "$MODELDIR"/vocab.$L1.spm "$MODELDIR"/vocab.$L2.spm \
        --layer-normalization --tied-embeddings-all \
        --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
        --early-stopping 10 --max-length 100 \
        --valid-freq 500 --save-freq 500 --disp-freq 10 \
        --cost-type ce-mean-words --valid-metrics ce-mean-words bleu-detok \
        --valid-sets "$DEVCORPUS.$L1" "$DEVCORPUS.$L2"  \
        --log "$TEMPDIR"/train.log --valid-log "$TEMPDIR"/validation.log --tempdir "$TEMPDIR" \
        --keep-best \
        --seed 1111 --exponential-smoothing \
        --normalize=0.6 --beam-size=6 --quiet-translation \
        $ALIGN_ARGS \
        --valid-translation-output "$TEMPDIR/validation-translation-output.txt" # \
        #--lr-decay-strategy stalled --lr-decay-start 1 --lr-report
done 

#generate synthetic corpus en->chr
#BOOKLIST="Frankenstien Pride-and-Prejudice Moby-Dick Dr-Jekyll Sherlock-Holmes Dracula Grimms-Fairy-Tales Jungle-Book Jungle-Book-2"
BOOKLIST="Jungle-Book"
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

# reset stalled count before running next training cycle
if [ -f "$MODELDIR/model-$L2-$L1.npz.progress.yml" ]; then
    sed -i 's/^stalled: .*$/stalled: 0/g' "$MODELDIR/model-$L2-$L1.npz.progress.yml"
fi

# train nmt model - backwards
nice $MARIAN/build/marian \
    --after-epochs $EPOCHS \
    --mini-batch-fit \
    --devices 0 \
    --no-restore-corpus \
    -w 4096 \
    --type s2s \
    --model "$MODELDIR"/model-$L2-$L1.npz \
    --dim-vocabs $L1COUNT $L2COUNT \
    --train-sets "$CORPUS.$L2" "$CORPUS.$L1" \
    --vocabs "$MODELDIR"/vocab.$L2.spm "$MODELDIR"/vocab.$L1.spm \
    --layer-normalization --tied-embeddings-all \
    --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 \
    --early-stopping 10 --max-length 100 \
    --valid-freq 100 --save-freq 100 --disp-freq 5 \
    --cost-type ce-mean-words --valid-metrics ce-mean-words bleu-detok \
    --valid-sets "$DEVCORPUS.$L2" "$DEVCORPUS.$L1"  \
    --log "$TEMPDIR"/train.log --valid-log "$TEMPDIR"/validation.log --tempdir "$TEMPDIR" \
    --keep-best \
    --seed 1111 --exponential-smoothing \
    --normalize=0.6 --beam-size=6 --quiet-translation \
    $ALIGN_ARGS \
    --valid-translation-output "$TEMPDIR/validation-translation-output.txt" #\
    #--lr-decay-strategy stalled --lr-decay-start 1 --lr-report

#generate synthetic corpus chr->en
cut -f 2 "$SYNTHETICCORPUS".$L1-$L2.tsv > "$SYNTHETICCORPUS".$L2
cp /dev/null "$SYNTHETICCORPUS".$L1-$L2.tsv
translateChrToEn "$SYNTHETICCORPUS".$L2 > "$SYNTHETICCORPUS".$L1
paste "$SYNTHETICCORPUS".$L1 "$SYNTHETICCORPUS".$L2 >> "$SYNTHETICCORPUS".$L1-$L2.tsv

done
