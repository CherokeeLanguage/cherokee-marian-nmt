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

# pre-train sentencepiece

L1COUNT=16384
L2COUNT=16384

cd "$MODELDIR"
rm "/data/model.spm/vocab.$L1.spm" 2> /dev/null || true
rm "/data/model.spm/vocab.$L2.spm" 2> /dev/null || true

for x in "$MONODIR/$L2/"*".$L2"; do
    cat "$x" >> "$MONODIR/corpus-sentencepiece.$L2"
done

sed -i 's/---//g' "$MONODIR/corpus-sentencepiece.$L2"
sed -i '/^\s*$/d' "$MONODIR/corpus-sentencepiece.$L2"

$MARIAN/build/spm_train --input="$MONODIR/corpus-sentencepiece.$L2" \
    --max_sentence_length 32768 \
    --model_prefix="/data/model.spm/vocab.$L2.spm" \
    --vocab_size=$L2COUNT --character_coverage=1.0

cp /dev/null "$MONODIR/corpus-sentencepiece.$L1"

function wgetIfNeeded {
    if [ -f "$1" ]; then return; fi
    wget -N -O /tmp/wget-temp.$$.txt "$2"
    mv -v "/tmp/wget-temp.$$.txt" "$1"
}

#Additional monolingual corpus for $L1 English
wgetIfNeeded "$MONODIR/$L1/Frankenstien.$L1" 'https://www.gutenberg.org/files/84/84-0.txt'
wgetIfNeeded "$MONODIR/$L1/Pride-and-Prejudice.$L1" 'https://www.gutenberg.org/files/1342/1342-0.txt'
wgetIfNeeded "$MONODIR/$L1/Beowulf.$L1" 'https://www.gutenberg.org/ebooks/16328.txt.utf-8'
wgetIfNeeded "$MONODIR/$L1/Edgar-Poe.$L1" 'https://www.gutenberg.org/ebooks/25525.txt.utf-8'
wgetIfNeeded "$MONODIR/$L1/Moby-Dick.$L1" 'https://www.gutenberg.org/files/2701/2701-0.txt'
wgetIfNeeded "$MONODIR/$L1/Dr-Jekyll.$L1" 'https://www.gutenberg.org/files/43/43-0.txt'
wgetIfNeeded "$MONODIR/$L1/Sherlock-Holmes.$L1" 'https://www.gutenberg.org/files/1661/1661-0.txt'
wgetIfNeeded "$MONODIR/$L1/Dracula.$L1" 'https://www.gutenberg.org/ebooks/345.txt.utf-8'
wgetIfNeeded "$MONODIR/$L1/Grimms-Fairy-Tales.$L1" 'https://www.gutenberg.org/files/2591/2591-0.txt'
wgetIfNeeded "$MONODIR/$L1/Jungle-Book.$L1" 'https://www.gutenberg.org/ebooks/35997.txt.utf-8'
wgetIfNeeded "$MONODIR/$L1/Jungle-Book-2.$L1" 'https://www.gutenberg.org/ebooks/37364.txt.utf-8'

for x in "$MONODIR/$L1/"*".$L1"; do
    cat "$x" >> "$MONODIR/corpus-sentencepiece.$L1"
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

# create alignment helper to improve word associations between languages faster.
# requires pretrained modles for SPM!
ALIGNEDCORPUS="$WORKDIR"/corpus.align.$L1-$L2
cat $CORPUS.$L1 | /git/marian/build/spm_encode --model "$MODELDIR/vocab.$L1.spm" > $TEMPDIR/align.temp.$L1
cat $CORPUS.$L2 | /git/marian/build/spm_encode --model "$MODELDIR/vocab.$L2.spm" > $TEMPDIR/align.temp.$L2
paste $TEMPDIR/align.temp.$L1 $TEMPDIR/align.temp.$L2 > $TEMPDIR/align.temp.$L1-$L2
sed -i 's/\t/ ||| /g' $TEMPDIR/align.temp.$L1-$L2
/git/fast_align/build/fast_align -vdo -i $TEMPDIR/align.temp.$L1-$L2 > $TEMPDIR/forward.align.$L1-$L2
/git/fast_align/build/fast_align -vdor -i $TEMPDIR/align.temp.$L1-$L2 > $TEMPDIR/reverse.align.$L1-$L2
/git/fast_align/build/atools -c grow-diag-final -i $TEMPDIR/forward.align.$L1-$L2 -j $TEMPDIR/reverse.align.$L1-$L2 > "$ALIGNEDCORPUS"

# train nmt model
nice $MARIAN/build/marian \
    --mini-batch-words $maxwords \
    --cpu-threads 16 \
    --after-epochs 1 \
    --no-restore-corpus \
    -w 1024 \
    --type s2s \
    --model "$MODELDIR"/model.npz \
    --dim-vocabs $L1COUNT $L2COUNT \
    --train-sets "$CORPUS.$L1" "$CORPUS.$L2" \
    --vocabs "$MODELDIR"/vocab.$L1.spm "$MODELDIR"/vocab.$L2.spm \
    --sentencepiece-options "--character_coverage=1.0" \
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
    # --overwrite

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
    ) | $MARIAN/build/marian-decoder -c "$MODELDIR"/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 16 --maxi-batch 64 --maxi-batch-sort src > "$WORKDIR/test-simple.$L2".output
    
