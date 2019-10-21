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

# pre-train sentencepiece

cd "$MODELDIR"
rm "/data/model.spm/vocab.$L1.spm" 2> /dev/null || true
rm "/data/model.spm/vocab.$L2.spm" 2> /dev/null || true

for x in "$MONODIR/$L2/"*".$L2"; do
    cat "$x" >> "$MONODIR/corpus-sentencepiece.$L2"
done

sed -i 's/---//g' "$MONODIR/corpus-sentencepiece.$L2"
sed -i '/^\s*$/d' "$MONODIR/corpus-sentencepiece.$L2"

$MARIAN/build/spm_train --input="$MONODIR/corpus-sentencepiece.$L2" \
    --model_prefix="/data/model.spm/vocab.$L2.spm" \
    --vocab_size=2000 --character_coverage=1.0 --hard_vocab_limit=false

cp /dev/null "$MONODIR/corpus-sentencepiece.$L1"
for x in "$MONODIR/$L1/"*".$L1"; do
    cat "$x" >> "$MONODIR/corpus-sentencepiece.$L1"
done

$MARIAN/build/spm_train --input="$MONODIR/corpus-sentencepiece.$L1" \
    --model_prefix="/data/model.spm/vocab.$L1.spm" \
    --vocab_size=8000 --character_coverage=1.0 --hard_vocab_limit=false

mv -v "/data/model.spm/vocab.$L1.spm.model" "/data/model.spm/vocab.$L1.spm"
mv -v "/data/model.spm/vocab.$L2.spm.model" "/data/model.spm/vocab.$L2.spm"

#copy in pretrained models for SPM
cp -v /data/model.spm/* "$MODELDIR/"

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
    --mini-batch 8 \
    --maxi-batch 24 \
    --cpu-threads 16 \
    --after-epochs 1 --allow-unk \
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
    # --overwrite

# translate dev set
cat "$CORPUSDEV.$L1" \
    | $MARIAN/build/marian-decoder -c model/model.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 64 --maxi-batch 100 --maxi-batch-sort src > "$CORPUSDEV.$L2".output

# translate test set
cat "$CORPUSTEST.$L1" \
    | $MARIAN/build/marian-decoder -c model/model.npz.npz.decoder.yml --cpu-threads 16 -b 6 -n0.6 \
      --mini-batch 64 --maxi-batch 100 --maxi-batch-sort src > "$CORPUSTEST.$L2".output

# get marian-nmt fork of sacrebleu
#cd /git
#git clone https://github.com/marian-nmt/sacreBLEU.git sacreBLEU

# calculate bleu scores on dev and test set
#/git/sacreBLEU/sacrebleu.py -t wmt16/dev -l $L1-$L2 < data/newsdev2016.$L2.output
#/git/sacreBLEU/sacrebleu.py -t wmt16 -l $L1-$L2 < data/newstest2016.$L2.output
