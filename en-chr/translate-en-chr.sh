#!/bin/bash -v

set -e
set -o pipefail

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

WORKDIR="/data/data.$L1-$L2"
if [ ! -d "$WORKDIR" ]; then echo "UNABLE TO TRANSLATE - NO DATA!"; exit -1; fi
MODELDIR="$WORKDIR/model.$L1-$L2"
if [ ! -d "$MODELDIR" ]; then echo "UNABLE TO TRANSLATE - NO MODEL!"; exit -1; fi
cd "$WORKDIR"

echo "$1" \
    | $MARIAN/build/marian-decoder -c $MODELDIR/model.npz.decoder.yml --cpu-threads 1 -b 6 -n0.6 \
      --mini-batch 64 --maxi-batch 100 --maxi-batch-sort src > "$WORKDIR/dev.$L2".output

