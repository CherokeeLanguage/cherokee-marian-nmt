#!/bin/bash

set -e
set -o pipefail

docker build -t spellcheck/cuda:latest -f spellcheck-cuda .

docker build -t marian-nmt/cuda:latest -f marian-nmt-cuda-1804 .

#docker build -t openblas-1804 -f openblas-1804 .

#docker build -t intelblas-1804 -f intelblas-1804 .

#docker build -t intelblas-1904 -f intelblas-1904 .
