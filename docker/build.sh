#!/bin/bash

set -e
set -o pipefail

docker build -t cuda-1804 -f cuda-1804 .

#docker build -t openblas-1804 -f openblas-1804 .

#docker build -t intelblas-1804 -f intelblas-1804 .

#docker build -t intelblas-1904 -f intelblas-1904 .
