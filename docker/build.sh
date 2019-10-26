#!/bin/bash

set -e
set -o pipefail

docker build -t cuda-mariannmt-20191025.01-1804 -f Dockerfile-cuda.1804 .

#docker build -t mariannmt-20191022.01-1804 -f Dockerfile-opencl.1804 .

#docker build -t mariannmt-20191014.01-1804 -f Dockerfile.1804 .

docker build -t mariannmt-20191014.01-1904 -f Dockerfile.1904 .
