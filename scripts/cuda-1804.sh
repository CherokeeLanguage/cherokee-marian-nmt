#!/bin/bash

set -e
set -o pipefail

cd "$(dirname "$0")"
cd ../en-chr

CURRENT_UID="$(id -u):$(id -g)"

docker run --gpus all --user "$CURRENT_UID" --mount type=bind,src="$(pwd)",target="/data" -it marian-nmt/cuda:latest

