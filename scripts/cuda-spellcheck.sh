#!/bin/bash

set -e
set -o pipefail

cd "$(dirname "$0")"
cd ../en-chr

CURRENT_UID="$(id -u):$(id -g)"

docker run --gpus all --mount type=bind,src="$(pwd)",target="/data" -it spellcheck/cuda:latest
