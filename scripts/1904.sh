#!/bin/bash

set -e
set -o pipefail

cd "$(dirname "$0")"

CURRENT_UID="$(id -u):$(id -g)"

docker run --user $CURRENT_UID --mount type=bind,src="$(pwd)",target="/data" -it mariannmt-20191014.01-1904

