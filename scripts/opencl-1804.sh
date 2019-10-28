#!/bin/bash

set -e
set -o pipefail

cd "$(dirname "$0")"

docker run --device /dev/dri --device /dev/kfd --mount type=bind,src="$(pwd)",target="/data" -it mariannmt-20191022.01-1804

