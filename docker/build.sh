#!/bin/bash

docker build -t mariannmt-20191014.01-1804 -f Dockerfile.1804 .

docker build -t mariannmt-20191014.01-1904 -f Dockerfile.1904 .
