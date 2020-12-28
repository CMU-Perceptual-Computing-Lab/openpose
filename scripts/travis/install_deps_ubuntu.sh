#!/bin/bash

# Install dependencies for Ubuntu
echo "Running on ${TRAVIS_OS_NAME} OS."

BASEDIR=$(dirname "$0")
source "$BASEDIR"/defaults.sh

if [[ "$WITH_CUDA" == "true" ]]
then
  source scripts/ubuntu/install_cuda.sh
fi
if [[ "$WITH_CUDNN" == "true" ]]
then
  source scripts/ubuntu/install_cudnn.sh
fi

source scripts/ubuntu/install_deps.sh

sudo apt-get -yq  install libopencv-dev
