#!/bin/bash

# Install dependencies for Ubuntu
echo "Running on ${TRAVIS_OS_NAME} OS."

BASEDIR=$(dirname "$0")
source "$BASEDIR"/defaults.sh

if [[ "$WITH_CUDA" == "true" ]]
then
  source scripts/ubuntu/install_deps_and_cuda.sh
else
  source scripts/ubuntu/install_deps.sh
fi
sudo apt-get -yq  install libatlas-base-dev
sudo apt-get -yq  install libopencv-dev
