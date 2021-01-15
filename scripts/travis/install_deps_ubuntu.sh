#!/bin/bash

# Install dependencies for Ubuntu
echo "Running on ${TRAVIS_OS_NAME} OS."

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CUDA == true ]]; then
  sudo bash $BASEDIR/../ubuntu/install_cuda.sh
fi
if [[ $WITH_CUDNN == true ]]; then
  sudo bash $BASEDIR/../ubuntu/install_cudnn.sh
fi

sudo bash $BASEDIR/../ubuntu/install_deps.sh

sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libopencv-dev
