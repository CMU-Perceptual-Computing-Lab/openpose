#!/bin/bash

# Install dependencies for Ubuntu
echo "Running on ${CI_OS_NAME} OS."

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CUDA == true ]]; then
  bash $BASEDIR/../ubuntu/install_cuda.sh
fi
if [[ $WITH_CUDNN == true ]]; then
  bash $BASEDIR/../ubuntu/install_cudnn_up_to_Ubuntu16.sh
fi

bash $BASEDIR/../ubuntu/install_deps.sh

sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libopencv-dev
