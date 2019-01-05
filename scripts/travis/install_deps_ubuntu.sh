#!/bin/bash

# Install dependencies for Ubuntu
echo "Running on ${TRAVIS_OS_NAME} OS."

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CUDA ]] ; then
  sudo bash ./3rdparty/ubuntu/install_ubuntu_deps_and_cuda.sh
else # if ! $WITH_CUDA ; then
  sudo bash ./3rdparty/ubuntu/install_ubuntu_deps.sh
fi
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libopencv-dev
