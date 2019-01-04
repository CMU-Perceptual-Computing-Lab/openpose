#!/bin/bash

# Install dependencies
echo "Running on OS ${TRAVIS_OS_NAME}."

# Ubuntu
if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
  BASEDIR=$(dirname $0)
  source $BASEDIR/defaults.sh

  if $WITH_CUDA ; then
    sudo bash ./3rdparty/ubuntu/install_ubuntu_deps_and_cuda.sh
  else # if ! $WITH_CUDA ; then
    sudo bash ./3rdparty/ubuntu/install_ubuntu_deps.sh
  fi
  sudo apt-get -y install libatlas-base-dev
  sudo apt-get -y install libopencv-dev



# OSX
elif [[ $TRAVIS_OS_NAME == 'osx' ]]; then
  echo "Unknown OS ${TRAVIS_OS_NAME}."
  # Install some custom requirements on OSX



# Windows
elif [[ $TRAVIS_OS_NAME == 'windows' ]]; then
  echo "Unknown OS ${TRAVIS_OS_NAME}."
  # Install some custom requirements on Windows



# Unknown
else
  echo "Unknown OS ${TRAVIS_OS_NAME}."
  exit 99
fi
