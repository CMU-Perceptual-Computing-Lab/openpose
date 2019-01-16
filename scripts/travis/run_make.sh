#!/bin/bash

# Build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CMAKE == true ]] ; then
  cd build
  if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then make -j1 ; fi
  if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then make -j`nproc` ; fi
else
  make all -j`nproc`
fi