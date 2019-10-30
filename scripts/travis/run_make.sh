#!/bin/bash

# Build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CMAKE == true ]] ; then
  cd build
  if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then make -j`nproc` ; fi
  if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then make -j`sysctl -n hw.logicalcpu` ; fi
else
  if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then make all -j`nproc` ; fi
  if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then make all -j`sysctl -n hw.logicalcpu` ; fi
fi
