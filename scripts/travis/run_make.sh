#!/bin/bash

# Build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CMAKE == true ]] ; then
  cd build
  make -j`nproc`
else
  make all -j`nproc`
fi
