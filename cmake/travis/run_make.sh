#!/bin/bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if $WITH_CMAKE ; then
  cd build
  make -j`nproc`
  # make --jobs $NUM_THREADS
else # if ! $WITH_CMAKE ; then
  make all -j`nproc`
  # make --jobs $NUM_THREADS all
fi
