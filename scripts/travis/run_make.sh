#!/bin/bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if $WITH_CMAKE ; then
  cd build
  make -j`nproc`
else # if ! $WITH_CMAKE ; then
  make all -j`nproc`
fi
