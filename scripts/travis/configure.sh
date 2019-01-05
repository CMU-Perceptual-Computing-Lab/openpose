#!/bin/bash

# Configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if [[ $WITH_CMAKE ]] ; then
  bash $BASEDIR/configure_cmake.sh
else # if ! $WITH_CMAKE ; then
  bash $BASEDIR/configure_make.sh
fi
