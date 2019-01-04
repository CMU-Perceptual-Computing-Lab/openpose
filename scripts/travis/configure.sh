#!/bin/bash
# configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if $WITH_CMAKE ; then
  source $BASEDIR/configure-cmake.sh
else # if ! $WITH_CMAKE ; then
  source $BASEDIR/configure-make.sh
fi
