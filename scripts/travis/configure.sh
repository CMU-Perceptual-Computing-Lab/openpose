#!/bin/bash

# Configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

echo "WITH_CMAKE = ${WITH_CMAKE}."
if [[ $WITH_CMAKE == true ]] ; then
  echo "Running CMake configuration..."
  source $BASEDIR/configure_cmake.sh
else
  echo "Running Makefile configuration..."
  source $BASEDIR/configure_make.sh
fi
