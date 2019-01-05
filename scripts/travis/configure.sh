#!/bin/bash

# Configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

echo "WITH_CMAKE = ${WITH_CMAKE}."
if [[ $WITH_CMAKE == true ]] ; then
  echo "Running CMake configuration..."
  bash $BASEDIR/configure_cmake.sh
else # if ! $WITH_CMAKE ; then
  echo "Running Makefile configuration..."
  bash $BASEDIR/configure_make.sh
fi
