#!/bin/bash

# Install dependencies for Mac OS
echo "Running on Mac OS."

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

source ./scripts/osx/install_brew.sh
source ./scripts/osx/install_deps.sh
