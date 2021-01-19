#!/bin/bash

# Install dependencies for Mac OS
echo "Running on Mac OS."

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

# To fix linking issue in CI during install of python as dep of opencv
[[ -f /usr/local/bin/2to3 ]] && rm -f /usr/local/bin/2to3

source ./scripts/osx/install_brew.sh
source ./scripts/osx/install_deps.sh
