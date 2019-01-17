#!/bin/bash

# Install dependencies for Mac OS
echo "Running on Mac OS."

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

bash scripts/osx/install_brew.sh
bash scripts/osx/install_deps.sh
