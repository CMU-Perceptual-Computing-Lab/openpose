#!/bin/bash

# Test the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

# Only for NAME="default-cmake-cpu" WITH_CUDA=false
if [[ $RUN_EXAMPLES ]] ; then
  echo "Running demos and tutorials..."

  echo "OpenPose demo..."
  ./build/examples/openpose/openpose.bin --net_resolution -1x64 --image_dir examples/media/ --write_json output/ --display 0 --render_pose 0

  echo "Demos and tutorials successfully finished!"
else
  echo "Skipping tests for non default-cmake-cpu version."
  exit 0
fi
