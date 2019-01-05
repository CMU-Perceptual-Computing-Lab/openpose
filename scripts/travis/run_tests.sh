#!/bin/bash

# Test the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

# Only for NAME="default-cmake-cpu" WITH_CUDA=false
if [[ $RUN_EXAMPLES == true ]] ; then
  echo "Running demos and tutorials..."
  echo " "

  echo "OpenPose demo..."
  ./build/examples/openpose/openpose.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --display 0 --render_pose 0
  echo " "

  echo "Tutorial Add Moduel: Example 1..."
  ./build/examples/tutorial_add_module/1_custom_post_processing.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --display 0 --render_pose 0
  echo " "

  # # Note: Examples 1-5 and 8-9 require GUI
  # echo "Tutorial API C++: Examples 1-5 and 8-9..."
  # ./build/examples/tutorial_api_cpp/1_body_from_image.bin
  # ./build/examples/tutorial_api_cpp/2_whole_body_from_image.bin
  # ./build/examples/tutorial_api_cpp/3_keypoints_from_image_configurable.bin --net_resolution -1x32
  # ./build/examples/tutorial_api_cpp/4_asynchronous_loop_custom_input_and_output.bin --net_resolution -1x32 --image_dir examples/media/
  # ./build/examples/tutorial_api_cpp/8_synchronous_custom_output.bin --net_resolution -1x32 --image_dir examples/media/
  # ./build/examples/tutorial_api_cpp/9_synchronous_custom_all.bin --net_resolution -1x32 --image_dir examples/media/
  # echo " "

  echo "Tutorial API C++: Example 6..."
  ./build/examples/tutorial_api_cpp/6_synchronous_custom_postprocessing.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --display 0 --render_pose 0
  echo " "

  echo "Tutorial API C++: Example 7..."
  ./build/examples/tutorial_api_cpp/7_synchronous_custom_input.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --display 0 --render_pose 0
  echo " "

  # Python examples
  if [[ $WITH_PYTHON == true ]] ; then
    echo "Python API C++: Example 1..."
    echo "TODO: Add Python examples in here..."
    echo " "
  fi

  echo "Demos and tutorials successfully finished!"

# Disable examples for all other Travis Build configurations
else
  echo "Skipping tests for non CPU-only versions."
  exit 0
fi
