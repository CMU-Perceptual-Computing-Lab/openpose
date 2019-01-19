#!/bin/bash

# Test the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

# Only for NAME="default-cmake-cpu" WITH_CUDA=false
if [[ $RUN_EXAMPLES == true ]] ; then
  echo "Running demos and tutorials..."
  echo " "

  echo "OpenPose demo..."
  ./build/examples/openpose/openpose.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --write_images output/ --display 0
  echo " "

  echo "Tutorial Add Module: Example 1..."
  ./build/examples/tutorial_add_module/1_custom_post_processing.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --write_images output/ --display 0
  echo " "

  # # Note: Examples 1-2 require the whole OpenPose resolution (too much RAM memory) and the GUI
  # echo "Tutorial API C++: Examples 1-2..."
  # ./build/examples/tutorial_api_cpp/01_body_from_image_default.bin
  # ./build/examples/tutorial_api_cpp/02_whole_body_from_image_default.bin
  # echo " "

  echo "Tutorial API C++: Example 3..."
  ./build/examples/tutorial_api_cpp/03_keypoints_from_image.bin --no_display --net_resolution -1x32 --write_json output/
  echo " "

  echo "Tutorial API C++: Example 4..."
  ./build/examples/tutorial_api_cpp/04_keypoints_from_images.bin --no_display --net_resolution -1x32 --write_json output/
  echo " "

  echo "Tutorial API C++: Example 5..."
  ./build/examples/tutorial_api_cpp/05_keypoints_from_images_multi_gpu.bin --no_display --net_resolution -1x32 --write_json output/
  echo " "

  echo "Tutorial API C++: Example 6..."
  ./build/examples/tutorial_api_cpp/06_asynchronous_custom_input.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --write_images output/ --display 0
  echo " "

  echo "Tutorial API C++: Example 7..."
  ./build/examples/tutorial_api_cpp/07_asynchronous_custom_output.bin --no_display --net_resolution -1x32 --image_dir examples/media/
  echo " "

  echo "Tutorial API C++: Example 8..."
  ./build/examples/tutorial_api_cpp/08_asynchronous_custom_input_output_and_datum.bin --no_display --net_resolution -1x32 --image_dir examples/media/
  echo " "

  echo "Tutorial API C++: Example 9..."
  ./build/examples/tutorial_api_cpp/09_face_from_image.bin --no_display --face_net_resolution 32x32
  echo " "

  echo "Tutorial API C++: Example 10..."
  ./build/examples/tutorial_api_cpp/10_hand_from_image.bin --no_display --hand_net_resolution 32x32
  echo " "

  echo "Tutorial API C++: Example 11..."
  ./build/examples/tutorial_api_cpp/11_synchronous_custom_input.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --write_images output/ --display 0
  echo " "

  echo "Tutorial API C++: Example 13..."
  ./build/examples/tutorial_api_cpp/13_synchronous_custom_postprocessing.bin --net_resolution -1x32 --image_dir examples/media/ --write_json output/ --write_images output/ --display 0
  echo " "

  echo "Tutorial API C++: Example 14..."
  ./build/examples/tutorial_api_cpp/14_synchronous_custom_output.bin  --no_display --net_resolution -1x32 --image_dir examples/media/
  echo " "

  echo "Tutorial API C++: Example 15..."
  ./build/examples/tutorial_api_cpp/15_synchronous_custom_all.bin  --no_display --net_resolution -1x32 --image_dir examples/media/
  echo " "

  # Python examples
  if [[ $WITH_PYTHON == true ]] ; then
    echo "Tutorial API Python: OpenPose demo..."
    cd build/examples/tutorial_api_python
    python openpose_python.py --net_resolution -1x32 --image_dir ../../../examples/media/ --write_json output/ --write_images output/ --display 0
    echo " "
    # Note: All Python examples require GUI
  fi

  echo "Demos and tutorials successfully finished!"

# Disable examples for all other Travis Build configurations
else
  echo "Skipping tests for non CPU-only versions."
  exit 0
fi
