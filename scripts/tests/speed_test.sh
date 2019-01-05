#!/bin/bash



# Script for internal use. We might completely change it continuously and we will not answer questions about it.

# USAGE EXAMPLE
# clear && clear && make all -j`nproc` && bash ./scripts/tests/speed_test.sh

# # Go back to main folder
# cd ../../

# Get model speed
~/devel/openpose_caffe_train/build/tools/caffe time -gpu 0 -model /mnt/DataUbuntu/openpose_train/training_results_light/pose/pose_training.prototxt
# ./3rdparty/caffe/build/tools/caffe time -gpu 0 -model /mnt/DataUbuntu/openpose_train/training_results_light/pose/pose_training.prototxt
