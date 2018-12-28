#!/bin/bash



# Script for internal use. We might completely change it continuously and we will not answer questions about it.

# Required
# sudo apt-get install graphviz

# USAGE EXAMPLE
# clear && clear && make all -j24 && bash ./examples/tests/speed_test.sh

# # Go back to main folder
# cd ../../

PROTO_TXT_PATH=/mnt/DataUbuntu/openpose_train/training_results/pose/pose_training.prototxt
OUTPUT_PNG_PATH=/mnt/DataUbuntu/openpose_train/training_results/pose/pose_training.png

# Get model speed
python ~/devel/openpose_caffe_train/python/draw_net.py $PROTO_TXT_PATH $OUTPUT_PNG_PATH
display $OUTPUT_PNG_PATH
rm $OUTPUT_PNG_PATH
