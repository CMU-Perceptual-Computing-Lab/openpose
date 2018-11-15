#!/bin/bash



# USAGE EXAMPLE
# clear && clear && make all -j`nproc` && bash ./examples/tests/pose_accuracy_coco_val.sh

# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# Parameters
IMAGE_FOLDER=~/devel/images/val2017/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin

    # 1 scale
$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1.json --display 0 --render_pose 0
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_max.json --display 0 --render_pose 0 --maximize_positives --model_pose BODY_25E

    # 1 scale - Debugging
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1.json --display 0 --write_images ~/Desktop/CppValidation/

#     # 3 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_3.json --display 0 --render_pose 0 --scale_number 3 --scale_gap 0.25

#     # 4 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_4.json --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_4_max.json --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --maximize_positives
