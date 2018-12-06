#!/bin/bash



# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# USAGE EXAMPLE
# See ./examples/tests/pose_accuracy_coco_test.sh

# Parameters
IMAGE_FOLDER=/home/gines/devel/images/val2017/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin

    # 1 scale
$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_foot_json ${JSON_FOLDER}1_foot.json --write_coco_json ${JSON_FOLDER}1.json --display 0 --render_pose 0

#     # 4 scales
# $OP_BIN --model_pose BODY_25 --image_dir $IMAGE_FOLDER --write_coco_foot_json ${JSON_FOLDER}1_4_foot.json --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
