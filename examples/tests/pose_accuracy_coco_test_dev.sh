#!/bin/bash



# USAGE EXAMPLE
# clear && clear && make all -j`nproc` && bash ./examples/tests/pose_accuracy_coco_test_dev.sh

# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# Parameters
IMAGE_FOLDER=/media/posefs3b/Users/gines/openpose_train/dataset/COCO/cocoapi/images/test2017_dev/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin

#     # 1 scale
# $OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_test.json
# $OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_test_max.json \
#     --maximize_positives

    # 4 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_4_test.json \
#     --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
$OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_4_test_max.json \
    --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --maximize_positives
