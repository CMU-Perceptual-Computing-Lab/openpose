#!/bin/bash



# USAGE EXAMPLE
# clear && clear && make all -j`nproc` && bash ./scripts/tests/pose_accuracy_coco_val.sh

# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# Parameters
IMAGE_FOLDER=~/devel/images/val2017/
IMAGE_FOOT_FOLDER=~/devel/images/val2017_foot/
JSON_FOLDER=../evaluation/coco_val_jsons/
# JSON_FOLDER=/media/posefs3b/Users/gines/openpose_train/training_results/2_23_51/best_702k/
OP_BIN=./build/examples/openpose/openpose.bin

    # 1 scale (body)
$OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1.json --write_coco_json_variants 1
# $OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_max.json --write_coco_json_variants 3 \
#     --maximize_positives

    # 1 scale (foot)
$OP_BIN --image_dir $IMAGE_FOOT_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1.json --write_coco_json_variants 2

#     # 4 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_4.json --write_coco_json_variants 3 \
#     --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
# $OP_BIN --image_dir $IMAGE_FOLDER --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json ${JSON_FOLDER}1_4_max.json --write_coco_json_variants 3 \
#     --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --maximize_positives
#     # \
#     # --model_pose BODY_23 --model_folder ${JSON_FOLDER}
