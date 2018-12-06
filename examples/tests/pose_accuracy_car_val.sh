#!/bin/bash



# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# USAGE EXAMPLE
# clear && clear && make all -j`nproc` && bash ./examples/tests/pose_accuracy_coco_test.sh

# # Go back to main folder
# cd ../../


# Write COCO-format JSON
# Last id:
    # ID 20671  <-->    #frames = 1471      -->     ~ 1.5 min at 15fps
    # ID 50006  <-->    #frames = 3559      -->     ~ 4 min at 15fps

# Parameters
IMAGE_FOLDER_CF=/home/gines/devel/images/car-fusion_val/
IMAGE_FOLDER_P3=/home/gines/devel/images/pascal3d+_val/
IMAGE_FOLDER_V7=/home/gines/devel/images/veri-776_val/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin
GPUS=-1
# GPUS=1

    # 1 scale
$OP_BIN --image_dir $IMAGE_FOLDER_CF --write_coco_json_variant 0 --write_coco_json ${JSON_FOLDER}processed_carfusion_val_1.json --model_pose CAR_22 --display 0 --render_pose 0 --num_gpu ${GPUS}
$OP_BIN --image_dir $IMAGE_FOLDER_P3 --write_coco_json_variant 1 --write_coco_json ${JSON_FOLDER}processed_pascal3dplus_val_1.json --model_pose CAR_22 --display 0 --render_pose 0 --num_gpu ${GPUS}
$OP_BIN --image_dir $IMAGE_FOLDER_V7 --write_coco_json_variant 2 --write_coco_json ${JSON_FOLDER}processed_veri776_val_1.json --model_pose CAR_22 --display 0 --render_pose 0 --num_gpu ${GPUS}

#     # 4 scales
# $OP_BIN --image_dir $IMAGE_FOLDER_CF --write_coco_json_variant 0 --write_coco_json ${JSON_FOLDER}processed_carfusion_val_4.json --model_pose CAR_22 --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --num_gpu ${GPUS}
# $OP_BIN --image_dir $IMAGE_FOLDER_P3 --write_coco_json_variant 1 --write_coco_json ${JSON_FOLDER}processed_pascal3dplus_val_4.json --model_pose CAR_22 --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --num_gpu ${GPUS}
# $OP_BIN --image_dir $IMAGE_FOLDER_V7 --write_coco_json_variant 2 --write_coco_json ${JSON_FOLDER}processed_veri776_val_4.json --model_pose CAR_22 --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --num_gpu ${GPUS}
