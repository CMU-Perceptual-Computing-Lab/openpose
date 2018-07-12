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
IMAGE_FOLDER=/mnt/DataUbuntu/openpose_train/dataset/COCO/car_dataset/Dataset/images_jpg_val/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin

    # 1 scale
$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}car_1.json --display 0 --render_pose 0 --model_pose CAR_12

#     # 4 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_4.json --display 0 --render_pose 0 --model_pose CAR_12 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_4.json --display 0 --render_pose 0 --model_pose CAR_12 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --frame_last 1005

# 	  # Debugging - Rendered frames saved
# $OP_BIN --image_dir $IMAGE_FOLDER --write_images ${JSON_FOLDER}frameOutput --display 0
