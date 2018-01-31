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
IMAGE_FOLDER=/home/ryaadhav/Downloads/val2017/
JSON_FOLDER=./evaluation/coco_val_jsons/mkl_656x368_3_0.25_200
OP_BIN=./build_mkl/examples/openpose/openpose.bin

    # 1 scale
#$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1.json --no_display --render_pose 1

    # 1 scale - Debugging
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1.json --no_display --write_images ~/Desktop/CppValidation/

#     # 3 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_3.json --no_display --render_pose 0 --scale_number 3 --scale_gap 0.25

#     # 4 scales
$OP_BIN  --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}.json --no_display --render_pose 0 --scale_number 3 --scale_gap 0.25 --net_resolution "656x368"  --frame_last 200

# 	  # Debugging - Rendered frames saved
# $OP_BIN --image_dir $IMAGE_FOLDER --write_images ${JSON_FOLDER}frameOutput --no_display
