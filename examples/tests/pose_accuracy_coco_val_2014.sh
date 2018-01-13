# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# USAGE EXAMPLE
# clear && clear && make all -j`nproc` && bash ./examples/tests/pose_accuracy_coco_test.sh

# # Go back to main folder
# cd ../../


# Write COCO-format JSON
# Note: `--frame_last 3558` --> total = 3559 frames
# Last id:
    # ID 20671  <-->    #frames = 1471      -->     ~ 1.5 min at 15fps
    # ID 50006  <-->    #frames = 3559      -->     ~ 4 min at 15fps

# Parameters
IMAGE_FOLDER=/home/gines/devel/images/val2014/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin

    # 1 scale
$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1.json --no_display --render_pose 0 --frame_last 3558

    # 1 scale - Debugging
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1.json --no_display --frame_last 3558 --write_images ~/Desktop/CppValidation/

#     # 3 scales
# $OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_3.json --no_display --render_pose 0 --scale_number 3 --scale_gap 0.25 --frame_last 3558

#     # 4 scales
# $OP_BIN --num_gpu 1 --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_4.json --no_display --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --frame_last 3558

# 	  # Debugging - Rendered frames saved
# $OP_BIN --image_dir $IMAGE_FOLDER --write_images ${JSON_FOLDER}frameOutput --no_display
