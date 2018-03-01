# Script for internal use. We might completely change it continuously and we will not answer questions about it.

clear && clear

# USAGE EXAMPLE
# See ./examples/tests/pose_accuracy_coco_test.sh

# Parameters
IMAGE_FOLDER=/media/posefs3b/Users/gines/openpose_train/dataset/COCO/images/test2017_dev/
JSON_FOLDER=../evaluation/coco_val_jsons/
OP_BIN=./build/examples/openpose/openpose.bin

    # 1 scale
$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_test.json --display 0 --render_pose 0

    # 4 scales
$OP_BIN --image_dir $IMAGE_FOLDER --write_coco_json ${JSON_FOLDER}1_4_test.json --display 0 --render_pose 0 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
