#!/bin/bash
# Script to extract COCO JSON file for each trained model
clear && clear && make all -j`nproc`

# Body/Foot
IMAGE_DIR="/home/gines/devel/images/val2017/"
IMAGE_DIR_FOOT="/home/gines/devel/images/val2017_foot/"
# Face
IMAGE_DIR_FRGC="/home/gines/devel/images/frgc_val/"
IMAGE_DIR_MPIE="/home/gines/devel/images/multipie_val/"
IMAGE_DIR_FACE_MASK_OUT="/home/gines/devel/images/face_mask_out_val/"
# Hand
IMAGE_DIR_HAND_DOME="/home/gines/devel/images/hand_dome_val/"
IMAGE_DIR_HAND_MPII="/home/gines/devel/images/hand_mpii_val/"

# cd OpenPose folder
cd /home/gines/Dropbox/Perceptual_Computing_Lab/openpose/openpose/

temporaryJsonFile1=~/Desktop/OpenPose_1.json
temporaryJsonFile4=~/Desktop/OpenPose_4.json
OP_COMAND="./build/examples/openpose/openpose.bin --face --hand --render_pose 0 --display 0 --cli_verbose 0.2"
OP_COMAND_1SCALE="${OP_COMAND} --write_coco_json ${temporaryJsonFile1} --num_gpu_start 1"
OP_COMAND_4SCALES="${OP_COMAND} --maximize_positives --net_resolution_dynamic -1 --scale_number 4 --scale_gap 0.25 --write_coco_json ${temporaryJsonFile4} --num_gpu_start 1"

# NOTE: Uncomment those tests you are interested into. By default, all disabled.

# # Body/foot 1 scale
# echo "Processing bodies/feet..."
# $OP_COMAND_1SCALE --image_dir ${IMAGE_DIR} --write_coco_json_variants 3

# # Faces
# echo "Processing faces..."
# temporaryJsonFile1Face=~/Desktop/OpenPose_1_face.json
# # Face FRGC processing
# $OP_COMAND_1SCALE --image_dir ${IMAGE_DIR_FRGC} --write_coco_json_variants 4
# # Face MPIE processing
# $OP_COMAND_1SCALE --image_dir ${IMAGE_DIR_MPIE} --write_coco_json_variants 4
# # Face Mask Out processing
# $OP_COMAND_1SCALE --image_dir ${IMAGE_DIR_FACE_MASK_OUT} --write_coco_json_variants 4

# # Hands
# echo "Processing hands..."
# # Hand Dome processing
# $OP_COMAND_1SCALE --image_dir ${IMAGE_DIR_HAND_DOME} --write_coco_json_variants 8
# Hand MPII processing
# $OP_COMAND_1SCALE --image_dir ${IMAGE_DIR_HAND_MPII} --write_coco_json_variants 16

# # Body/foot 4 scales
# $OP_COMAND_4SCALES --image_dir ${IMAGE_DIR} --write_coco_json_variants 3 --net_resolution "1312x736"

echo " "
echo "Finished! Exiting script..."
echo " "
