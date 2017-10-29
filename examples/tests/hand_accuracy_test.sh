# Script for internal use. We might completely change it continuously and we will not answer questions about it.

# Clear terminal screen
clear && clear



HAND_TESTING_FOLDER="/media/posefs3b/Users/gines/openpose_training/dataset/hand_testing/5_keypointJson/"
IMAGES_FOLDER=${HAND_TESTING_FOLDER}"0_images/"
IMAGES_BB_FOLDER=${HAND_TESTING_FOLDER}"3_images_bounding_box"
IMAGES_BB_FOLDER=${HAND_TESTING_FOLDER}"4_hand_detections"
KEYPOINT_JSON_FOLDER=${HAND_TESTING_FOLDER}"5_keypointJson/"

SCALES=6
SUFFIX="_test${SCALES}"



# Given bounding box
HAND_RESULTS_FOLDER_BB=${KEYPOINT_JSON_FOLDER}"hand_keypoints_estimated"${SUFFIX}"_bounding_box"
echo "Output on ${HAND_RESULTS_FOLDER_BB}"
rm -rf $HAND_RESULTS_FOLDER_BB
# 1 scale
./build/examples/tests/handFromJsonTest.bin \
    --hand_scale_number ${SCALES} --hand_scale_range 0.4 \
    --image_dir ${IMAGES_BB_FOLDER} \
    --hand_ground_truth ${IMAGES_BB_FOLDER} \
    --write_keypoint_json $HAND_RESULTS_FOLDER_BB \
    --no_display



# No bounding box
HAND_RESULTS_FOLDER_NO_BB=${KEYPOINT_JSON_FOLDER}"hand_keypoints_estimated"${SUFFIX}
echo "Output on ${HAND_RESULTS_FOLDER_NO_BB}"
rm -rf $HAND_RESULTS_FOLDER_NO_BB
# 1 scale
./build/examples/openpose/openpose.bin \
    --hand logging_level 3 \
    --hand_scale_number ${SCALES} --hand_scale_range 0.4 \
    --image_dir ${IMAGES_FOLDER} \
    --write_keypoint_json $HAND_RESULTS_FOLDER_NO_BB \
    --no_display
