#!/bin/bash



# Disclaimer:
# Script for internal use. We might make continuous changess on it and we will not answer questions about it.

# Full usage description:
    # Step 1 - Getting JSON output
        # Running it:
        # Run it from OpenPose main folder with the following command:
        # clear && clear && make all -j`nproc` && bash ./examples/tests/hand_accuracy_test.sh

        # Image paths:
        # Read that script for details about all the paths and change them for your own paths.

        # Careful:
        # If you are using the NAS, please do not override my files, i.e., please change the output paths (corresponding to the ones indicated by `--write_json`, which is ).

        # In order to generate the JSON output:
        # Uncomment the commented lines starting by `--write_json` and `--display 0`

    # Step 2 - Running JSON output to get accuracy
        # Once you have the JSON files, run them with the script Tomas prepared for it, which in my case I use:
        # From Matlab, `cd /media/posefs3b/Users/gines/openpose_train/dataset/hand_testing`
        # Run `b_keypointJsonToMatAndImage` to generate your new file (you can run the current code to try it, I commented everything but test 1)
        # Run `c_plot_save_results` to plot and save the results. Just modify `models` and `texts` with your new model path and desired name.

# Clear terminal screen
clear && clear



# Fix paths
HAND_TESTING_FOLDER="/media/posefs3b/Users/gines/openpose_train/dataset/hand_testing/"
IMAGES_FOLDER=${HAND_TESTING_FOLDER}"0_images/"
IMAGES_BB_FOLDER=${HAND_TESTING_FOLDER}"3_images_bounding_box"
HAND_GROUND_TRUTH_FOLDER=${HAND_TESTING_FOLDER}"4_hand_detections"
PEOPLE_JSON_FOLDER=${HAND_TESTING_FOLDER}"5_keypointJson/"

# Variable paths
SCALES=6
SUFFIX="_${SCALES}"
HAND_RESULTS_FOLDER_BASE=${PEOPLE_JSON_FOLDER}"hand_keypoints_estimated"
HAND_RESULTS_FOLDER_NO_BB=${HAND_RESULTS_FOLDER_BASE}"_old"${SUFFIX}
HAND_RESULTS_FOLDER_BB=${HAND_RESULTS_FOLDER_BASE}"_BBox"${SUFFIX}
HAND_RESULTS_FOLDER_BODY_59=${HAND_RESULTS_FOLDER_BASE}"_BODY_59"



# Given bounding box
echo "Output on ${HAND_RESULTS_FOLDER_BB}"
rm -rf $HAND_RESULTS_FOLDER_BB
# 1 scale
./build/examples/tests/handFromJsonTest.bin \
    --hand_scale_number ${SCALES} --hand_scale_range 0.4 \
    --image_dir ${IMAGES_BB_FOLDER} \
    --hand_ground_truth ${HAND_GROUND_TRUTH_FOLDER} \
    --write_json $HAND_RESULTS_FOLDER_BB \
    --display 0



# No bounding box
echo "Output on ${HAND_RESULTS_FOLDER_NO_BB}"
rm -rf $HAND_RESULTS_FOLDER_NO_BB
# 1 scale
./build/examples/openpose/openpose.bin \
    --hand \
    --hand_scale_number ${SCALES} --hand_scale_range 0.4 \
    --image_dir ${IMAGES_FOLDER} \
    --write_json $HAND_RESULTS_FOLDER_NO_BB \
    --display 0



# No bounding box BODY_59
echo "Output on ${HAND_RESULTS_FOLDER_BODY_59}"
rm -rf $HAND_RESULTS_FOLDER_BODY_59
# 1 scale
./build/examples/openpose/openpose.bin \
    --model_pose BODY_59 \
    --image_dir ${IMAGES_FOLDER} \
    --write_json $HAND_RESULTS_FOLDER_BODY_59 \
    --display 0
