# Script for internal use. We might completely change it continuously and we will not answer questions about it.

# # Go back to main folder
# cd ../../

# Performance results
PROTOTXT_PATH=/home/gines/Dropbox/Perceptual_Computing_Lab/openpose/openpose/models/pose/coco/pose_deploy_linevec.prototxt

gedit $0
# First: Add 656 x 368 as input_dim in:
gedit $PROTOTXT_PATH
./3rdparty/caffe/build/tools/caffe time -model $PROTOTXT_PATH -gpu 0 -phase TEST
gedit $PROTOTXT_PATH
