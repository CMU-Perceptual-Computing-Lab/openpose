# Script for internal use. We might completely change it continuously and we will not answer questions about it.

# # Go back to main folder
# cd ../../

# Re-build
clear && clear && make all -j12

# Performance results (~1400)
./build/examples/openpose/openpose.bin --video soccer.mp4 --frame_last 1500
# Including 2nd graphics card (~3500)
# ./build/examples/openpose/openpose.bin --video soccer.mp4 --frame_last 3750
