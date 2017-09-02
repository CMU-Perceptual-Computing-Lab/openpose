# Script for internal use. We might completely change it continuously and we will not answer questions about it.

# USAGE EXAMPLE
# clear && clear && make all -j24 && bash ./examples/tests/pose_accuracy_coco_test.sh

# # Go back to main folder
# cd ../../

# Write COCO-format JSON
# Note: `--frame_last 3558` --> total = 3559 frames
# Last id:
    # ID 20671  <-->    #frames = 1471      -->     ~ 1.5 min at 15fps
    # ID 50006  <-->    #frames = 3559      -->     ~ 4 min at 15fps

    # 1 scale
./build/examples/openpose/openpose.bin --image_dir "/home/gines/devel/images/val2014" --write_coco_json ../evaluation/coco/results/openpose/1.json --no_display --render_pose 0 --frame_last 3558

#     # 3 scales
# ./build/examples/openpose/openpose.bin --image_dir "/home/gines/devel/images/val2014" --write_coco_json ../evaluation/coco/results/openpose/1_3.json --no_display --render_pose 0 --scale_number 3 --scale_gap 0.25 --frame_last 3558

#     # 4 scales
# ./build/examples/openpose/openpose.bin --num_gpu 1 --image_dir "/home/gines/devel/images/val2014" --write_coco_json ../evaluation/coco/results/openpose/1_4.json --no_display --render_pose 0 --num_gpu 1 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736" --frame_last 3558

# Debugging - Rendered frames saved
#  ./build/examples/openpose/openpose.bin --image_dir "/home/gines/devel/images/val2014" --write_images ../evaluation/coco/results/openpose/frameOutput --no_display
