#!/bin/sh

xhost +local:root

docker run \
  -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --device /dev/video0:/dev/video0 \
  cmupcl/openpose:cpu ./build/examples/openpose/rtpose.bin

xhost -local:root

# TODO: add a CPU version of openpose

# TODO: instruct users run get_models locally on host
# then mount them into the container using volumes
# --volume="/${PWD}/../models:/root/openpose/models" \
