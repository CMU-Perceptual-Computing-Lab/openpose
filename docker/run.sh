#!/bin/sh

xhost +local:root

nvidia-docker run \
  -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/${PWD}/../models:/root/openpose/models" \
  --device /dev/video0:/dev/video0 \
  openpose ./build/examples/openpose/rtpose.bin

xhost -local:root
