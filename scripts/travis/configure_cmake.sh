#!/bin/bash

# CMake configuration

mkdir build
cd build

ARGS="-DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF"
# ARGS="-DBUILD_CAFFE=ON -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF -DDOWNLOAD_BODY_MPI_MODEL=OFF"

if [ $WITH_PYTHON ] ; then
  ARGS="$ARGS -DBUILD_PYTHON=On"
fi

# CUDA version
if [ $WITH_CUDA ] ; then
  # Only build SM50
  ARGS="$ARGS -DGPU_MODE=CUDA -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=\"52\" -DCUDA_ARCH_PTX=\"50\""
# OpenCL version
elif [ $WITH_OPEN_CL ] ; then
  echo "OpenCL version not implemented for Travis Build yet."
  exit 99
# CPU version
else
  ARGS="$ARGS -DGPU_MODE=CPU_ONLY"
  # MKL (Intel Caffe)
  if [ $WITH_MKL ] ; then
    ARGS="$ARGS -DUSE_MKL=On"
  else
    ARGS="$ARGS -DUSE_MKL=Off"
  fi
fi

if [ $WITH_CUDNN ] ; then
  ARGS="$ARGS -DUSE_CUDNN=On"
else
  ARGS="$ARGS -DUSE_CUDNN=Off"
fi

cmake .. $ARGS