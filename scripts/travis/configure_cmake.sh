#!/bin/bash

# CMake configuration

mkdir build
cd build

echo "RUN_EXAMPLES = ${RUN_EXAMPLES}."
if [[ $RUN_EXAMPLES == true ]] ; then
  ARGS="-DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF"
else
  ARGS="-DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF"
  # ARGS="-DBUILD_CAFFE=ON -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF -DDOWNLOAD_BODY_MPI_MODEL=OFF"
fi

echo "WITH_PYTHON = ${WITH_PYTHON}."
if [[ $WITH_PYTHON == true ]] ; then
  ARGS="$ARGS -DBUILD_PYTHON=On"
fi

# CUDA version
echo "WITH_CUDA = ${WITH_CUDA}."
echo "WITH_OPEN_CL = ${WITH_OPEN_CL}."
echo "WITH_MKL = ${WITH_MKL}."
if [[ $WITH_CUDA == true ]] ; then
  # Only build SM50
  ARGS="$ARGS -DGPU_MODE=CUDA -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=\"52\" -DCUDA_ARCH_PTX=\"\""
# OpenCL version
elif [[ $WITH_OPEN_CL == true ]] ; then
  echo "OpenCL version not implemented for Travis Build yet."
  exit 99
# CPU version
else
  ARGS="$ARGS -DGPU_MODE=CPU_ONLY"
  # MKL (Intel Caffe)
  if [[ $WITH_MKL == true ]] ; then
    ARGS="$ARGS -DUSE_MKL=On"
  else
    ARGS="$ARGS -DUSE_MKL=Off"
  fi
fi

echo "WITH_CUDNN = ${WITH_CUDNN}."
if [[ $WITH_CUDNN == true ]] ; then
  ARGS="$ARGS -DUSE_CUDNN=On"
else
  ARGS="$ARGS -DUSE_CUDNN=Off"
fi

echo "ARGS = ${ARGS}."

cmake .. $ARGS
