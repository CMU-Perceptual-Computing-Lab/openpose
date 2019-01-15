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
if [[ $WITH_PYTHON == true && "$TRAVIS_OS_NAME" == "linux" ]] ; then
  ARGS="$ARGS -DBUILD_PYTHON=On -DPYTHON_EXECUTABLE=/usr/bin/python2.7 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7m.so"
fi
if [[ $WITH_PYTHON == true && "$TRAVIS_OS_NAME" == "osx" ]] ; then
  ARGS="$ARGS -DBUILD_PYTHON=On -DPYTHON_EXECUTABLE=/usr/local/bin/python2.7 -DPYTHON_LIBRARY=/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7m.dylib"
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

# Run Cmake twice for pybind to register
if [[ $WITH_PYTHON == true ]] ; then
  cmake .. $ARGS
fi
