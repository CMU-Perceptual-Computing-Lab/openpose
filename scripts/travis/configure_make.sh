#!/bin/bash

# Raw Makefile configuration
# # All in one line
# bash 3rdparty/ubuntu_deprecated/install_caffe_and_openpose_if_cuda8.sh

if [[ $MAKE_CONFIG_COMPATIBLE == false ]] ; then
  echo "Configuration not compatible for Makefile."
  exit 99
fi

LINE () {
  echo "$@" >> Makefile.config
  echo "$@" >> 3rdparty/caffe/Makefile.config
}

# Install Caffe
cd ./3rdparty/caffe
bash install_caffe_if_cuda8.sh
cd ../..

# Generate Makefile files for OpenPose
bash 3rdparty/ubuntu_deprecated/copy_makefile_files.sh

# Modifying Makefile.config file
echo "WITH_PYTHON = ${WITH_PYTHON}."
if [[ $WITH_PYTHON == true ]] ; then
  # TODO(lukeyeager) this path is currently disabled because of test errors like:
  #   ImportError: dynamic module does not define init function (PyInit__caffe)
  LINE "PYTHON_LIBRARIES := python3.4m boost_python-py34"
  LINE "PYTHON_INCLUDE := /usr/include/python3.4 /usr/lib/python3/dist-packages/numpy/core/include"
  LINE "INCLUDE_DIRS := \$(INCLUDE_DIRS) \$(PYTHON_INCLUDE)"
fi

echo "WITH_CUDA = ${WITH_CUDA}."
if [[ $WITH_CUDA == true ]] ; then
  # Only build SM50
  LINE "CUDA_ARCH := -gencode arch=compute_50,code=sm_50"
else
  LINE "CPU_ONLY := 1"
fi

echo "WITH_CUDNN = ${WITH_CUDNN}."
if [[ $WITH_CUDNN == true ]] ; then
  LINE "USE_CUDNN := 1"
fi
