#!/bin/bash

# Raw Makefile configuration
# # All in one line
# bash scripts/ubuntu_deprecated/install_caffe_and_openpose_if_cuda8.sh

if [[ $MAKE_CONFIG_COMPATIBLE == false ]] ; then
  echo "Configuration not compatible for Makefile."
  exit 99
fi

LINE () {
  echo "$@" >> Makefile.config
  echo "$@" >> 3rdparty/caffe/Makefile.config
}

# Install Caffe
echo "Installing Caffe..."
cd ./3rdparty/caffe
bash install_caffe_if_cuda8.sh
cd ../..

# Generate Makefile files for OpenPose
echo "Generating Makefile files for OpenPose..."
bash scripts/ubuntu_deprecated/copy_makefile_files.sh

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
