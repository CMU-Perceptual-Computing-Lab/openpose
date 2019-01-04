#!/bin/bash

### INSTALL PREREQUISITES

# Install CUDA 8.0
bash 3rdparty/ubuntu/install_cuda.sh

# Install cuDNN 5.1
bash 3rdparty/ubuntu/install_cudnn.sh

# Caffe prerequisites
bash 3rdparty/ubuntu/install_ubuntu_deps.sh
