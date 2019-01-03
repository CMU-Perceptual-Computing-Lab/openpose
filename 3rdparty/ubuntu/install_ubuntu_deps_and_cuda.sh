#!/bin/bash

### INSTALL PREREQUISITES

# Install CUDA 8.0
bash install_cuda.sh

# Install cuDNN 5.1
bash install_cudnn.sh

# Caffe prerequisites
bash install_ubuntu_deps.sh
