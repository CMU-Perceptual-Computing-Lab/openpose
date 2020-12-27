#!/bin/bash

### INSTALL PREREQUISITES

# Install CUDA 8.0
source scripts/ubuntu/install_cuda.sh

# Install cuDNN 5.1
source scripts/ubuntu/install_cudnn.sh

# Caffe prerequisites
source scripts/ubuntu/install_deps.sh
