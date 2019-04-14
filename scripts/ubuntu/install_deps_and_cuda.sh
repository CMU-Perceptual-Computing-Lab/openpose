#!/bin/bash

### INSTALL PREREQUISITES

# Install CUDA 8.0
bash scripts/ubuntu/install_cuda.sh

# Install cuDNN 5.1
bash scripts/ubuntu/install_cudnn.sh

# Caffe prerequisites
bash scripts/ubuntu/install_deps.sh
