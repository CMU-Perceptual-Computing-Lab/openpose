#!/bin/bash

### INSTALL PREREQUISITES

ubuntu_version="$(lsb_release -r)"
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
if [[ $ubuntu_version == *"14."* ]]; then
  wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb"
  sudo dpkg --install cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
elif [[ $ubuntu_version == *"16."* ]]; then
  wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb"
  sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
fi
sudo apt-get update
sudo apt-get install cuda-8-0

# Install cuDNN 5.1
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget -c ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig

# Basic
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install build-essential
# General dependencies
sudo apt-get --assume-yes install libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get --assume-yes install --no-install-recommends libboost-all-dev
# Remaining dependencies, 14.04
sudo apt-get --assume-yes install libgflags-dev libgoogle-glog-dev liblmdb-dev
# Python libs
sudo -H pip install --upgrade numpy protobuf
# OpenCV 2.4 -> Added as option
# sudo apt-get --assume-yes install libopencv-dev
