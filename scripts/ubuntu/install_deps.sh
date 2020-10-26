#!/bin/bash

### INSTALL PREREQUISITES
ubuntu_version="$(lsb_release -r)"

# Basic
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install build-essential
# General dependencies
sudo apt-get --assume-yes install libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get --assume-yes install --no-install-recommends libboost-all-dev
# Remaining dependencies, 14.04
sudo apt-get --assume-yes install libgflags-dev libgoogle-glog-dev liblmdb-dev
if [[ $ubuntu_version != *"20."* ]]; then
  # Python2 libs - Skip due to Ubuntu 20 dropped Python2
  sudo apt-get --assume-yes install python-setuptools python-dev build-essential
  sudo easy_install pip
  sudo -H pip install --upgrade numpy protobuf opencv-python
fi
# Python3 libs
sudo apt-get --assume-yes install python3-setuptools python3-dev build-essential
sudo apt-get --assume-yes install python3-pip
sudo -H pip3 install --upgrade numpy protobuf opencv-python
# OpenCV 2.4 -> Added as option
# # sudo apt-get --assume-yes install libopencv-dev
# OpenCL Generic
sudo apt-get --assume-yes install opencl-headers ocl-icd-opencl-dev
sudo apt-get --assume-yes install libviennacl-dev
