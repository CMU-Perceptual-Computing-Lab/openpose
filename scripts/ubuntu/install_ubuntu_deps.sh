#!/bin/bash

### INSTALL PREREQUISITES

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
