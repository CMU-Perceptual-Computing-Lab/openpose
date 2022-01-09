#!/bin/bash

### INSTALL PREREQUISITES

UBUNTU_VERSION="$(lsb_release -r)"

# Basic
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install build-essential
# General dependencies
sudo apt-get --assume-yes install libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get --assume-yes install --no-install-recommends libboost-all-dev
# Remaining dependencies
sudo apt-get --assume-yes install libgflags-dev libgoogle-glog-dev
# LMDB is needed for Caffe training, but very likely not for inference-only
sudo apt-get --assume-yes install liblmdb-dev

# Python2 libs (Official Ubuntu support dropped after Ubuntu 20)
if [[ $UBUNTU_VERSION == *"14."* ]] || [[ $UBUNTU_VERSION == *"15."* ]] || [[ $UBUNTU_VERSION == *"16."* ]] || [[ $UBUNTU_VERSION == *"17."* ]] || [[ $UBUNTU_VERSION == *"18."* ]]; then
    echo "Installing Python2 libs..."
    sudo apt-get --assume-yes install python-setuptools python-dev build-essential
    hash pip2 2> /dev/null || sudo apt-get --assume-yes install python-pip
    sudo -H python2 -m pip install pip --upgrade
    if [[ $CI == true ]]; then
        sudo -H python2 -m pip install --upgrade "numpy<1.17" protobuf
        python2 -m pip install --user "opencv-python<4.3"
    else
        sudo -H python2 -m pip install --upgrade "numpy<1.17" protobuf "opencv-python<4.3"
    fi
fi
# Python3 libs
echo "Installing Python3 libs..."
sudo apt-get --assume-yes install python3-setuptools python3-dev build-essential
hash pip3 2> /dev/null || sudo apt-get --assume-yes install python3-pip
sudo -H python3 -m pip install pip --upgrade
if [[ $CI == true ]]; then
    sudo -H python3 -m pip install --upgrade numpy protobuf
    python3 -m pip install --user opencv-python
else
    sudo -H python3 -m pip install --upgrade numpy protobuf opencv-python
fi

# OpenCL Generic (Official OpenPose support dropped after Ubuntu 20)
if [[ $UBUNTU_VERSION == *"14."* ]] || [[ $UBUNTU_VERSION == *"15."* ]] || [[ $UBUNTU_VERSION == *"16."* ]] || [[ $UBUNTU_VERSION == *"17."* ]] || [[ $UBUNTU_VERSION == *"18."* ]]; then
	sudo apt-get --assume-yes install opencl-headers ocl-icd-opencl-dev
	sudo apt-get --assume-yes install libviennacl-dev
fi
# OpenCV 2.4 -> Added as option
# # sudo apt-get --assume-yes install libopencv-dev
